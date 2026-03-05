"""
realtime_v2.py - Realtime VSL Recognition (208 dim — với finger curl)
======================================================================
BƯỚC 4 trong pipeline

Thay đổi so với v3:
  - Dùng model 208 dim (thêm finger curl)
  - Detect emotion tự động từ mặt (FacialExpressionAnalyzer)
  - Hiển thị finger curl realtime: thấy trực tiếp ngón nào đang co
  - Tương thích với train_bilstm_v2.py (HandAwareVSLClassifier)
    hoặc BiLSTMClassifier cũ

Phím tắt:
  1-7      : Gán emotion thủ công (override auto-detect)
  A        : Toggle auto emotion detect
  C        : Toggle hiển thị curl debug
  +/-      : Tăng/giảm ngưỡng confidence
  F        : Toggle fullscreen
  R        : Reset buffer
  Q/ESC    : Thoát

Chạy:
  python realtime_v4.py
  python realtime_v4.py --model checkpoints/bilstm_v4_best.pt
  python realtime_v4.py --model checkpoints/bilstm_v3_best.pt --feat_dim 178
"""

import os
import json
import math
import time
import argparse
import threading
import collections
import numpy as np
import torch
import torch.nn as nn
import cv2
from pathlib import Path
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request

from collector import (
    FullBodyDrawer, draw_text_bg,
    FacialExpressionAnalyzer,
    FramingChecker,
)


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

class Config:
    SEQ_LEN  = 64

    # v2: 208 dim | v1: 178 dim (backward compatible)
    FEAT_DIM_V2 = 208
    FEAT_DIM_V1 = 178

    POSE_KEY_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    EMOTIONS     = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    EMOTION_COLORS = {
        "angry":    (0,   0,   255),
        "disgust":  (0,   128, 128),
        "fear":     (128, 0,   200),
        "happy":    (0,   255, 100),
        "sad":      (255, 150, 0),
        "surprise": (0,   200, 255),
        "neutral":  (180, 180, 180),
    }

    FINGER_ORDER = ['thumb', 'index', 'middle', 'ring', 'pinky']
    FINGER_JOINTS = {
        'thumb':  (1, 2, 3, 4),
        'index':  (5, 6, 7, 8),
        'middle': (9, 10, 11, 12),
        'ring':   (13, 14, 15, 16),
        'pinky':  (17, 18, 19, 20),
    }
    WRIST_IDX  = 0
    MIDDLE_MCP = 9

    CONF_THRESHOLD_DEFAULT = 0.6
    CONF_THRESHOLD_MIN     = 0.3
    CONF_THRESHOLD_MAX     = 0.95
    CONF_THRESHOLD_STEP    = 0.05
    STABLE_FRAMES = 3
    TOP_K = 5

    EXPR_TO_EMOTION = {
        "happy":     "happy",   "sad":       "sad",
        "angry":     "angry",   "surprised": "surprise",
        "surprise":  "surprise","disgusted": "disgust",
        "disgust":   "disgust", "fearful":   "fear",
        "fear":      "fear",    "neutral":   "neutral",
        "unknown":   "neutral",
    }


cfg = Config()

_PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_URLS = {
    'hand_landmarker.task':
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    'pose_landmarker_heavy.task':
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
    'face_landmarker.task':
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
}

def ensure_model(name):
    path = _PROJECT_ROOT / name
    if not path.exists():
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(MODEL_URLS[name], str(path))
    return str(path)


# ═══════════════════════════════════════════════════════════════════
# FINGER CURL CALCULATOR (nhất quán với video_to_npy_v2.py)
# ═══════════════════════════════════════════════════════════════════

def compute_angle(a, b, c):
    ba = a - b; bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.arccos(np.clip(cos_a, -1.0, 1.0))


def compute_finger_curl_features(landmarks: np.ndarray) -> np.ndarray:
    """
    landmarks: (21, 3)
    returns:   (15,) — 5 ngón × [curl_ratio, bend_angle, tip_dist_norm]
    """
    curl_feats = np.zeros(15, dtype=np.float32)
    wrist   = landmarks[cfg.WRIST_IDX]
    mid_mcp = landmarks[cfg.MIDDLE_MCP]
    scale   = np.linalg.norm(mid_mcp - wrist) + 1e-8

    for i, finger in enumerate(cfg.FINGER_ORDER):
        mcp_i, pip_i, dip_i, tip_i = cfg.FINGER_JOINTS[finger]
        mcp = landmarks[mcp_i]; pip = landmarks[pip_i]
        dip = landmarks[dip_i]; tip = landmarks[tip_i]

        dist_wrist_tip = np.linalg.norm(tip - wrist)
        dist_wrist_mcp = np.linalg.norm(mcp - wrist) + 1e-8
        curl_ratio     = np.clip(dist_wrist_tip / dist_wrist_mcp / 2.0, 0.0, 1.0)

        bend_angle = compute_angle(mcp, pip, dip)
        bend_norm  = np.clip((np.pi - bend_angle) / np.pi, 0.0, 1.0)

        tip_dist_norm = np.clip(dist_wrist_tip / scale / 3.0, 0.0, 1.0)

        base = i * 3
        curl_feats[base]     = curl_ratio
        curl_feats[base + 1] = bend_norm
        curl_feats[base + 2] = tip_dist_norm

    return curl_feats


# ═══════════════════════════════════════════════════════════════════
# MODEL (tự động detect v3 BiLSTM hoặc v4 HandAware)
# ═══════════════════════════════════════════════════════════════════

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, lstm_out):
        scores  = self.attn(lstm_out).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)
        return context, weights


class BiLSTMClassifier(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers, num_classes,
                 dropout_lstm=0.3, dropout_fc=0.4,
                 bidirectional=True, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.num_dirs = 2 if bidirectional else 1
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(0.1))
        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout_lstm if num_layers > 1 else 0,
            bidirectional=bidirectional)
        lstm_out_dim = hidden_dim * self.num_dirs
        if use_attention:
            self.attention = AttentionLayer(lstm_out_dim)
            fc_in = lstm_out_dim * 2
        else:
            fc_in = lstm_out_dim
        mid = max(num_classes * 4, 128)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fc_in),
            nn.Linear(fc_in, mid),    nn.GELU(), nn.Dropout(dropout_fc),
            nn.Linear(mid, mid // 2), nn.GELU(), nn.Dropout(dropout_fc / 2),
            nn.Linear(mid // 2, num_classes))

    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, (hn, _) = self.lstm(x)
        last_h = (torch.cat([hn[-2], hn[-1]], dim=-1)
                  if self.num_dirs == 2 else hn[-1])
        if self.use_attention:
            ctx, _ = self.attention(lstm_out)
            feat = torch.cat([ctx, last_h], dim=-1)
        else:
            feat = last_h
        return self.classifier(feat)


def load_model(ckpt_path, device):
    ckpt       = torch.load(ckpt_path, map_location=device)
    label_map  = ckpt['label_map']
    model_cfg  = ckpt.get('cfg', {})
    model_type = ckpt.get('model_type', 'BiLSTMClassifier')
    feat_dim   = model_cfg.get('FEAT_DIM', cfg.FEAT_DIM_V2)

    if model_type == 'HandAwareVSLClassifier':
        # Import từ train_bilstm_v4
        try:
            sys_path = str(_PROJECT_ROOT)
            import sys
            if sys_path not in sys.path:
                sys.path.insert(0, sys_path)
            from train_bilstm_v4 import HandAwareVSLClassifier, Config as CfgV4
            cfgv4 = CfgV4()
            cfgv4.FINGER_DIM      = model_cfg.get('FINGER_DIM',      32)
            cfgv4.TEMPORAL_DIM    = model_cfg.get('TEMPORAL_DIM',    256)
            cfgv4.TEMPORAL_HEADS  = model_cfg.get('TEMPORAL_HEADS',  8)
            cfgv4.TEMPORAL_LAYERS = model_cfg.get('TEMPORAL_LAYERS', 4)
            cfgv4.GRAPH_HEADS     = model_cfg.get('GRAPH_HEADS',     4)
            cfgv4.FEAT_DIM        = feat_dim
            model = HandAwareVSLClassifier(num_classes=len(label_map), cfg=cfgv4)
            print("  [Model] HandAwareVSLClassifier v4")
        except ImportError:
            print("  [WARN] Không tìm thấy train_bilstm_v4.py, dùng BiLSTM fallback")
            model = BiLSTMClassifier(
                feat_dim=feat_dim,
                hidden_dim=model_cfg.get('HIDDEN_DIM', 256),
                num_layers=model_cfg.get('NUM_LAYERS', 3),
                num_classes=len(label_map))
    else:
        model = BiLSTMClassifier(
            feat_dim=feat_dim,
            hidden_dim=model_cfg.get('HIDDEN_DIM', 256),
            num_layers=model_cfg.get('NUM_LAYERS', 3),
            num_classes=len(label_map))
        print(f"  [Model] BiLSTMClassifier (feat_dim={feat_dim})")

    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    idx2label = {v: k for k, v in label_map.items()}
    print(f"  [Model] Labels: {list(label_map.keys())}")
    print(f"  [Model] Val acc: {ckpt.get('val_acc', 0)*100:.2f}%")
    print(f"  [Model] Feat dim: {feat_dim}")
    return model, idx2label, feat_dim


# ═══════════════════════════════════════════════════════════════════
# REALTIME EXTRACTOR (v2 — thêm curl + face)
# ═══════════════════════════════════════════════════════════════════

class RealtimeExtractor:
    def __init__(self):
        print("  Init RealtimeExtractor v2...")
        self._latest = {
            'pose': None, 'hands': [], 'handedness': [],
            'face': None, 'blendshapes': None,
        }
        self._ts   = 0
        self._lock = threading.Lock()

        self.hand_detector = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=ensure_model('hand_landmarker.task')),
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=self._on_hand))

        self.pose_detector = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=ensure_model('pose_landmarker_heavy.task')),
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=self._on_pose))

        self.face_detector = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=ensure_model('face_landmarker.task')),
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,
                result_callback=self._on_face))

        print("  RealtimeExtractor v2 ready!")

    def _on_pose(self, result, image, ts):
        with self._lock:
            self._latest['pose'] = (result.pose_landmarks[0]
                                    if result.pose_landmarks else None)

    def _on_hand(self, result, image, ts):
        with self._lock:
            hands = []
            hdness = []
            if result.hand_landmarks and result.handedness:
                for i, hand in enumerate(result.hand_landmarks):
                    if i < len(result.handedness):
                        hands.append(hand)
                        hdness.append(result.handedness[i])
            self._latest['hands']      = hands
            self._latest['handedness'] = hdness

    def _on_face(self, result, image, ts):
        with self._lock:
            self._latest['face'] = (result.face_landmarks[0]
                                    if result.face_landmarks else None)
            self._latest['blendshapes'] = (result.face_blendshapes[0]
                                           if result.face_blendshapes else None)

    def send_frame(self, rgb):
        self._ts += 33
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        for det in [self.pose_detector, self.hand_detector, self.face_detector]:
            try:
                det.detect_async(mp_img, self._ts)
            except Exception:
                pass

    def extract_features_v2(self):
        """
        Extract (201,) = pose(45) + hands_xyz(126) + curl(30).
        Emotion ghép sau.
        """
        with self._lock:
            feat = np.zeros(201, dtype=np.float32)

            # Pose (0:45)
            if self._latest['pose']:
                for i, idx in enumerate(cfg.POSE_KEY_INDICES):
                    lm = self._latest['pose'][idx]
                    feat[i*3:(i+1)*3] = [lm.x, lm.y, lm.z]

            hands  = self._latest['hands']
            if hands:
                sorted_hands = sorted(hands,
                    key=lambda h: np.mean([lm.x for lm in h]))
                xyz_slots  = [45, 108]
                curl_slots = [171, 186]
                for xyz_s, curl_s, hand in zip(
                        xyz_slots, curl_slots, sorted_hands[:2]):
                    # XYZ
                    for j, lm in enumerate(hand):
                        feat[xyz_s + j*3: xyz_s + j*3 + 3] = [lm.x, lm.y, lm.z]
                    # Curl
                    lms_arr = np.array([[lm.x, lm.y, lm.z] for lm in hand])
                    feat[curl_s: curl_s + 15] = compute_finger_curl_features(lms_arr)

            return feat, {
                'hands':      list(self._latest['hands']),
                'pose':       self._latest['pose'],
                'face':       self._latest['face'],
                'blendshapes': self._latest['blendshapes'],
            }

    def extract_features_v1(self):
        """Backward compat: extract (171,) — dùng khi model 178 dim."""
        with self._lock:
            feat = np.zeros(171, dtype=np.float32)
            if self._latest['pose']:
                for i, idx in enumerate(cfg.POSE_KEY_INDICES):
                    lm = self._latest['pose'][idx]
                    feat[i*3:(i+1)*3] = [lm.x, lm.y, lm.z]
            hands = self._latest['hands']
            if hands:
                sorted_hands = sorted(hands,
                    key=lambda h: np.mean([lm.x for lm in h]))
                for slot_start, hand in zip([45, 108], sorted_hands[:2]):
                    for j, lm in enumerate(hand):
                        feat[slot_start + j*3: slot_start + j*3 + 3] = [lm.x, lm.y, lm.z]
            return feat, {
                'hands':      list(self._latest['hands']),
                'pose':       self._latest['pose'],
                'face':       self._latest['face'],
                'blendshapes': self._latest['blendshapes'],
            }

    def has_hands(self):
        with self._lock:
            return len(self._latest['hands']) > 0

    def get_curl_for_display(self):
        """Lấy curl values của tay hiện tại để hiển thị debug."""
        with self._lock:
            if not self._latest['hands']:
                return None
            hand = self._latest['hands'][0]
            lms  = np.array([[lm.x, lm.y, lm.z] for lm in hand])
            return compute_finger_curl_features(lms)   # (15,)

    def close(self):
        self.hand_detector.close()
        self.pose_detector.close()
        self.face_detector.close()


# ═══════════════════════════════════════════════════════════════════
# NORMALIZE (nhất quán với video_to_npy_v2)
# ═══════════════════════════════════════════════════════════════════

def normalize_frame_v2(feat: np.ndarray) -> np.ndarray:
    """Normalize pose + hands xyz. Curl (171:201) không chạm."""
    f = feat.copy()
    pose = f[:45].reshape(-1, 3)
    if np.any(pose != 0):
        hip_mid = (pose[13] + pose[14]) / 2
        pose    = pose - hip_mid
        sd      = np.linalg.norm(pose[1] - pose[2])
        if sd > 1e-6:
            pose = pose / sd
        f[:45] = pose.flatten()
    for start, end in [(45, 108), (108, 171)]:
        hand = f[start:end].reshape(-1, 3)
        if np.any(hand != 0):
            hand  = hand - hand[0]
            scale = np.linalg.norm(hand[9])
            if scale > 1e-6:
                hand = hand / scale
            f[start:end] = hand.flatten()
    return f


def normalize_frame_v1(feat):
    f = feat.copy()
    pose = f[:45].reshape(-1, 3)
    if np.any(pose != 0):
        hip_mid = (pose[13] + pose[14]) / 2
        pose   -= hip_mid
        s = np.linalg.norm(pose[1] - pose[2])
        if s > 1e-6: pose /= s
        f[:45] = pose.flatten()
    for start, end in [(45, 108), (108, 171)]:
        hand = f[start:end].reshape(-1, 3)
        if np.any(hand != 0):
            hand -= hand[0]
            s = np.linalg.norm(hand[9])
            if s > 1e-6: hand /= s
            f[start:end] = hand.flatten()
    return f


# ═══════════════════════════════════════════════════════════════════
# COLORS & UI
# ═══════════════════════════════════════════════════════════════════

class Colors:
    BG_DARK    = (15,  15,  15)
    BG_PANEL   = (30,  30,  30)
    BG_CARD    = (45,  45,  45)
    TEXT_PRI   = (255, 255, 255)
    TEXT_SEC   = (180, 180, 180)
    TEXT_MUTED = (100, 100, 100)
    PRIMARY    = (0,   200, 150)
    SUCCESS    = (0,   255, 100)
    WARNING    = (0,   200, 255)
    DANGER     = (0,   100, 255)


def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1):
    x1, y1 = pt1; x2, y2 = pt2
    if thickness == -1:
        cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, -1)
        for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                       (x1+radius, y2-radius), (x2-radius, y2-radius)]:
            cv2.circle(img, (cx, cy), radius, color, -1)


def draw_finger_curl_panel(img, curl_feats, x, y, panel_w=220):
    """
    Vẽ thanh curl cho từng ngón tay.
    curl_feats: (15,) — [curl_ratio, bend_angle, tip_dist] × 5 ngón
    curl_ratio gần 0 = ngón co (màu đỏ), gần 1 = ngón duỗi (màu xanh)
    """
    font    = cv2.FONT_HERSHEY_SIMPLEX
    bar_w   = panel_w - 80
    bar_h   = 10
    fingers = ['Cai', 'Tro', 'Giua', 'Nhan', 'Ut']

    draw_rounded_rect(img, (x, y), (x + panel_w, y + 10 + len(fingers) * 28),
                      Colors.BG_PANEL, radius=8)

    cv2.putText(img, "CURL NGON TAY", (x + 8, y + 16),
                font, 0.38, Colors.TEXT_MUTED, 1)

    for i, fname in enumerate(fingers):
        base         = i * 3
        curl_ratio   = float(curl_feats[base])      # 0=co, 1=duoi
        bend_angle   = float(curl_feats[base + 1])  # 0=thang, 1=co

        bar_y = y + 22 + i * 28

        # Label
        cv2.putText(img, fname, (x + 8, bar_y + 9),
                    font, 0.38, Colors.TEXT_SEC, 1)

        # Background bar
        bx = x + 42
        cv2.rectangle(img, (bx, bar_y), (bx + bar_w, bar_y + bar_h),
                      (60, 60, 60), -1)

        # Fill: curl_ratio (0=co→đỏ, 1=duỗi→xanh)
        fill_w = int(bar_w * curl_ratio)
        r = int(255 * (1 - curl_ratio))
        g = int(200 * curl_ratio)
        bar_color = (0, g, r)
        if fill_w > 0:
            cv2.rectangle(img, (bx, bar_y), (bx + fill_w, bar_y + bar_h),
                          bar_color, -1)

        # Co/Duỗi label
        state_txt = "DUOI" if curl_ratio > 0.6 else ("CO" if curl_ratio < 0.4 else "...")
        state_col = Colors.SUCCESS if curl_ratio > 0.6 else \
                    Colors.DANGER  if curl_ratio < 0.4 else Colors.TEXT_MUTED
        cv2.putText(img, state_txt, (bx + bar_w + 5, bar_y + 9),
                    font, 0.32, state_col, 1)


def draw_confidence_bar(img, x, y, width, height, conf, threshold, label, rank):
    font = cv2.FONT_HERSHEY_SIMPLEX
    draw_rounded_rect(img, (x, y), (x+width, y+height), Colors.BG_CARD, radius=5)
    bar_w  = int((width-20) * conf)
    bar_x  = x + 10
    bar_y  = y + height - 15
    color  = (Colors.SUCCESS if conf >= threshold else
              Colors.WARNING if conf >= threshold * 0.7 else Colors.DANGER)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+width-20, bar_y+8), (60,60,60), -1)
    if bar_w > 0:
        cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+8), color, -1)
    thresh_x = bar_x + int((width-20) * threshold)
    cv2.line(img, (thresh_x, bar_y-2), (thresh_x, bar_y+10), Colors.WARNING, 2)
    rank_colors = [Colors.SUCCESS, Colors.PRIMARY, Colors.WARNING,
                   Colors.TEXT_SEC, Colors.TEXT_MUTED]
    cv2.circle(img, (x+20, y+20), 11, rank_colors[min(rank, 4)], -1)
    cv2.putText(img, str(rank+1), (x+15, y+25), font, 0.45, Colors.BG_DARK, 2)
    lc = Colors.TEXT_PRI if conf >= threshold else Colors.TEXT_SEC
    cv2.putText(img, label, (x+38, y+26), font, 0.55, lc, 1 if rank > 0 else 2)
    cv2.putText(img, f"{conf*100:.1f}%", (x+width-65, y+26), font, 0.45, color, 1)


# ═══════════════════════════════════════════════════════════════════
# MAIN CLASS
# ═══════════════════════════════════════════════════════════════════

class RealtimeInferenceV4:

    def __init__(self, model_path, feat_dim_override=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n{'='*60}")
        print(" VSL REALTIME v4 ".center(60, "="))
        print("="*60)
        print(f"  Device: {self.device}")

        self.model, self.idx2label, self.feat_dim = load_model(model_path, self.device)
        if feat_dim_override:
            self.feat_dim = feat_dim_override

        self.use_v2 = (self.feat_dim == cfg.FEAT_DIM_V2)
        print(f"  Mode: {'v2 (208 dim — with curl)' if self.use_v2 else 'v1 (178 dim — no curl)'}")

        # Display names
        self.display_names = self._load_display_names()

        self.extractor = RealtimeExtractor()
        self.buffer    = deque(maxlen=cfg.SEQ_LEN)

        # Emotion state
        self.auto_emotion    = True      # True = detect từ mặt
        self.emotion_key     = 7
        self.emotion_name    = "neutral"
        self.emotion_buffer  = deque(maxlen=10)   # smooth emotion

        # Prediction state
        self.conf_threshold  = cfg.CONF_THRESHOLD_DEFAULT
        self.top_predictions = []
        self.current_label   = ""
        self.current_conf    = 0.0
        self.stable_count    = 0
        self.last_pred       = ""
        self.pred_history    = deque(maxlen=5)

        # Display flags
        self.is_fullscreen    = True
        self.show_curl_debug  = True
        self.show_mesh        = False

        # FPS
        self.fps         = 0
        self.frame_times = deque(maxlen=30)

    def _load_display_names(self):
        for p in ['data/processed_v2/display_names.json',
                  'data/processed/display_names.json',
                  'data/display_names.json']:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    return json.load(f)
        return {}

    # ── Emotion ─────────────────────────────────────────────

    def _detect_emotion_from_face(self, blendshapes, face_lms) -> str:
        expr = None
        if blendshapes:
            try:
                expr = FacialExpressionAnalyzer.analyze_blendshapes(blendshapes)
            except Exception:
                pass
        if expr is None and face_lms:
            try:
                expr = FacialExpressionAnalyzer.analyze_landmarks(face_lms, 640, 480)
            except Exception:
                pass
        if expr is None:
            return "neutral"
        raw = expr.get('expression_label', 'neutral')
        return cfg.EXPR_TO_EMOTION.get(raw.lower(), "neutral")

    def _smooth_emotion(self, emotion: str) -> str:
        self.emotion_buffer.append(emotion)
        return collections.Counter(self.emotion_buffer).most_common(1)[0][0]

    def _get_emotion_vec(self, emotion_name: str) -> np.ndarray:
        vec = np.zeros(7, dtype=np.float32)
        vec[cfg.EMOTIONS.index(emotion_name)] = 1.0
        return vec

    # ── Feature extraction ───────────────────────────────────

    def _build_full_feature(self, raw, emotion_name: str) -> np.ndarray:
        if self.use_v2:
            # (201,) → normalize → ghép emotion → (208,)
            feat_201  = raw['feat']
            feat_norm = normalize_frame_v2(feat_201)
            emo_vec   = self._get_emotion_vec(emotion_name)
            return np.concatenate([feat_norm, emo_vec])   # (208,)
        else:
            # (171,) → normalize → ghép emotion → (178,)
            feat_171  = raw['feat']
            feat_norm = normalize_frame_v1(feat_171)
            emo_vec   = self._get_emotion_vec(emotion_name)
            return np.concatenate([feat_norm, emo_vec])   # (178,)

    # ── Inference ────────────────────────────────────────────

    def process_frame(self, frame):
        t0 = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.extractor.send_frame(rgb)

        # Extract features
        if self.use_v2:
            feat, raw_data = self.extractor.extract_features_v2()
        else:
            feat, raw_data = self.extractor.extract_features_v1()
        raw_data['feat'] = feat

        # Emotion
        if self.auto_emotion:
            raw_emo     = self._detect_emotion_from_face(
                raw_data.get('blendshapes'), raw_data.get('face'))
            smooth_emo  = self._smooth_emotion(raw_emo)
            self.emotion_name = smooth_emo
        else:
            smooth_emo = self.emotion_name

        # Build full feature
        full_feat = self._build_full_feature(raw_data, smooth_emo)
        self.buffer.append(full_feat)

        if len(self.buffer) < cfg.SEQ_LEN:
            return "", 0.0, raw_data, []

        if not self.extractor.has_hands():
            self.buffer.clear()
            return "[NO HANDS]", 0.0, raw_data, []

        seq = np.array(list(self.buffer), dtype=np.float32)
        x   = torch.from_numpy(seq).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=-1)[0]
            top_k  = torch.topk(probs, min(cfg.TOP_K, len(probs)))
            top_predictions = [
                (self.idx2label[i.item()], c.item())
                for i, c in zip(top_k.indices, top_k.values)
            ]

        label = self.idx2label[top_k.indices[0].item()]
        conf  = top_k.values[0].item()

        self.top_predictions = top_predictions
        self.pred_history.append((label, conf))

        # Smooth prediction (majority vote)
        if len(self.pred_history) >= 3:
            counts = collections.Counter(lb for lb, c in self.pred_history
                                         if c >= self.conf_threshold)
            if counts:
                best = counts.most_common(1)[0][0]
                avg_conf = np.mean([c for lb, c in self.pred_history if lb == best])
                self.current_label = best
                self.current_conf  = float(avg_conf)

        self.frame_times.append(time.time() - t0)
        self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times) + 1e-8)

        return label, conf, raw_data, top_predictions

    # ── UI ───────────────────────────────────────────────────

    def draw_ui(self, frame, raw_data):
        h, w  = frame.shape[:2]
        font  = cv2.FONT_HERSHEY_SIMPLEX
        overlay = frame.copy()

        # Hand landmarks
        if raw_data.get('hands'):
            for hi, hand in enumerate(raw_data['hands']):
                col = (0, 255, 200) if hi == 0 else (255, 200, 0)
                pts = [(int(lm.x*w), int(lm.y*h)) for lm in hand]
                conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                         (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
                         (15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
                for a, b in conns:
                    if a < len(pts) and b < len(pts):
                        cv2.line(overlay, pts[a], pts[b], (80, 160, 120), 2)
                for i, pt in enumerate(pts):
                    r = 6 if i in [0,4,8,12,16,20] else 4
                    cv2.circle(overlay, pt, r, col, -1)

        # TOP BAR
        cv2.rectangle(overlay, (0,0), (w, 75), Colors.BG_PANEL, -1)
        cv2.putText(overlay, "VSL RECOGNITION v4", (15, 32), font, 0.8, Colors.PRIMARY, 2)

        # Emotion badge (top bar)
        emo_col = cfg.EMOTION_COLORS.get(self.emotion_name, (180,180,180))
        mode_txt = "AUTO" if self.auto_emotion else "MANUAL"
        draw_rounded_rect(overlay, (15, 42), (220, 68), (40,40,40), radius=6)
        cv2.putText(overlay, f"EMO [{mode_txt}]: {self.emotion_name.upper()}",
                    (22, 60), font, 0.42, emo_col, 1)

        cv2.putText(overlay, f"FPS:{self.fps:.0f}",
                    (w-90, 32), font, 0.5, Colors.TEXT_SEC, 1)
        buf_pct = len(self.buffer) / cfg.SEQ_LEN
        bx = w - 170; by = 48
        cv2.rectangle(overlay, (bx,by), (bx+150,by+10), (60,60,60), -1)
        cv2.rectangle(overlay, (bx,by),
                      (bx+int(150*buf_pct),by+10), Colors.PRIMARY, -1)
        cv2.putText(overlay, f"Buf:{len(self.buffer)}/{cfg.SEQ_LEN}",
                    (bx, by-4), font, 0.32, Colors.TEXT_MUTED, 1)

        # RIGHT PANEL — Predictions
        pw   = 310
        px   = w - pw - 15
        py   = 90
        ph   = h - 115
        draw_rounded_rect(overlay, (px, py), (w-15, py+ph), Colors.BG_PANEL, radius=12)
        cv2.putText(overlay, "PREDICTIONS", (px+12, py+28), font, 0.55, Colors.TEXT_PRI, 1)
        cv2.putText(overlay, f"Threshold: {self.conf_threshold*100:.0f}%  [+/-]",
                    (px+12, py+50), font, 0.38, Colors.WARNING, 1)
        cv2.line(overlay, (px+12, py+60), (w-28, py+60), (60,60,60), 1)

        for i, (lbl, cf) in enumerate(self.top_predictions):
            iy = py + 70 + i * 52
            draw_confidence_bar(overlay, px+8, iy, pw-20, 48,
                                cf, self.conf_threshold, lbl, i)

        # MAIN prediction card
        card_x, card_y = 15, 90
        card_w = px - 30
        card_h = 105
        draw_rounded_rect(overlay, (card_x, card_y),
                          (card_x+card_w, card_y+card_h), Colors.BG_PANEL, radius=12)

        if self.current_label and self.current_conf >= self.conf_threshold:
            disp = self.display_names.get(self.current_label, self.current_label)
            col  = Colors.SUCCESS
            cv2.putText(overlay, disp.upper(), (card_x+15, card_y+65),
                        font, 1.6, col, 3)
            cv2.putText(overlay, f"{self.current_conf*100:.1f}%",
                        (card_x+15, card_y+90), font, 0.55, Colors.TEXT_SEC, 1)
            cv2.circle(overlay, (card_x+card_w-30, card_y+50), 14, col, -1)
            cv2.putText(overlay, "OK", (card_x+card_w-42, card_y+55),
                        font, 0.38, Colors.BG_DARK, 1)
        else:
            pulse = int(80 + 60 * math.sin(time.time() * 5))
            cv2.putText(overlay, "Detecting...", (card_x+15, card_y+58),
                        font, 1.1, (pulse, pulse, pulse), 2)

        # FINGER CURL DEBUG PANEL
        if self.show_curl_debug and self.use_v2:
            curl_data = self.extractor.get_curl_for_display()
            if curl_data is not None:
                draw_finger_curl_panel(overlay, curl_data,
                                       x=card_x, y=card_y+card_h+10,
                                       panel_w=card_w)

        # BOTTOM BAR
        bottom_y = h - 38
        cv2.rectangle(overlay, (0, bottom_y), (w, h), Colors.BG_PANEL, -1)

        controls = ("[A] Auto-emo  [C] Curl  [M] Mesh  "
                    "[1-7] Emo  [+/-] Thresh  [F] Full  [R] Reset  [Q] Quit")
        cv2.putText(overlay, controls, (10, h-12),
                    font, 0.35, Colors.TEXT_MUTED, 1)

        hands_n = len(raw_data.get('hands', []))
        hcol = Colors.SUCCESS if hands_n else Colors.DANGER
        cv2.putText(overlay, f"Hands:{hands_n}",
                    (w-90, h-12), font, 0.38, hcol, 1)

        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
        return frame

    # ── Main loop ────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  [ERROR] Không mở được webcam!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        win = "VSL Realtime v4"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        if self.is_fullscreen:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print(f"\n  Phím tắt:")
        print(f"    [A]  Toggle auto emotion detect")
        print(f"    [C]  Toggle curl debug panel")
        print(f"    [1-7] Gán emotion thủ công")
        print(f"    [+/-] Điều chỉnh threshold")
        print(f"    [R]  Reset buffer  |  [Q] Thoát\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            frame   = cv2.resize(frame, (1280, 720))
            label, conf, raw_data, _ = self.process_frame(frame)
            display = self.draw_ui(frame.copy(), raw_data)
            cv2.imshow(win, display)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('a'), ord('A')):
                self.auto_emotion = not self.auto_emotion
                print(f"  Auto emotion: {'ON' if self.auto_emotion else 'OFF'}")
            elif key in (ord('c'), ord('C')):
                self.show_curl_debug = not self.show_curl_debug
            elif key in (ord('m'), ord('M')):
                self.show_mesh = not self.show_mesh
            elif key in (ord('r'), ord('R')):
                self.buffer.clear(); self.pred_history.clear()
                self.current_label = ""; self.current_conf = 0.0
                print("  Buffer reset.")
            elif key in (ord('+'), ord('=')):
                self.conf_threshold = min(cfg.CONF_THRESHOLD_MAX,
                    self.conf_threshold + cfg.CONF_THRESHOLD_STEP)
                print(f"  Threshold: {self.conf_threshold*100:.0f}%")
            elif key in (ord('-'), ord('_')):
                self.conf_threshold = max(cfg.CONF_THRESHOLD_MIN,
                    self.conf_threshold - cfg.CONF_THRESHOLD_STEP)
                print(f"  Threshold: {self.conf_threshold*100:.0f}%")
            elif key in (ord('f'), ord('F')):
                self.is_fullscreen = not self.is_fullscreen
                prop = (cv2.WINDOW_FULLSCREEN if self.is_fullscreen
                        else cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, prop)
            elif ord('1') <= key <= ord('7'):
                k = key - ord('0')
                self.emotion_key  = k
                self.emotion_name = cfg.EMOTIONS[k - 1]
                self.auto_emotion = False
                print(f"  Emotion: {self.emotion_name} (manual)")

        cap.release()
        self.extractor.close()
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VSL Realtime v4 — finger curl + auto emotion",
        epilog="""
Ví dụ:
  python realtime_v4.py
  python realtime_v4.py --model checkpoints/bilstm_v4_best.pt
  python realtime_v4.py --model checkpoints/bilstm_v3_best.pt --feat_dim 178
        """
    )
    parser.add_argument("--model",    default="checkpoints/bilstm_v2_best.pt")
    parser.add_argument("--feat_dim", type=int, default=None,
                        help="Override feat dim (mặc định: đọc từ checkpoint)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        # Fallback sang v3
        fallback = "checkpoints/bilstm_v3_best.pt"
        if os.path.exists(fallback):
            print(f"  [INFO] Không có v2, dùng: {fallback}")
            args.model    = fallback
            args.feat_dim = args.feat_dim or 178
        else:
            print(f"  [ERROR] Không tìm thấy model: {args.model}")
            print("  Chạy train_bilstm_v2.py trước!")
            return

    engine = RealtimeInferenceV4(args.model, feat_dim_override=args.feat_dim)
    try:
        engine.run()
    finally:
        engine.extractor.close()


if __name__ == "__main__":
    main()