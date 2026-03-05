"""
realtime.py - Realtime VSL Recognition
========================================
BƯỚC 4 trong pipeline

Tự động detect model 208-dim (với finger curl) hoặc 178-dim (legacy).

Tính năng:
  - Giao diện fullscreen đẹp
  - TOP-5 predictions với thanh confidence
  - Điều chỉnh ngưỡng confidence realtime
  - Auto-detect emotion từ khuôn mặt (FaceLandmarker)
  - Finger curl panel: thấy trực tiếp ngón nào đang co/duỗi
  - Hand tracking visualization

Phím tắt:
  1-7   : Gán emotion thủ công (tắt auto)
  A     : Toggle auto emotion detect
  C     : Toggle hiển thị curl debug
  +/-   : Tăng/giảm ngưỡng confidence
  F     : Toggle fullscreen
  R     : Reset buffer
  Q/ESC : Thoát

Chạy:
  python realtime.py
  python realtime.py --model checkpoints/best.pt
  python realtime.py --model checkpoints/old178.pt --feat_dim 178
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


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

class Config:
    SEQ_LEN      = 64
    FEAT_DIM_208 = 208   # model mới: với finger curl
    FEAT_DIM_178 = 178   # model cũ: không có curl (backward compat)

    POSE_KEY_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
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

    # Mapping từ FaceLandmarker expression → emotion name
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
# FINGER CURL CALCULATOR (nhất quán với video_to_npy.py)
# ═══════════════════════════════════════════════════════════════════

def compute_angle(a, b, c):
    ba = a - b; bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def compute_finger_curl_features(landmarks: np.ndarray) -> np.ndarray:
    """
    landmarks: (21, 3) → (15,): [curl_ratio, bend_angle, tip_dist_norm] × 5 ngón
    curl_ratio ≈ 0 = ngón co, ≈ 1 = ngón duỗi
    """
    curl = np.zeros(15, dtype=np.float32)
    wrist   = landmarks[cfg.WRIST_IDX]
    mid_mcp = landmarks[cfg.MIDDLE_MCP]
    scale   = np.linalg.norm(mid_mcp - wrist) + 1e-8

    for i, fname in enumerate(cfg.FINGER_ORDER):
        mcp_i, pip_i, dip_i, tip_i = cfg.FINGER_JOINTS[fname]
        mcp = landmarks[mcp_i]; pip = landmarks[pip_i]
        dip = landmarks[dip_i]; tip = landmarks[tip_i]

        d_tip_wrist = np.linalg.norm(tip - wrist)
        d_mcp_wrist = np.linalg.norm(mcp - wrist) + 1e-8
        curl_ratio   = float(np.clip(d_tip_wrist / d_mcp_wrist / 2.0, 0.0, 1.0))
        angle_pip    = compute_angle(mcp, pip, dip)
        bend_norm    = float(np.clip((np.pi - angle_pip) / np.pi, 0.0, 1.0))
        tip_dist     = float(np.clip(d_tip_wrist / scale / 3.0, 0.0, 1.0))

        base = i * 3
        curl[base] = curl_ratio; curl[base+1] = bend_norm; curl[base+2] = tip_dist

    return curl


# ═══════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS (phải match với train.py)
# ═══════════════════════════════════════════════════════════════════

# ── Legacy BiLSTM (cho checkpoint 178-dim cũ) ─────────────────────

class _AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, lstm_out):
        w = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=-1)
        return (lstm_out * w.unsqueeze(-1)).sum(dim=1), w


class BiLSTMClassifier(nn.Module):
    """Legacy model cho checkpoint 178-dim."""
    def __init__(self, feat_dim, hidden_dim, num_layers, num_classes,
                 dropout_lstm=0.3, dropout_fc=0.4, bidirectional=True):
        super().__init__()
        dirs = 2 if bidirectional else 1
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(0.1))
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout_lstm if num_layers > 1 else 0)
        od = hidden_dim * dirs
        self.attention  = _AttentionLayer(od)
        mid = max(num_classes * 4, 128)
        self.classifier = nn.Sequential(
            nn.LayerNorm(od * 2), nn.Linear(od*2, mid), nn.GELU(), nn.Dropout(dropout_fc),
            nn.Linear(mid, mid//2), nn.GELU(), nn.Dropout(dropout_fc/2),
            nn.Linear(mid//2, num_classes))
    def forward(self, x):
        x = self.input_proj(x)
        out, (hn, _) = self.lstm(x)
        last = torch.cat([hn[-2], hn[-1]], dim=-1)
        ctx, _ = self.attention(out)
        return self.classifier(torch.cat([ctx, last], dim=-1))


# ── HandAware model (208-dim, từ train.py) ────────────────────────

# Config ngón tay (phải nhất quán với train.py)
_FINGER_GROUPS = {
    'wrist': [0], 'thumb': [1,2,3,4], 'index': [5,6,7,8],
    'middle': [9,10,11,12], 'ring': [13,14,15,16], 'pinky': [17,18,19,20],
}
_FINGER_NAMES = list(_FINGER_GROUPS.keys())
_NUM_FINGERS  = len(_FINGER_GROUPS)
_FINGER_EDGES = [(0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(2,3),(3,4),(4,5)]


class _FingerEncoder(nn.Module):
    def __init__(self, out_dim=32, dropout=0.1):
        super().__init__()
        self.out_dim  = out_dim
        self.encoders = nn.ModuleDict({
            fname: nn.Sequential(
                nn.Linear(len(idx)*3, out_dim*2), nn.LayerNorm(out_dim*2),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(out_dim*2, out_dim),
                nn.LayerNorm(out_dim))
            for fname, idx in _FINGER_GROUPS.items()
        })
    def forward(self, hand):  # (B,T,21,3) → (B,T,6,out_dim)
        B, T = hand.shape[:2]
        return torch.stack([
            self.encoders[fn](hand[:, :, _FINGER_GROUPS[fn], :].reshape(B, T, -1))
            for fn in _FINGER_NAMES], dim=2)


class _CurlEncoder(nn.Module):
    def __init__(self, out_dim=32, dropout=0.1):
        super().__init__()
        per = out_dim // 5
        rem = out_dim - per * 4
        self.finger_encs = nn.ModuleList([
            nn.Sequential(nn.Linear(3, 16), nn.GELU(),
                          nn.Linear(16, rem if i == 4 else per))
            for i in range(5)])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(out_dim), nn.Linear(out_dim, out_dim),
            nn.GELU(), nn.Dropout(dropout))
    def forward(self, curl):  # (B,T,15) → (B,T,out_dim)
        return self.out_proj(torch.cat([
            self.finger_encs[i](curl[:, :, i*3:i*3+3]) for i in range(5)], dim=-1))


class _FingerGraphAttn(nn.Module):
    def __init__(self, feat_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = feat_dim // num_heads
        adj = torch.zeros(_NUM_FINGERS, _NUM_FINGERS)
        for i, j in _FINGER_EDGES:
            adj[i,j] = adj[j,i] = 1.0
        adj = adj + torch.eye(_NUM_FINGERS)
        self.register_buffer('adj', adj / adj.sum(-1, keepdim=True).clamp(1))
        self.W_q = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_k = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_v = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_o = nn.Linear(feat_dim, feat_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feat_dim)
    def forward(self, x):  # (B,T,6,D)
        B, T, N, D = x.shape; BT = B*T
        xf = x.reshape(BT, N, D)
        Q = self.W_q(xf).reshape(BT, N, self.num_heads, self.head_dim).transpose(1,2)
        K = self.W_k(xf).reshape(BT, N, self.num_heads, self.head_dim).transpose(1,2)
        V = self.W_v(xf).reshape(BT, N, self.num_heads, self.head_dim).transpose(1,2)
        s = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.head_dim)
        s = s.masked_fill(self.adj.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        out = torch.matmul(self.drop(torch.softmax(s, -1)), V)
        out = self.W_o(out.transpose(1,2).reshape(BT, N, D)).reshape(B, T, N, D)
        return self.norm(out + x)


class _HandCrossAttn(nn.Module):
    def __init__(self, feat_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_lr = nn.MultiheadAttention(feat_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_rl = nn.MultiheadAttention(feat_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_l   = nn.LayerNorm(feat_dim)
        self.norm_r   = nn.LayerNorm(feat_dim)
    def forward(self, left, right):
        B, T, N, D = left.shape; BT = B*T
        lf = left.reshape(BT, N, D); rf = right.reshape(BT, N, D)
        lc, _ = self.cross_lr(lf, rf, rf); rc, _ = self.cross_rl(rf, lf, lf)
        return self.norm_l(lf+lc).reshape(B,T,N,D), self.norm_r(rf+rc).reshape(B,T,N,D)


class _HandAggregator(nn.Module):
    def __init__(self, finger_dim, curl_dim, pose_dim, emo_dim, out_dim, dropout=0.1):
        super().__init__()
        self.finger_pool = nn.Linear(_NUM_FINGERS, 1, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(finger_dim*2 + curl_dim*2 + pose_dim + emo_dim, out_dim),
            nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout))
    def forward(self, lf, rf, cl, cr, pose, emo):
        lp = self.finger_pool(lf.transpose(-1,-2)).squeeze(-1)
        rp = self.finger_pool(rf.transpose(-1,-2)).squeeze(-1)
        return self.proj(torch.cat([lp, rp, cl, cr, pose, emo], dim=-1))


class _TemporalTransformer(nn.Module):
    def __init__(self, dim, heads=8, layers=4, dropout=0.1):
        super().__init__()
        pe  = torch.zeros(256, dim)
        pos = torch.arange(256).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        lay = nn.TransformerEncoderLayer(dim, heads, dim*4, dropout,
                                         batch_first=True, activation='gelu', norm_first=True)
        self.encoder = nn.TransformerEncoder(lay, layers)
        self.norm    = nn.LayerNorm(dim)
    def forward(self, x):
        return self.norm(self.encoder(x + self.pe[:, :x.size(1)]))


class HandAwareVSLClassifier(nn.Module):
    """Model 208-dim từ train.py (phải match kiến trúc)."""
    def __init__(self, num_classes, finger_dim=32, curl_dim=32, temporal_dim=256,
                 temporal_heads=8, temporal_layers=4, graph_heads=4, dropout=0.2):
        super().__init__()
        # Tên attribute phải khớp chính xác với train.py
        self.finger_encoder   = _FingerEncoder(finger_dim, dropout)
        self.curl_encoder     = _CurlEncoder(curl_dim, dropout)
        self.graph_attn_left  = _FingerGraphAttn(finger_dim, graph_heads, dropout)
        self.graph_attn_right = _FingerGraphAttn(finger_dim, graph_heads, dropout)
        self.cross_attn       = _HandCrossAttn(finger_dim, graph_heads, dropout)
        self.aggregator       = _HandAggregator(finger_dim, curl_dim, 45, 7,
                                                temporal_dim, dropout)
        self.temporal         = _TemporalTransformer(temporal_dim, temporal_heads,
                                                     temporal_layers, dropout)
        self.temporal_pool    = nn.Linear(temporal_dim, 1)
        mid = max(num_classes * 4, 256)
        self.classifier  = nn.Sequential(
            nn.LayerNorm(temporal_dim), nn.Linear(temporal_dim, mid), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(mid, mid//2), nn.GELU(),
            nn.Dropout(dropout/2), nn.Linear(mid//2, num_classes))

    def forward(self, x):
        B, T, _ = x.shape
        pose       = x[:, :, 0:45]
        lh         = x[:, :, 45:108].view(B, T, 21, 3)
        rh         = x[:, :, 108:171].view(B, T, 21, 3)
        curl_l     = x[:, :, 171:186]
        curl_r     = x[:, :, 186:201]
        emo        = x[:, :, 201:208]

        lf = self.graph_attn_left(self.finger_encoder(lh))
        rf = self.graph_attn_right(self.finger_encoder(rh))
        lf, rf = self.cross_attn(lf, rf)
        cl = self.curl_encoder(curl_l); cr = self.curl_encoder(curl_r)
        t  = self.temporal(self.aggregator(lf, rf, cl, cr, pose, emo))
        w  = torch.softmax(self.temporal_pool(t).squeeze(-1), dim=-1)
        return self.classifier((t * w.unsqueeze(-1)).sum(1))


# ═══════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════════

def load_model(ckpt_path, device, feat_dim_override=None):
    ckpt       = torch.load(ckpt_path, map_location=device)
    label_map  = ckpt['label_map']
    model_cfg  = ckpt.get('cfg', {})
    model_type = ckpt.get('model_type', 'BiLSTMClassifier')
    feat_dim   = feat_dim_override or model_cfg.get('FEAT_DIM', cfg.FEAT_DIM_208)

    if model_type == 'HandAwareVSLClassifier' or feat_dim == cfg.FEAT_DIM_208:
        model = HandAwareVSLClassifier(
            num_classes    = len(label_map),
            finger_dim     = model_cfg.get('FINGER_DIM',      32),
            curl_dim       = model_cfg.get('CURL_EMBED_DIM',  32),
            temporal_dim   = model_cfg.get('TEMPORAL_DIM',   256),
            temporal_heads = model_cfg.get('TEMPORAL_HEADS',   8),
            temporal_layers= model_cfg.get('TEMPORAL_LAYERS',  4),
            graph_heads    = model_cfg.get('GRAPH_HEADS',       4),
            dropout        = model_cfg.get('DROPOUT',         0.2),
        )
        print(f"  [Model] HandAwareVSLClassifier (208-dim)")
    else:
        model = BiLSTMClassifier(
            feat_dim   = feat_dim,
            hidden_dim = model_cfg.get('HIDDEN_DIM', 256),
            num_layers = model_cfg.get('NUM_LAYERS',   3),
            num_classes= len(label_map),
        )
        print(f"  [Model] BiLSTMClassifier (legacy {feat_dim}-dim)")

    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    idx2label = {v: k for k, v in label_map.items()}
    print(f"  [Model] Labels   : {list(label_map.keys())}")
    print(f"  [Model] Val acc  : {ckpt.get('val_acc', 0)*100:.2f}%")
    print(f"  [Model] Feat dim : {feat_dim}")
    return model, idx2label, feat_dim


# ═══════════════════════════════════════════════════════════════════
# REALTIME EXTRACTOR
# ═══════════════════════════════════════════════════════════════════

class RealtimeExtractor:
    """
    Async landmark extraction qua MediaPipe LIVE_STREAM.
    Hỗ trợ: Hand, Pose, Face (face dùng cho auto emotion).
    """

    def __init__(self):
        print("  Init RealtimeExtractor...")
        self._data = {'pose': None, 'hands': [], 'blendshapes': None}
        self._ts   = 0
        self._lock = threading.Lock()

        self.hand_det = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=ensure_model('hand_landmarker.task')),
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=self._on_hand))

        self.pose_det = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=ensure_model('pose_landmarker_heavy.task')),
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=self._on_pose))

        # Face: optional (có thể fail nếu thiếu model)
        self.face_det = None
        try:
            self.face_det = mp_vision.FaceLandmarker.create_from_options(
                mp_vision.FaceLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=ensure_model('face_landmarker.task')),
                    running_mode=mp_vision.RunningMode.LIVE_STREAM,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    output_face_blendshapes=True,
                    result_callback=self._on_face))
        except Exception as e:
            print(f"  [WARN] Face detector unavailable: {e}")

        print("  RealtimeExtractor ready.")

    def _on_pose(self, result, image, ts):
        with self._lock:
            self._data['pose'] = (result.pose_landmarks[0]
                                  if result.pose_landmarks else None)

    def _on_hand(self, result, image, ts):
        with self._lock:
            if result.hand_landmarks:
                self._data['hands'] = list(result.hand_landmarks)
            else:
                self._data['hands'] = []

    def _on_face(self, result, image, ts):
        with self._lock:
            self._data['blendshapes'] = (result.face_blendshapes[0]
                                         if result.face_blendshapes else None)

    def send_frame(self, rgb: np.ndarray):
        self._ts += 33
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        for det in [self.pose_det, self.hand_det]:
            try:
                det.detect_async(mp_img, self._ts)
            except Exception:
                pass
        if self.face_det:
            try:
                self.face_det.detect_async(mp_img, self._ts)
            except Exception:
                pass

    def extract_208(self):
        """Extract (201,) = pose(45) + hands_xyz(126) + curl(30). Emotion ghép sau.
        1 tay → LUÔN slot 0, không phân biệt trái/phải (nhất quán với lúc train).
        """
        with self._lock:
            feat = np.zeros(201, dtype=np.float32)

            if self._data['pose']:
                for i, idx in enumerate(cfg.POSE_KEY_INDICES):
                    lm = self._data['pose'][idx]
                    feat[i*3:(i+1)*3] = [lm.x, lm.y, lm.z]

            hands = self._data['hands']
            if hands:
                if len(hands) == 1:
                    # 1 tay → luôn slot 0, bất kể trái/phải
                    hand = hands[0]
                    for j, lm in enumerate(hand):
                        feat[45 + j*3 : 45 + j*3 + 3] = [lm.x, lm.y, lm.z]
                    lms = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
                    feat[171 : 186] = compute_finger_curl_features(lms)
                else:
                    # 2 tay → sort theo x
                    sorted_h = sorted(hands, key=lambda h: np.mean([lm.x for lm in h]))
                    for hand, xyz_s, curl_s in zip(sorted_h[:2], [45, 108], [171, 186]):
                        for j, lm in enumerate(hand):
                            feat[xyz_s + j*3 : xyz_s + j*3 + 3] = [lm.x, lm.y, lm.z]
                        lms = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
                        feat[curl_s : curl_s+15] = compute_finger_curl_features(lms)

            return feat, dict(self._data)

    def extract_178(self):
        """Extract (171,) = pose(45) + hands_xyz(126). Legacy mode.
        1 tay → LUÔN slot 0, không phân biệt trái/phải.
        """
        with self._lock:
            feat = np.zeros(171, dtype=np.float32)

            if self._data['pose']:
                for i, idx in enumerate(cfg.POSE_KEY_INDICES):
                    lm = self._data['pose'][idx]
                    feat[i*3:(i+1)*3] = [lm.x, lm.y, lm.z]

            hands = self._data['hands']
            if hands:
                if len(hands) == 1:
                    # 1 tay → luôn slot 0
                    hand = hands[0]
                    for j, lm in enumerate(hand):
                        feat[45 + j*3 : 45 + j*3 + 3] = [lm.x, lm.y, lm.z]
                else:
                    sorted_h = sorted(hands, key=lambda h: np.mean([lm.x for lm in h]))
                    for hand, slot in zip(sorted_h[:2], [45, 108]):
                        for j, lm in enumerate(hand):
                            feat[slot + j*3 : slot + j*3 + 3] = [lm.x, lm.y, lm.z]

            return feat, dict(self._data)

    def get_curl_display(self):
        """Lấy curl (15,) của tay đầu tiên để hiển thị debug."""
        with self._lock:
            if not self._data['hands']:
                return None
            lms = np.array([[lm.x, lm.y, lm.z] for lm in self._data['hands'][0]],
                           dtype=np.float32)
            return compute_finger_curl_features(lms)

    def has_hands(self):
        with self._lock:
            return len(self._data['hands']) > 0

    def close(self):
        self.hand_det.close()
        self.pose_det.close()
        if self.face_det:
            self.face_det.close()


# ═══════════════════════════════════════════════════════════════════
# FEATURE NORMALIZATION
# ═══════════════════════════════════════════════════════════════════

def normalize_208(feat: np.ndarray) -> np.ndarray:
    """Normalize pose + hands xyz; curl (171:201) giữ nguyên."""
    f = feat.copy()
    pose = f[:45].reshape(-1, 3)
    if np.any(pose != 0):
        hip  = (pose[13] + pose[14]) / 2
        pose = pose - hip
        sd   = np.linalg.norm(pose[1] - pose[2])
        if sd > 1e-6: pose /= sd
        f[:45] = pose.flatten()
    for s, e in [(45, 108), (108, 171)]:
        hand = f[s:e].reshape(-1, 3)
        if np.any(hand != 0):
            hand = hand - hand[0]
            sc   = np.linalg.norm(hand[9])
            if sc > 1e-6: hand /= sc
            f[s:e] = hand.flatten()
    return f


def normalize_178(feat: np.ndarray) -> np.ndarray:
    """Normalize pose + hands xyz (171 dim total)."""
    return normalize_208(feat)   # same logic, curl không có nên không ảnh hưởng


# ═══════════════════════════════════════════════════════════════════
# AUTO EMOTION DETECTION
# ═══════════════════════════════════════════════════════════════════

_BLENDSHAPE_MAP = {
    # key blendshape → (emotion, weight)
    'mouthSmileLeft':    ('happy',   1.0),
    'mouthSmileRight':   ('happy',   1.0),
    'browDownLeft':      ('angry',   0.8),
    'browDownRight':     ('angry',   0.8),
    'eyeSquintLeft':     ('disgust', 0.6),
    'eyeSquintRight':    ('disgust', 0.6),
    'mouthFrownLeft':    ('sad',     0.9),
    'mouthFrownRight':   ('sad',     0.9),
    'browInnerUp':       ('surprise',0.9),
    'jawOpen':           ('surprise',0.7),
    'eyeWideLeft':       ('fear',    0.8),
    'eyeWideRight':      ('fear',    0.8),
}

def detect_emotion_from_blendshapes(blendshapes) -> str:
    """Phân tích FaceBlendshapes → tên emotion."""
    if blendshapes is None:
        return "neutral"
    scores = {e: 0.0 for e in cfg.EMOTIONS}
    try:
        for bs in blendshapes:
            name  = bs.category_name
            value = float(bs.score)
            if name in _BLENDSHAPE_MAP:
                emo, w = _BLENDSHAPE_MAP[name]
                scores[emo] += value * w
    except Exception:
        return "neutral"
    best = max(scores, key=scores.get)
    return best if scores[best] > 0.3 else "neutral"


# ═══════════════════════════════════════════════════════════════════
# UI COMPONENTS
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
    INFO       = (255, 200, 0)


def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1):
    x1, y1 = pt1; x2, y2 = pt2
    if thickness == -1:
        cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, -1)
        for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                       (x1+radius, y2-radius), (x2-radius, y2-radius)]:
            cv2.circle(img, (cx, cy), radius, color, -1)
    else:
        cv2.line(img, (x1+radius,y1), (x2-radius,y1), color, thickness)
        cv2.line(img, (x1+radius,y2), (x2-radius,y2), color, thickness)
        cv2.line(img, (x1,y1+radius), (x1,y2-radius), color, thickness)
        cv2.line(img, (x2,y1+radius), (x2,y2-radius), color, thickness)
        for ang, cx, cy in [(180,x1+radius,y1+radius),(270,x2-radius,y1+radius),
                             (90,x1+radius,y2-radius),(0,x2-radius,y2-radius)]:
            cv2.ellipse(img, (cx,cy), (radius,radius), ang, 0, 90, color, thickness)


def draw_confidence_bar(img, x, y, w, h, conf, threshold, label, rank):
    font = cv2.FONT_HERSHEY_SIMPLEX
    draw_rounded_rect(img, (x,y), (x+w,y+h), Colors.BG_CARD, radius=5)
    bx = x+10; by = y+h-14; bw = w-20; bh = 8
    col = (Colors.SUCCESS if conf >= threshold else
           Colors.WARNING if conf >= threshold*0.7 else Colors.DANGER)
    cv2.rectangle(img, (bx,by), (bx+bw,by+bh), (60,60,60), -1)
    fw = int(bw * conf)
    if fw > 0:
        cv2.rectangle(img, (bx,by), (bx+fw,by+bh), col, -1)
    tx = bx + int(bw * threshold)
    cv2.line(img, (tx,by-2), (tx,by+bh+2), Colors.WARNING, 2)
    rank_col = [Colors.SUCCESS, Colors.PRIMARY, Colors.INFO,
                Colors.TEXT_SEC, Colors.TEXT_MUTED][min(rank, 4)]
    cv2.circle(img, (x+18, y+18), 11, rank_col, -1)
    cv2.putText(img, str(rank+1), (x+13, y+23), font, 0.42, Colors.BG_DARK, 2)
    lc = Colors.TEXT_PRI if conf >= threshold else Colors.TEXT_SEC
    cv2.putText(img, label, (x+36, y+24), font, 0.52, lc, 1 if rank > 0 else 2)
    cv2.putText(img, f"{conf*100:.1f}%", (x+w-62, y+24), font, 0.42, col, 1)


def draw_hand_landmarks(img, hands, w, h):
    conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
             (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
             (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
    for hi, hand in enumerate(hands):
        col = (0, 255, 200) if hi == 0 else (255, 200, 0)
        pts = [(int(lm.x*w), int(lm.y*h)) for lm in hand]
        for a, b in conns:
            if a < len(pts) and b < len(pts):
                cv2.line(img, pts[a], pts[b], (80, 160, 120), 2)
        for i, pt in enumerate(pts):
            r = 6 if i in [0,4,8,12,16,20] else 4
            cv2.circle(img, pt, r, col, -1)
            cv2.circle(img, pt, r, Colors.BG_DARK, 1)


def draw_curl_panel(img, curl, x, y, panel_w=230):
    """Hiển thị trạng thái co/duỗi của 5 ngón tay."""
    if curl is None:
        return
    font    = cv2.FONT_HERSHEY_SIMPLEX
    bar_w   = panel_w - 90
    names   = ['Cai', 'Tro', 'Giua', 'Nhan', 'Ut']
    n       = len(names)
    ph      = 16 + n * 28

    draw_rounded_rect(img, (x, y), (x+panel_w, y+ph), Colors.BG_PANEL, radius=8)
    cv2.putText(img, "CURL NGON TAY", (x+8, y+13), font, 0.35, Colors.TEXT_MUTED, 1)

    for i, name in enumerate(names):
        ratio = float(curl[i * 3])       # curl_ratio: 0=co, 1=duỗi
        by    = y + 18 + i * 28

        cv2.putText(img, name, (x+8, by+10), font, 0.38, Colors.TEXT_SEC, 1)

        bx = x + 44
        cv2.rectangle(img, (bx, by), (bx+bar_w, by+10), (60,60,60), -1)
        fw    = int(bar_w * ratio)
        r_val = int(255 * (1 - ratio))
        g_val = int(200 * ratio)
        if fw > 0:
            cv2.rectangle(img, (bx, by), (bx+fw, by+10), (0, g_val, r_val), -1)

        state = "DUOI" if ratio > 0.6 else ("CO" if ratio < 0.4 else "~")
        s_col = Colors.SUCCESS if ratio > 0.6 else \
                Colors.DANGER  if ratio < 0.4 else Colors.TEXT_MUTED
        cv2.putText(img, state, (bx+bar_w+5, by+10), font, 0.32, s_col, 1)


# ═══════════════════════════════════════════════════════════════════
# MAIN INFERENCE CLASS
# ═══════════════════════════════════════════════════════════════════

class RealtimeInference:

    def __init__(self, model_path, feat_dim_override=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n{'='*60}")
        print(" VSL REALTIME ".center(60, "="))
        print("="*60)
        print(f"  Device: {self.device}")

        self.model, self.idx2label, self.feat_dim = \
            load_model(model_path, self.device, feat_dim_override)

        self.use_curl = (self.feat_dim == cfg.FEAT_DIM_208)
        print(f"  Mode: {'208-dim (with finger curl)' if self.use_curl else '178-dim (legacy)'}")

        # Load display names nếu có
        self.display_names = {}
        for p in ['data/processed/display_names.json', 'data/display_names.json']:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    self.display_names = json.load(f)
                break

        self.extractor = RealtimeExtractor()
        self.buffer    = deque(maxlen=cfg.SEQ_LEN)

        # Emotion state
        self.auto_emotion   = True
        self.emotion_name   = "neutral"
        self.emotion_buf    = deque(maxlen=10)

        # Prediction state
        self.conf_threshold  = cfg.CONF_THRESHOLD_DEFAULT
        self.top_predictions = []
        self.current_label   = ""
        self.current_conf    = 0.0
        self.pred_history    = deque(maxlen=5)

        # Display
        self.is_fullscreen   = True
        self.show_curl       = True

        # FPS
        self.fps         = 0
        self.frame_times = deque(maxlen=30)

    def _emotion_vec(self) -> np.ndarray:
        vec = np.zeros(7, dtype=np.float32)
        vec[cfg.EMOTIONS.index(self.emotion_name)] = 1.0
        return vec

    def _smooth_emotion(self, emo: str) -> str:
        self.emotion_buf.append(emo)
        return collections.Counter(self.emotion_buf).most_common(1)[0][0]

    def process_frame(self, frame: np.ndarray):
        t0  = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.extractor.send_frame(rgb)

        # Extract raw features
        if self.use_curl:
            feat, raw = self.extractor.extract_208()   # (201,)
            feat      = normalize_208(feat)
        else:
            feat, raw = self.extractor.extract_178()   # (171,)
            feat      = normalize_178(feat)

        # Auto emotion
        if self.auto_emotion:
            raw_emo = detect_emotion_from_blendshapes(raw.get('blendshapes'))
            self.emotion_name = self._smooth_emotion(raw_emo)

        # Build full feature
        full = np.concatenate([feat, self._emotion_vec()])  # 208 or 178
        self.buffer.append(full)

        if len(self.buffer) < cfg.SEQ_LEN:
            return "", 0.0, raw, []

        if not self.extractor.has_hands():
            self.buffer.clear()
            return "[NO HANDS]", 0.0, raw, []

        seq = np.array(list(self.buffer), dtype=np.float32)
        x   = torch.from_numpy(seq).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=-1)[0]
            topk   = torch.topk(probs, min(cfg.TOP_K, len(probs)))
            preds  = [(self.idx2label[i.item()], c.item())
                      for i, c in zip(topk.indices, topk.values)]

        self.top_predictions = preds
        label = preds[0][0]; conf = preds[0][1]
        self.pred_history.append((label, conf))

        # Majority vote smoothing
        if len(self.pred_history) >= 3:
            counted = collections.Counter(
                lb for lb, c in self.pred_history if c >= self.conf_threshold)
            if counted:
                best = counted.most_common(1)[0][0]
                avg  = float(np.mean([c for lb, c in self.pred_history if lb == best]))
                self.current_label = best
                self.current_conf  = avg

        self.frame_times.append(time.time() - t0)
        self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times) + 1e-8)
        return label, conf, raw, preds

    def draw_ui(self, frame: np.ndarray, raw: dict) -> np.ndarray:
        h, w  = frame.shape[:2]
        font  = cv2.FONT_HERSHEY_SIMPLEX
        ol    = frame.copy()

        # ── Hand landmarks ────────────────────────────────────────
        if raw.get('hands'):
            draw_hand_landmarks(ol, raw['hands'], w, h)

        # ── TOP BAR ───────────────────────────────────────────────
        cv2.rectangle(ol, (0,0), (w,75), Colors.BG_PANEL, -1)
        cv2.putText(ol, "VSL RECOGNITION", (15, 32), font, 0.8, Colors.PRIMARY, 2)

        emo_col  = cfg.EMOTION_COLORS.get(self.emotion_name, (180,180,180))
        mode_txt = "AUTO" if self.auto_emotion else "MANUAL"
        draw_rounded_rect(ol, (15,42), (230,68), (40,40,40), radius=6)
        cv2.putText(ol, f"EMO [{mode_txt}]: {self.emotion_name.upper()}",
                    (22, 60), font, 0.4, emo_col, 1)

        cv2.putText(ol, f"FPS:{self.fps:.0f}", (w-90,32), font, 0.5, Colors.TEXT_SEC, 1)

        buf_pct = len(self.buffer) / cfg.SEQ_LEN
        bx, by  = w-170, 48
        cv2.rectangle(ol, (bx,by), (bx+150,by+10), (60,60,60), -1)
        cv2.rectangle(ol, (bx,by), (bx+int(150*buf_pct),by+10), Colors.PRIMARY, -1)
        cv2.putText(ol, f"Buf:{len(self.buffer)}/{cfg.SEQ_LEN}",
                    (bx,by-4), font, 0.32, Colors.TEXT_MUTED, 1)

        # ── RIGHT PANEL — Predictions ─────────────────────────────
        pw = 310; px = w-pw-15; py = 85
        draw_rounded_rect(ol, (px,py), (w-15,h-38), Colors.BG_PANEL, radius=12)
        cv2.putText(ol, "PREDICTIONS", (px+12,py+28), font, 0.55, Colors.TEXT_PRI, 1)
        cv2.putText(ol, f"Threshold: {self.conf_threshold*100:.0f}%  [+/-]",
                    (px+12,py+50), font, 0.38, Colors.WARNING, 1)
        cv2.line(ol, (px+12,py+58), (w-28,py+58), (60,60,60), 1)
        for i, (lbl, cf) in enumerate(self.top_predictions):
            draw_confidence_bar(ol, px+8, py+64+i*52, pw-20, 48,
                                cf, self.conf_threshold, lbl, i)

        # ── MAIN prediction card ──────────────────────────────────
        cx, cy = 15, 85; cw = px-30; cch = 105
        draw_rounded_rect(ol, (cx,cy), (cx+cw,cy+cch), Colors.BG_PANEL, radius=12)

        if self.current_label and self.current_conf >= self.conf_threshold:
            disp = self.display_names.get(self.current_label, self.current_label)
            cv2.putText(ol, disp.upper(), (cx+15,cy+65), font, 1.5, Colors.SUCCESS, 3)
            cv2.putText(ol, f"{self.current_conf*100:.1f}%", (cx+15,cy+90),
                        font, 0.55, Colors.TEXT_SEC, 1)
            cv2.circle(ol, (cx+cw-30,cy+50), 13, Colors.SUCCESS, -1)
            cv2.putText(ol, "OK", (cx+cw-42,cy+55), font, 0.38, Colors.BG_DARK, 1)
        else:
            pulse = int(80 + 60 * math.sin(time.time() * 5))
            cv2.putText(ol, "Detecting...", (cx+15,cy+58), font, 1.0,
                        (pulse,pulse,pulse), 2)

        # ── Curl debug panel ──────────────────────────────────────
        if self.show_curl and self.use_curl:
            curl_data = self.extractor.get_curl_display()
            if curl_data is not None:
                draw_curl_panel(ol, curl_data, x=cx, y=cy+cch+10, panel_w=cw)

        # ── BOTTOM BAR ────────────────────────────────────────────
        cv2.rectangle(ol, (0,h-35), (w,h), Colors.BG_PANEL, -1)
        cv2.putText(ol,
            "[A] Auto-emo  [C] Curl  [1-7] Emo  [+/-] Thresh  [F] Full  [R] Reset  [Q] Quit",
            (10, h-12), font, 0.35, Colors.TEXT_MUTED, 1)
        nh  = len(raw.get('hands', []))
        hcl = Colors.SUCCESS if nh else Colors.DANGER
        cv2.putText(ol, f"Hands:{nh}", (w-90,h-12), font, 0.38, hcl, 1)

        cv2.addWeighted(ol, 0.92, frame, 0.08, 0, frame)
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  [ERROR] Không mở được webcam!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        win = "VSL Realtime"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        if self.is_fullscreen:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print(f"\n  Controls:")
        print(f"    [A]   Toggle auto emotion detect")
        print(f"    [C]   Toggle curl debug panel")
        print(f"    [1-7] Gán emotion thủ công")
        print(f"    [+/-] Điều chỉnh confidence threshold")
        print(f"    [R]   Reset buffer  |  [Q] Thoát\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            frame  = cv2.resize(frame, (1280, 720))
            _, _, raw, _ = self.process_frame(frame)
            display = self.draw_ui(frame.copy(), raw)
            cv2.imshow(win, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('a'), ord('A')):
                self.auto_emotion = not self.auto_emotion
                print(f"  Auto emotion: {'ON' if self.auto_emotion else 'OFF'}")
            elif key in (ord('c'), ord('C')):
                self.show_curl = not self.show_curl
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
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if self.is_fullscreen else cv2.WINDOW_NORMAL)
            elif ord('1') <= key <= ord('7'):
                k = key - ord('0')
                self.emotion_name  = cfg.EMOTIONS[k - 1]
                self.auto_emotion  = False
                print(f"  Emotion: {self.emotion_name} (manual)")

        cap.release()
        self.extractor.close()
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VSL Realtime Inference (auto-detect 208/178 dim)",
        epilog="""
Ví dụ:
  python realtime.py
  python realtime.py --model checkpoints/best.pt
  python realtime.py --model checkpoints/old.pt --feat_dim 178
        """
    )
    parser.add_argument("--model",    default="checkpoints/best.pt")
    parser.add_argument("--feat_dim", type=int, default=None,
                        help="Override feat dim (mặc định: đọc từ checkpoint)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        # Thử tìm checkpoint nào đó
        for fallback in ['checkpoints/best.pt', 'checkpoints/bilstm_v3_best.pt',
                         'checkpoints/bilstm_v2_best.pt']:
            if os.path.exists(fallback):
                print(f"  [INFO] Dùng fallback: {fallback}")
                args.model = fallback
                break
        else:
            print(f"  [ERROR] Không tìm thấy model: {args.model}")
            print("  Chạy train.py trước!")
            return

    engine = RealtimeInference(args.model, feat_dim_override=args.feat_dim)
    try:
        engine.run()
    finally:
        engine.extractor.close()


if __name__ == "__main__":
    main()