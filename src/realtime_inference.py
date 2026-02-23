"""
VSL REALTIME INFERENCE - Giao diện test model ngôn ngữ ký hiệu
==============================================================
Dùng model DualTransformer đã train để nhận diện ký hiệu realtime từ webcam.


Cách chạy:
    python realtime_inference.py
    python realtime_inference.py --checkpoint checkpoints/best_xxx.pt
    python realtime_inference.py --checkpoint checkpoints/best_xxx.pt --top_k 3

Phím tắt trong cửa sổ:
    [Q]       - Thoat
    [SPACE]   - Tam dung / Tiep tuc
    [C]       - Xoa lich su
    [S]       - Luu screenshot
    [+/-]     - Tang/giam nguong tin cay
"""

import cv2
import numpy as np
import os
import json
import math
import time
import argparse
import urllib.request
from collections import deque
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as Func

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ═══════════════════════════════════════════════════════════
# CẤU HÌNH (phải khớp với train_dual_transformer.py)
# ═══════════════════════════════════════════════════════════

class Config:
    SEQ_LEN          = 30
    FEAT_DIM         = 339      # 75+90+126+17+31 (interact mở rộng ngón trỏ→mặt)
    D_MODEL          = 256
    SPATIAL_HEADS    = 8
    SPATIAL_LAYERS   = 3
    SPATIAL_FF_DIM   = 512
    SPATIAL_DROPOUT  = 0.1
    TEMPORAL_HEADS   = 8
    TEMPORAL_LAYERS  = 4
    TEMPORAL_FF_DIM  = 512
    TEMPORAL_DROPOUT = 0.1
    CLASSIFIER_HIDDEN = 256
    DROPOUT_FINAL    = 0.3

    POSE_START,    POSE_END    = 0,   75
    FACE_START,    FACE_END    = 75,  165
    HAND_START,    HAND_END    = 165, 291
    BLEND_START,   BLEND_END   = 291, 308
    INTERACT_START,INTERACT_END= 308, 339

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg = Config()

# 30 face landmarks + blendshapes quan trọng (từ video_to_npy)
FACE_KEY_INDICES = [
    70,63,105,66,107, 336,296,334,293,300,
    33,159,145,133,   263,386,374,362,
    13,14,61,291,0,17,78,308,
    1,4,10,152,
]
KEY_BLENDSHAPES = [
    'jawOpen','mouthSmileLeft','mouthSmileRight',
    'mouthFrownLeft','mouthFrownRight','mouthPucker','cheekPuff',
    'eyeBlinkLeft','eyeBlinkRight','eyeWideLeft','eyeWideRight',
    'eyeSquintLeft','eyeSquintRight','browInnerUp',
    'browDownLeft','browDownRight','noseSneerLeft',
]

MODEL_URLS = {
    'hand_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'),
    'pose_landmarker_heavy.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'),
    'face_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'face_landmarker/face_landmarker/float16/1/face_landmarker.task'),
}


def download_model(filename):
    if not os.path.exists(filename):
        print(f"  Dang tai {filename}...")
        urllib.request.urlretrieve(MODEL_URLS[filename], filename)
    return filename


# ═══════════════════════════════════════════════════════════
# DUAL TRANSFORMER MODEL (copy từ train để standalone)
# ═══════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class SpatialTransformer(nn.Module):
    NUM_TOKENS = 6
    def __init__(self, feat_dim, d_model, nhead, num_layers, ff_dim, dropout):
        super().__init__()
        self.d_model = d_model
        group_dims = [
            cfg.POSE_END - cfg.POSE_START,
            cfg.FACE_END - cfg.FACE_START,
            63, 63,
            cfg.BLEND_END  - cfg.BLEND_START,
            cfg.INTERACT_END - cfg.INTERACT_START,
        ]
        self.group_projs = nn.ModuleList([nn.Linear(d, d_model) for d in group_dims])
        self.token_embed = nn.Embedding(self.NUM_TOKENS, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers,
            norm=nn.LayerNorm(d_model))
        self.cls_token = nn.Parameter(torch.randn(1,1,d_model)*0.02)

    def _split(self, x):
        return [
            x[:,:,cfg.POSE_START   :cfg.POSE_END],
            x[:,:,cfg.FACE_START   :cfg.FACE_END],
            x[:,:,cfg.HAND_START   :cfg.HAND_START+63],
            x[:,:,cfg.HAND_START+63:cfg.HAND_END],
            x[:,:,cfg.BLEND_START  :cfg.BLEND_END],
            x[:,:,cfg.INTERACT_START:cfg.INTERACT_END],
        ]

    def forward(self, x):
        B, T, _ = x.shape
        toks = []
        for i,(g,proj) in enumerate(zip(self._split(x), self.group_projs)):
            tok = proj(g.reshape(B*T,-1)) + self.token_embed.weight[i]
            toks.append(tok.unsqueeze(1))
        tokens = torch.cat([self.cls_token.expand(B*T,-1,-1)] + toks, dim=1)
        return self.transformer(tokens)[:,0,:].reshape(B,T,self.d_model)


class TemporalTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, ff_dim, dropout, seq_len):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len+1, dropout=dropout)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers,
            norm=nn.LayerNorm(d_model))
        self.cls_token = nn.Parameter(torch.randn(1,1,d_model)*0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.pos_enc(torch.cat([self.cls_token.expand(B,-1,-1), x], dim=1))
        return self.transformer(x)[:,0,:]


class DualTransformer(nn.Module):
    def __init__(self, feat_dim, seq_len, num_classes, cfg):
        super().__init__()
        d = cfg.D_MODEL
        self.spatial  = SpatialTransformer(feat_dim, d, cfg.SPATIAL_HEADS,
            cfg.SPATIAL_LAYERS, cfg.SPATIAL_FF_DIM, cfg.SPATIAL_DROPOUT)
        self.temporal = TemporalTransformer(d, cfg.TEMPORAL_HEADS,
            cfg.TEMPORAL_LAYERS, cfg.TEMPORAL_FF_DIM, cfg.TEMPORAL_DROPOUT, seq_len)
        self.classifier = nn.Sequential(
            nn.Linear(d*2, cfg.CLASSIFIER_HIDDEN), nn.GELU(),
            nn.Dropout(cfg.DROPOUT_FINAL),
            nn.Linear(cfg.CLASSIFIER_HIDDEN, cfg.CLASSIFIER_HIDDEN//2), nn.GELU(),
            nn.Dropout(cfg.DROPOUT_FINAL/2),
            nn.Linear(cfg.CLASSIFIER_HIDDEN//2, num_classes),
        )

    def forward(self, x):
        s = self.spatial(x)
        t = self.temporal(s)
        return self.classifier(torch.cat([s.mean(1), t], dim=-1))


# ═══════════════════════════════════════════════════════════
# FEATURE EXTRACTOR (realtime, IMAGE mode)
# ═══════════════════════════════════════════════════════════

class RealtimeExtractor:
    def __init__(self):
        self._latest = dict(pose=None, face=None, hands=None, blendshapes=None)
        self._ts = 0

        hand_m = download_model('hand_landmarker.task')
        pose_m = download_model('pose_landmarker_heavy.task')
        face_m = download_model('face_landmarker.task')

        def _on_pose(r, img, ts):
            self._latest['pose'] = r.pose_landmarks[0] if r.pose_landmarks else None

        def _on_hand(r, img, ts):
            left = right = None
            if r.hand_landmarks and r.handedness:
                for i, hlms in enumerate(r.hand_landmarks):
                    cat = r.handedness[i][0].category_name
                    if cat == 'Left':  right = hlms
                    else:              left  = hlms
            self._latest['hands'] = (left, right)

        def _on_face(r, img, ts):
            self._latest['face'] = r.face_landmarks[0] if r.face_landmarks else None
            self._latest['blendshapes'] = r.face_blendshapes[0] if r.face_blendshapes else None

        self.pose_det = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=pose_m),
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_poses=1, min_pose_detection_confidence=0.4,
                min_pose_presence_confidence=0.4, min_tracking_confidence=0.4,
                result_callback=_on_pose))

        self.hand_det = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=hand_m),
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_hands=2, min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4, min_tracking_confidence=0.4,
                result_callback=_on_hand))

        self.face_det = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=face_m),
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_faces=1, min_face_detection_confidence=0.4,
                min_face_presence_confidence=0.4, min_tracking_confidence=0.4,
                output_face_blendshapes=True, result_callback=_on_face))

    def send_frame(self, rgb_frame):
        self._ts += 33
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        try: self.pose_det.detect_async(mp_img, self._ts)
        except: pass
        try: self.hand_det.detect_async(mp_img, self._ts)
        except: pass
        try: self.face_det.detect_async(mp_img, self._ts)
        except: pass

    def extract_features(self):
        """Trích xuất feature vector 339-dim từ kết quả mới nhất"""
        pose_lms    = self._latest['pose']
        face_lms    = self._latest['face']
        blendshapes = self._latest['blendshapes']
        hands       = self._latest['hands'] or (None, None)
        left_hand, right_hand = hands

        # ── Pose (75) ──
        pose_arr = np.zeros(75, dtype=np.float32)
        if pose_lms:
            for i in range(min(25, len(pose_lms))):
                pose_arr[i*3:i*3+3] = [pose_lms[i].x, pose_lms[i].y, pose_lms[i].z]

        # ── Face (90) ──
        face_arr = np.zeros(90, dtype=np.float32)
        if face_lms:
            for j, idx in enumerate(FACE_KEY_INDICES):
                if idx < len(face_lms):
                    face_arr[j*3:j*3+3] = [face_lms[idx].x,
                                            face_lms[idx].y,
                                            face_lms[idx].z]

        # ── Hands (126) ──
        hand_arr = np.zeros(126, dtype=np.float32)
        for hlms, offset in [(left_hand, 0), (right_hand, 63)]:
            if hlms:
                for k, lm in enumerate(hlms):
                    hand_arr[offset+k*3:offset+k*3+3] = [lm.x, lm.y, lm.z]

        # ── Blendshapes (17) ──
        blend_arr = np.zeros(17, dtype=np.float32)
        if blendshapes:
            bs = {c.category_name: c.score for c in blendshapes}
            for j, name in enumerate(KEY_BLENDSHAPES):
                blend_arr[j] = bs.get(name, 0.0)

        # ── Interactions (31) ──
        interact_arr = self._compute_interactions(pose_lms, left_hand, right_hand)

        feats = np.concatenate([pose_arr, face_arr, hand_arr,
                                 blend_arr, interact_arr])

        # Normalize theo shoulder center
        ls = feats[33:36]; rs = feats[36:39]
        center = (ls + rs) / 2
        if np.sum(np.abs(center)) > 1e-6:
            for i in range(25):
                feats[i*3]   -= center[0]
                feats[i*3+1] -= center[1]
            fs = 75
            for j in range(30):
                feats[fs+j*3]   -= center[0]
                feats[fs+j*3+1] -= center[1]
            hs = 165
            for k in range(42):
                feats[hs+k*3]   -= center[0]
                feats[hs+k*3+1] -= center[1]

        return feats

    def _compute_interactions(self, pose_lms, left_hand, right_hand):
        """
        31 features: cổ tay→cơ thể + ngón trỏ→mặt
          Mỗi tay: 7 dist cổ tay + 2 rel + 6 dist ngón trỏ→mặt = 15
          15 × 2 tay + 1 two-hand dist = 31
        """
        result = np.zeros(31, dtype=np.float32)
        if pose_lms is None:
            return result

        def xy(lm): return np.array([lm.x, lm.y], dtype=np.float32)

        # ── Vùng cơ thể ──
        head  = xy(pose_lms[0])
        l_ear = xy(pose_lms[7])  if pose_lms[7].visibility  > 0.3 else head.copy()
        r_ear = xy(pose_lms[8])  if pose_lms[8].visibility  > 0.3 else head.copy()
        ls    = xy(pose_lms[11]) if pose_lms[11].visibility > 0.3 else np.zeros(2)
        rs    = xy(pose_lms[12]) if pose_lms[12].visibility > 0.3 else np.zeros(2)
        chest = (ls + rs) / 2
        belly = (
            (ls + rs + xy(pose_lms[23]) + xy(pose_lms[24])) / 4
            if pose_lms[23].visibility > 0.3 and pose_lms[24].visibility > 0.3
            else chest + np.array([0.0, 0.15])
        )
        body_regions = [head, l_ear, r_ear, chest, belly, ls, rs]

        # ── Vùng mặt từ face landmarks ──
        face_lms = self._latest.get('face')
        if face_lms is not None and len(face_lms) >= 468:
            face_regions = [
                xy(face_lms[50]),    # má phải
                xy(face_lms[280]),   # má trái
                xy(face_lms[159]),   # mắt phải (giữa)
                xy(face_lms[386]),   # mắt trái (giữa)
                xy(face_lms[4]),     # mũi tip
                xy(face_lms[13]),    # môi trên
            ]
        else:
            # Fallback ước lượng từ pose head
            face_regions = [
                head + np.array([ 0.06,  0.02]),  # má phải
                head + np.array([-0.06,  0.02]),  # má trái
                head + np.array([ 0.03, -0.03]),  # mắt phải
                head + np.array([-0.03, -0.03]),  # mắt trái
                head + np.array([ 0.00,  0.02]),  # mũi
                head + np.array([ 0.00,  0.05]),  # môi
            ]

        idx = 0
        for hlms in [right_hand, left_hand]:
            wrist     = xy(hlms[0]) if hlms else np.zeros(2)
            index_tip = xy(hlms[8]) if hlms else np.zeros(2)  # ngón trỏ tip

            # 7 khoảng cách cổ tay → cơ thể
            for reg in body_regions:
                result[idx] = float(np.linalg.norm(wrist - reg)); idx += 1

            # 2 relative so với ngực
            result[idx] = float(wrist[0] - chest[0]); idx += 1
            result[idx] = float(wrist[1] - chest[1]); idx += 1

            # 6 khoảng cách ngón trỏ → vùng mặt [MỚI]
            for freg in face_regions:
                result[idx] = float(np.linalg.norm(index_tip - freg)); idx += 1

        # Khoảng cách 2 tay
        if right_hand and left_hand:
            result[idx] = float(np.linalg.norm(
                xy(right_hand[0]) - xy(left_hand[0])))
        idx += 1
        return result

    def get_latest(self):
        return self._latest

    def close(self):
        self.pose_det.close()
        self.hand_det.close()
        self.face_det.close()


# ═══════════════════════════════════════════════════════════
# UI RENDERER
# ═══════════════════════════════════════════════════════════

class UIRenderer:
    """Vẽ toàn bộ overlay UI lên frame OpenCV"""

    # Màu sắc
    BG_DARK   = (18, 18, 28)
    ACCENT    = (0, 220, 160)      # xanh lá mint
    ACCENT2   = (60, 160, 255)     # xanh dương
    WHITE     = (240, 240, 240)
    GRAY      = (140, 140, 150)
    RED       = (60, 60, 220)
    YELLOW    = (0, 200, 230)
    FONT      = cv2.FONT_HERSHEY_SIMPLEX

    @staticmethod
    def draw_rounded_rect(frame, x1, y1, x2, y2, color, radius=10,
                          filled=True, alpha=0.75):
        overlay = frame.copy()
        if filled:
            cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, -1)
            cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, -1)
            for cx, cy in [(x1+radius, y1+radius),(x2-radius, y1+radius),
                            (x1+radius, y2-radius),(x2-radius, y2-radius)]:
                cv2.circle(overlay, (cx, cy), radius, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

    @staticmethod
    def draw_bar(frame, x, y, w, h, value, color, bg=(50,50,60)):
        cv2.rectangle(frame, (x, y), (x+w, y+h), bg, -1)
        filled_w = max(0, int(w * min(value, 1.0)))
        if filled_w > 0:
            cv2.rectangle(frame, (x, y), (x+filled_w, y+h), color, -1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (80,80,90), 1)

    @classmethod
    def draw_header(cls, frame, w, h, fps, paused):
        UIRenderer.draw_rounded_rect(frame, 0, 0, w, 56,
                                      cls.BG_DARK, radius=0, alpha=0.88)
        # Title
        cv2.putText(frame, "VSL  REALTIME  RECOGNITION",
                    (16, 36), cls.FONT, 0.75, cls.ACCENT, 2)
        # FPS
        fps_col = cls.ACCENT if fps >= 20 else cls.YELLOW if fps >= 12 else cls.RED
        cv2.putText(frame, f"FPS {fps:4.1f}", (w-140, 36),
                    cls.FONT, 0.6, fps_col, 2)
        # Paused
        if paused:
            cv2.putText(frame, "[ PAUSED ]", (w//2-60, 36),
                        cls.FONT, 0.65, cls.YELLOW, 2)

    @classmethod
    def draw_prediction_panel(cls, frame, w, h, top_preds,
                               history, confidence_thr):
        """
        top_preds: list of (label, prob) sorted by prob desc
        history  : deque of recent confirmed labels
        """
        px, py = 12, 68
        pw, ph = 320, 220

        UIRenderer.draw_rounded_rect(frame, px, py, px+pw, py+ph,
                                      cls.BG_DARK, radius=12, alpha=0.82)

        # Header
        cv2.putText(frame, "PREDICTION", (px+12, py+26),
                    cls.FONT, 0.55, cls.ACCENT2, 1)
        cv2.line(frame, (px+8, py+32), (px+pw-8, py+32),
                 (60,60,80), 1)

        if not top_preds:
            cv2.putText(frame, "No detection", (px+12, py+70),
                        cls.FONT, 0.5, cls.GRAY, 1)
            return

        # Top prediction (lớn)
        top_label, top_prob = top_preds[0]
        label_col = cls.ACCENT if top_prob >= confidence_thr else cls.YELLOW
        cv2.putText(frame, top_label.upper().replace('_', ' '),
                    (px+12, py+68), cls.FONT, 0.85, label_col, 2)
        # Confidence bar top1
        cls.draw_bar(frame, px+12, py+76, pw-24, 14,
                     top_prob, label_col)
        cv2.putText(frame, f"{top_prob*100:.1f}%", (px+pw-60, py+88),
                    cls.FONT, 0.45, label_col, 1)

        # Top 2-N predictions
        for i, (lbl, prob) in enumerate(top_preds[1:], 1):
            ry = py + 100 + (i-1)*34
            if ry + 30 > py + ph:
                break
            cv2.putText(frame, f"{i+1}. {lbl.replace('_',' ')}",
                        (px+12, ry+14), cls.FONT, 0.48, cls.WHITE, 1)
            cls.draw_bar(frame, px+12, ry+18, pw-80, 9,
                         prob, cls.GRAY)
            cv2.putText(frame, f"{prob*100:.1f}%", (px+pw-66, ry+26),
                        cls.FONT, 0.4, cls.GRAY, 1)

        # Confidence threshold line (indicator)
        thr_x = px + 12 + int((pw-80) * confidence_thr)
        cv2.line(frame, (thr_x, py+76), (thr_x, py+76+14),
                 (100, 100, 220), 1)

    @classmethod
    def draw_history_panel(cls, frame, w, h, history):
        """Panel lịch sử nhận diện phía dưới bên trái"""
        px, py = 12, h - 220
        pw, ph = 320, 200

        UIRenderer.draw_rounded_rect(frame, px, py, px+pw, py+ph,
                                      cls.BG_DARK, radius=12, alpha=0.82)
        cv2.putText(frame, "HISTORY", (px+12, py+22),
                    cls.FONT, 0.5, cls.ACCENT2, 1)
        cv2.line(frame, (px+8, py+28), (px+pw-8, py+28),
                 (60,60,80), 1)

        hist_list = list(history)[-8:][::-1]  # 8 gần nhất, mới nhất trên
        for i, (ts_str, label, conf) in enumerate(hist_list):
            ty = py + 48 + i * 19
            if ty > py + ph - 10:
                break
            age_alpha = max(100, 220 - i * 20)
            c = (age_alpha, age_alpha, age_alpha)
            cv2.putText(frame, f"{ts_str}  {label.replace('_',' '):<18} {conf*100:4.0f}%",
                        (px+12, ty), cls.FONT, 0.38, c, 1)

    @classmethod
    def draw_skeleton(cls, frame, latest, w, h):
        """Vẽ keypoints nhẹ để biết model đang detect gì"""
        pose = latest.get('pose')
        hands = latest.get('hands') or (None, None)

        POSE_CONN = [(11,13),(13,15),(12,14),(14,16),(11,12),(11,23),(12,24)]
        if pose:
            for i, j in POSE_CONN:
                if (i < len(pose) and j < len(pose) and
                        pose[i].visibility > 0.5 and pose[j].visibility > 0.5):
                    p1 = (int(pose[i].x*w), int(pose[i].y*h))
                    p2 = (int(pose[j].x*w), int(pose[j].y*h))
                    cv2.line(frame, p1, p2, (0, 180, 120), 1)
            for idx in [0,11,12,13,14,15,16]:
                if idx < len(pose) and pose[idx].visibility > 0.5:
                    cv2.circle(frame, (int(pose[idx].x*w), int(pose[idx].y*h)),
                               4, cls.ACCENT, -1)

        HAND_CONN = [(0,1),(1,2),(2,3),(3,4),
                     (0,5),(5,6),(6,7),(7,8),
                     (0,9),(9,10),(10,11),(11,12),
                     (0,13),(13,14),(14,15),(15,16),
                     (0,17),(17,18),(18,19),(19,20),
                     (5,9),(9,13),(13,17)]
        colors_hand = [(0,255,140), (60,160,255)]
        for hlms, hc in zip(hands, colors_hand):
            if hlms is None: continue
            for i, j in HAND_CONN:
                p1 = (int(hlms[i].x*w), int(hlms[i].y*h))
                p2 = (int(hlms[j].x*w), int(hlms[j].y*h))
                cv2.line(frame, p1, p2, hc, 1)
            for lm in hlms:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 2, hc, -1)

    @classmethod
    def draw_buffer_bar(cls, frame, w, h, buf_len, seq_len):
        """Thanh tiến trình buffer thu thập frame"""
        bw   = w - 24
        bh   = 8
        bx   = 12
        by   = h - 28
        ratio = buf_len / seq_len

        UIRenderer.draw_rounded_rect(frame, bx-2, by-2, bx+bw+2, by+bh+2,
                                      cls.BG_DARK, radius=4, alpha=0.7)

        col = cls.ACCENT if ratio >= 1.0 else cls.ACCENT2
        cls.draw_bar(frame, bx, by, bw, bh, ratio, col)

        label = "READY" if ratio >= 1.0 else f"Buffer {buf_len}/{seq_len}"
        cv2.putText(frame, label,
                    (bx + bw//2 - 50, by - 4),
                    cls.FONT, 0.38, col, 1)

    @classmethod
    def draw_footer(cls, frame, w, h, confidence_thr):
        UIRenderer.draw_rounded_rect(frame, 0, h-18, w, h,
                                      cls.BG_DARK, radius=0, alpha=0.88)
        hints = (f"[Q] Thoat   [SPACE] Pause   [C] Clear   "
                 f"[S] Screenshot   [+/-] Threshold: {confidence_thr*100:.0f}%")
        cv2.putText(frame, hints, (10, h-4),
                    cls.FONT, 0.38, cls.GRAY, 1)

    @classmethod
    def draw_status_dot(cls, frame, w, h, recording):
        """Chấm đỏ nhấp nháy khi đang thu thập buffer"""
        if recording and int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (w - 22, 28), 7, cls.RED, -1)


# ═══════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════

class InferenceEngine:
    def __init__(self, model, label_map, device, seq_len,
                 confidence_thr=0.6, top_k=5,
                 smooth_window=5):
        self.model          = model
        self.idx2label      = {v: k for k, v in label_map.items()}
        self.device         = device
        self.seq_len        = seq_len
        self.confidence_thr = confidence_thr
        self.top_k          = top_k

        # Buffer frames
        self.frame_buffer = deque(maxlen=seq_len)

        # Smoothing: trung bình xác suất qua nhiều lần inference
        self.prob_history  = deque(maxlen=smooth_window)
        self.last_result   = None
        self.last_time     = 0

    def push_frame(self, feature_vec):
        self.frame_buffer.append(feature_vec)

    @property
    def buffer_ready(self):
        return len(self.frame_buffer) >= self.seq_len

    def predict(self):
        """Chạy inference và trả về top_k predictions"""
        if not self.buffer_ready:
            return None

        # Lấy seq_len frames gần nhất
        seq = np.stack(list(self.frame_buffer)[-self.seq_len:])  # (T, D)
        x   = torch.from_numpy(seq).unsqueeze(0).to(self.device)  # (1,T,D)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs  = Func.softmax(logits, dim=-1).cpu().numpy()[0]

        # Smoothing: trung bình prob qua smooth_window lần
        self.prob_history.append(probs)
        smooth_probs = np.mean(self.prob_history, axis=0)

        # Top-K
        top_idx  = np.argsort(smooth_probs)[::-1][:self.top_k]
        top_preds = [(self.idx2label.get(i, f'cls_{i}'),
                       float(smooth_probs[i]))
                      for i in top_idx]

        self.last_result = top_preds
        return top_preds


# ═══════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════

class RealtimeApp:
    def __init__(self, checkpoint_path, top_k=5,
                 confidence_thr=0.6, smooth_window=5):
        self.confidence_thr = confidence_thr
        self.top_k          = top_k

        # ── Load checkpoint ──
        print(f"\n  Loading checkpoint: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Khong tim thay checkpoint: {checkpoint_path}\n"
                "Hay train model truoc voi train_dual_transformer.py")

        ckpt      = torch.load(checkpoint_path, map_location=cfg.DEVICE)
        label_map = ckpt.get('label_map', {})
        if not label_map:
            raise ValueError("Checkpoint khong co label_map!")

        num_classes = len(label_map)
        print(f"  Classes ({num_classes}): {list(label_map.keys())}")

        model = DualTransformer(cfg.FEAT_DIM, cfg.SEQ_LEN,
                                num_classes, cfg)
        model.load_state_dict(ckpt['model_state'])
        model.to(cfg.DEVICE).eval()
        print(f"  Model loaded (epoch={ckpt.get('epoch','?')}, "
              f"val_acc={ckpt.get('val_acc',0)*100:.1f}%)")

        # ── Extractor + Engine ──
        print("\n  Khoi tao MediaPipe detectors...")
        self.extractor = RealtimeExtractor()
        self.engine    = InferenceEngine(
            model, label_map, cfg.DEVICE, cfg.SEQ_LEN,
            confidence_thr=confidence_thr, top_k=top_k,
            smooth_window=smooth_window,
        )

        # ── State ──
        self.history         = deque(maxlen=50)
        self.paused          = False
        self.last_confirmed  = None
        self.confirm_counter = 0
        self.CONFIRM_FRAMES  = 8   # cần N frame liên tục cùng kết quả mới log

        # FPS
        self._fps_buf = deque(maxlen=30)
        self._t_prev  = time.time()

        os.makedirs('screenshots', exist_ok=True)

    def _update_fps(self):
        now = time.time()
        self._fps_buf.append(1.0 / max(now - self._t_prev, 1e-6))
        self._t_prev = now
        return np.mean(self._fps_buf)

    def _maybe_log(self, top_preds):
        """Chỉ ghi vào history khi kết quả ổn định nhiều frame"""
        if not top_preds:
            self.confirm_counter = 0
            return

        top_label, top_prob = top_preds[0]
        if top_prob < self.confidence_thr:
            self.confirm_counter = 0
            return

        if top_label == self.last_confirmed:
            self.confirm_counter += 1
        else:
            self.confirm_counter = 1
            self.last_confirmed  = top_label

        # Ghi khi đạt ngưỡng confirm
        if self.confirm_counter == self.CONFIRM_FRAMES:
            ts = datetime.now().strftime('%H:%M:%S')
            self.history.append((ts, top_label, top_prob))
            print(f"  [{ts}] Detected: {top_label}  ({top_prob*100:.1f}%)")

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  LOI: Khong mo duoc webcam!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n  Camera: {w}x{h}")
        print("  ─" * 30)
        print("  Phim: [Q] Thoat  [SPACE] Pause  "
              "[C] Clear  [S] Screenshot  [+/-] Threshold")
        print("  ─" * 30 + "\n")

        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            fps   = self._update_fps()

            if not self.paused:
                # Gửi frame đến MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.extractor.send_frame(rgb)

                # Trích xuất và đẩy vào buffer
                feats = self.extractor.extract_features()
                self.engine.push_frame(feats)

                # Chạy inference
                top_preds = self.engine.predict()

                # Ghi history
                self._maybe_log(top_preds)
            else:
                top_preds = self.engine.last_result

            # ── Vẽ UI ──
            latest = self.extractor.get_latest()
            UIRenderer.draw_skeleton(frame, latest, w, h)
            UIRenderer.draw_header(frame, w, h, fps, self.paused)
            UIRenderer.draw_prediction_panel(
                frame, w, h, top_preds or [],
                self.history, self.confidence_thr)
            UIRenderer.draw_history_panel(frame, w, h, self.history)
            UIRenderer.draw_buffer_bar(
                frame, w, h,
                len(self.engine.frame_buffer), cfg.SEQ_LEN)
            UIRenderer.draw_footer(frame, w, h, self.confidence_thr)
            UIRenderer.draw_status_dot(frame, w, h, not self.paused)

            cv2.imshow('VSL Realtime Inference', frame)

            # ── Phím bấm ──
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                break

            elif key == ord(' '):
                self.paused = not self.paused
                print(f"  {'PAUSED' if self.paused else 'RESUMED'}")

            elif key == ord('c') or key == ord('C'):
                self.history.clear()
                self.engine.frame_buffer.clear()
                self.engine.prob_history.clear()
                self.confirm_counter = 0
                self.last_confirmed  = None
                print("  History cleared")

            elif key == ord('s') or key == ord('S'):
                ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
                fn  = f"screenshots/screenshot_{ts}.png"
                cv2.imwrite(fn, frame)
                print(f"  Screenshot: {fn}")

            elif key in (ord('+'), ord('=')):
                self.confidence_thr = min(0.99, self.confidence_thr + 0.05)
                self.engine.confidence_thr = self.confidence_thr
                print(f"  Threshold: {self.confidence_thr*100:.0f}%")

            elif key == ord('-'):
                self.confidence_thr = max(0.05, self.confidence_thr - 0.05)
                self.engine.confidence_thr = self.confidence_thr
                print(f"  Threshold: {self.confidence_thr*100:.0f}%")

        cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()
        print("\n  Da thoat.\n")


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

def find_latest_checkpoint(ckpt_dir='checkpoints'):
    """Tự động tìm checkpoint mới nhất nếu không chỉ định"""
    if not os.path.isdir(ckpt_dir):
        return None
    pts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
    return os.path.join(ckpt_dir, pts[-1]) if pts else None


def main():
    parser = argparse.ArgumentParser(
        description='VSL Realtime Inference')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Duong dan den file .pt (tu dong tim neu bo trong)')
    parser.add_argument('--top_k',      type=int,   default=5,
                        help='So ket qua top-K hien thi (mac dinh 5)')
    parser.add_argument('--threshold',  type=float, default=0.60,
                        help='Nguong tin cay (0-1, mac dinh 0.60)')
    parser.add_argument('--smooth',     type=int,   default=5,
                        help='So frame smoothing (mac dinh 5)')
    args = parser.parse_args()

    ckpt = args.checkpoint or find_latest_checkpoint()
    if ckpt is None:
        print("\n  LOI: Khong tim thay checkpoint!")
        print("  Hay train model truoc hoac chi dinh --checkpoint <path>\n")
        return

    print("\n" + "="*60)
    print(" VSL REALTIME INFERENCE ".center(60, "="))
    print("="*60)
    print(f"  Checkpoint : {ckpt}")
    print(f"  Top-K      : {args.top_k}")
    print(f"  Threshold  : {args.threshold*100:.0f}%")
    print(f"  Smoothing  : {args.smooth} frames")
    print(f"  Device     : {cfg.DEVICE}")

    app = RealtimeApp(
        checkpoint_path = ckpt,
        top_k           = args.top_k,
        confidence_thr  = args.threshold,
        smooth_window   = args.smooth,
    )
    app.run()


if __name__ == '__main__':
    main()