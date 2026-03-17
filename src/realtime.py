"""
realtime.py - Realtime VSL Recognition + Live Emotion Detection
================================================================
BƯỚC 4 trong pipeline

Tích hợp 2 model:
  ┌─ checkpoints/best.pt           : HandAwareVSLClassifier (208-dim)
  └─ checkpoints/emotion_cnn_best.pth : EfficientNet-B2 (7 emotion classes)

Hand Symmetry — 3 giải pháp đảm bảo tay trái = tay phải cùng nhận diện:
  1. Dominant Hand Slot  : 1 tay → luôn slot[45], không phân biệt trái/phải
  2. Mirror Augmentation : inference 2 lần (gốc + flip-x), merge conf
  3. Dual-pass Fusion    : lấy max confidence giữa 2 pass, smooth qua history

Pipeline mỗi frame:
  Camera → MediaPipe (Hand + Pose) → feat(201,) → normalize
         → [Pass A: gốc] + [Pass B: mirror flip-x] → merge top-K → display
  Camera → Haar → crop face → EmotionCNN (async thread) → emotion bar

Phím tắt:
  A     : Toggle auto/manual emotion
  C     : Toggle curl panel
  E     : Toggle emotion bar chart
  M     : Toggle mirror mode (bật/tắt dual-pass)
  1-7   : Gán emotion thủ công
  +/-   : Confidence threshold
  F     : Toggle fullscreen
  R     : Reset buffer
  Q/ESC : Thoát

Chạy:
  python realtime.py
  python realtime.py --model checkpoints/best.pt
  python realtime.py --model checkpoints/best.pt --emotion_model checkpoints/emotion_cnn_best.pth
  python realtime.py --no_emotion
  python realtime.py --no_mirror     # tắt mirror augmentation
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
import urllib.request
from pathlib import Path
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from torchvision import transforms
from torchvision.models import efficientnet_b2


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

class Config:
    SEQ_LEN      = 64
    FEAT_DIM     = 208

    POSE_KEY_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    EMOTIONS   = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    EMOTION_ID = {e: i for i, e in enumerate(EMOTIONS)}

    CNN_REMAP = {
        "angry": "angry", "cry": "sad",   "disgust": "disgust",
        "neutral": "neutral", "scare": "fear", "smile": "happy",
        "surprise": "surprise", "happy": "happy", "sad": "sad", "fear": "fear",
    }

    EMOTION_COLORS = {
        "angry":    (50,  50,  230),
        "disgust":  (128, 128, 0  ),
        "fear":     (200, 0,   128),
        "happy":    (40,  230, 40 ),
        "sad":      (230, 160, 0  ),
        "surprise": (230, 200, 0  ),
        "neutral":  (170, 170, 170),
    }

    FINGER_ORDER  = ['thumb', 'index', 'middle', 'ring', 'pinky']
    FINGER_JOINTS = {
        'thumb':  (1, 2, 3, 4),    'index':  (5, 6, 7, 8),
        'middle': (9, 10, 11, 12), 'ring':   (13, 14, 15, 16),
        'pinky':  (17, 18, 19, 20),
    }
    WRIST_IDX  = 0
    MIDDLE_MCP = 9

    CONF_THRESHOLD_DEFAULT = 0.60
    CONF_THRESHOLD_MIN     = 0.30
    CONF_THRESHOLD_MAX     = 0.95
    CONF_THRESHOLD_STEP    = 0.05
    STABLE_FRAMES          = 3
    TOP_K                  = 5
    EMOTION_EVERY_N        = 3

    # Mirror: inference mỗi N frame để tiết kiệm CPU
    # Frame lẻ dùng kết quả mirror cache từ frame chẵn trước
    MIRROR_EVERY_N         = 2


cfg = Config()

_PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_URLS = {
    'hand_landmarker.task':
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
        'hand_landmarker/float16/1/hand_landmarker.task',
    'pose_landmarker_heavy.task':
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
        'pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
}


def ensure_mediapipe_model(name):
    path = _PROJECT_ROOT / name
    if not path.exists():
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(MODEL_URLS[name], str(path))
    return str(path)


# ═══════════════════════════════════════════════════════════════════
# FINGER CURL CALCULATOR
# ═══════════════════════════════════════════════════════════════════

def _angle(a, b, c):
    ba = a - b; bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def compute_finger_curl(lms: np.ndarray) -> np.ndarray:
    """lms: (21,3) → curl (15,): [curl_ratio, bend_angle, tip_dist] × 5 ngón."""
    curl  = np.zeros(15, dtype=np.float32)
    wrist = lms[cfg.WRIST_IDX]
    scale = np.linalg.norm(lms[cfg.MIDDLE_MCP] - wrist) + 1e-8
    for i, fname in enumerate(cfg.FINGER_ORDER):
        mcp_i, pip_i, dip_i, tip_i = cfg.FINGER_JOINTS[fname]
        mcp = lms[mcp_i]; pip = lms[pip_i]
        dip = lms[dip_i]; tip = lms[tip_i]
        d_tip = np.linalg.norm(tip - wrist)
        d_mcp = np.linalg.norm(mcp - wrist) + 1e-8
        angle = _angle(mcp, pip, dip)
        b = i * 3
        curl[b]   = float(np.clip(d_tip / d_mcp / 2.0, 0, 1))
        curl[b+1] = float(np.clip((np.pi - angle) / np.pi, 0, 1))
        curl[b+2] = float(np.clip(d_tip / scale / 3.0, 0, 1))
    return curl


# ═══════════════════════════════════════════════════════════════════
# HAND SYMMETRY — MIRROR FEATURE
# ═══════════════════════════════════════════════════════════════════

def mirror_feat(feat: np.ndarray) -> np.ndarray:
    """
    Tạo phiên bản mirror (tay ngược) của feature vector (201,).

    Thực hiện 3 bước:
      1. Flip x-coordinate của cả 2 tay: x → 1-x  (đối xứng qua trục dọc)
      2. Hoán đổi slot tay trái ↔ tay phải (45:108 ↔ 108:171)
      3. Hoán đổi curl trái ↔ phải (171:186 ↔ 186:201)
      4. Flip x của pose keypoints (11 cặp vai/tay)

    Kết quả: nếu data train dùng tay phải, mirror sẽ
    biến input tay trái thành "như thể tay phải" → model nhận diện được.
    """
    m = feat.copy()

    # 1. Flip x tay trái (slot 45:108) — 21 landmarks × 3
    for j in range(21):
        m[45 + j*3] = 1.0 - feat[45 + j*3]

    # 2. Flip x tay phải (slot 108:171)
    for j in range(21):
        m[108 + j*3] = 1.0 - feat[108 + j*3]

    # 3. Hoán đổi 2 slot tay (xyz)
    left_xyz  = m[45:108].copy()
    right_xyz = m[108:171].copy()
    m[45:108]  = right_xyz
    m[108:171] = left_xyz

    # 4. Hoán đổi curl trái ↔ phải
    curl_l = m[171:186].copy()
    curl_r = m[186:201].copy()
    m[171:186] = curl_r
    m[186:201] = curl_l

    # 5. Flip x pose keypoints (index 0→14, mỗi lm có 3 giá trị xyz)
    for i in range(15):
        m[i*3] = 1.0 - feat[i*3]   # chỉ flip x, giữ y và z

    return m


# ═══════════════════════════════════════════════════════════════════
# EMOTION CNN
# ═══════════════════════════════════════════════════════════════════

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = efficientnet_b2(weights=None)
        in_f = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_f, 512),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


_FACE_PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_HAAR = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def load_emotion_cnn(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        print(f"  [WARN] Không tìm thấy emotion model: {ckpt_path}")
        return None, None
    ckpt = torch.load(ckpt_path, map_location=device)
    cnn_labels = ckpt.get('emotions',
                  ['angry', 'cry', 'disgust', 'neutral', 'scare', 'smile', 'surprise'])
    model = EmotionCNN(num_classes=len(cnn_labels)).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    bal = ckpt.get('bal_acc', ckpt.get('val_acc', 0))
    print(f"  [EmoCNN] {ckpt_path}  bal_acc={bal*100:.1f}%  labels={cnn_labels}")
    return model, cnn_labels


def _crop_face(frame_bgr):
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _HAAR.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    pad = int(0.15 * w)
    x1 = max(0, x-pad); y1 = max(0, y-pad)
    x2 = min(frame_bgr.shape[1], x+w+pad)
    y2 = min(frame_bgr.shape[0], y+h+pad)
    face = frame_bgr[y1:y2, x1:x2]
    return cv2.cvtColor(face, cv2.COLOR_BGR2RGB) if face.size > 0 else None


class EmotionDetector:
    """Async worker thread cho EmotionCNN."""

    def __init__(self, model, cnn_labels, device):
        self.model      = model
        self.cnn_labels = cnn_labels
        self.device     = device
        self._lock    = threading.Lock()
        self._result  = ("neutral", {e: 0.0 for e in cfg.EMOTIONS})
        self._q       = deque(maxlen=1)
        self._running = True
        threading.Thread(target=self._worker, daemon=True).start()

    def push_frame(self, frame_bgr):
        self._q.append(frame_bgr.copy())

    def get_result(self):
        with self._lock:
            return self._result

    def _worker(self):
        while self._running:
            if not self._q:
                time.sleep(0.005)
                continue
            frame = self._q.pop()
            try:
                face = _crop_face(frame)
                if face is None:
                    time.sleep(0.01)
                    continue
                t = _FACE_PREPROCESS(face).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    probs = torch.softmax(self.model(t), dim=1)[0].cpu().numpy()
                mapped = {e: 0.0 for e in cfg.EMOTIONS}
                for i, lbl in enumerate(self.cnn_labels):
                    tgt = cfg.CNN_REMAP.get(lbl, lbl)
                    if tgt in mapped:
                        mapped[tgt] += float(probs[i])
                total = sum(mapped.values()) + 1e-8
                mapped = {k: v/total for k, v in mapped.items()}
                best = max(mapped, key=mapped.get)
                with self._lock:
                    self._result = (best, mapped)
            except Exception:
                pass

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════════════════════
# VSL MODEL — HandAwareVSLClassifier (208-dim)
# Tên attribute phải match chính xác train.py
# ═══════════════════════════════════════════════════════════════════

_FG = {'wrist': [0], 'thumb': [1,2,3,4], 'index': [5,6,7,8],
       'middle': [9,10,11,12], 'ring': [13,14,15,16], 'pinky': [17,18,19,20]}
_FN = list(_FG.keys())
_NF = 6
_FE = [(0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(2,3),(3,4),(4,5)]


class _FingerEncoder(nn.Module):
    def __init__(self, out_dim=32, dr=0.1):
        super().__init__()
        self.register_buffer('wrist_idx',  torch.tensor([0], dtype=torch.long))
        self.register_buffer('thumb_idx',  torch.tensor([1,2,3,4], dtype=torch.long))
        self.register_buffer('index_idx',  torch.tensor([5,6,7,8], dtype=torch.long))
        self.register_buffer('middle_idx', torch.tensor([9,10,11,12], dtype=torch.long))
        self.register_buffer('ring_idx',   torch.tensor([13,14,15,16], dtype=torch.long))
        self.register_buffer('pinky_idx',  torch.tensor([17,18,19,20], dtype=torch.long))
        self.encoders = nn.ModuleDict({
            fn: nn.Sequential(
                nn.Linear(len(idx)*3, out_dim*2), nn.LayerNorm(out_dim*2),
                nn.GELU(), nn.Dropout(dr), nn.Linear(out_dim*2, out_dim),
                nn.LayerNorm(out_dim))
            for fn, idx in _FG.items()})

    def forward(self, hand):
        B, T = hand.shape[:2]
        return torch.stack([
            self.encoders[fn](hand[:,:,_FG[fn],:].reshape(B,T,-1))
            for fn in _FN], dim=2)


class _CurlEncoder(nn.Module):
    def __init__(self, out_dim=32, dr=0.1):
        super().__init__()
        per = out_dim // 5; rem = out_dim - per*4
        self.finger_encs = nn.ModuleList([
            nn.Sequential(nn.Linear(3,16), nn.GELU(),
                          nn.Linear(16, rem if i==4 else per))
            for i in range(5)])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(out_dim), nn.Linear(out_dim,out_dim),
            nn.GELU(), nn.Dropout(dr))

    def forward(self, c):
        return self.out_proj(torch.cat(
            [self.finger_encs[i](c[:,:,i*3:i*3+3]) for i in range(5)], -1))


class _FGA(nn.Module):
    def __init__(self, d, h=4, dr=0.1):
        super().__init__()
        self.h = h; self.hd = d // h
        adj = torch.zeros(_NF, _NF)
        for i, j in _FE: adj[i,j] = adj[j,i] = 1.0
        adj += torch.eye(_NF)
        self.register_buffer('adj', adj / adj.sum(-1, keepdim=True).clamp(1))
        self.W_q = nn.Linear(d,d,bias=False); self.W_k = nn.Linear(d,d,bias=False)
        self.W_v = nn.Linear(d,d,bias=False); self.W_o = nn.Linear(d,d)
        self.drop = nn.Dropout(dr); self.norm = nn.LayerNorm(d)

    def forward(self, x):
        B,T,N,D = x.shape; BT = B*T; xf = x.reshape(BT,N,D)
        Q = self.W_q(xf).reshape(BT,N,self.h,self.hd).transpose(1,2)
        K = self.W_k(xf).reshape(BT,N,self.h,self.hd).transpose(1,2)
        V = self.W_v(xf).reshape(BT,N,self.h,self.hd).transpose(1,2)
        s = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.hd)
        s = s.masked_fill(self.adj.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        out = torch.matmul(self.drop(torch.softmax(s,-1)), V)
        return self.norm(self.W_o(out.transpose(1,2).reshape(BT,N,D)).reshape(B,T,N,D) + x)


class _HCA(nn.Module):
    def __init__(self, d, h=4, dr=0.1):
        super().__init__()
        self.cross_lr = nn.MultiheadAttention(d,h,dropout=dr,batch_first=True)
        self.cross_rl = nn.MultiheadAttention(d,h,dropout=dr,batch_first=True)
        self.norm_l = nn.LayerNorm(d); self.norm_r = nn.LayerNorm(d)

    def forward(self, l, r):
        B,T,N,D = l.shape; BT = B*T
        lf = l.reshape(BT,N,D); rf = r.reshape(BT,N,D)
        lc,_ = self.cross_lr(lf,rf,rf); rc,_ = self.cross_rl(rf,lf,lf)
        return self.norm_l(lf+lc).reshape(B,T,N,D), self.norm_r(rf+rc).reshape(B,T,N,D)


class _HAgg(nn.Module):
    def __init__(self, fd, cd, pd, ed, od, dr=0.1):
        super().__init__()
        self.finger_pool = nn.Linear(_NF, 1, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(fd*2+cd*2+pd+ed, od), nn.LayerNorm(od),
            nn.GELU(), nn.Dropout(dr))

    def forward(self, lf, rf, cl, cr, p, e):
        lp = self.finger_pool(lf.transpose(-1,-2)).squeeze(-1)
        rp = self.finger_pool(rf.transpose(-1,-2)).squeeze(-1)
        return self.proj(torch.cat([lp,rp,cl,cr,p,e], -1))


class _TT(nn.Module):
    def __init__(self, d, h=8, nl=4, dr=0.1):
        super().__init__()
        pe  = torch.zeros(256, d)
        pos = torch.arange(256).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d,2).float() * (-math.log(10000)/d))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
        lay = nn.TransformerEncoderLayer(d,h,d*4,dr,batch_first=True,
                                         activation='gelu', norm_first=True)
        self.encoder = nn.TransformerEncoder(lay, nl)
        self.norm    = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(self.encoder(x + self.pe[:,:x.size(1)]))


class HandAwareVSLClassifier(nn.Module):
    def __init__(self, num_classes, finger_dim=32, curl_dim=32, temporal_dim=256,
                 temporal_heads=8, temporal_layers=4, graph_heads=4, dropout=0.2):
        super().__init__()
        self.finger_encoder   = _FingerEncoder(finger_dim, dropout)
        self.curl_encoder     = _CurlEncoder(curl_dim, dropout)
        self.graph_attn_left  = _FGA(finger_dim, graph_heads, dropout)
        self.graph_attn_right = _FGA(finger_dim, graph_heads, dropout)
        self.cross_attn       = _HCA(finger_dim, graph_heads, dropout)
        self.aggregator       = _HAgg(finger_dim, curl_dim, 45, 7,
                                      temporal_dim, dropout)
        self.temporal      = _TT(temporal_dim, temporal_heads, temporal_layers, dropout)
        self.temporal_pool = nn.Linear(temporal_dim, 1)
        mid = max(num_classes*4, 256)
        self.classifier = nn.Sequential(
            nn.LayerNorm(temporal_dim), nn.Linear(temporal_dim, mid), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(mid, mid//2), nn.GELU(),
            nn.Dropout(dropout/2), nn.Linear(mid//2, num_classes))

    def forward(self, x):
        B, T, _ = x.shape
        pose = x[:,:,0:45];    lh = x[:,:,45:108].view(B,T,21,3)
        rh   = x[:,:,108:171].view(B,T,21,3)
        cl   = x[:,:,171:186]; cr = x[:,:,186:201]; emo = x[:,:,201:208]
        lf = self.graph_attn_left(self.finger_encoder(lh))
        rf = self.graph_attn_right(self.finger_encoder(rh))
        lf, rf = self.cross_attn(lf, rf)
        t = self.temporal(self.aggregator(
            lf, rf, self.curl_encoder(cl), self.curl_encoder(cr), pose, emo))
        w = torch.softmax(self.temporal_pool(t).squeeze(-1), -1)
        return self.classifier((t * w.unsqueeze(-1)).sum(1))


def load_vsl_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    lm   = ckpt['label_map']
    mc   = ckpt.get('cfg', {})
    model = HandAwareVSLClassifier(
        num_classes    = len(lm),
        finger_dim     = mc.get('FINGER_DIM',      32),
        curl_dim       = mc.get('CURL_EMBED_DIM',  32),
        temporal_dim   = mc.get('TEMPORAL_DIM',   256),
        temporal_heads = mc.get('TEMPORAL_HEADS',   8),
        temporal_layers= mc.get('TEMPORAL_LAYERS',  4),
        graph_heads    = mc.get('GRAPH_HEADS',       4),
        dropout        = mc.get('DROPOUT',         0.2),
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    idx2label = {v: k for k, v in lm.items()}
    print(f"  [VSL] {ckpt_path}  val_acc={ckpt.get('val_acc',0)*100:.2f}%  labels={len(lm)}")
    return model, idx2label


# ═══════════════════════════════════════════════════════════════════
# MEDIAPIPE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════

class LandmarkExtractor:
    def __init__(self):
        print("  Init MediaPipe...")
        self._lock = threading.Lock()
        self._data = {'pose': None, 'hands': []}
        self._ts   = 0

        self.hand_det = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=ensure_mediapipe_model('hand_landmarker.task')),
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=self._on_hand))

        self.pose_det = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=ensure_mediapipe_model('pose_landmarker_heavy.task')),
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=self._on_pose))
        print("  MediaPipe ready.")

    def _on_pose(self, result, img, ts):
        with self._lock:
            self._data['pose'] = (result.pose_landmarks[0]
                                   if result.pose_landmarks else None)

    def _on_hand(self, result, img, ts):
        with self._lock:
            self._data['hands'] = list(result.hand_landmarks) \
                                   if result.hand_landmarks else []

    def send(self, rgb):
        self._ts += 33
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        for det in [self.pose_det, self.hand_det]:
            try: det.detect_async(mp_img, self._ts)
            except Exception: pass

    def extract(self):
        """
        Returns (feat_201, raw, hand_side)
        feat_201: pose(45) + slot_A_xyz(63) + slot_B_xyz(63) + curl_A(15) + curl_B(15)

        DOMINANT HAND SLOT (Giải pháp 1):
          - 1 tay → slot[45] (slot "tay chính"), không phân biệt trái/phải
          - 2 tay → sort by x: tay nhỏ x → slot[45], tay lớn x → slot[108]
          Điều này nhất quán với video_to_npy.py lúc train.

        hand_side: 'left' | 'right' | 'both' | 'none'
        """
        with self._lock:
            feat  = np.zeros(201, dtype=np.float32)
            pose  = self._data['pose']
            hands = list(self._data['hands'])
            raw   = {'pose': pose, 'hands': hands}

        hand_side = 'none'

        if pose:
            for i, idx in enumerate(cfg.POSE_KEY_INDICES):
                lm = pose[idx]
                feat[i*3:(i+1)*3] = [lm.x, lm.y, lm.z]

        if hands:
            if len(hands) == 1:
                # DOMINANT SLOT: 1 tay → luôn slot[45]
                h = hands[0]
                for j, lm in enumerate(h):
                    feat[45+j*3:45+j*3+3] = [lm.x, lm.y, lm.z]
                lms = np.array([[lm.x, lm.y, lm.z] for lm in h], np.float32)
                feat[171:186] = compute_finger_curl(lms)
                # Xác định tay nào đang dùng dựa vào x trung bình
                mean_x = np.mean([lm.x for lm in h])
                hand_side = 'right' if mean_x > 0.5 else 'left'
            else:
                # 2 tay: sort by x
                sorted_h = sorted(hands, key=lambda h: np.mean([lm.x for lm in h]))
                for h, xs, cs in zip(sorted_h[:2], [45, 108], [171, 186]):
                    for j, lm in enumerate(h):
                        feat[xs+j*3:xs+j*3+3] = [lm.x, lm.y, lm.z]
                    lms = np.array([[lm.x,lm.y,lm.z] for lm in h], np.float32)
                    feat[cs:cs+15] = compute_finger_curl(lms)
                hand_side = 'both'

        return feat, raw, hand_side

    def has_hands(self):
        with self._lock: return len(self._data['hands']) > 0

    def get_curl(self):
        with self._lock:
            if not self._data['hands']: return None
            lms = np.array([[lm.x,lm.y,lm.z] for lm in self._data['hands'][0]], np.float32)
            return compute_finger_curl(lms)

    def close(self):
        self.hand_det.close()
        self.pose_det.close()


# ═══════════════════════════════════════════════════════════════════
# FEATURE NORMALIZATION
# ═══════════════════════════════════════════════════════════════════

def normalize_feat(feat: np.ndarray) -> np.ndarray:
    """Normalize pose(0:45) + hand_xyz(45:171). Curl(171:201) giữ nguyên."""
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


# ═══════════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════════

class C:
    BG_DARK  = (12,  12,  12)
    BG_PANEL = (28,  28,  28)
    BG_CARD  = (45,  45,  45)
    WHITE    = (255, 255, 255)
    MUTED    = (100, 100, 100)
    GRAY     = (170, 170, 170)
    GREEN    = (40,  220, 80 )
    TEAL     = (0,   200, 160)
    ORANGE   = (0,   190, 255)
    RED      = (50,  50,  230)
    YELLOW   = (0,   220, 220)
    PURPLE   = (200, 80,  200)


def rrect(img, p1, p2, col, r=8, t=-1):
    x1,y1 = p1; x2,y2 = p2
    if t == -1:
        cv2.rectangle(img, (x1+r,y1), (x2-r,y2), col, -1)
        cv2.rectangle(img, (x1,y1+r), (x2,y2-r), col, -1)
        for cx,cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.circle(img, (cx,cy), r, col, -1)
    else:
        cv2.line(img,(x1+r,y1),(x2-r,y1),col,t)
        cv2.line(img,(x1+r,y2),(x2-r,y2),col,t)
        cv2.line(img,(x1,y1+r),(x1,y2-r),col,t)
        cv2.line(img,(x2,y1+r),(x2,y2-r),col,t)
        for a,cx,cy in [(180,x1+r,y1+r),(270,x2-r,y1+r),(90,x1+r,y2-r),(0,x2-r,y2-r)]:
            cv2.ellipse(img,(cx,cy),(r,r),a,0,90,col,t)


def put(img, text, pos, scale=0.5, col=C.WHITE, thick=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def draw_hand_skeleton(img, hands, W, H):
    CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
    TIPS = {4, 8, 12, 16, 20}
    for hi, hand in enumerate(hands):
        pts = [(int(lm.x*W), int(lm.y*H)) for lm in hand]
        ec  = (0, 210, 170) if hi == 0 else (210, 170, 0)
        for a, b in CONN:
            if a < len(pts) and b < len(pts):
                cv2.line(img, pts[a], pts[b], (60,140,110), 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            r = 6 if i in TIPS or i == 0 else 3
            cv2.circle(img, pt, r, ec, -1, cv2.LINE_AA)
            cv2.circle(img, pt, r, (10,10,10), 1, cv2.LINE_AA)


def draw_conf_bar(img, x, y, w, h, conf, thresh, label, rank):
    rrect(img, (x,y), (x+w,y+h), C.BG_CARD, r=5)
    bx = x+10; by = y+h-14; bw = w-20; bh = 7
    col = (C.GREEN if conf >= thresh else C.ORANGE if conf >= thresh*0.7 else C.RED)
    cv2.rectangle(img, (bx,by), (bx+bw,by+bh), (55,55,55), -1)
    fw = int(bw * conf)
    if fw > 0: cv2.rectangle(img, (bx,by), (bx+fw,by+bh), col, -1)
    tx = bx + int(bw * thresh)
    cv2.line(img, (tx,by-2), (tx,by+bh+2), C.ORANGE, 2)
    rank_cols = [C.GREEN, C.TEAL, C.YELLOW, C.GRAY, C.MUTED]
    cv2.circle(img, (x+16,y+18), 10, rank_cols[min(rank,4)], -1)
    put(img, str(rank+1), (x+11,y+23), 0.38, C.BG_DARK, 2)
    lc = C.WHITE if conf >= thresh else C.GRAY
    put(img, label[:22], (x+33,y+23), 0.50, lc, 2 if rank==0 else 1)
    put(img, f"{conf*100:.1f}%", (x+w-58,y+23), 0.40, col, 1)


def draw_emotion_panel(img, x, y, w, probs, current):
    N = len(cfg.EMOTIONS); rh = 26; ph = 18 + N*rh + 10
    bar_w = w - 120
    rrect(img, (x,y), (x+w, y+ph), C.BG_PANEL, r=8)
    put(img, "EMOTION CNN", (x+8, y+14), 0.37, C.MUTED)
    for i, emo in enumerate(cfg.EMOTIONS):
        by = y + 20 + i*rh
        p  = float(probs.get(emo, 0))
        ec = cfg.EMOTION_COLORS.get(emo, (170,170,170))
        is_top = (emo == current)
        lc = ec if is_top else C.GRAY
        put(img, emo[:7], (x+8, by+16), 0.40, lc, 2 if is_top else 1)
        bx = x + 62
        cv2.rectangle(img, (bx, by+4), (bx+bar_w, by+18), (55,55,55), -1)
        fw = int(bar_w * p)
        if fw > 1: cv2.rectangle(img, (bx, by+4), (bx+fw, by+18), ec, -1)
        put(img, f"{p*100:.0f}%", (bx+bar_w+5, by+16), 0.38, C.GRAY)
        if is_top: cv2.circle(img, (x+55, by+11), 4, ec, -1)


def draw_curl_panel(img, curl, x, y, w=225):
    if curl is None: return
    names = ['Cai','Tro','Giua','Nhan','Ut']
    bw = w - 88; ph = 16 + len(names)*26
    rrect(img, (x,y), (x+w, y+ph), C.BG_PANEL, r=8)
    put(img, "CURL NGON TAY", (x+8, y+13), 0.33, C.MUTED)
    for i, name in enumerate(names):
        ratio = float(curl[i*3])
        by = y + 18 + i*26
        put(img, name, (x+6, by+10), 0.38, C.GRAY)
        bx = x + 42
        cv2.rectangle(img,(bx,by),(bx+bw,by+10),(55,55,55),-1)
        fw = int(bw*ratio)
        rc = int(255*(1-ratio)); gc = int(200*ratio)
        if fw > 0: cv2.rectangle(img,(bx,by),(bx+fw,by+10),(0,gc,rc),-1)
        st = "DUOI" if ratio>0.6 else ("CO" if ratio<0.4 else "~")
        sc = C.GREEN if ratio>0.6 else C.RED if ratio<0.4 else C.MUTED
        put(img, st, (bx+bw+5, by+10), 0.30, sc)


# ═══════════════════════════════════════════════════════════════════
# REALTIME ENGINE
# ═══════════════════════════════════════════════════════════════════

class RealtimeEngine:

    def __init__(self, vsl_path, emo_path, no_emotion=False, no_mirror=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n{'='*62}")
        print(" VSL + EMOTION REALTIME (Hand Symmetry) ".center(62, '='))
        print('='*62)
        print(f"  Device      : {self.device}")

        self.vsl_model, self.idx2label = load_vsl_model(vsl_path, self.device)

        self.emo_detector = None
        self.use_emo_cnn  = False
        if not no_emotion and emo_path:
            emo_model, cnn_labels = load_emotion_cnn(emo_path, self.device)
            if emo_model is not None:
                self.emo_detector = EmotionDetector(emo_model, cnn_labels, self.device)
                self.use_emo_cnn  = True

        self.extractor = LandmarkExtractor()

        self.display_names = {}
        for p in ['data/processed/display_names.json', 'data/display_names.json']:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    self.display_names = json.load(f)
                break

        # ── State ─────────────────────────────────────────────────
        self.buffer       = deque(maxlen=cfg.SEQ_LEN)
        self.pred_history = deque(maxlen=5)

        self.conf_thresh  = cfg.CONF_THRESHOLD_DEFAULT
        self.top_preds    = []
        self.cur_label    = ""
        self.cur_conf     = 0.0
        self.hand_side    = 'none'   # 'left' | 'right' | 'both' | 'none'

        # Emotion
        self.auto_emo   = True
        self.emo_name   = "neutral"
        self.emo_probs  = {e: 0.0 for e in cfg.EMOTIONS}
        self.emo_probs["neutral"] = 1.0
        self.emo_manual = "neutral"
        self.emo_smooth = deque(maxlen=8)

        # Mirror mode (Giải pháp 2 + 3)
        self.use_mirror      = not no_mirror
        self._mirror_cache   = {}   # cache kết quả pass mirror để không infer mỗi frame
        self._mirror_counter = 0

        # UI
        self.show_curl    = True
        self.show_emo_bar = True
        self.fullscreen   = True

        # FPS
        self.fps         = 0.0
        self.ft          = deque(maxlen=30)
        self.frame_count = 0

        print(f"  Mirror mode : {'ON (dual-pass fusion)' if self.use_mirror else 'OFF'}")
        print(f"  Emotion CNN : {'ON' if self.use_emo_cnn else 'OFF'}")

    # ── Helpers ───────────────────────────────────────────────────

    def _emo_vec(self):
        vec = np.zeros(7, np.float32)
        vec[cfg.EMOTION_ID.get(self.emo_name, 6)] = 1.0
        return vec

    def _update_emotion(self, frame_bgr):
        if not self.auto_emo:
            self.emo_name = self.emo_manual
            return
        if self.use_emo_cnn:
            if self.frame_count % cfg.EMOTION_EVERY_N == 0:
                self.emo_detector.push_frame(frame_bgr)
            best, probs = self.emo_detector.get_result()
            self.emo_probs = probs
            self.emo_smooth.append(best)
            self.emo_name = collections.Counter(self.emo_smooth).most_common(1)[0][0]
        else:
            self.emo_name = "neutral"

    def _run_vsl(self, buffer_frames: list) -> dict:
        """
        Chạy VSL model trên buffer, trả về {label: conf} dict.
        buffer_frames: list of np.ndarray (208,)
        """
        seq = np.array(buffer_frames, dtype=np.float32)
        x   = torch.from_numpy(seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.vsl_model(x)
            probs  = torch.softmax(logits, -1)[0]
            topk   = torch.topk(probs, min(cfg.TOP_K, len(probs)))
        return {
            self.idx2label[i.item()]: float(c)
            for i, c in zip(topk.indices, topk.values)
        }

    def _infer_with_mirror(self, feat_201_norm: np.ndarray) -> list:
        """
        Dual-pass fusion (Giải pháp 2 + 3):

        Pass A — Original:
          Buffer hiện tại (đã có feat gốc ở frame cuối)

        Pass B — Mirror:
          Thay frame cuối bằng mirror(feat), inference lần 2
          Chỉ chạy mỗi MIRROR_EVERY_N frame, các frame còn lại dùng cache.

        Fusion: với mỗi nhãn, lấy max(conf_A, conf_B * mirror_weight)
        mirror_weight = 0.9 để ưu tiên pass gốc nhẹ hơn.

        Returns: sorted list [(label, conf), ...]
        """
        MIRROR_W = 0.9   # weight cho pass mirror

        # ── Pass A: gốc (buffer đã có full gốc) ──────────────────
        buf_list  = list(self.buffer)
        result_A  = self._run_vsl(buf_list)

        # ── Pass B: mirror (cache để tiết kiệm CPU) ──────────────
        if self.use_mirror:
            self._mirror_counter += 1
            if self._mirror_counter % cfg.MIRROR_EVERY_N == 0:
                # Build frame mirror: thay frame cuối bằng mirror version
                feat_mirror = mirror_feat(feat_201_norm)
                full_mirror = np.concatenate([feat_mirror, self._emo_vec()])
                buf_mirror  = buf_list[:-1] + [full_mirror]
                self._mirror_cache = self._run_vsl(buf_mirror)

            result_B = self._mirror_cache
        else:
            result_B = {}

        # ── Fusion: max(A, B * weight) ────────────────────────────
        merged = dict(result_A)
        for label, conf_b in result_B.items():
            weighted = conf_b * MIRROR_W
            if label not in merged or weighted > merged[label]:
                merged[label] = weighted

        sorted_preds = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        return sorted_preds[:cfg.TOP_K]

    # ── Process frame ─────────────────────────────────────────────

    def process_frame(self, frame_bgr):
        t0  = time.time()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.extractor.send(rgb)
        self._update_emotion(frame_bgr)

        feat_201, raw, self.hand_side = self.extractor.extract()
        feat_201_norm = normalize_feat(feat_201)

        # Append frame gốc vào buffer
        full = np.concatenate([feat_201_norm, self._emo_vec()])
        self.buffer.append(full)
        self.frame_count += 1

        if len(self.buffer) < cfg.SEQ_LEN:
            return raw

        if not self.extractor.has_hands():
            self.buffer.clear()
            self._mirror_cache = {}
            self.cur_label = "[NO HANDS]"
            self.cur_conf  = 0.0
            self.top_preds = []
            return raw

        # ── Inference với dual-pass mirror fusion ─────────────────
        self.top_preds = self._infer_with_mirror(feat_201_norm)

        label, conf = self.top_preds[0]
        self.pred_history.append((label, conf))

        # Majority vote
        if len(self.pred_history) >= cfg.STABLE_FRAMES:
            voted = collections.Counter(
                lb for lb, c in self.pred_history if c >= self.conf_thresh)
            if voted:
                best_lb = voted.most_common(1)[0][0]
                avg_c   = float(np.mean(
                    [c for lb, c in self.pred_history if lb == best_lb]))
                self.cur_label = best_lb
                self.cur_conf  = avg_c

        self.ft.append(time.time() - t0)
        self.fps = 1.0 / (sum(self.ft) / len(self.ft) + 1e-8)
        return raw

    # ── Draw UI ───────────────────────────────────────────────────

    def draw(self, frame, raw):
        H, W = frame.shape[:2]
        ol   = frame.copy()

        # Hand skeleton
        if raw.get('hands'):
            draw_hand_skeleton(ol, raw['hands'], W, H)

        # ── TOP BAR ───────────────────────────────────────────────
        cv2.rectangle(ol, (0,0), (W,72), (28,28,28), -1)
        put(ol, "VSL RECOGNITION", (15,34), 0.80, C.TEAL, 2)

        # Emotion badge
        ec  = cfg.EMOTION_COLORS.get(self.emo_name, (170,170,170))
        src = "CNN" if (self.auto_emo and self.use_emo_cnn) else "MAN"
        rrect(ol, (15,44), (230,70), (45,45,45), r=5)
        put(ol, f"EMO [{src}]: {self.emo_name.upper()}", (22,62), 0.40, ec, 1)

        # Hand side indicator + mirror status
        side_map  = {'left': 'TAY TRAI', 'right': 'TAY PHAI',
                     'both': '2 TAY', 'none': 'NO HANDS'}
        side_txt  = side_map.get(self.hand_side, '')
        mirror_on = self.use_mirror
        mirror_col = C.PURPLE if mirror_on else C.MUTED
        rrect(ol, (238,44), (480,70), (40,40,40), r=5)
        put(ol, f"{side_txt}", (246, 57), 0.38, C.TEAL, 1)
        put(ol, f"[M] MIRROR:{'ON' if mirror_on else 'OFF'}",
            (246, 68), 0.32, mirror_col, 1)

        # Buffer progress + FPS
        bp = len(self.buffer) / cfg.SEQ_LEN
        bx = W-175; by = 50
        cv2.rectangle(ol, (bx,by), (bx+155,by+10), (55,55,55), -1)
        cv2.rectangle(ol, (bx,by), (bx+int(155*bp),by+10), C.TEAL, -1)
        put(ol, f"Buf {len(self.buffer)}/{cfg.SEQ_LEN}", (bx, by-4), 0.31, C.MUTED)
        put(ol, f"FPS:{self.fps:.0f}", (W-82,32), 0.50, C.GRAY)

        # ── RIGHT PANEL — TOP-5 ───────────────────────────────────
        pw = 305; px = W-pw-15; py = 82
        rrect(ol, (px,py), (W-15,H-38), (28,28,28), r=12)
        put(ol, "PREDICTIONS", (px+12,py+28), 0.55, C.WHITE)
        put(ol, f"Thresh: {self.conf_thresh*100:.0f}%  [+/-]",
            (px+12,py+48), 0.36, C.ORANGE)
        cv2.line(ol, (px+12,py+56), (W-28,py+56), (60,60,60), 1)
        for i, (lbl, cf) in enumerate(self.top_preds[:cfg.TOP_K]):
            draw_conf_bar(ol, px+8, py+62+i*52, pw-20, 48,
                          cf, self.conf_thresh, lbl, i)

        # ── MAIN PREDICTION CARD ──────────────────────────────────
        cx=15; cy=82; cw=px-30; cch=100
        rrect(ol, (cx,cy), (cx+cw,cy+cch), (28,28,28), r=12)

        if self.cur_label and self.cur_conf >= self.conf_thresh:
            disp = self.display_names.get(self.cur_label, self.cur_label)
            put(ol, disp.upper()[:20], (cx+15,cy+62), 1.4, C.GREEN, 3)
            put(ol, f"{self.cur_conf*100:.1f}%", (cx+15,cy+88), 0.52, C.GRAY)
            cv2.circle(ol, (cx+cw-30,cy+50), 12, C.GREEN, -1)
            put(ol, "OK", (cx+cw-42,cy+55), 0.36, C.BG_DARK)
        elif self.cur_label == "[NO HANDS]":
            put(ol, "NO HANDS", (cx+15,cy+58), 1.0, C.RED, 2)
        else:
            pulse = int(80 + 60*math.sin(time.time()*5))
            put(ol, "Detecting...", (cx+15,cy+58), 1.0, (pulse,pulse,pulse), 2)

        # ── LEFT PANELS ───────────────────────────────────────────
        py_cur = cy + cch + 10

        if self.show_emo_bar:
            draw_emotion_panel(ol, cx, py_cur, cw, self.emo_probs, self.emo_name)
            py_cur += 18 + len(cfg.EMOTIONS)*26 + 16

        if self.show_curl:
            curl = self.extractor.get_curl()
            draw_curl_panel(ol, curl, cx, py_cur, cw)

        # ── BOTTOM BAR ────────────────────────────────────────────
        cv2.rectangle(ol, (0,H-35), (W,H), (28,28,28), -1)
        put(ol,
            "[A] Emo  [E] EmoBar  [C] Curl  [M] Mirror  [1-7] Manual  [+/-] Thresh  [R] Reset  [Q] Quit",
            (10,H-12), 0.31, C.MUTED)
        nh = len(raw.get('hands', []))
        hc = C.GREEN if nh > 0 else C.RED
        put(ol, f"Hands:{nh}", (W-85,H-12), 0.36, hc)

        cv2.addWeighted(ol, 0.92, frame, 0.08, 0, frame)
        return frame

    # ── Main loop ─────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  [ERROR] Không mở được webcam!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        WIN = "VSL + Emotion Realtime"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)

        print(f"\n  Controls:")
        print(f"    [A]     Toggle auto emotion (CNN ↔ manual)")
        print(f"    [E]     Toggle emotion bar chart")
        print(f"    [C]     Toggle curl debug panel")
        print(f"    [M]     Toggle mirror mode (dual-pass fusion)")
        print(f"    [1-7]   Gán emotion thủ công")
        print(f"    [+/-]   Confidence threshold")
        print(f"    [R]     Reset buffer")
        print(f"    [Q/ESC] Thoát\n")

        while True:
            ret, frame = cap.read()
            if not ret: break

            frame   = cv2.flip(frame, 1)
            frame   = cv2.resize(frame, (1280, 720))
            raw     = self.process_frame(frame)
            display = self.draw(frame.copy(), raw)
            cv2.imshow(WIN, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('a'), ord('A')):
                self.auto_emo = not self.auto_emo
                print(f"  Auto emotion: {'ON' if self.auto_emo else 'OFF'}")
            elif key in (ord('e'), ord('E')):
                self.show_emo_bar = not self.show_emo_bar
            elif key in (ord('c'), ord('C')):
                self.show_curl = not self.show_curl
            elif key in (ord('m'), ord('M')):
                self.use_mirror = not self.use_mirror
                self._mirror_cache = {}
                print(f"  Mirror mode: {'ON' if self.use_mirror else 'OFF'}")
            elif key in (ord('r'), ord('R')):
                self.buffer.clear(); self.pred_history.clear()
                self._mirror_cache = {}
                self.cur_label = ""; self.cur_conf = 0.0; self.top_preds = []
                print("  Buffer reset.")
            elif key in (ord('+'), ord('=')):
                self.conf_thresh = min(cfg.CONF_THRESHOLD_MAX,
                    self.conf_thresh + cfg.CONF_THRESHOLD_STEP)
                print(f"  Threshold: {self.conf_thresh*100:.0f}%")
            elif key in (ord('-'), ord('_')):
                self.conf_thresh = max(cfg.CONF_THRESHOLD_MIN,
                    self.conf_thresh - cfg.CONF_THRESHOLD_STEP)
                print(f"  Threshold: {self.conf_thresh*100:.0f}%")
            elif key in (ord('f'), ord('F')):
                self.fullscreen = not self.fullscreen
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
            elif ord('1') <= key <= ord('7'):
                idx = key - ord('1')
                self.emo_manual = cfg.EMOTIONS[idx]
                self.emo_name   = self.emo_manual
                self.auto_emo   = False
                print(f"  Manual emotion: {self.emo_manual}")

        cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()
        if self.emo_detector:
            self.emo_detector.stop()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VSL Realtime + Emotion CNN + Hand Symmetry",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Ví dụ:
  python realtime.py
  python realtime.py --model checkpoints/best.pt
  python realtime.py --model checkpoints/best.pt \\
                     --emotion_model checkpoints/emotion_cnn_best.pth
  python realtime.py --no_emotion     # tắt emotion CNN
  python realtime.py --no_mirror      # tắt mirror augmentation
        """
    )
    parser.add_argument("--model",
                        default="checkpoints/best.pt")
    parser.add_argument("--emotion_model",
                        default="checkpoints/emotion_cnn_best.pth")
    parser.add_argument("--no_emotion",
                        action="store_true",
                        help="Tắt emotion CNN")
    parser.add_argument("--no_mirror",
                        action="store_true",
                        help="Tắt mirror augmentation (chỉ dùng 1 pass)")
    args = parser.parse_args()

    # Tìm VSL checkpoint
    if not os.path.exists(args.model):
        fallbacks = ['checkpoints/best.pt',
                     'checkpoints/bilstm_v3_best.pt',
                     'checkpoints/bilstm_v2_best.pt']
        for fb in fallbacks:
            if os.path.exists(fb):
                print(f"  [INFO] Fallback: {fb}")
                args.model = fb
                break
        else:
            print(f"  [ERROR] Không tìm thấy: {args.model}")
            print("  Chạy train.py trước!")
            return

    # Emotion model
    emo_path = None
    if not args.no_emotion:
        if os.path.exists(args.emotion_model):
            emo_path = args.emotion_model
        else:
            print(f"  [WARN] Không tìm thấy emotion model: {args.emotion_model}")
            print("  → Chạy không có emotion CNN")

    engine = RealtimeEngine(
        vsl_path   = args.model,
        emo_path   = emo_path,
        no_emotion = args.no_emotion,
        no_mirror  = args.no_mirror,
    )
    try:
        engine.run()
    finally:
        engine.extractor.close()
        if engine.emo_detector:
            engine.emo_detector.stop()


if __name__ == "__main__":
    main()