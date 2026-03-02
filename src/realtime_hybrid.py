"""
realtime_hybrid.py - Hybrid VSL: Static MLP + BiLSTM + DualWindowGate
======================================================================
Chạy MediaPipe 1 lần, dùng kết quả cho cả 2 model.

Luồng:
    Frame → RealtimeExtractor (pose+face+hand)
                │
                ▼
        DualWindowGate (wrist velocity)
         │              │              │
      STATIC          DYNAMIC     TRANSITIONING
         │              │              │
    StaticMLP      BiLSTM buffer   Hiển thị "..."
    (96-dim)       (64×346-dim)
         │              │
         └──── Unified UI + Log ────

Chạy:
    python src/realtime_hybrid.py
    python src/realtime_hybrid.py --static_ckpt checkpoints/static_mlp_best_X.pt
                                  --dynamic_ckpt checkpoints/bilstm_best_X.pt
                                  --conf 0.65

Phím tắt:
    Q / ESC  – Thoát
    S        – Screenshot
    C        – Xóa log
    +  / -   – Tăng/giảm confidence threshold
    G        – Hiện/ẩn debug Gate info
"""

import sys
import os
import glob
import json
import time
import datetime
import math
import argparse
from pathlib import Path
from collections import deque, Counter
from itertools import combinations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Fix sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / 'src')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vsl.extractor import RealtimeExtractor
from vsl.config import cfg as vsl_cfg


# ══════════════════════════════════════════════════════════════════
# STATIC MODEL (copy từ train_static_mlp.py)
# ══════════════════════════════════════════════════════════════════

class StaticMLP(nn.Module):
    def __init__(self, feat_dim, num_classes,
                 hidden_1=256, hidden_2=128, hidden_3=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_1), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(hidden_1, hidden_2), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(hidden_2, hidden_3), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(hidden_3, num_classes),
        )
    def forward(self, x): return self.net(x)


# ══════════════════════════════════════════════════════════════════
# DYNAMIC MODEL (copy từ train_bilstm.py)
# ══════════════════════════════════════════════════════════════════

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        w = torch.softmax(self.attn(x).squeeze(-1), dim=-1)
        return (x * w.unsqueeze(-1)).sum(1), w


class BiLSTMClassifier(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers, num_classes,
                 dropout_lstm=0.0, dropout_fc=0.0,
                 bidirectional=True, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        dirs = 2 if bidirectional else 1
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout_fc))
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout_lstm if num_layers > 1 else 0.0)
        out_dim = hidden_dim * dirs
        self.attention  = AttentionLayer(out_dim) if use_attention else None
        fc_in = out_dim * 2 if use_attention else out_dim
        mid   = max(num_classes * 4, 128)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fc_in),
            nn.Linear(fc_in, mid),    nn.GELU(), nn.Dropout(dropout_fc),
            nn.Linear(mid, mid // 2), nn.GELU(), nn.Dropout(dropout_fc / 2),
            nn.Linear(mid // 2, num_classes))
        self._dirs = dirs

    def forward(self, x):
        x = self.input_proj(x)
        out, (hn, _) = self.lstm(x)
        last = torch.cat([hn[-2], hn[-1]], -1) if self._dirs == 2 else hn[-1]
        if self.use_attention and self.attention:
            ctx, _ = self.attention(out)
            feat   = torch.cat([ctx, last], -1)
        else:
            feat = last
        return self.classifier(feat)


# ══════════════════════════════════════════════════════════════════
# HAND FEATURE EXTRACTION (dùng cho StaticMLP từ hand landmarks)
# ══════════════════════════════════════════════════════════════════

FINGERTIPS    = [4, 8, 12, 16, 20]
FINGER_BASES  = [2, 5, 9, 13, 17]
FINGER_CHAINS = [
    [0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12],
    [0,13,14,15,16], [0,17,18,19,20],
]

def _angle(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return 0.0
    return float(math.acos(np.clip(np.dot(v1,v2)/(n1*n2), -1, 1)))

def extract_static_features(hand_lms) -> np.ndarray | None:
    """
    hand_lms: list of 21 NormalizedLandmark (từ MediaPipe)
    → vector 96-dim cho StaticMLP
    """
    if hand_lms is None or len(hand_lms) < 21:
        return None
    lm = np.array([[h.x, h.y, h.z] for h in hand_lms], dtype=np.float32)
    # Normalize
    lm -= lm[0]
    scale = np.linalg.norm(lm[9])
    if scale > 1e-6: lm /= scale
    coords  = lm.flatten()                                               # 63
    angles  = np.array([
        _angle(lm[c[i-1]]-lm[c[i]], lm[c[i+1]]-lm[c[i]])
        for c in FINGER_CHAINS for i in range(1, 4)
    ])                                                                    # 15
    lengths = np.array([np.linalg.norm(lm[t]-lm[b])
                        for b,t in zip(FINGER_BASES, FINGERTIPS)])       # 5
    tips    = lm[FINGERTIPS]
    dists   = np.array([np.linalg.norm(tips[i]-tips[j])
                        for i,j in combinations(range(5), 2)])           # 10
    v1, v2  = lm[5]-lm[0], lm[17]-lm[0]
    n       = np.cross(v1, v2)
    norm    = np.linalg.norm(n)
    palm_n  = (n/norm if norm > 1e-6 else n).astype(np.float32)         # 3
    return np.concatenate([coords, angles, lengths, dists, palm_n]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# DUAL WINDOW GATE
# ══════════════════════════════════════════════════════════════════

class DualWindowGate:
    """
    Dùng wrist velocity từ pose landmarks để phân loại:
        STATIC       → gọi StaticMLP
        DYNAMIC      → gọi BiLSTM
        TRANSITIONING → không gọi model nào, chờ ổn định

    Wrist lấy từ feature vector 346-dim:
        Pose wrist trái  = feat[45:47]  (pose idx 15, x,y)
        Pose wrist phải  = feat[48:50]  (pose idx 16, x,y)
    """
    STATIC        = 'static'
    DYNAMIC       = 'dynamic'
    TRANSITIONING = 'transitioning'

    def __init__(self,
                 short_window:  int   = 4,
                 long_window:   int   = 20,
                 threshold:     float = 0.012,
                 hysteresis:    int   = 3):
        self.short_buf   = deque(maxlen=short_window)
        self.long_buf    = deque(maxlen=long_window)
        self.threshold   = threshold
        self.hysteresis  = hysteresis

        self._state      = self.DYNAMIC   # trạng thái hiện tại
        self._state_cnt  = 0              # frames liên tiếp cùng state

        # Debug info
        self.v_short     = 0.0
        self.v_long      = 0.0
        self.raw_state   = self.DYNAMIC

    def _velocity(self, buf) -> float:
        """Tính mean velocity trong buffer."""
        if len(buf) < 2: return 0.0
        pts   = list(buf)
        total = sum(np.linalg.norm(pts[i] - pts[i-1])
                    for i in range(1, len(pts)))
        return total / len(pts)

    def push(self, feat_346: np.ndarray) -> str:
        """
        Nhận feature vector 346-dim, cập nhật gate.
        Trả về state: 'static' | 'dynamic' | 'transitioning'
        """
        # Lấy wrist từ pose (index 15=wrist_L, 16=wrist_R trong pose)
        # pose layout: 25 landmarks × 3, bắt đầu từ feat[0]
        # wrist_L = pose[15] → feat[45:47] (chỉ lấy x,y)
        # wrist_R = pose[16] → feat[48:50]
        wrist_l = feat_346[45:47].copy()
        wrist_r = feat_346[48:50].copy()

        # Dùng wrist nào đang active (khác 0)
        if np.any(wrist_r != 0):
            wrist = wrist_r
        elif np.any(wrist_l != 0):
            wrist = wrist_l
        else:
            # Không có pose → giữ nguyên state cũ
            return self._state

        self.short_buf.append(wrist)
        self.long_buf.append(wrist)

        v_s = self._velocity(self.short_buf)
        v_l = self._velocity(self.long_buf)
        self.v_short = v_s
        self.v_long  = v_l
        thr = self.threshold

        # Logic 4 trạng thái
        if v_s < thr and v_l < thr:
            new = self.STATIC
        elif v_s >= thr and v_l >= thr:
            new = self.DYNAMIC
        else:
            new = self.TRANSITIONING   # 2 window mâu thuẫn

        self.raw_state = new

        # Hysteresis — cần N frames liên tiếp mới đổi state
        if new == self._state:
            self._state_cnt += 1
        else:
            self._state_cnt = max(0, self._state_cnt - 1)
            if self._state_cnt == 0:
                self._state     = new
                self._state_cnt = 0

        return self._state

    @property
    def state(self) -> str:
        return self._state

    def reset(self):
        self.short_buf.clear()
        self.long_buf.clear()
        self._state     = self.DYNAMIC
        self._state_cnt = 0


# ══════════════════════════════════════════════════════════════════
# CHECKPOINT LOADERS
# ══════════════════════════════════════════════════════════════════

def _latest(pattern: str) -> str | None:
    files = glob.glob(str(_PROJECT_ROOT / 'checkpoints' / pattern))
    return max(files, key=os.path.getmtime) if files else None


def load_static_model(path: str, device: str):
    ckpt      = torch.load(path, map_location=device, weights_only=False)
    label_map = ckpt['label_map']
    mcfg      = ckpt.get('cfg', {})
    model = StaticMLP(
        feat_dim    = mcfg.get('FEAT_DIM',  96),
        num_classes = len(label_map),
        hidden_1    = mcfg.get('HIDDEN_1', 256),
        hidden_2    = mcfg.get('HIDDEN_2', 128),
        hidden_3    = mcfg.get('HIDDEN_3',  64),
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    acc = ckpt.get('val_acc', 0) * 100
    print(f"  [Static]  {Path(path).name} | "
          f"classes={len(label_map)} | val_acc={acc:.1f}%")
    return model, label_map


def load_dynamic_model(path: str, device: str):
    ckpt      = torch.load(path, map_location=device, weights_only=False)
    label_map = ckpt['label_map']
    mcfg      = ckpt.get('cfg', {})
    model = BiLSTMClassifier(
        feat_dim   = mcfg.get('FEAT_DIM',   vsl_cfg.FEAT_DIM),
        hidden_dim = mcfg.get('HIDDEN_DIM', 256),
        num_layers = mcfg.get('NUM_LAYERS',   3),
        num_classes= len(label_map),
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    acc = ckpt.get('val_acc', 0) * 100
    print(f"  [Dynamic] {Path(path).name} | "
          f"classes={len(label_map)} | val_acc={acc:.1f}%")
    return model, label_map


# ══════════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════════

class HybridLogger:
    def __init__(self):
        log_dir  = _PROJECT_ROOT / 'logs'
        log_dir.mkdir(exist_ok=True)
        ts        = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.path = str(log_dir / f'hybrid_realtime_{ts}.txt')
        self.entries   = []
        self.last_key  = ''   # "model:label" — tránh log lặp

        with open(self.path, 'w', encoding='utf-8') as f:
            f.write(f"Hybrid Realtime Log — {ts}\n")
            f.write("=" * 55 + "\n")
            f.write(f"{'Time':<10} {'Model':<8} {'Label':<25} {'Conf':>6}\n")
            f.write("-" * 55 + "\n")
        print(f"  [Log] {self.path}")

    def log(self, model_type: str, label: str, conf: float):
        key = f"{model_type}:{label}"
        if key == self.last_key:
            return
        self.last_key = key
        ts  = datetime.datetime.now().strftime('%H:%M:%S')
        self.entries.append({'time': ts, 'model': model_type,
                             'label': label, 'conf': round(conf, 4)})
        line = f"{ts:<10} {model_type:<8} {label:<25} {conf*100:>5.1f}%\n"
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(line)
        print(f"  [LOG] {ts}  [{model_type}]  {label}  {conf*100:.1f}%")

    def clear(self):
        self.entries  = []
        self.last_key = ''
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- CLEARED {datetime.datetime.now().strftime('%H:%M:%S')} ---\n")
        print("  [Log cleared]")

    def summary(self):
        if not self.entries: return
        counts = Counter(f"[{e['model']}] {e['label']}" for e in self.entries)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 55 + "\n")
            f.write("SUMMARY\n")
            f.write(f"Total: {len(self.entries)}\n")
            for k, v in counts.most_common():
                f.write(f"  {k:<35} {v}\n")
        print(f"\n  [Log saved] {self.path}")
        print(f"  Total detections: {len(self.entries)}")


# ══════════════════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════════════════

FONT  = cv2.FONT_HERSHEY_DUPLEX
_C    = dict(
    white =(240,240,240), gray  =(100,100,110),
    green =(50, 210, 60), orange=(30, 160,255),
    yellow=(20, 220,220), red   =(50,  50,220),
    black =(8,    8,  8), teal  =(0,  200,180),
    purple=(180,  80,220),
)

def _put(img, text, pos, scale=0.6, color=_C['white'], thick=1):
    x, y = pos
    cv2.putText(img, text, (x+1,y+1), FONT, scale, _C['black'], thick+1, cv2.LINE_AA)
    cv2.putText(img, text, pos,        FONT, scale, color,       thick,   cv2.LINE_AA)

def _pill(img, x1,y1,x2,y2, color, alpha=0.80, r=10):
    ov = img.copy()
    cv2.rectangle(ov, (x1+r,y1),(x2-r,y2), color, -1)
    cv2.rectangle(ov, (x1,y1+r),(x2,y2-r), color, -1)
    for cx,cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(ov, (cx,cy), r, color, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)

def _bar(img, x,y,w,h, ratio, color, bg=(35,35,45)):
    cv2.rectangle(img,(x,y),(x+w,y+h), bg, -1)
    fw = max(0, int(w*min(ratio,1.0)))
    if fw: cv2.rectangle(img,(x,y),(x+fw,y+h), color, -1)
    cv2.rectangle(img,(x,y),(x+w,y+h),(65,65,75),1)

# Màu theo model type
_MODEL_COLOR = {
    'static':  (50, 210,  60),   # xanh lá
    'dynamic': (30, 160, 255),   # cam
}
_STATE_COLOR = {
    'static':        (50,  210,  60),
    'dynamic':       (30,  160, 255),
    'transitioning': (20,  220, 220),
}
_STATE_LABEL = {
    'static':        'STATIC',
    'dynamic':       'DYNAMIC',
    'transitioning': 'WAIT...',
}


def draw_hybrid_overlay(frame, state, model_type, label, conf,
                        conf_thr, fps, buf_fill, seq_len,
                        log_count, notif, notif_ts,
                        recent_logs, show_gate_debug,
                        v_short, v_long, gate_thr):
    h, w = frame.shape[:2]
    above = conf >= conf_thr and bool(label)

    # ── Header ───────────────────────────────────────────────────
    cv2.rectangle(frame, (0,0), (w,50), (18,18,26), -1)
    _put(frame, "VSL Hybrid", (10,32), 0.70, _C['white'], 2)
    fps_c = _C['green'] if fps>=20 else (_C['yellow'] if fps>=10 else _C['orange'])
    _put(frame, f"FPS {fps:.0f}", (w-120,30), 0.55, fps_c)
    _put(frame, f"thr {conf_thr:.2f}", (w-120,46), 0.38, _C['gray'])

    # ── Gate state badge ─────────────────────────────────────────
    sc = _STATE_COLOR.get(state, _C['gray'])
    sl = _STATE_LABEL.get(state, state.upper())
    _pill(frame, 10, 58, 130, 82, sc, alpha=0.75, r=6)
    _put(frame,  sl, (18, 75), 0.45, _C['black'], 2)

    # ── BiLSTM buffer bar (chỉ hiện khi dynamic) ─────────────────
    if state == 'dynamic':
        bx, by, bw2 = 140, 58, 200
        _put(frame, f"buf {buf_fill}/{seq_len}", (bx, 75), 0.38, _C['orange'])
        _bar(frame, bx, 60, bw2, 10, buf_fill/seq_len, _C['orange'])

    # ── Gate debug (phím G) ───────────────────────────────────────
    if show_gate_debug:
        dy = 95
        _put(frame, f"v_short={v_short:.4f}", (10,dy),    0.38, _C['teal'])
        _put(frame, f"v_long ={v_long:.4f}",  (10,dy+18), 0.38, _C['teal'])
        _put(frame, f"thr    ={gate_thr:.4f}", (10,dy+36), 0.38, _C['teal'])
        # Visual bars
        _bar(frame, 140, dy-8,    150, 7, min(v_short/gate_thr,2)/2, _C['teal'])
        _bar(frame, 140, dy+10,   150, 7, min(v_long/gate_thr,2)/2,  _C['purple'])

    # ── Log count ────────────────────────────────────────────────
    _put(frame, f"Log: {log_count}", (10, h-100), 0.40, _C['gray'])

    # ── PREDICTION BLOCK ─────────────────────────────────────────
    if above and model_type:
        mc = _MODEL_COLOR.get(model_type, _C['white'])
        (tw,_),_ = cv2.getTextSize(label, FONT, 1.1, 2)
        cx  = w // 2
        px1 = cx - tw//2 - 30
        px2 = cx + tw//2 + 30
        py1 = h - 160
        py2 = h - 65

        # Pill màu theo model
        _pill(frame, px1, py1, px2, py2, (10,50,10) if model_type=='static'
              else (10,40,70), alpha=0.85)
        _put(frame, label, (cx - tw//2, h-85), scale=1.1, color=_C['white'], thick=2)

        # Confidence bar
        bw = px2 - px1 - 10
        _bar(frame, px1+5, h-68, bw, 12, conf, mc)
        _put(frame, f"{conf*100:.1f}%", (px2+6, h-60), 0.52, mc)

        # Model badge
        badge_txt = "STATIC" if model_type=='static' else "DYNAMIC"
        _pill(frame, px1, py1-26, px1+100, py1-4, mc, alpha=0.85, r=5)
        _put(frame, f"[{badge_txt}]", (px1+6, py1-10), 0.38, _C['black'], 2)

    elif state == 'transitioning':
        _put(frame, "...", (w//2-20, h//2), 0.9, _C['yellow'], 2)

    elif not label:
        hint = ("Gio tay..." if state == 'dynamic'
                else "Giu tay yen..." if state == 'static'
                else "...")
        _put(frame, hint, (w//2-100, h//2), 0.65, _C['orange'])

    # ── Recent log panel ─────────────────────────────────────────
    if recent_logs:
        px = w - 240
        _put(frame, "Recent:", (px, 75), 0.42, _C['gray'])
        for i, (mt, lb, cf) in enumerate(recent_logs[-7:]):
            mc = _MODEL_COLOR.get(mt, _C['gray'])
            c  = mc if cf >= conf_thr else _C['gray']
            _put(frame, f"[{mt[0].upper()}] {lb}  {cf*100:.0f}%",
                 (px, 95+i*20), 0.38, c)

    # ── Notification toast ────────────────────────────────────────
    if notif and (time.time() - notif_ts < 2.0):
        (nw,_),_ = cv2.getTextSize(notif, FONT, 0.55, 1)
        nx = w//2 - nw//2
        _pill(frame, nx-12, h//2+20, nx+nw+12, h//2+52, (40,40,60), alpha=0.85, r=7)
        _put(frame, notif, (nx, h//2+42), 0.55, _C['yellow'])

    # ── Hint bar ─────────────────────────────────────────────────
    cv2.rectangle(frame, (0,h-22), (w,h), (18,18,26), -1)
    _put(frame, "Q:Quit  S:Shot  C:Clear  +/-:Thr  G:Gate debug",
         (8, h-6), 0.32, (70,70,80))


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--static_ckpt',  default=None,
                    help='Path static MLP checkpoint (default: auto-find)')
    ap.add_argument('--dynamic_ckpt', default=None,
                    help='Path BiLSTM checkpoint (default: auto-find)')
    ap.add_argument('--conf',   type=float, default=0.60)
    ap.add_argument('--gate',   type=float, default=0.012,
                    help='Velocity threshold cho Gate (default: 0.012)')
    ap.add_argument('--source', default='0')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEQ_LEN = vsl_cfg.SEQ_LEN   # 64

    print(f"\n{'='*55}")
    print("  VSL Hybrid (Static MLP + BiLSTM + DualWindowGate)")
    print(f"{'='*55}")
    print(f"  Device : {device}")

    # ── Load Static model ─────────────────────────────────────────
    s_ckpt = args.static_ckpt or _latest('static_mlp_best_*.pt')
    if not s_ckpt or not os.path.exists(s_ckpt):
        print("\n  [ERROR] Khong tim thay static_mlp_best_*.pt")
        print("  Train truoc: python src/train_static_mlp.py")
        sys.exit(1)
    static_model, static_map = load_static_model(s_ckpt, device)
    static_idx2lbl = {v: k for k, v in static_map.items()}

    # ── Load Dynamic model ────────────────────────────────────────
    d_ckpt = args.dynamic_ckpt or _latest('bilstm_best_*.pt')
    if not d_ckpt or not os.path.exists(d_ckpt):
        print("\n  [ERROR] Khong tim thay bilstm_best_*.pt")
        print("  Train truoc: python src/train_bilstm.py")
        sys.exit(1)
    dynamic_model, dynamic_map = load_dynamic_model(d_ckpt, device)
    dynamic_idx2lbl = {v: k for k, v in dynamic_map.items()}

    # ── Extractor + Gate ──────────────────────────────────────────
    print("\n  Khoi tao MediaPipe...")
    extractor = RealtimeExtractor()

    gate = DualWindowGate(
        short_window=4,
        long_window =20,
        threshold   =args.gate,
        hysteresis  =3,
    )

    # ── Video source ──────────────────────────────────────────────
    src = args.source
    try: src = int(src)
    except ValueError: pass
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"  [ERROR] Khong mo duoc: {src}")
        sys.exit(1)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video  : {fw}×{fh}")

    # ── State ─────────────────────────────────────────────────────
    logger       = HybridLogger()
    dynamic_buf  = deque(maxlen=SEQ_LEN)   # buffer BiLSTM
    smooth_s     = deque(maxlen=5)         # smoothing static
    smooth_d     = deque(maxlen=5)         # smoothing dynamic
    fps_buf      = deque(maxlen=30)
    t_prev       = time.time()

    label      = ''
    conf       = 0.0
    model_type = ''
    notif      = ''
    notif_ts   = 0.0
    recent_logs = []
    show_gate  = False
    conf_thr   = args.conf

    ss_dir = _PROJECT_ROOT / 'screenshots'
    ss_dir.mkdir(exist_ok=True)

    WIN = "VSL Hybrid"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    print(f"\n  Dang chay... (Q/ESC de thoat)\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        # FPS
        t_now = time.time()
        fps_buf.append(1.0 / max(t_now - t_prev, 1e-9))
        t_prev = t_now
        fps    = float(np.mean(fps_buf))

        # ── Extract features (1 lần cho cả 2 model) ──────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        extractor.send_frame(rgb)
        feat_346 = extractor.extract_features()   # (346,)
        latest   = extractor.get_latest()
        left_h   = latest['hands'][0] if latest['hands'] else None
        right_h  = latest['hands'][1] if latest['hands'] else None

        # ── Gate decision ─────────────────────────────────────────
        state = gate.push(feat_346)

        # ── STATIC branch ─────────────────────────────────────────
        if state == 'static':
            dynamic_buf.clear()   # reset BiLSTM buffer khi chuyển sang static
            smooth_d.clear()

            # Lấy tay tốt nhất (ưu tiên tay phải)
            hand_lms = right_h if right_h else left_h
            s_feat   = extract_static_features(hand_lms)

            if s_feat is not None:
                with torch.no_grad():
                    x      = torch.from_numpy(s_feat).unsqueeze(0).to(device)
                    logits = static_model(x)
                    proba  = F.softmax(logits, dim=-1)[0].cpu().numpy()
                top_idx = int(np.argmax(proba))
                raw_c   = float(proba[top_idx])
                raw_l   = static_idx2lbl.get(top_idx, str(top_idx))

                smooth_s.append(raw_l)
                label      = Counter(smooth_s).most_common(1)[0][0]
                conf       = raw_c
                model_type = 'static'

                if conf >= conf_thr:
                    logger.log('static', label, conf)
                    if not recent_logs or recent_logs[-1][1] != label:
                        recent_logs.append(('static', label, conf))
                        if len(recent_logs) > 20: recent_logs.pop(0)
            else:
                smooth_s.clear()
                label = ''; conf = 0.0; model_type = ''

        # ── DYNAMIC branch ────────────────────────────────────────
        elif state == 'dynamic':
            smooth_s.clear()
            dynamic_buf.append(feat_346)

            if len(dynamic_buf) == SEQ_LEN:
                seq = np.stack(list(dynamic_buf), axis=0)   # (64, 346)
                with torch.no_grad():
                    x      = torch.from_numpy(seq).unsqueeze(0).to(device)
                    logits = dynamic_model(x)
                    proba  = F.softmax(logits, dim=-1)[0].cpu().numpy()
                top_idx = int(np.argmax(proba))
                raw_c   = float(proba[top_idx])
                raw_l   = dynamic_idx2lbl.get(top_idx, str(top_idx))

                smooth_d.append(raw_l)
                label      = Counter(smooth_d).most_common(1)[0][0]
                conf       = raw_c
                model_type = 'dynamic'

                if conf >= conf_thr:
                    logger.log('dynamic', label, conf)
                    if not recent_logs or recent_logs[-1][1] != label:
                        recent_logs.append(('dynamic', label, conf))
                        if len(recent_logs) > 20: recent_logs.pop(0)
            else:
                label = ''; conf = 0.0; model_type = ''

        # ── TRANSITIONING ─────────────────────────────────────────
        else:
            smooth_s.clear()
            smooth_d.clear()
            label = ''; conf = 0.0; model_type = ''

        # ── Draw ──────────────────────────────────────────────────
        draw_hybrid_overlay(
            frame       = frame,
            state       = state,
            model_type  = model_type,
            label       = label,
            conf        = conf,
            conf_thr    = conf_thr,
            fps         = fps,
            buf_fill    = len(dynamic_buf),
            seq_len     = SEQ_LEN,
            log_count   = len(logger.entries),
            notif       = notif,
            notif_ts    = notif_ts,
            recent_logs = recent_logs,
            show_gate_debug = show_gate,
            v_short     = gate.v_short,
            v_long      = gate.v_long,
            gate_thr    = gate.threshold,
        )

        cv2.imshow(WIN, frame)

        # ── Keys ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('s'), ord('S')):
            ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            path = str(ss_dir / f'hybrid_{ts}.png')
            cv2.imwrite(path, frame)
            notif, notif_ts = f"Saved: hybrid_{ts}.png", time.time()
        elif key in (ord('c'), ord('C')):
            logger.clear()
            recent_logs.clear()
            notif, notif_ts = "Log cleared", time.time()
        elif key in (ord('g'), ord('G')):
            show_gate = not show_gate
            notif, notif_ts = f"Gate debug: {'ON' if show_gate else 'OFF'}", time.time()
        elif key in (ord('+'), ord('=')):
            conf_thr = min(0.99, round(conf_thr + 0.05, 2))
            notif, notif_ts = f"Threshold → {conf_thr:.2f}", time.time()
        elif key == ord('-'):
            conf_thr = max(0.05, round(conf_thr - 0.05, 2))
            notif, notif_ts = f"Threshold → {conf_thr:.2f}", time.time()

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    logger.summary()
    print("  Thoat.")


if __name__ == '__main__':
    main()