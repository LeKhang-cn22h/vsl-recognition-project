"""
realtime_bilstm.py - Realtime VSL Sign Language Recognition (BiLSTM)
=====================================================================
Cấu trúc project:
    vsl-recognition-project/
    ├── src/
    │   ├── vsl/        ← config.py, extractor, ...
    │   └── lstm/       ← realtime_bilstm.py (file này)
    └── checkpoints/    ← bilstm_best_*.pt

Chạy từ thư mục gốc project:
    # Webcam
    python -m src.realtime_bilstm

    # File video
    python -m src.realtime_bilstm --source video.mp4

    # Chỉ định checkpoint + threshold
    python -m src.realtime_bilstm --checkpoint checkpoints/bilstm_best_20240101.pt --conf 0.70

Phím tắt:
    Q / ESC  – Thoát
    R        – Reset buffer
    +  / -   – Tăng/giảm confidence threshold
    P        – Pause / Resume
    S        – Lưu screenshot
"""

import sys
import os
import glob
import argparse
import time
import datetime
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Fix sys.path ──────────────────────────────────────────────────
# File nằm ở src/realtime_bilstm.py
# parents[0] = src/
# parents[1] = vsl-recognition-project/  ← project root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR      = _PROJECT_ROOT / 'src'

for _p in [str(_PROJECT_ROOT), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import vsl.config — dùng relative từ src/ (hoạt động cả 2 cách chạy)
from vsl.config import cfg as vsl_cfg

# ── Feature extractor ─────────────────────────────────────────────
try:
    from vsl.extractor import FeatureExtractor
    _extractor = FeatureExtractor()
    HAS_EXTRACTOR = True
    print("  [OK] FeatureExtractor loaded")
except Exception as e:
    HAS_EXTRACTOR = False
    print(f"  [WARN] Khong load duoc FeatureExtractor ({e})")
    print("         → Chay demo mode voi random features")


# ══════════════════════════════════════════════════════════════════
# MODEL (inline để tránh phụ thuộc import)
# ══════════════════════════════════════════════════════════════════

class AttentionLayer(nn.Module):
    """Tên class + attribute khớp với train_bilstm.py → "attention.attn" trong checkpoint."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)   # key: attention.attn.weight / .bias

    def forward(self, x):
        w = torch.softmax(self.attn(x).squeeze(-1), dim=-1)
        return (x * w.unsqueeze(-1)).sum(1), w


class BiLSTMClassifier(nn.Module):
    """Kiến trúc khớp 100% với train_bilstm.py để load checkpoint đúng."""
    def __init__(self, feat_dim, hidden_dim, num_layers, num_classes,
                 dropout_lstm=0.0, dropout_fc=0.0,
                 bidirectional=True, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        dirs = 2 if bidirectional else 1

        # key trong checkpoint: "input_proj.X.*"
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_fc))

        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout_lstm if num_layers > 1 else 0.0)

        out_dim = hidden_dim * dirs

        # key trong checkpoint: "attention.attn.*"
        self.attention = AttentionLayer(out_dim) if use_attention else None

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
            feat = torch.cat([ctx, last], -1)
        else:
            feat = last
        return self.classifier(feat)


# ══════════════════════════════════════════════════════════════════
# CHECKPOINT
# ══════════════════════════════════════════════════════════════════

def find_latest_checkpoint():
    files = glob.glob(str(_PROJECT_ROOT / 'checkpoints' / 'bilstm_best_*.pt'))
    return max(files, key=os.path.getmtime) if files else None


def load_model(ckpt_path, device):
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    label_map = ckpt['label_map']
    mcfg      = ckpt.get('cfg', {})
    model = BiLSTMClassifier(
        feat_dim    = mcfg.get('FEAT_DIM',   vsl_cfg.FEAT_DIM),
        hidden_dim  = mcfg.get('HIDDEN_DIM', 256),
        num_layers  = mcfg.get('NUM_LAYERS', 3),
        num_classes = len(label_map),
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    val_acc = ckpt.get('val_acc', 0) * 100
    print(f"  [OK] Loaded | classes={len(label_map)} "
          f"| epoch={ckpt.get('epoch','?')} | val_acc={val_acc:.1f}%")
    return model, label_map


# ══════════════════════════════════════════════════════════════════
# FRAME BUFFER
# ══════════════════════════════════════════════════════════════════

class FrameBuffer:
    def __init__(self, seq_len, feat_dim):
        self.seq_len  = seq_len
        self.feat_dim = feat_dim
        self.buf      = deque(maxlen=seq_len)

    def push(self, vec):
        self.buf.append(np.asarray(vec, dtype=np.float32))

    @property
    def ready(self):
        return len(self.buf) == self.seq_len

    @property
    def ratio(self):
        return len(self.buf) / self.seq_len

    def to_tensor(self, device):
        return torch.from_numpy(
            np.stack(list(self.buf))).unsqueeze(0).to(device)

    def reset(self):
        self.buf.clear()


# ══════════════════════════════════════════════════════════════════
# DRAWING  ── chỉ hiện nổi bật khi conf >= threshold
# ══════════════════════════════════════════════════════════════════

FONT = cv2.FONT_HERSHEY_DUPLEX

# BGR colors
GREEN  = (55,  210,  55)
ORANGE = (30,  160, 255)
YELLOW = (20,  220, 220)
WHITE  = (240, 240, 240)
GRAY   = (120, 120, 120)
BLACK  = (8,    8,   8)


def _put(img, text, pos, scale, color, thick=1):
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), FONT, scale, BLACK,  thick+1, cv2.LINE_AA)
    cv2.putText(img, text,  pos,        FONT, scale, color,  thick,   cv2.LINE_AA)


def _pill(img, x1, y1, x2, y2, color, alpha=0.78, r=10):
    ov = img.copy()
    cv2.rectangle(ov, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(ov, (x1, y1+r), (x2, y2-r), color, -1)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(ov, (cx, cy), r, color, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)


def _bar(img, x, y, w, h, ratio, color, bg=(35, 35, 45)):
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
    fw = max(0, int(w * min(ratio, 1.0)))
    if fw:
        cv2.rectangle(img, (x, y), (x+fw, y+h), color, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (65, 65, 75), 1)


def draw_overlay(frame, label, conf, conf_thr, buf_ratio, fps,
                 paused, notif, notif_ts):
    h, w = frame.shape[:2]
    above = conf >= conf_thr and bool(label)

    # ── Buffer bar (top) ─────────────────────────────────────────
    bc = GREEN if buf_ratio >= 1.0 else ORANGE
    _bar(frame, 8, 8, w - 16, 7, buf_ratio, bc)
    _put(frame, "READY" if buf_ratio >= 1.0 else f"{int(buf_ratio*100)}%",
         (10, 30), 0.42, bc)

    # ── FPS + threshold (top-right) ───────────────────────────────
    fps_c = GREEN if fps >= 20 else (YELLOW if fps >= 10 else ORANGE)
    _put(frame, f"FPS {fps:.0f}", (w-100, 28), 0.50, fps_c)
    _put(frame, f"thr {conf_thr:.2f}", (w-100, 52), 0.40, GRAY)

    # ════════════════════════════════════════════════════════════
    # PREDICTION BLOCK
    # ════════════════════════════════════════════════════════════
    if above:
        # ── Measure text width để căn giữa ──
        (tw, th), _ = cv2.getTextSize(label, FONT, 1.05, 2)
        cx   = w // 2
        px1  = cx - tw // 2 - 28
        px2  = cx + tw // 2 + 28
        py1  = h - 145
        py2  = h - 60

        # Pill nền xanh
        _pill(frame, px1, py1, px2, py2, (15, 72, 15), alpha=0.80)

        # Nhãn ký hiệu
        _put(frame, label, (cx - tw//2, h - 82),
             scale=1.05, color=WHITE, thick=2)

        # Confidence bar
        bw = px2 - px1 - 10
        _bar(frame, px1+5, h - 68, bw, 11, conf, GREEN)

        # % bên phải bar
        _put(frame, f"{conf*100:.0f}%", (px2 + 6, h - 60),
             scale=0.52, color=GREEN)

        # Badge "DETECTED"
        _pill(frame, px2 - 90, py1 - 22, px2 + 2, py1 - 4,
              (18, 100, 18), alpha=0.88, r=5)
        _put(frame, "✓ DETECTED", (px2 - 85, py1 - 8),
             scale=0.35, color=GREEN, thick=1)

    elif label and buf_ratio >= 1.0:
        # Buffer đầy nhưng chưa đủ confidence → hiện mờ nhỏ
        _put(frame, f"? {label}   {conf*100:.0f}%",
             (10, h - 18), scale=0.48, color=GRAY)

    # ── Paused banner ─────────────────────────────────────────────
    if paused:
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (w,h), (0,0,0), -1)
        cv2.addWeighted(ov, 0.42, frame, 0.58, 0, frame)
        _put(frame, "PAUSED", (w//2-60, h//2), 1.3, YELLOW, thick=3)
        _put(frame, "Press P to resume",
             (w//2-110, h//2+42), 0.52, GRAY)

    # ── Notification toast ────────────────────────────────────────
    if notif and (time.time() - notif_ts < 2.0):
        (nw, _), _ = cv2.getTextSize(notif, FONT, 0.58, 1)
        nx = w//2 - nw//2
        _pill(frame, nx-14, h//2-26, nx+nw+14, h//2+8,
              (38, 38, 58), alpha=0.85, r=7)
        _put(frame, notif, (nx, h//2), 0.58, YELLOW)

    # ── Hint bar (bottom) ─────────────────────────────────────────
    _put(frame,
         "Q:Quit   R:Reset   P:Pause   +/-:Threshold   S:Screenshot",
         (8, h - 8), 0.32, (65, 65, 72), thick=1)


# ══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*55}")
    print("  VSL BiLSTM – Realtime Inference")
    print(f"{'='*55}")
    print(f"  Device     : {device}")
    print(f"  Source     : {args.source}")
    print(f"  Conf thr   : {args.conf}")

    # ── Checkpoint ───────────────────────────────────────────────
    ckpt = args.checkpoint or find_latest_checkpoint()
    if not ckpt or not os.path.exists(ckpt):
        print("\n  [ERROR] Khong tim thay checkpoint bilstm_best_*.pt")
        print("  Train truoc: python -m src.lstm.train_bilstm")
        sys.exit(1)
    print(f"  Checkpoint : {Path(ckpt).name}\n")
    model, label_map = load_model(ckpt, device)
    idx2label = {v: k for k, v in label_map.items()}

    # ── Video source ─────────────────────────────────────────────
    src = args.source
    try:
        src = int(src)
    except ValueError:
        pass
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"  [ERROR] Khong mo duoc: {src}")
        sys.exit(1)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video      : {fw}×{fh}")

    # ── State ────────────────────────────────────────────────────
    buf        = FrameBuffer(vsl_cfg.SEQ_LEN, vsl_cfg.FEAT_DIM)
    smooth_buf = deque(maxlen=args.smooth)
    fps_buf    = deque(maxlen=30)
    conf_thr   = args.conf
    paused     = False
    notif      = ''
    notif_ts   = 0.0
    label      = ''
    conf       = 0.0
    t_prev     = time.time()
    frame_idx  = 0
    ss_dir     = _PROJECT_ROOT / 'screenshots'
    ss_dir.mkdir(exist_ok=True)

    WIN = "VSL BiLSTM – Realtime"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    print(f"\n  Dang chay... (Q/ESC de thoat)\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if isinstance(src, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    buf.reset(); smooth_buf.clear()
                    continue
                break

            frame_idx += 1

            # FPS
            t_now = time.time()
            fps_buf.append(1.0 / max(t_now - t_prev, 1e-9))
            t_prev = t_now

            # Feature extraction
            if HAS_EXTRACTOR:
                try:
                    feat = _extractor.extract(frame)
                except Exception:
                    feat = np.zeros(vsl_cfg.FEAT_DIM, dtype=np.float32)
            else:
                feat = np.random.randn(vsl_cfg.FEAT_DIM).astype(np.float32) * 0.05

            buf.push(feat)

            # Inference
            if buf.ready and frame_idx % args.skip == 0:
                with torch.no_grad():
                    logits = model(buf.to_tensor(device))
                    proba  = F.softmax(logits, dim=-1)[0].cpu().numpy()
                top_idx = int(np.argmax(proba))
                conf    = float(proba[top_idx])
                raw_lbl = idx2label.get(top_idx, str(top_idx))
                smooth_buf.append(raw_lbl)
                label = Counter(smooth_buf).most_common(1)[0][0]

        fps = float(np.mean(fps_buf)) if fps_buf else 0.0

        # Draw
        draw_overlay(frame, label, conf, conf_thr,
                     buf.ratio, fps, paused, notif, notif_ts)
        cv2.imshow(WIN, frame)

        # Keyboard
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            break

        elif key in (ord('r'), ord('R')):
            buf.reset(); smooth_buf.clear()
            label, conf = '', 0.0
            notif, notif_ts = "Buffer reset", time.time()

        elif key in (ord('p'), ord('P')):
            paused = not paused
            notif, notif_ts = ("PAUSED" if paused else "RESUMED"), time.time()

        elif key in (ord('s'), ord('S')):
            ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            path = str(ss_dir / f'bilstm_{ts}.png')
            cv2.imwrite(path, frame)
            notif, notif_ts = f"Saved: bilstm_{ts}.png", time.time()

        elif key in (ord('+'), ord('=')):
            conf_thr = min(0.99, round(conf_thr + 0.05, 2))
            notif, notif_ts = f"Threshold → {conf_thr:.2f}", time.time()

        elif key == ord('-'):
            conf_thr = max(0.05, round(conf_thr - 0.05, 2))
            notif, notif_ts = f"Threshold → {conf_thr:.2f}", time.time()

    cap.release()
    cv2.destroyAllWindows()
    print("  Thoat.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='VSL BiLSTM Realtime')
    ap.add_argument('--source',     default='0',
                    help='0=webcam, 1, 2, hoac path/to/video.mp4')
    ap.add_argument('--checkpoint', default=None,
                    help='Path .pt (mac dinh: tim bilstm_best_*.pt moi nhat)')
    ap.add_argument('--conf',       type=float, default=0.60,
                    help='Confidence threshold (default 0.60)')
    ap.add_argument('--smooth',     type=int,   default=5,
                    help='Smoothing window (default 5 frames)')
    ap.add_argument('--skip',       type=int,   default=2,
                    help='Inference moi N frames (default 2)')
    main(ap.parse_args())