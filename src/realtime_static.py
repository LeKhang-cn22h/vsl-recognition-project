"""
realtime_static.py - Realtime test Static MLP (fingerspelling VSL)
===================================================================
Chạy:
    python src/realtime_static.py

Phím tắt:
    Q / ESC  – Thoát
    S        – Screenshot
    C        – Xóa log hiện tại
    +  / -   – Tăng/giảm confidence threshold
"""

import sys
import os
import glob
import json
import time
import datetime
import math
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

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request

# ══════════════════════════════════════════════════════════════════
# MODEL (copy từ train_static_mlp.py)
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
    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════
# HAND FEATURE EXTRACTION (copy từ video_to_npy_static.py)
# ══════════════════════════════════════════════════════════════════

FINGERTIPS   = [4, 8, 12, 16, 20]
FINGER_BASES = [2, 5, 9, 13, 17]
FINGER_CHAINS = [
    [0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12],
    [0,13,14,15,16], [0,17,18,19,20],
]

def _angle(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))

def extract_hand_features(landmarks_21: np.ndarray) -> np.ndarray:
    """21 landmarks (21,3) → vector 96 dims."""
    lm = landmarks_21.copy()
    # Normalize về wrist = origin
    lm -= lm[0]
    scale = np.linalg.norm(lm[9])
    if scale > 1e-6:
        lm /= scale

    coords  = lm.flatten()                                             # 63
    angles  = np.array([                                               # 15
        _angle(lm[c[i-1]] - lm[c[i]], lm[c[i+1]] - lm[c[i]])
        for c in FINGER_CHAINS for i in range(1, 4)
    ])
    lengths = np.array([                                               # 5
        np.linalg.norm(lm[t] - lm[b])
        for b, t in zip(FINGER_BASES, FINGERTIPS)
    ])
    tips    = lm[FINGERTIPS]
    dists   = np.array([                                               # 10
        np.linalg.norm(tips[i] - tips[j])
        for i, j in combinations(range(5), 2)
    ])
    v1, v2  = lm[5] - lm[0], lm[17] - lm[0]
    n       = np.cross(v1, v2)
    norm    = np.linalg.norm(n)
    palm_n  = (n / norm if norm > 1e-6 else n).astype(np.float32)    # 3

    return np.concatenate([coords, angles, lengths, dists, palm_n]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# MEDIAPIPE HAND DETECTOR
# ══════════════════════════════════════════════════════════════════

_HAND_MODEL_URL  = (
    'https://storage.googleapis.com/mediapipe-models/'
    'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')
_HAND_MODEL_PATH = str(_PROJECT_ROOT / 'hand_landmarker.task')

def _ensure_hand_model():
    if not os.path.exists(_HAND_MODEL_PATH):
        print("  Dang tai hand_landmarker.task ...")
        urllib.request.urlretrieve(_HAND_MODEL_URL, _HAND_MODEL_PATH)
    return _HAND_MODEL_PATH


class HandDetector:
    """
    LIVE_STREAM mode — giống webcam_collector.py
    Callback trả về landmarks bất đồng bộ.
    """
    def __init__(self):
        self._latest = None   # (21, 3) hoặc None
        self._ts     = 0

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=_ensure_hand_model()),
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._on_result,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

    def _on_result(self, result, image, ts):
        if result.hand_landmarks:
            self._latest = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]],
                dtype=np.float32)   # (21, 3)
        else:
            self._latest = None

    def detect_async(self, frame_bgr: np.ndarray):
        self._ts += 33
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            self.detector.detect_async(mp_img, self._ts)
        except Exception:
            pass

    def get_landmarks(self):
        """Trả về (21,3) hoặc None."""
        return self._latest

    def draw(self, frame, w, h):
        """Vẽ 21 landmarks + connections lên frame."""
        lm = self._latest
        if lm is None:
            return
        # Connections
        connections = [
            (0,1),(1,2),(2,3),(3,4),          # thumb
            (0,5),(5,6),(6,7),(7,8),           # index
            (0,9),(9,10),(10,11),(11,12),      # middle
            (0,13),(13,14),(14,15),(15,16),    # ring
            (0,17),(17,18),(18,19),(19,20),    # pinky
            (5,9),(9,13),(13,17),              # palm
        ]
        pts = [(int(lm[i,0]*w), int(lm[i,1]*h)) for i in range(21)]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], (0, 200, 100), 1)
        for i, (x, y) in enumerate(pts):
            r = 4 if i in FINGERTIPS else 3
            cv2.circle(frame, (x, y), r, (0, 255, 150), -1)

    def close(self):
        self.detector.close()


# ══════════════════════════════════════════════════════════════════
# CHECKPOINT LOADER
# ══════════════════════════════════════════════════════════════════

def find_latest_checkpoint():
    files = glob.glob(str(_PROJECT_ROOT / 'checkpoints' / 'static_mlp_best_*.pt'))
    return max(files, key=os.path.getmtime) if files else None


def load_model(ckpt_path, device):
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    label_map = ckpt['label_map']
    mcfg      = ckpt.get('cfg', {})
    model = StaticMLP(
        feat_dim    = mcfg.get('FEAT_DIM',   96),
        num_classes = len(label_map),
        hidden_1    = mcfg.get('HIDDEN_1',  256),
        hidden_2    = mcfg.get('HIDDEN_2',  128),
        hidden_3    = mcfg.get('HIDDEN_3',   64),
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    val_acc = ckpt.get('val_acc', 0) * 100
    print(f"  [OK] Loaded | classes={len(label_map)} "
          f"| epoch={ckpt.get('epoch','?')} | val_acc={val_acc:.1f}%")
    return model, label_map


# ══════════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════════

class ResultLogger:
    """Ghi log kết quả nhận dạng ra file txt."""
    def __init__(self):
        log_dir = _PROJECT_ROOT / 'logs'
        log_dir.mkdir(exist_ok=True)
        ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.path = str(log_dir / f'static_realtime_{ts}.txt')
        self.entries  = []
        self.last_log = ''   # tránh log lặp liên tiếp

        with open(self.path, 'w', encoding='utf-8') as f:
            f.write(f"Static MLP Realtime Log — {ts}\n")
            f.write("=" * 50 + "\n")
            f.write(f"{'Time':<12} {'Label':<25} {'Conf':>6}\n")
            f.write("-" * 50 + "\n")
        print(f"  [Log] {self.path}")

    def log(self, label: str, conf: float):
        """Chỉ log khi label thay đổi."""
        if label == self.last_log:
            return
        self.last_log = label
        ts  = datetime.datetime.now().strftime('%H:%M:%S')
        row = {'time': ts, 'label': label, 'conf': round(conf, 4)}
        self.entries.append(row)

        line = f"{ts:<12} {label:<25} {conf*100:>5.1f}%\n"
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(line)
        print(f"  [LOG] {ts}  {label}  {conf*100:.1f}%")

    def summary(self):
        """Ghi tóm tắt cuối file."""
        if not self.entries:
            return
        counts = Counter(e['label'] for e in self.entries)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 50 + "\n")
            f.write("SUMMARY\n")
            f.write(f"Total detections : {len(self.entries)}\n")
            f.write("Per label:\n")
            for lb, cnt in counts.most_common():
                f.write(f"  {lb:<25} {cnt}\n")
        print(f"\n  [Log saved] {self.path}")
        print(f"  Total detections: {len(self.entries)}")

    def clear(self):
        self.entries  = []
        self.last_log = ''
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- CLEARED {datetime.datetime.now().strftime('%H:%M:%S')} ---\n")
        print("  [Log cleared]")


# ══════════════════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════════════════

FONT   = cv2.FONT_HERSHEY_DUPLEX
GREEN  = (55,  210,  55)
ORANGE = (30,  160, 255)
YELLOW = (20,  220, 220)
WHITE  = (240, 240, 240)
GRAY   = (110, 110, 110)
BLACK  = (8,     8,   8)
RED    = (50,   50, 220)


def _put(img, text, pos, scale=0.6, color=WHITE, thick=1):
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), FONT, scale, BLACK, thick+1, cv2.LINE_AA)
    cv2.putText(img, text, pos,         FONT, scale, color, thick,   cv2.LINE_AA)


def _pill(img, x1, y1, x2, y2, color, alpha=0.78, r=10):
    ov = img.copy()
    cv2.rectangle(ov, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(ov, (x1, y1+r), (x2, y2-r), color, -1)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(ov, (cx, cy), r, color, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)


def _bar(img, x, y, w, h, ratio, color, bg=(35,35,45)):
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
    fw = max(0, int(w * min(ratio, 1.0)))
    if fw:
        cv2.rectangle(img, (x, y), (x+fw, y+h), color, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (65,65,75), 1)


def draw_overlay(frame, label, conf, conf_thr, fps,
                 hand_detected, log_count, notif, notif_ts,
                 recent_logs):
    h, w = frame.shape[:2]
    above = conf >= conf_thr and bool(label) and hand_detected

    # ── Header bar ───────────────────────────────────────────────
    cv2.rectangle(frame, (0,0), (w, 48), (20,20,28), -1)
    _put(frame, "Static MLP – Realtime", (10, 30), 0.65, WHITE, 2)

    # FPS
    fps_c = GREEN if fps >= 20 else (YELLOW if fps >= 10 else ORANGE)
    _put(frame, f"FPS {fps:.0f}", (w-110, 30), 0.55, fps_c)

    # Threshold
    _put(frame, f"thr {conf_thr:.2f}", (w-110, 44), 0.38, GRAY)

    # ── Hand status (top-left dưới header) ───────────────────────
    if hand_detected:
        _put(frame, "Hand: OK", (10, 68), 0.45, GREEN)
    else:
        _put(frame, "Hand: NOT FOUND", (10, 68), 0.45, ORANGE)

    # Log count
    _put(frame, f"Log: {log_count}", (10, 88), 0.40, GRAY)

    # ── PREDICTION BLOCK ─────────────────────────────────────────
    if above:
        (tw, _), _ = cv2.getTextSize(label, FONT, 1.1, 2)
        cx   = w // 2
        px1  = cx - tw//2 - 30
        px2  = cx + tw//2 + 30
        py1  = h - 155
        py2  = h - 65

        _pill(frame, px1, py1, px2, py2, (15,72,15), alpha=0.82)
        _put(frame, label, (cx - tw//2, h - 85),
             scale=1.1, color=WHITE, thick=2)

        bw = px2 - px1 - 10
        _bar(frame, px1+5, h-70, bw, 12, conf, GREEN)
        _put(frame, f"{conf*100:.1f}%", (px2+6, h-61), 0.52, GREEN)

        # Badge LOG
        _pill(frame, px1, py1-24, px1+90, py1-4,
              (15,100,15), alpha=0.88, r=5)
        _put(frame, "✓ DETECTED", (px1+4, py1-9), 0.36, GREEN)

    elif hand_detected and label:
        # Có tay nhưng chưa đủ confidence
        _put(frame, f"? {label}  {conf*100:.1f}%",
             (10, h-20), 0.48, GRAY)

    elif not hand_detected:
        _put(frame, "Gioi tay vao khung hinh...",
             (w//2 - 140, h//2), 0.65, ORANGE)

    # ── Recent log panel (bên phải) ───────────────────────────────
    if recent_logs:
        panel_x = w - 230
        _put(frame, "Recent:", (panel_x, 75), 0.42, GRAY)
        for i, (lb, cf) in enumerate(recent_logs[-6:]):
            color = GREEN if cf >= conf_thr else GRAY
            _put(frame, f"{lb}  {cf*100:.0f}%",
                 (panel_x, 95 + i*20), 0.40, color)

    # ── Notification toast ────────────────────────────────────────
    if notif and (time.time() - notif_ts < 2.0):
        (nw, _), _ = cv2.getTextSize(notif, FONT, 0.55, 1)
        nx = w//2 - nw//2
        _pill(frame, nx-12, h//2-24, nx+nw+12, h//2+8,
              (40,40,60), alpha=0.85, r=7)
        _put(frame, notif, (nx, h//2), 0.55, YELLOW)

    # ── Hint bar (bottom) ─────────────────────────────────────────
    cv2.rectangle(frame, (0, h-22), (w, h), (20,20,28), -1)
    _put(frame,
         "Q:Quit   S:Screenshot   C:Clear log   +/-:Threshold",
         (8, h-6), 0.32, (70,70,80), thick=1)


# ══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--conf',       type=float, default=0.60)
    ap.add_argument('--smooth',     type=int,   default=5)
    ap.add_argument('--source',     default='0')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*50}")
    print("  Static MLP – Realtime Inference")
    print(f"{'='*50}")
    print(f"  Device : {device}")

    # Load checkpoint
    ckpt = args.checkpoint or find_latest_checkpoint()
    if not ckpt or not os.path.exists(ckpt):
        print("\n  [ERROR] Khong tim thay static_mlp_best_*.pt")
        print("  Chay train truoc: python src/train_static_mlp.py")
        sys.exit(1)
    print(f"  Ckpt   : {Path(ckpt).name}")
    model, label_map = load_model(ckpt, device)
    idx2label = {v: k for k, v in label_map.items()}

    # Video source
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

    # Init
    detector    = HandDetector()
    logger      = ResultLogger()
    smooth_buf  = deque(maxlen=args.smooth)
    fps_buf     = deque(maxlen=30)
    conf_thr    = args.conf
    label       = ''
    conf        = 0.0
    notif       = ''
    notif_ts    = 0.0
    t_prev      = time.time()
    recent_logs = []   # list of (label, conf)

    ss_dir = _PROJECT_ROOT / 'screenshots'
    ss_dir.mkdir(exist_ok=True)

    WIN = "Static MLP – Realtime"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    print(f"\n  Dang chay... (Q/ESC de thoat)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # FPS
        t_now = time.time()
        fps_buf.append(1.0 / max(t_now - t_prev, 1e-9))
        t_prev = t_now
        fps    = float(np.mean(fps_buf))

        # Detect hand (async)
        detector.detect_async(frame)
        lm = detector.get_landmarks()   # (21,3) hoặc None

        hand_detected = lm is not None

        if hand_detected:
            # Extract features
            feat = extract_hand_features(lm)   # (96,)

            # Inference
            with torch.no_grad():
                x      = torch.from_numpy(feat).unsqueeze(0).to(device)
                logits = model(x)
                proba  = F.softmax(logits, dim=-1)[0].cpu().numpy()

            top_idx = int(np.argmax(proba))
            conf    = float(proba[top_idx])
            raw_lbl = idx2label.get(top_idx, str(top_idx))

            # Smoothing
            smooth_buf.append(raw_lbl)
            label = Counter(smooth_buf).most_common(1)[0][0]

            # Log nếu đủ confidence
            if conf >= conf_thr:
                logger.log(label, conf)
                if not recent_logs or recent_logs[-1][0] != label:
                    recent_logs.append((label, conf))
                    if len(recent_logs) > 20:
                        recent_logs.pop(0)
        else:
            smooth_buf.clear()
            label = ''
            conf  = 0.0

        # Vẽ hand skeleton
        detector.draw(frame, fw, fh)

        # Vẽ overlay
        draw_overlay(
            frame, label, conf, conf_thr, fps,
            hand_detected, len(logger.entries),
            notif, notif_ts, recent_logs)

        cv2.imshow(WIN, frame)

        # Keyboard
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            break

        elif key in (ord('s'), ord('S')):
            ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            path = str(ss_dir / f'static_{ts}.png')
            cv2.imwrite(path, frame)
            notif, notif_ts = f"Saved: static_{ts}.png", time.time()

        elif key in (ord('c'), ord('C')):
            logger.clear()
            recent_logs.clear()
            notif, notif_ts = "Log cleared", time.time()

        elif key in (ord('+'), ord('=')):
            conf_thr = min(0.99, round(conf_thr + 0.05, 2))
            notif, notif_ts = f"Threshold → {conf_thr:.2f}", time.time()

        elif key == ord('-'):
            conf_thr = max(0.05, round(conf_thr - 0.05, 2))
            notif, notif_ts = f"Threshold → {conf_thr:.2f}", time.time()

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    logger.summary()
    print("  Thoat.")


if __name__ == '__main__':
    main()