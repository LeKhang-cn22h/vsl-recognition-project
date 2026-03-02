"""
realtime_inference.py - Nhận diện VSL realtime từ webcam
=========================================================
Cách chạy:
    python realtime_inference.py
    python realtime_inference.py --checkpoint checkpoints/best_model.pt
    python realtime_inference.py --top_k 3 --threshold 0.7

Phím tắt:
    [Q]     - Thoat
    [SPACE] - Tam dung / Tiep tuc
    [C]     - Xoa lich su + reset buffer
    [S]     - Luu screenshot
    [G]     - Bat/tat hien thi gate values
    [T]     - Bat/tat hien thi touch zones
    [+/-]   - Tang/giam nguong tin cay

Thay đổi v2:
    - Hiển thị touch detection (vùng tay đang chạm)
    - Hiển thị soft gate values (debug mode)
    - Dùng forward_with_gates() khi gate mode bật
"""

import os
import cv2
import time
import argparse
import numpy as np
import torch
from collections import deque
from datetime import datetime

from vsl import (cfg, load_model, load_display_names,
                  get_display_name, RealtimeExtractor,
                  InferenceEngine, UIRenderer)
from vsl.extractor import detect_touch


# ══════════════════════════════════════════════════════════
# TOUCH DISPLAY HELPER
# ══════════════════════════════════════════════════════════

def draw_touch_info(frame, touch_info: dict, w: int, h: int):
    """Vẽ thông tin vùng tay đang chạm lên frame."""
    if not touch_info or touch_info.get('hand') is None:
        return

    lines = []
    if touch_info.get('face_zone_vn'):
        hand = touch_info['hand']
        hand_txt = {'right': 'Tay P', 'left': 'Tay T', 'both': '2 Tay'}.get(hand, '')
        lines.append(f"{hand_txt} cham: {touch_info['face_zone_vn']}")
    if touch_info.get('body_zone_vn'):
        lines.append(f"Body: {touch_info['body_zone_vn']}")

    if not lines:
        return

    x, y = 10, h - 120
    for line in lines:
        # Background
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x-4, y-th-4), (x+tw+4, y+4),
                      (0, 60, 80), -1)
        cv2.putText(frame, line, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
        y += 25


def draw_gate_panel(frame, gate_dict: dict, w: int, h: int):
    """Vẽ soft gate values dạng thanh bar."""
    if not gate_dict:
        return

    panel_x = w - 180
    panel_y = 80
    cv2.rectangle(frame, (panel_x - 5, panel_y - 20),
                  (w - 5, panel_y + len(gate_dict) * 22 + 5),
                  (20, 20, 20), -1)
    cv2.putText(frame, "Soft Gate", (panel_x, panel_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    GATE_COLORS = {
        'pose':       (100, 200, 100),
        'face':       (200, 150, 50),
        'left_hand':  (50, 150, 255),
        'right_hand': (50, 200, 255),
        'interact':   (200, 100, 200),
    }

    for i, (name, val) in enumerate(gate_dict.items()):
        y = panel_y + 10 + i * 22
        bar_max = 120
        bar_w   = int(val * bar_max)
        color   = GATE_COLORS.get(name, (150, 150, 150))

        # Label
        short = {'pose': 'Pose', 'face': 'Face',
                 'left_hand': 'L.Hand', 'right_hand': 'R.Hand',
                 'interact': 'Interact'}.get(name, name)
        cv2.putText(frame, f"{short[:7]:<7}", (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        # Bar
        bx = panel_x + 58
        cv2.rectangle(frame, (bx, y-8), (bx+bar_max, y+2), (50,50,50), -1)
        if bar_w > 0:
            cv2.rectangle(frame, (bx, y-8), (bx+bar_w, y+2), color, -1)
        cv2.putText(frame, f"{val:.2f}", (bx+bar_max+4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


# ══════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════

class RealtimeApp:
    def __init__(self, checkpoint_path: str,
                 top_k: int          = cfg.TOP_K,
                 confidence_thr: float = cfg.CONFIDENCE_THR,
                 smooth_window: int  = cfg.SMOOTH_WINDOW):

        self.confidence_thr = confidence_thr
        self.show_gates     = False   # [G] toggle
        self.show_touch     = True    # [T] toggle

        print(f"\n  Loading checkpoint: {checkpoint_path}")
        self.model, label_map, epoch, val_acc = load_model(checkpoint_path)

        print("\n  Khoi tao MediaPipe detectors...")
        self.extractor = RealtimeExtractor()
        self.engine    = InferenceEngine(
            self.model, label_map,
            confidence_thr = confidence_thr,
            top_k          = top_k,
            smooth_window  = smooth_window,
        )

        self.history   = deque(maxlen=50)
        self.paused    = False
        self._fps_buf  = deque(maxlen=30)
        self._t_prev   = time.time()
        self._gate_dict = {}

        os.makedirs('screenshots', exist_ok=True)

    def _fps(self) -> float:
        now = time.time()
        self._fps_buf.append(1.0 / max(now - self._t_prev, 1e-6))
        self._t_prev = now
        return float(np.mean(self._fps_buf))

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  LOI: Khong mo duoc webcam!"); return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n  Camera: {w}x{h}")
        print("  " + "─"*50)
        print("  [Q] Thoat  [SPACE] Pause  [C] Clear")
        print("  [G] Gate   [T] Touch       [S] Screenshot  [+/-] Threshold")
        print("  " + "─"*50 + "\n")

        touch_info = {}
        top_preds  = None

        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            fps   = self._fps()

            if not self.paused:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.extractor.send_frame(rgb)
                feats = self.extractor.extract_features()
                self.engine.push_frame(feats)

                # Predict — dùng forward_with_gates nếu show_gates bật
                if self.show_gates:
                    buf = self.engine.frame_buffer
                    if len(buf) == cfg.SEQ_LEN:
                        x = torch.tensor(
                            np.array(buf), dtype=torch.float32
                        ).unsqueeze(0).to(cfg.DEVICE)
                        with torch.no_grad():
                            _, self._gate_dict = \
                                self.model.forward_with_gates(x)

                top_preds = self.engine.predict(curr_features=feats)

                # Touch detection
                if self.show_touch:
                    touch_info = self.extractor.get_touch_info()

                if self.engine._just_confirmed:
                    label = self.engine._just_confirmed
                    ts    = datetime.now().strftime('%H:%M:%S')
                    conf  = top_preds[0][1] if top_preds else 0.0
                    touch_str = ""
                    if touch_info.get('face_zone_vn'):
                        touch_str = f" | {touch_info['face_zone_vn']}"
                    self.history.append((ts, label, conf))
                    print(f"  [{ts}] {get_display_name(label)}"
                          f"  ({conf*100:.1f}%){touch_str}")
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

            # Touch info
            if self.show_touch:
                draw_touch_info(frame, touch_info, w, h)

            # Gate values
            if self.show_gates and self._gate_dict:
                draw_gate_panel(frame, self._gate_dict, w, h)

            # Indicator góc trên trái
            indicators = []
            if self.show_touch: indicators.append("TOUCH:ON")
            if self.show_gates: indicators.append("GATE:ON")
            if indicators:
                cv2.putText(frame, "  ".join(indicators),
                            (10, h - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (100, 255, 100), 1)

            cv2.imshow('VSL Realtime Inference', frame)

            # ── Phím bấm ──
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key == ord(' '):
                self.paused = not self.paused
                print(f"  {'PAUSED' if self.paused else 'RESUMED'}")
            elif key in (ord('c'), ord('C')):
                self.engine.reset()
                self.history.clear()
                touch_info = {}
                self._gate_dict = {}
                print("  History + buffer cleared")
            elif key in (ord('g'), ord('G')):
                self.show_gates = not self.show_gates
                print(f"  Gate display: {'ON' if self.show_gates else 'OFF'}")
            elif key in (ord('t'), ord('T')):
                self.show_touch = not self.show_touch
                print(f"  Touch display: {'ON' if self.show_touch else 'OFF'}")
            elif key in (ord('s'), ord('S')):
                fn = f"screenshots/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
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


def main():
    parser = argparse.ArgumentParser(description='VSL Realtime Inference')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best_model.pt')
    parser.add_argument('--top_k',     type=int,   default=cfg.TOP_K)
    parser.add_argument('--threshold', type=float, default=cfg.CONFIDENCE_THR)
    parser.add_argument('--smooth',    type=int,   default=cfg.SMOOTH_WINDOW)
    args = parser.parse_args()

    load_display_names()

    print("\n" + "="*60)
    print(" VSL REALTIME INFERENCE ".center(60, "="))
    print("="*60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Top-K      : {args.top_k}")
    print(f"  Threshold  : {args.threshold*100:.0f}%")
    print(f"  Smoothing  : {args.smooth} frames")
    print(f"  Device     : {cfg.DEVICE}")
    print(f"  FEAT_DIM   : {cfg.FEAT_DIM}")

    app = RealtimeApp(
        checkpoint_path = args.checkpoint,
        top_k           = args.top_k,
        confidence_thr  = args.threshold,
        smooth_window   = args.smooth,
    )
    app.run()


if __name__ == '__main__':
    main()