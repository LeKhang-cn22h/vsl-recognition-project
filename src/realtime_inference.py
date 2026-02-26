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
    [+/-]   - Tang/giam nguong tin cay
"""

import os
import cv2
import time
import argparse
import numpy as np
from collections import deque
from datetime import datetime

from vsl import (cfg, load_model, load_display_names,
                  get_display_name, RealtimeExtractor,
                  InferenceEngine, UIRenderer)


class RealtimeApp:
    def __init__(self, checkpoint_path: str,
                 top_k: int = cfg.TOP_K,
                 confidence_thr: float = cfg.CONFIDENCE_THR,
                 smooth_window: int = cfg.SMOOTH_WINDOW):

        self.confidence_thr = confidence_thr

        print(f"\n  Loading checkpoint: {checkpoint_path}")
        model, label_map, epoch, val_acc = load_model(checkpoint_path)

        print("\n  Khoi tao MediaPipe detectors...")
        self.extractor = RealtimeExtractor()
        self.engine    = InferenceEngine(
            model, label_map,
            confidence_thr = confidence_thr,
            top_k          = top_k,
            smooth_window  = smooth_window,
        )

        self.history  = deque(maxlen=50)
        self.paused   = False
        self._fps_buf = deque(maxlen=30)
        self._t_prev  = time.time()

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
        print("  " + "─"*30)
        print("  [Q] Thoat  [SPACE] Pause  [C] Clear  [S] Screenshot  [+/-] Threshold")
        print("  " + "─"*30 + "\n")

        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            fps   = self._fps()
            top_preds = None

            if not self.paused:
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.extractor.send_frame(rgb)
                feats = self.extractor.extract_features()
                self.engine.push_frame(feats)
                top_preds = self.engine.predict(curr_features=feats)

                if self.engine._just_confirmed:
                    label = self.engine._just_confirmed
                    ts    = datetime.now().strftime('%H:%M:%S')
                    conf  = top_preds[0][1] if top_preds else 0.0
                    self.history.append((ts, label, conf))
                    print(f"  [{ts}] Detected: "
                          f"{get_display_name(label)}  ({conf*100:.1f}%)")
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
            if key in (ord('q'), ord('Q')):
                break
            elif key == ord(' '):
                self.paused = not self.paused
                print(f"  {'PAUSED' if self.paused else 'RESUMED'}")
            elif key in (ord('c'), ord('C')):
                self.engine.reset()
                self.history.clear()
                print("  History + buffer cleared")
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

    app = RealtimeApp(
        checkpoint_path = args.checkpoint,
        top_k           = args.top_k,
        confidence_thr  = args.threshold,
        smooth_window   = args.smooth,
    )
    app.run()


if __name__ == '__main__':
    main()