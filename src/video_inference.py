"""
video_inference.py - Test model VSL với file video
===================================================
Cách chạy:
    python video_inference.py --video path/to/video.mp4
    python video_inference.py --video video.mp4 --label buoi_sang
    python video_inference.py --folder data/videos/buoi_sang --label buoi_sang
    python video_inference.py --video video.mp4 --slowdown 3

Phím tắt:
    [Q]     - Thoat
    [SPACE] - Tam dung / Tiep tuc
    [R]     - Xem lai tu dau
    [S]     - Luu screenshot
    [+/-]   - Tang/giam nguong tin cay
"""

import os
import cv2
import argparse
import numpy as np
from collections import deque, Counter
from datetime import datetime

import torch
import torch.nn.functional as F

from vsl import (cfg, load_model, load_display_names,
                  get_display_name, is_idle_label,
                  VideoExtractor, resample_sequence)


class VideoInferenceApp:
    def __init__(self, checkpoint_path: str,
                 confidence_thr: float = cfg.CONFIDENCE_THR,
                 top_k: int            = cfg.TOP_K,
                 slowdown: int         = 1):

        self.confidence_thr = confidence_thr
        self.top_k          = top_k
        self.slowdown       = slowdown

        print(f"\n  Loading checkpoint: {checkpoint_path}")
        self.model, label_map, epoch, val_acc = load_model(checkpoint_path)
        self.idx2label = {v: k for k, v in label_map.items()}

        print("  Khoi tao MediaPipe detectors...")
        self.extractor = VideoExtractor()
        os.makedirs('screenshots', exist_ok=True)

    # ── Predict từ sequence ──────────────────────────────────

    def _predict_sequence(self, sequence: np.ndarray, prob_buf: deque):
        if len(sequence) < cfg.SEQ_LEN:
            seq = resample_sequence(sequence, cfg.SEQ_LEN)
        else:
            seq = np.stack(sequence[-cfg.SEQ_LEN:])

        x = torch.from_numpy(seq).unsqueeze(0).to(cfg.DEVICE)
        with torch.no_grad():
            probs = F.softmax(self.model(x), dim=-1).cpu().numpy()[0]

        prob_buf.append(probs)
        smooth = np.mean(prob_buf, axis=0)
        top_idx = np.argsort(smooth)[::-1][:self.top_k]
        return [(self.idx2label.get(i, f'cls_{i}'), float(smooth[i]))
                for i in top_idx]

    # ── Draw UI ──────────────────────────────────────────────

    def _draw_ui(self, frame, top_preds, frame_idx,
                 total_frames, fps_video, landmarks,
                 result_log, ground_truth=None):
        h, w = frame.shape[:2]
        F_  = cv2.FONT_HERSHEY_SIMPLEX

        def bg(x1, y1, x2, y2, alpha=0.75):
            ov = frame.copy()
            cv2.rectangle(ov, (x1,y1), (x2,y2), (18,18,28), -1)
            cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

        # Header
        bg(0, 0, w, 52)
        cv2.putText(frame, "VSL VIDEO INFERENCE", (14, 34), F_, 0.75, (0,220,160), 2)
        cv2.putText(frame, f"Frame {frame_idx}/{total_frames}",
                    (w-200, 22), F_, 0.5, (200,200,200), 1)
        cv2.putText(frame, f"x{self.slowdown} slow",
                    (w-200, 42), F_, 0.45, (200,200,200), 1)

        if ground_truth:
            cv2.putText(frame,
                        f"Label that: {get_display_name(ground_truth)}",
                        (14, 68), F_, 0.55, (0,255,200), 1)

        # Prediction panel
        py = 78 if not ground_truth else 90
        bg(10, py, 340, py+200)
        cv2.putText(frame, "PREDICTION", (22, py+22), F_, 0.5, (60,160,255), 1)
        cv2.line(frame, (14, py+28), (334, py+28), (60,60,80), 1)

        if top_preds:
            show = [(l,p) for l,p in top_preds if not is_idle_label(l)] or top_preds
            tl, tp = show[0]
            col = (0,220,160) if tp >= self.confidence_thr else (0,200,230)
            cv2.putText(frame, get_display_name(tl).upper(),
                        (22, py+58), F_, 0.85, col, 2)
            # Bar
            bw = int(300 * min(tp, 1.0))
            cv2.rectangle(frame, (22, py+64), (322, py+78), (50,50,60), -1)
            if bw > 0:
                cv2.rectangle(frame, (22, py+64), (22+bw, py+78), col, -1)
            cv2.putText(frame, f"{tp*100:.1f}%", (270, py+90), F_, 0.45, col, 1)

            for i, (lbl, prob) in enumerate(show[1:4], 1):
                ry = py + 98 + (i-1)*30
                cv2.putText(frame, f"{i+1}. {get_display_name(lbl)}",
                            (22, ry+14), F_, 0.45, (240,240,240), 1)
                bw2 = int(240 * min(prob, 1.0))
                cv2.rectangle(frame, (22, ry+16), (262, ry+24), (50,50,60), -1)
                if bw2 > 0:
                    cv2.rectangle(frame, (22, ry+16), (22+bw2, ry+24), (140,140,150), -1)
                cv2.putText(frame, f"{prob*100:.1f}%",
                            (268, ry+24), F_, 0.38, (140,140,150), 1)

        # Result log
        lpy = h - 180
        bg(10, lpy, 340, h-8)
        cv2.putText(frame, "DETECTED", (22, lpy+20), F_, 0.5, (60,160,255), 1)
        cv2.line(frame, (14, lpy+26), (334, lpy+26), (60,60,80), 1)
        for i, (ts, label, conf) in enumerate(list(result_log)[-7:][::-1]):
            ty  = lpy + 44 + i*20
            age = max(100, 220 - i*20)
            cv2.putText(frame,
                        f"{ts}  {get_display_name(label):<15} {conf*100:4.0f}%",
                        (22, ty), F_, 0.38, (age,age,age), 1)

        # Progress bar
        bg(0, h-22, w, h, alpha=0.88)
        if total_frames > 0:
            pw = int((w-4) * frame_idx / total_frames)
            cv2.rectangle(frame, (2, h-18), (2+pw, h-6), (0,220,160), -1)
        cv2.putText(frame,
                    "[Q]Thoat [SPACE]Pause [R]Replay [S]Screenshot [+/-]Threshold",
                    (10, h-4), F_, 0.35, (140,140,150), 1)

        # Skeleton
        pose = landmarks.get('pose')
        lh   = landmarks.get('left_hand')
        rh   = landmarks.get('right_hand')
        POSE_CONN = [(11,13),(13,15),(12,14),(14,16),(11,12),(11,23),(12,24)]
        if pose:
            for i, j in POSE_CONN:
                if (i < len(pose) and j < len(pose) and
                        pose[i].visibility > 0.5 and pose[j].visibility > 0.5):
                    cv2.line(frame,
                             (int(pose[i].x*w), int(pose[i].y*h)),
                             (int(pose[j].x*w), int(pose[j].y*h)),
                             (0,180,120), 1)
        HAND_CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                     (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                     (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
        for hlms, hc in [(lh,(255,100,0)),(rh,(0,255,140))]:
            if hlms is None: continue
            for i, j in HAND_CONN:
                cv2.line(frame,
                         (int(hlms[i].x*w), int(hlms[i].y*h)),
                         (int(hlms[j].x*w), int(hlms[j].y*h)), hc, 1)

        return frame

    # ── Run single video ─────────────────────────────────────

    def run_video(self, video_path: str, ground_truth: str = None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  LOI: Khong mo duoc: {video_path}"); return None

        fps_v   = cap.get(cv2.CAP_PROP_FPS) or 30
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\n  Video: {os.path.basename(video_path)}")
        print(f"  Frames: {n_total} | FPS: {fps_v:.1f}")
        if ground_truth:
            print(f"  Label that: {get_display_name(ground_truth)}")

        # Bước 1: Extract
        print(f"  [1/2] Trich xuat features...")
        all_feats, all_frames, all_lms = [], [], []
        while True:
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feats, lms = self.extractor.extract_frame(rgb)
            all_feats.append(feats)
            all_frames.append(frame.copy())
            all_lms.append(lms)
            if len(all_feats) % 30 == 0:
                print(f"    {len(all_feats)}/{n_total}...", end='\r')
        cap.release()
        print(f"    {len(all_feats)} frames xong!      ")

        if len(all_feats) < 5:
            print("  LOI: Video qua ngan!"); return None

        # Bước 2: Sliding window inference
        print(f"  [2/2] Inference (sliding window)...")
        frame_preds = []
        prob_buf    = deque(maxlen=cfg.SMOOTH_WINDOW)
        for i in range(len(all_feats)):
            start  = max(0, i - cfg.SEQ_LEN + 1)
            window = all_feats[start:i+1]
            if len(window) < cfg.SEQ_LEN:
                pad    = [window[0]] * (cfg.SEQ_LEN - len(window))
                window = pad + list(window)
            frame_preds.append(
                self._predict_sequence(np.stack(window), prob_buf))

        # Bước 3: Playback
        result_log   = deque(maxlen=20)
        last_confirm = None
        cand         = None
        cand_cnt     = 0
        paused       = False
        cur          = 0
        delay        = max(1, int(1000 / fps_v / self.slowdown))

        while True:
            if not paused and cur < len(all_frames):
                frame     = all_frames[cur].copy()
                top_preds = frame_preds[cur]

                # Consecutive logic
                if top_preds:
                    tl, tc = top_preds[0]
                    if not is_idle_label(tl) and tc >= self.confidence_thr:
                        if tl == cand:
                            cand_cnt += 1
                        else:
                            cand = tl; cand_cnt = 1
                        if cand_cnt >= cfg.CONSEC_THR and tl != last_confirm:
                            result_log.append((f"{cur:04d}", tl, tc))
                            last_confirm = tl; cand_cnt = 0
                            mark = ("✓" if ground_truth and tl == ground_truth
                                    else "✗" if ground_truth else "")
                            print(f"  [Frame {cur:04d}] {mark} "
                                  f"{get_display_name(tl)} ({tc*100:.1f}%)")
                    else:
                        cand = None; cand_cnt = 0
                        if top_preds and not is_idle_label(top_preds[0][0]):
                            last_confirm = None

                frame = self._draw_ui(frame, top_preds, cur,
                                      len(all_frames), fps_v,
                                      all_lms[cur], result_log, ground_truth)
                cv2.imshow('VSL Video Inference', frame)
                cur += 1

                if cur >= len(all_frames):
                    paused = True
                    print("\n  Het video. [R] xem lai, [Q] thoat.")

            key = cv2.waitKey(delay) & 0xFF
            if key in (ord('q'), ord('Q')): break
            elif key == ord(' '):
                paused = not paused
            elif key in (ord('r'), ord('R')):
                cur = 0; result_log.clear(); last_confirm = None
                cand = None; cand_cnt = 0; prob_buf.clear(); paused = False
            elif key in (ord('s'), ord('S')) and cur > 0:
                fn = f"screenshots/vid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(fn, all_frames[cur-1])
                print(f"  Screenshot: {fn}")
            elif key in (ord('+'), ord('=')):
                self.confidence_thr = min(0.99, self.confidence_thr + 0.05)
                print(f"  Threshold: {self.confidence_thr*100:.0f}%")
            elif key == ord('-'):
                self.confidence_thr = max(0.05, self.confidence_thr - 0.05)
                print(f"  Threshold: {self.confidence_thr*100:.0f}%")

        cv2.destroyAllWindows()

        # Tổng kết
        print(f"\n{'='*50}")
        print(f"  KET QUA: {os.path.basename(video_path)}")
        if ground_truth:
            print(f"  Label that: {get_display_name(ground_truth)}")
        if result_log:
            cnt  = Counter([l for _,l,_ in result_log])
            best = cnt.most_common(1)[0]
            print(f"  Du doan: {get_display_name(best[0])} ({best[1]} lan)")
            if ground_truth:
                print("  So sanh: " + ("DUNG ✓" if best[0] == ground_truth else "SAI ✗"))
        else:
            print("  Khong nhan dien duoc (giam threshold?)")
        print(f"{'='*50}\n")
        return result_log

    # ── Run folder ───────────────────────────────────────────

    def run_folder(self, folder_path: str, label_name: str = None):
        exts   = {'.mp4','.avi','.mov','.mkv','.webm'}
        videos = sorted([f for f in os.listdir(folder_path)
                         if os.path.splitext(f)[1].lower() in exts])
        if not videos:
            print(f"  Khong co video trong: {folder_path}"); return

        print(f"\n  Folder: {folder_path} ({len(videos)} videos)")
        correct = 0; results = []
        for i, vf in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] {vf}")
            log = self.run_video(os.path.join(folder_path, vf), label_name)
            if log and label_name:
                cnt  = Counter([l for _,l,_ in log])
                pred = cnt.most_common(1)[0][0] if cnt else None
                ok   = (pred == label_name)
                correct += int(ok); results.append((vf, pred, ok))

        if label_name and results:
            print(f"\n  TONG KET: {correct}/{len(results)} dung "
                  f"({correct/len(results)*100:.1f}%)")

    def close(self):
        self.extractor.close()


def main():
    parser = argparse.ArgumentParser(description='VSL Video Inference')
    parser.add_argument('--video',      type=str,   default=None)
    parser.add_argument('--folder',     type=str,   default=None)
    parser.add_argument('--label',      type=str,   default=None)
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best_model.pt')
    parser.add_argument('--threshold',  type=float,
                        default=cfg.CONFIDENCE_THR)
    parser.add_argument('--top_k',      type=int,   default=cfg.TOP_K)
    parser.add_argument('--slowdown',   type=int,   default=1)
    args = parser.parse_args()

    if not args.video and not args.folder:
        vp = input("\n  Nhap duong dan video (Enter de thoat): ").strip()
        if not vp: return
        args.video = vp
        args.label = input("  Label that (Enter neu khong biet): ").strip() or None

    load_display_names()

    print("\n" + "="*55)
    print(" VSL VIDEO INFERENCE ".center(55, "="))
    print("="*55)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Threshold  : {args.threshold*100:.0f}%")
    print(f"  Slowdown   : {args.slowdown}x")
    print(f"  Device     : {cfg.DEVICE}")

    app = VideoInferenceApp(
        checkpoint_path = args.checkpoint,
        confidence_thr  = args.threshold,
        top_k           = args.top_k,
        slowdown        = args.slowdown,
    )
    try:
        if args.video:
            app.run_video(args.video, ground_truth=args.label)
        elif args.folder:
            app.run_folder(args.folder, label_name=args.label)
    finally:
        app.close()


if __name__ == '__main__':
    main()