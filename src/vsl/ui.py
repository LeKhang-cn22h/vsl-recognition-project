"""
vsl/ui.py - Vẽ overlay UI lên frame OpenCV
==========================================
    from vsl.ui import UIRenderer
    UIRenderer.draw_header(frame, w, h, fps, paused)
"""

import time
import cv2
import numpy as np

from vsl.config import cfg
from vsl.utils  import get_display_name


class UIRenderer:
    """Tất cả hàm vẽ UI đều là @staticmethod / @classmethod để dễ dùng."""

    # ── Palette ──
    BG_DARK  = (18,  18,  28)
    ACCENT   = (0,  220, 160)   # xanh mint
    ACCENT2  = (60, 160, 255)   # xanh dương
    WHITE    = (240, 240, 240)
    GRAY     = (140, 140, 150)
    RED      = (60,  60,  220)
    YELLOW   = (0,  200, 230)
    FONT     = cv2.FONT_HERSHEY_SIMPLEX

    # ── Primitives ────────────────────────────────────────────

    @staticmethod
    def rect_alpha(frame, x1, y1, x2, y2, color, alpha=0.75):
        """Vẽ hình chữ nhật bán trong suốt."""
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

    @staticmethod
    def progress_bar(frame, x, y, w, h, value,
                     color, bg=(50, 50, 60)):
        """Thanh tiến trình [0..1]."""
        cv2.rectangle(frame, (x, y), (x+w, y+h), bg, -1)
        fw = max(0, int(w * min(value, 1.0)))
        if fw > 0:
            cv2.rectangle(frame, (x, y), (x+fw, y+h), color, -1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (80, 80, 90), 1)

    # ── Header ────────────────────────────────────────────────

    @classmethod
    def draw_header(cls, frame, w, h, fps, paused):
        cls.rect_alpha(frame, 0, 0, w, 56, cls.BG_DARK, alpha=0.88)
        cv2.putText(frame, "VSL  REALTIME  RECOGNITION",
                    (16, 36), cls.FONT, 0.75, cls.ACCENT, 2)
        fps_col = (cls.ACCENT if fps >= 20
                   else cls.YELLOW if fps >= 12 else cls.RED)
        cv2.putText(frame, f"FPS {fps:4.1f}",
                    (w-140, 36), cls.FONT, 0.6, fps_col, 2)
        if paused:
            cv2.putText(frame, "[ PAUSED ]",
                        (w//2-60, 36), cls.FONT, 0.65, cls.YELLOW, 2)

    # ── Prediction panel ──────────────────────────────────────

    @classmethod
    def draw_prediction_panel(cls, frame, w, h,
                               top_preds, history, confidence_thr):
        px, py, pw, ph = 12, 68, 320, 220
        cls.rect_alpha(frame, px, py, px+pw, py+ph,
                       cls.BG_DARK, alpha=0.82)
        cv2.putText(frame, "PREDICTION",
                    (px+12, py+26), cls.FONT, 0.55, cls.ACCENT2, 1)
        cv2.line(frame, (px+8, py+32), (px+pw-8, py+32), (60,60,80), 1)

        if not top_preds:
            cv2.putText(frame, "No detection",
                        (px+12, py+70), cls.FONT, 0.5, cls.GRAY, 1)
            return

        top_label, top_prob = top_preds[0]
        label_col = cls.ACCENT if top_prob >= confidence_thr else cls.YELLOW
        cv2.putText(frame, get_display_name(top_label).upper(),
                    (px+12, py+68), cls.FONT, 0.85, label_col, 2)
        cls.progress_bar(frame, px+12, py+76, pw-24, 14,
                         top_prob, label_col)
        cv2.putText(frame, f"{top_prob*100:.1f}%",
                    (px+pw-60, py+88), cls.FONT, 0.45, label_col, 1)
        # Vạch threshold
        tx = px + 12 + int((pw-24) * confidence_thr)
        cv2.line(frame, (tx, py+76), (tx, py+90), (100,100,220), 1)

        for i, (lbl, prob) in enumerate(top_preds[1:], 1):
            ry = py + 100 + (i-1)*34
            if ry + 30 > py + ph:
                break
            cv2.putText(frame, f"{i+1}. {get_display_name(lbl)}",
                        (px+12, ry+14), cls.FONT, 0.48, cls.WHITE, 1)
            cls.progress_bar(frame, px+12, ry+18, pw-80, 9,
                             prob, cls.GRAY)
            cv2.putText(frame, f"{prob*100:.1f}%",
                        (px+pw-66, ry+26), cls.FONT, 0.4, cls.GRAY, 1)

    # ── History panel ─────────────────────────────────────────

    @classmethod
    def draw_history_panel(cls, frame, w, h, history):
        px, py, pw, ph = 12, h-220, 320, 200
        cls.rect_alpha(frame, px, py, px+pw, py+ph,
                       cls.BG_DARK, alpha=0.82)
        cv2.putText(frame, "HISTORY",
                    (px+12, py+22), cls.FONT, 0.5, cls.ACCENT2, 1)
        cv2.line(frame, (px+8, py+28), (px+pw-8, py+28), (60,60,80), 1)

        for i, (ts_str, label, conf) in enumerate(
                list(history)[-8:][::-1]):
            ty = py + 48 + i * 19
            if ty > py + ph - 10:
                break
            age = max(100, 220 - i*20)
            cv2.putText(
                frame,
                f"{ts_str}  {get_display_name(label):<18} {conf*100:4.0f}%",
                (px+12, ty), cls.FONT, 0.38, (age, age, age), 1)

    # ── Skeleton ──────────────────────────────────────────────

    @classmethod
    def draw_skeleton(cls, frame, latest: dict, w, h):
        pose  = latest.get('pose')
        hands = latest.get('hands') or (None, None)

        POSE_CONN = [(11,13),(13,15),(12,14),(14,16),
                     (11,12),(11,23),(12,24)]
        if pose:
            for i, j in POSE_CONN:
                if (i < len(pose) and j < len(pose) and
                        pose[i].visibility > 0.5 and
                        pose[j].visibility > 0.5):
                    cv2.line(frame,
                             (int(pose[i].x*w), int(pose[i].y*h)),
                             (int(pose[j].x*w), int(pose[j].y*h)),
                             (0, 180, 120), 1)
            for idx in [0, 11, 12, 13, 14, 15, 16]:
                if idx < len(pose) and pose[idx].visibility > 0.5:
                    cv2.circle(frame,
                               (int(pose[idx].x*w), int(pose[idx].y*h)),
                               4, cls.ACCENT, -1)

        HAND_CONN = [(0,1),(1,2),(2,3),(3,4),
                     (0,5),(5,6),(6,7),(7,8),
                     (0,9),(9,10),(10,11),(11,12),
                     (0,13),(13,14),(14,15),(15,16),
                     (0,17),(17,18),(18,19),(19,20),
                     (5,9),(9,13),(13,17)]
        for hlms, hc in zip(hands, [(0,255,140),(60,160,255)]):
            if hlms is None: continue
            for i, j in HAND_CONN:
                cv2.line(frame,
                         (int(hlms[i].x*w), int(hlms[i].y*h)),
                         (int(hlms[j].x*w), int(hlms[j].y*h)), hc, 1)
            for lm in hlms:
                cv2.circle(frame,
                           (int(lm.x*w), int(lm.y*h)), 2, hc, -1)

    # ── Buffer bar ────────────────────────────────────────────

    @classmethod
    def draw_buffer_bar(cls, frame, w, h, buf_len, seq_len):
        bx, by, bw, bh = 12, h-28, w-24, 8
        ratio = buf_len / seq_len
        cls.rect_alpha(frame, bx-2, by-2, bx+bw+2, by+bh+2,
                       cls.BG_DARK, alpha=0.7)
        col = cls.ACCENT if ratio >= 1.0 else cls.ACCENT2
        cls.progress_bar(frame, bx, by, bw, bh, ratio, col)
        label = "READY" if ratio >= 1.0 else f"Buffer {buf_len}/{seq_len}"
        cv2.putText(frame, label,
                    (bx + bw//2 - 50, by-4), cls.FONT, 0.38, col, 1)

    # ── Footer ────────────────────────────────────────────────

    @classmethod
    def draw_footer(cls, frame, w, h, confidence_thr):
        cls.rect_alpha(frame, 0, h-18, w, h, cls.BG_DARK, alpha=0.88)
        hints = (f"[Q] Thoat   [SPACE] Pause   [C] Clear   "
                 f"[S] Screenshot   [+/-] Threshold: "
                 f"{confidence_thr*100:.0f}%")
        cv2.putText(frame, hints, (10, h-4),
                    cls.FONT, 0.38, cls.GRAY, 1)

    # ── Recording dot ─────────────────────────────────────────

    @classmethod
    def draw_status_dot(cls, frame, w, h, recording):
        if recording and int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (w-22, 28), 7, cls.RED, -1)