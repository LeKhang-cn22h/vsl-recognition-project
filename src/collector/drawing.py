"""
Vẽ keypoints lên frame
"""

import cv2
import numpy as np


# hàm tiện ích
def lm_to_px(lm, w, h):
    """NormalizedLandmark → pixel (x, y)"""
    return int(lm.x * w), int(lm.y * h)


def lm_dist(a, b):
    """Khoảng cách normalized giữa 2 landmark"""
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


def draw_text_bg(frame, text, pos, scale=0.6,
                 color=(255, 255, 255), bg=(0, 0, 0), thick=1, pad=5):
    """Vẽ text có nền mờ."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = pos
    cv2.rectangle(frame, (x-pad, y-th-pad), (x+tw+pad, y+bl+pad), bg, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thick)


# ── FullBodyDrawer ─────────────────────────────────────────

class FullBodyDrawer:
    """Vẽ Pose, Face Mesh, Hands lên frame."""

    POSE_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,7), (0,4),(4,5),(5,6),(6,8), (9,10),
        (11,12), (11,13),(13,15), (12,14),(14,16),
        (11,23),(12,24),(23,24),
        (15,17),(15,19),(15,21), (16,18),(16,20),(16,22),
    ]

    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]

    POSE_COLORS = {
        **{i: (255,200,0)   for i in range(7)},
        **{i: (255,100,0)   for i in [7, 8]},
        **{i: (255,0,100)   for i in [9, 10]},
        **{i: (0,255,0)     for i in [11, 12]},
        **{i: (0,200,255)   for i in [13, 14]},
        **{i: (0,100,255)   for i in [15, 16]},
        **{i: (200,200,200) for i in range(17, 23)},
        **{i: (200,0,200)   for i in [23, 24]},
    }

    @staticmethod
    def draw_pose(frame, pose_lms, w, h):
        if pose_lms is None:
            return frame
        for i, j in FullBodyDrawer.POSE_CONNECTIONS:
            if (i < len(pose_lms) and j < len(pose_lms) and
                    pose_lms[i].visibility > 0.5 and
                    pose_lms[j].visibility > 0.5):
                cv2.line(frame, lm_to_px(pose_lms[i], w, h),
                         lm_to_px(pose_lms[j], w, h), (0, 200, 200), 2)
        for idx in range(min(25, len(pose_lms))):
            if pose_lms[idx].visibility > 0.5:
                px, py = lm_to_px(pose_lms[idx], w, h)
                c = FullBodyDrawer.POSE_COLORS.get(idx, (200, 200, 200))
                cv2.circle(frame, (px, py), 5, c, -1)
                cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)
        for idx, name in [(7, "Tai T"), (8, "Tai P")]:
            if idx < len(pose_lms) and pose_lms[idx].visibility > 0.5:
                px, py = lm_to_px(pose_lms[idx], w, h)
                cv2.putText(frame, name, (px+8, py-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,150,0), 1)
        return frame

    @staticmethod
    def draw_face_mesh(frame, face_lms, w, h):
        if face_lms is None or len(face_lms) < 468:
            return frame

        oval = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,
                379,378,400,377,152,148,176,149,150,136,172,58,132,93,
                234,127,162,21,54,103,67,109,10]
        for k in range(len(oval)-1):
            cv2.line(frame, lm_to_px(face_lms[oval[k]], w, h),
                     lm_to_px(face_lms[oval[k+1]], w, h), (180,180,180), 1)

        le = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33]
        for k in range(len(le)-1):
            cv2.line(frame, lm_to_px(face_lms[le[k]], w, h),
                     lm_to_px(face_lms[le[k+1]], w, h), (0,255,0), 1)

        re = [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466,263]
        for k in range(len(re)-1):
            cv2.line(frame, lm_to_px(face_lms[re[k]], w, h),
                     lm_to_px(face_lms[re[k+1]], w, h), (0,255,0), 1)

        for brow in [[70,63,105,66,107], [300,293,334,296,336]]:
            for k in range(len(brow)-1):
                cv2.line(frame, lm_to_px(face_lms[brow[k]], w, h),
                         lm_to_px(face_lms[brow[k+1]], w, h), (0,200,255), 2)

        lips = [61,146,91,181,84,17,314,405,321,375,291,
                409,270,269,267,0,37,39,40,185,61]
        for k in range(len(lips)-1):
            cv2.line(frame, lm_to_px(face_lms[lips[k]], w, h),
                     lm_to_px(face_lms[lips[k+1]], w, h), (0,0,255), 1)

        nose = [168,6,197,195,5,4,1]
        for k in range(len(nose)-1):
            cv2.line(frame, lm_to_px(face_lms[nose[k]], w, h),
                     lm_to_px(face_lms[nose[k+1]], w, h), (200,200,0), 1)

        return frame

    @staticmethod
    def draw_hand(frame, hand_lms, w, h, label='R'):
        if hand_lms is None:
            return frame
        color = (0, 255, 100) if label == 'R' else (255, 100, 0)
        for i, j in FullBodyDrawer.HAND_CONNECTIONS:
            cv2.line(frame, lm_to_px(hand_lms[i], w, h),
                     lm_to_px(hand_lms[j], w, h), color, 2)
        for idx, lm in enumerate(hand_lms):
            px, py = lm_to_px(lm, w, h)
            r = 6 if idx in [4, 8, 12, 16, 20] else 3
            cv2.circle(frame, (px, py), r, color, -1)
            if idx in [4, 8, 12, 16, 20]:
                cv2.circle(frame, (px, py), r+2, (255,255,255), 1)
        wx, wy = lm_to_px(hand_lms[0], w, h)
        cv2.putText(frame, "Tay P" if label == 'R' else "Tay T",
                    (wx-15, wy+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame