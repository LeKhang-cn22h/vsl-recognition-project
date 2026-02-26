"""
collector/interaction.py - Hiển thị vùng tương tác tay ↔ cơ thể
=================================================================
    from collector.interaction import InteractionVisualizer

    frame, interactions = InteractionVisualizer.draw(
        frame, pose_lms, face_lms, left_hand_lms, right_hand_lms, w, h)
"""

import cv2
import numpy as np


class InteractionVisualizer:
    """Vẽ vùng cơ thể và highlight khi tay chạm/gần."""

    TOUCH_THR = 0.08   # khoảng cách normalized = chạm
    NEAR_THR  = 0.15   # khoảng cách normalized = gần

    # Màu cho từng vùng
    REGION_COLORS = {
        'Dau':      (255, 200, 0),
        'Tran':     (255, 220, 100),
        'Cam':      (255, 180, 50),
        'Ma trai':  (220, 170, 255),
        'Ma phai':  (220, 170, 255),
        'Tai trai': (255, 150, 0),
        'Tai phai': (255, 150, 0),
        'Nguc':     (0, 200, 255),
        'Tim':      (0, 0, 255),
        'Bung':     (0, 255, 200),
        'Vai trai': (200, 200, 0),
        'Vai phai': (200, 200, 0),
    }

    @staticmethod
    def _build_regions(pose_lms, face_lms) -> dict:
        """Tính tọa độ normalized của các vùng cơ thể."""
        regions = {}

        # Từ pose
        regions['Dau'] = (pose_lms[0].x, pose_lms[0].y)

        for idx, name in [(7, 'Tai trai'), (8, 'Tai phai')]:
            if pose_lms[idx].visibility > 0.5:
                regions[name] = (pose_lms[idx].x, pose_lms[idx].y)

        ls, rs = pose_lms[11], pose_lms[12]
        if ls.visibility > 0.5 and rs.visibility > 0.5:
            cx = (ls.x + rs.x) / 2
            cy = (ls.y + rs.y) / 2
            regions['Nguc'] = (cx, cy)
            regions['Tim']  = ((rs.x + cx) / 2, (rs.y + cy) / 2)

        if all(pose_lms[i].visibility > 0.5 for i in [11, 12, 23, 24]):
            regions['Bung'] = (
                sum(pose_lms[i].x for i in [11,12,23,24]) / 4,
                sum(pose_lms[i].y for i in [11,12,23,24]) / 4,
            )

        for idx, name in [(11, 'Vai trai'), (12, 'Vai phai')]:
            if pose_lms[idx].visibility > 0.5:
                regions[name] = (pose_lms[idx].x, pose_lms[idx].y)

        # Từ face landmarks (chính xác hơn pose cho vùng mặt)
        if face_lms is not None and len(face_lms) >= 468:
            regions['Tran']    = (face_lms[10].x,  face_lms[10].y)
            regions['Cam']     = (face_lms[152].x, face_lms[152].y)
            regions['Ma trai'] = (face_lms[50].x,  face_lms[50].y)
            regions['Ma phai'] = (face_lms[280].x, face_lms[280].y)

        return regions

    @classmethod
    def draw(cls, frame, pose_lms, face_lms,
             left_hand_lms, right_hand_lms, w, h):
        """
        Vẽ vùng cơ thể + đường kết nối tay↔vùng.
        Trả về (frame, list[str]) — danh sách tương tác đang xảy ra.
        """
        if pose_lms is None:
            return frame, []

        regions      = cls._build_regions(pose_lms, face_lms)
        interactions = []

        # Vẽ markers vùng cơ thể
        for name, (rx, ry) in regions.items():
            px, py = int(rx * w), int(ry * h)
            c = cls.REGION_COLORS.get(name, (200, 200, 200))
            cv2.circle(frame, (px, py), 8, c, 2)
            cv2.putText(frame, name, (px+10, py-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

        # Kiểm tra tương tác từng tay
        hand_list = []
        if right_hand_lms: hand_list.append(('T.Phai', right_hand_lms[0]))
        if left_hand_lms:  hand_list.append(('T.Trai', left_hand_lms[0]))

        for hname, wrist in hand_list:
            hx, hy = wrist.x, wrist.y
            hp = (int(hx * w), int(hy * h))

            for rname, (rx, ry) in regions.items():
                dist = np.sqrt((hx - rx)**2 + (hy - ry)**2)
                rp   = (int(rx * w), int(ry * h))

                if dist < cls.TOUCH_THR:
                    # Chạm — highlight đỏ
                    cv2.line(frame, hp, rp, (0, 0, 255), 3)
                    cv2.circle(frame, rp, 15, (0, 0, 255), 3)
                    overlay = frame.copy()
                    cv2.circle(overlay, rp, 25, (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                    interactions.append(f"CHAM: {hname}->{rname}")

                elif dist < cls.NEAR_THR:
                    # Gần — đường đứt nét vàng
                    for t in range(0, 100, 10):
                        r1 = t / 100.0
                        r2 = (t + 5) / 100.0
                        p1 = (int(hp[0] + r1*(rp[0]-hp[0])),
                              int(hp[1] + r1*(rp[1]-hp[1])))
                        p2 = (int(hp[0] + r2*(rp[0]-hp[0])),
                              int(hp[1] + r2*(rp[1]-hp[1])))
                        cv2.line(frame, p1, p2, (0, 255, 255), 1)

        return frame, interactions  