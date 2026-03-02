"""
collector/framing.py - Kiểm tra góc quay
=========================================
    from collector.framing import FramingChecker
"""


class FramingChecker:
    """Kiểm tra người quay nằm đúng khung hình."""

    MARGIN = 0.03  # 3% mép

    @staticmethod
    def check(pose_lms, face_lms, hand_results, w, h):
        """
        Trả về dict {ok, warnings, details}

        pose_lms     : list[NormalizedLandmark] (33) hoặc None
        face_lms     : list[NormalizedLandmark] (478) hoặc None
        hand_results : (left_hand_lms, right_hand_lms)
        """
        left_hand_lms, right_hand_lms = hand_results
        warnings = []
        details = dict(
            face_visible=True, upper_body_visible=True,
            left_arm_visible=True, right_arm_visible=True,
            left_hand_visible=True, right_hand_visible=True,
            too_close=False, too_far=False,
        )

        mx = w * FramingChecker.MARGIN
        my = h * FramingChecker.MARGIN

        def in_frame(lm):
            return mx < lm.x * w < w - mx and my < lm.y * h < h - my

        def vis_ok(lm, thr=0.5):
            return hasattr(lm, 'visibility') and lm.visibility > thr

        # Không thấy người
        if pose_lms is None:
            details.update(face_visible=False, upper_body_visible=False,
                           left_arm_visible=False, right_arm_visible=False)
            return dict(ok=False,
                        warnings=['KHONG THAY NGUOI - Hay vao khung hinh!'],
                        details=details)

        # 1. Khuôn mặt
        if sum(1 for i in [0, 7, 8]
               if vis_ok(pose_lms[i]) and in_frame(pose_lms[i])) < 2:
            details['face_visible'] = False
            warnings.append('KHONG THAY KHUON MAT - Dieu chinh camera!')

        # 2. Thân trên
        if sum(1 for i in [11, 12, 23, 24]
               if vis_ok(pose_lms[i]) and in_frame(pose_lms[i])) < 3:
            details['upper_body_visible'] = False
            warnings.append('THAN TREN BI CAT - Lui ra xa hon!')

        # 3. Cánh tay trái
        if sum(1 for i in [11, 13, 15]
               if vis_ok(pose_lms[i]) and in_frame(pose_lms[i])) < 2:
            details['left_arm_visible'] = False
            warnings.append('TAY TRAI BI CAT - Dua tay vao khung hinh!')

        # 4. Cánh tay phải
        if sum(1 for i in [12, 14, 16]
               if vis_ok(pose_lms[i]) and in_frame(pose_lms[i])) < 2:
            details['right_arm_visible'] = False
            warnings.append('TAY PHAI BI CAT - Dua tay vao khung hinh!')

        # 5. Bàn tay
        details['left_hand_visible']  = left_hand_lms is not None
        details['right_hand_visible'] = right_hand_lms is not None
        if left_hand_lms is None and right_hand_lms is None:
            warnings.append('KHONG THAY BAN TAY - Dua tay len!')

        # 6. Khoảng cách camera
        if vis_ok(pose_lms[11]) and vis_ok(pose_lms[12]):
            sw = abs(pose_lms[11].x - pose_lms[12].x)
            if sw > 0.55:
                details['too_close'] = True
                warnings.append('QUA GAN CAMERA - Lui ra xa!')
            elif sw < 0.15:
                details['too_far'] = True
                warnings.append('QUA XA CAMERA - Lai gan hon!')

        return dict(ok=len(warnings) == 0, warnings=warnings, details=details)