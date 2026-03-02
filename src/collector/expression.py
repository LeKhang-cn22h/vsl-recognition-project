"""
collector/expression.py - Phân tích biểu cảm khuôn mặt
========================================================
Ưu tiên dùng Blendshapes (chính xác hơn), fallback sang landmark distances.

    from collector.expression import FacialExpressionAnalyzer

    # Cách dùng:
    expr = FacialExpressionAnalyzer.analyze_blendshapes(blendshapes)  # ưu tiên
    expr = FacialExpressionAnalyzer.analyze_landmarks(face_lms, w, h) # fallback
"""

import numpy as np


class FacialExpressionAnalyzer:
    """
    Phân tích biểu cảm bằng 52 ARKit Blendshapes (ưu tiên)
    hoặc khoảng cách landmark (fallback khi không có blendshapes).

    Các blendshape quan trọng nhất cho VSL:
      - browInnerUp / browDownLeft/Right  → nhíu mày vs nhướn mày
      - eyeBlinkLeft/Right                → nhắm mắt
      - eyeWideLeft/Right                 → mở to mắt (ngạc nhiên)
      - jawOpen                           → há miệng
      - mouthSmileLeft/Right              → cười
      - mouthFrownLeft/Right              → mím/bĩu môi
      - mouthPucker                       → chúm môi
      - cheekPuff                         → phồng má
    """

    @staticmethod
    def analyze_blendshapes(blendshapes) -> dict | None:
        """
        Phân tích từ blendshape scores (52 scores từ FaceLandmarker).
        blendshapes: list[Category] — .category_name và .score
        """
        if not blendshapes:
            return None

        bs = {c.category_name: c.score for c in blendshapes}

        jaw_open     = bs.get('jawOpen', 0)
        smile        = (bs.get('mouthSmileLeft', 0) + bs.get('mouthSmileRight', 0)) / 2
        frown        = (bs.get('mouthFrownLeft', 0) + bs.get('mouthFrownRight', 0)) / 2
        blink_l      = bs.get('eyeBlinkLeft', 0)
        blink_r      = bs.get('eyeBlinkRight', 0)
        eye_wide     = (bs.get('eyeWideLeft', 0) + bs.get('eyeWideRight', 0)) / 2
        eye_squint   = (bs.get('eyeSquintLeft', 0) + bs.get('eyeSquintRight', 0)) / 2
        brow_inner   = bs.get('browInnerUp', 0)
        brow_down    = (bs.get('browDownLeft', 0) + bs.get('browDownRight', 0)) / 2
        nose_sneer   = (bs.get('noseSneerLeft', 0) + bs.get('noseSneerRight', 0)) / 2
        cheek_puff   = bs.get('cheekPuff', 0)
        pucker       = bs.get('mouthPucker', 0)

        # Xác định biểu cảm chính
        label = "Binh thuong"
        if blink_l > 0.6 and blink_r > 0.6:
            label = "Nham mat"
        elif smile > 0.4 and brow_inner > 0.2:
            label = "Vui / Cuoi"
        elif smile > 0.3:
            label = "Cuoi nhe"
        elif frown > 0.3 and brow_down > 0.3:
            label = "Buon / Khong vui"
        elif brow_inner > 0.4 and eye_wide > 0.3:
            label = "Ngac nhien"
        elif brow_down > 0.4 and eye_squint > 0.3:
            label = "Nhiu may / Gian"
        elif brow_down > 0.3 and nose_sneer > 0.3:
            label = "Kho chiu"
        elif jaw_open > 0.5:
            label = "Mieng mo to"
        elif pucker > 0.4:
            label = "Chum moi"
        elif cheek_puff > 0.4:
            label = "Phong ma"
        elif brow_inner > 0.35:
            label = "Nhuon may"

        return dict(
            mouth_open=round(jaw_open, 2),
            mouth_smile=round(smile, 2),
            mouth_frown=round(frown, 2),
            left_eye_open=round(1.0 - blink_l, 2),
            right_eye_open=round(1.0 - blink_r, 2),
            brow_up=round(brow_inner, 2),
            brow_down=round(brow_down, 2),
            eye_wide=round(eye_wide, 2),
            eye_squint=round(eye_squint, 2),
            nose_sneer=round(nose_sneer, 2),
            cheek_puff=round(cheek_puff, 2),
            pucker=round(pucker, 2),
            expression_label=label,
            source='blendshapes',
        )

    @staticmethod
    def analyze_landmarks(face_lms, w, h) -> dict | None:
        """Fallback: tính từ khoảng cách landmark khi không có blendshapes."""
        if face_lms is None or len(face_lms) < 468:
            return None

        def d(i, j):
            return np.sqrt((face_lms[i].x - face_lms[j].x)**2 +
                           (face_lms[i].y - face_lms[j].y)**2)

        eye_ref = max(d(133, 362), 1e-6)

        mouth_open  = float(np.clip(d(13, 14) / eye_ref * 3.0, 0, 1))
        mouth_smile = float(np.clip(d(291, 61) / eye_ref * 1.5 - 0.8, 0, 1))
        l_eye       = float(np.clip(d(159, 145) / eye_ref * 6.0, 0, 1))
        r_eye       = float(np.clip(d(386, 374) / eye_ref * 6.0, 0, 1))
        brow_val    = ((face_lms[159].y - face_lms[107].y) +
                       (face_lms[386].y - face_lms[336].y)) / 2 / eye_ref * 5.0

        label = "Binh thuong"
        if l_eye < 0.2 and r_eye < 0.2:    label = "Nham mat"
        elif mouth_smile > 0.5 and brow_val > 0.3: label = "Vui / Cuoi"
        elif brow_val < -0.1:               label = "Nhiu may"
        elif brow_val > 0.5:                label = "Ngac nhien"
        elif mouth_open > 0.4:              label = "Mieng mo"

        return dict(
            mouth_open=round(mouth_open, 2),
            mouth_smile=round(mouth_smile, 2),
            mouth_frown=0.0,
            left_eye_open=round(l_eye, 2),
            right_eye_open=round(r_eye, 2),
            brow_up=round(max(brow_val, 0), 2),
            brow_down=round(max(-brow_val, 0), 2),
            eye_wide=0.0, eye_squint=0.0,
            nose_sneer=0.0, cheek_puff=0.0, pucker=0.0,
            expression_label=label,
            source='landmarks',
        )