"""
WEBCAM VIDEO COLLECTOR - FULL BODY VERSION (MediaPipe Tasks API)
Thu thập video ngôn ngữ ký hiệu VSL từ webcam

Sử dụng MediaPipe Tasks API mới:
  - PoseLandmarker   (33 keypoints)
  - HandLandmarker   (21 × 2 keypoints)
  - FaceLandmarker   (478 keypoints + blendshapes)

Tính năng:
  - Hiển thị keypoints: Face Mesh + Pose + Hands
  - Cảnh báo góc quay: kiểm tra nửa thân trên, cánh tay, khuôn mặt
  - Hiển thị vùng tương tác: tay↔ngực, tay↔bụng, tay↔đầu, tay↔tai
  - Hiển thị biểu cảm khuôn mặt (miệng, mắt, lông mày)
  - Quay/Dừng linh hoạt, lưu video .mp4 (frame gốc sạch)
  - Tự động upload lên HuggingFace sau mỗi video
  - Tự động lưu metadata


Cấu hình HuggingFace (tạo file .env):
    HF_TOKEN=hf_xxxxxxxxxxxxxx
    HF_REPO_ID=KhangCN/Video_VSL
    HF_SIGNER_ID=khang

Model files sẽ được tự động tải về lần chạy đầu tiên.
"""

import cv2
import numpy as np
import os
import json
import time
import urllib.request
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from dotenv import load_dotenv
from huggingface_hub import HfApi

# ═══════════════════════════════════════════════════════════
# HUGGING FACE CONFIG
# ═══════════════════════════════════════════════════════════

load_dotenv()
_hf_token = os.getenv("HF_TOKEN")
hf_api = HfApi(token=_hf_token) if _hf_token else None

HF_REPO_ID = os.getenv("HF_REPO_ID", "KhangCN/Video_VSL")

if hf_api:
    print(f"  HuggingFace: OK (repo={HF_REPO_ID}")
else:
    print("  HuggingFace: KHONG CO TOKEN - Video chi luu local")
    print("  (Tao file .env voi HF_TOKEN=hf_xxx de bat upload)")


def upload_to_hf(local_path, label_name):
    """Upload 1 video len HuggingFace. Tra ve True neu thanh cong."""
    if hf_api is None:
        return False
    try:
        filename = os.path.basename(local_path)
        remote_path = f"videos/{label_name}/{filename}"
        hf_api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
        )
        print(f"  HF Upload: {remote_path} OK")
        return True
    except Exception as e:
        print(f"  HF Upload LOI: {e}")
        return False


# ═══════════════════════════════════════════════════════════
# TẢI MODEL
# ═══════════════════════════════════════════════════════════

MODEL_URLS = {
    'hand_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    ),
    'pose_landmarker_heavy.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
    ),
    'face_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    ),
}


def download_model(filename):
    """Tải model nếu chưa có"""
    if os.path.exists(filename):
        return filename
    url = MODEL_URLS[filename]
    print(f"  Dang tai {filename} ...")
    urllib.request.urlretrieve(url, filename)
    print(f"  Da tai xong {filename}")
    return filename


# ═══════════════════════════════════════════════════════════
# CÁC HÀM TIỆN ÍCH
# ═══════════════════════════════════════════════════════════

def lm_to_px(lm, w, h):
    """NormalizedLandmark → pixel (x, y)"""
    return int(lm.x * w), int(lm.y * h)


def lm_dist(a, b):
    """Khoảng cách normalized giữa 2 landmark"""
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


def draw_text_bg(frame, text, pos, scale=0.6, color=(255, 255, 255),
                 bg=(0, 0, 0), thick=1, pad=5):
    """Vẽ text có nền"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = pos
    cv2.rectangle(frame, (x - pad, y - th - pad),
                  (x + tw + pad, y + bl + pad), bg, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thick)


# ═══════════════════════════════════════════════════════════
# FramingChecker – Kiểm tra góc quay
# ═══════════════════════════════════════════════════════════

class FramingChecker:
    """Kiểm tra người quay nằm đúng khung hình"""

    MARGIN = 0.03  # 3 % mép

    @staticmethod
    def check(pose_lms, face_lms, hand_results, w, h):
        """
        pose_lms  : list[NormalizedLandmark] (33) hoặc None
        face_lms  : list[NormalizedLandmark] (478) hoặc None
        hand_results: (left_hand_lms, right_hand_lms) – mỗi cái là list hoặc None
        Trả về dict {ok, warnings, details}
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
            px, py = lm.x * w, lm.y * h
            return mx < px < w - mx and my < py < h - my

        def vis_ok(lm, thr=0.5):
            return hasattr(lm, 'visibility') and lm.visibility > thr

        if pose_lms is None:
            details['face_visible'] = False
            details['upper_body_visible'] = False
            details['left_arm_visible'] = False
            details['right_arm_visible'] = False
            return dict(ok=False,
                        warnings=['KHONG THAY NGUOI - Hay vao khung hinh!'],
                        details=details)

        # 1. Khuôn mặt (pose idx 0=nose, 7=left_ear, 8=right_ear)
        face_cnt = sum(1 for i in [0, 7, 8]
                       if vis_ok(pose_lms[i]) and in_frame(pose_lms[i]))
        if face_cnt < 2:
            details['face_visible'] = False
            warnings.append('KHONG THAY KHUON MAT - Dieu chinh camera!')

        # 2. Thân trên (11,12=shoulders  23,24=hips)
        upper_cnt = sum(1 for i in [11, 12, 23, 24]
                        if vis_ok(pose_lms[i]) and in_frame(pose_lms[i]))
        if upper_cnt < 3:
            details['upper_body_visible'] = False
            warnings.append('THAN TREN BI CAT - Lui ra xa hon!')

        # 3. Cánh tay trái (11,13,15)
        la = sum(1 for i in [11, 13, 15]
                 if vis_ok(pose_lms[i]) and in_frame(pose_lms[i]))
        if la < 2:
            details['left_arm_visible'] = False
            warnings.append('TAY TRAI BI CAT - Dua tay vao khung hinh!')

        # 4. Cánh tay phải (12,14,16)
        ra = sum(1 for i in [12, 14, 16]
                 if vis_ok(pose_lms[i]) and in_frame(pose_lms[i]))
        if ra < 2:
            details['right_arm_visible'] = False
            warnings.append('TAY PHAI BI CAT - Dua tay vao khung hinh!')

        # 5. Bàn tay
        if left_hand_lms is None:
            details['left_hand_visible'] = False
        if right_hand_lms is None:
            details['right_hand_visible'] = False
        if left_hand_lms is None and right_hand_lms is None:
            warnings.append('KHONG THAY BAN TAY - Dua tay len!')

        # 6. Quá gần / xa
        if vis_ok(pose_lms[11]) and vis_ok(pose_lms[12]):
            sw = abs(pose_lms[11].x - pose_lms[12].x)
            if sw > 0.55:
                details['too_close'] = True
                warnings.append('QUA GAN CAMERA - Lui ra xa!')
            elif sw < 0.15:
                details['too_far'] = True
                warnings.append('QUA XA CAMERA - Lai gan hon!')

        return dict(ok=len(warnings) == 0, warnings=warnings, details=details)


# ═══════════════════════════════════════════════════════════
# FacialExpressionAnalyzer  (Blendshapes + Landmark fallback)
# ═══════════════════════════════════════════════════════════
#
# FaceLandmarker với output_face_blendshapes=True trả về 52 blendshape
# scores (0.0 → 1.0). Đây là cách CHÍNH XÁC NHẤT để bắt cảm xúc vì
# Google đã train model riêng cho việc này, thay vì tự tính khoảng cách
# giữa các landmark (dễ bị sai do góc mặt, khoảng cách camera ...).
#
# 52 Blendshapes gồm (tên theo ARKit convention):
#   browDownLeft, browDownRight, browInnerUp, browOuterUpLeft, browOuterUpRight,
#   cheekPuff, cheekSquintLeft, cheekSquintRight,
#   eyeBlinkLeft, eyeBlinkRight, eyeLookDownLeft, eyeLookDownRight,
#   eyeLookInLeft, eyeLookInRight, eyeLookOutLeft, eyeLookOutRight,
#   eyeLookUpLeft, eyeLookUpRight, eyeSquintLeft, eyeSquintRight,
#   eyeWideLeft, eyeWideRight,
#   jawForward, jawLeft, jawOpen, jawRight,
#   mouthClose, mouthDimpleLeft, mouthDimpleRight, mouthFrownLeft,
#   mouthFrownRight, mouthFunnel, mouthLeft, mouthLowerDownLeft,
#   mouthLowerDownRight, mouthPressLeft, mouthPressRight, mouthPucker,
#   mouthRight, mouthRollLower, mouthRollUpper, mouthShrugLower,
#   mouthShrugUpper, mouthSmileLeft, mouthSmileRight, mouthStretchLeft,
#   mouthStretchRight, mouthUpperUpLeft, mouthUpperUpRight,
#   noseSneerLeft, noseSneerRight, _neutral
#
# Với VSL, các blendshape quan trọng nhất:
#   - browInnerUp / browDownLeft/Right  → nhíu mày vs nhướn mày
#   - eyeBlinkLeft/Right                → nhắm mắt
#   - eyeWideLeft/Right                 → mở to mắt (ngạc nhiên)
#   - jawOpen                           → há miệng
#   - mouthSmileLeft/Right              → cười
#   - mouthFrownLeft/Right              → mím/bĩu môi (buồn)
#   - mouthPucker                       → chúm môi
#   - cheekPuff                         → phồng má
#

class FacialExpressionAnalyzer:
    """Phân tích biểu cảm bằng Blendshapes (ưu tiên) hoặc landmark distances (fallback)"""

    @staticmethod
    def analyze_blendshapes(blendshapes):
        """
        Phân tích từ blendshape scores (52 scores từ FaceLandmarker).
        blendshapes: list[Category]  — mỗi item có .category_name và .score
        """
        if blendshapes is None or len(blendshapes) == 0:
            return None

        # Chuyển thành dict {name: score}
        bs = {}
        for cat in blendshapes:
            bs[cat.category_name] = cat.score

        # --- Trích xuất giá trị ---
        jaw_open       = bs.get('jawOpen', 0)
        smile_l        = bs.get('mouthSmileLeft', 0)
        smile_r        = bs.get('mouthSmileRight', 0)
        smile          = (smile_l + smile_r) / 2
        frown_l        = bs.get('mouthFrownLeft', 0)
        frown_r        = bs.get('mouthFrownRight', 0)
        frown          = (frown_l + frown_r) / 2
        mouth_pucker   = bs.get('mouthPucker', 0)
        cheek_puff     = bs.get('cheekPuff', 0)

        blink_l        = bs.get('eyeBlinkLeft', 0)
        blink_r        = bs.get('eyeBlinkRight', 0)
        eye_wide_l     = bs.get('eyeWideLeft', 0)
        eye_wide_r     = bs.get('eyeWideRight', 0)
        eye_squint_l   = bs.get('eyeSquintLeft', 0)
        eye_squint_r   = bs.get('eyeSquintRight', 0)

        brow_inner_up  = bs.get('browInnerUp', 0)
        brow_down_l    = bs.get('browDownLeft', 0)
        brow_down_r    = bs.get('browDownRight', 0)
        brow_outer_l   = bs.get('browOuterUpLeft', 0)
        brow_outer_r   = bs.get('browOuterUpRight', 0)

        nose_sneer_l   = bs.get('noseSneerLeft', 0)
        nose_sneer_r   = bs.get('noseSneerRight', 0)

        # --- Tính features tổng hợp ---
        mouth_open     = round(jaw_open, 2)
        mouth_smile    = round(smile, 2)
        mouth_frown    = round(frown, 2)
        left_eye_open  = round(1.0 - blink_l, 2)
        right_eye_open = round(1.0 - blink_r, 2)
        brow_up        = round(brow_inner_up, 2)
        brow_down      = round((brow_down_l + brow_down_r) / 2, 2)
        eye_wide       = round((eye_wide_l + eye_wide_r) / 2, 2)
        eye_squint     = round((eye_squint_l + eye_squint_r) / 2, 2)
        nose_sneer     = round((nose_sneer_l + nose_sneer_r) / 2, 2)
        cheek_puff     = round(cheek_puff, 2)
        pucker         = round(mouth_pucker, 2)

        # --- Xác định biểu cảm chính ---
        label = "Binh thuong"

        if blink_l > 0.6 and blink_r > 0.6:
            label = "Nham mat"
        elif smile > 0.4 and brow_inner_up > 0.2:
            label = "Vui / Cuoi"
        elif smile > 0.3:
            label = "Cuoi nhe"
        elif frown > 0.3 and brow_down > 0.3:
            label = "Buon / Khong vui"
        elif brow_inner_up > 0.4 and eye_wide > 0.3:
            label = "Ngac nhien"
        elif brow_down > 0.4 and eye_squint > 0.3:
            label = "Nhiu may / Gian"
        elif brow_down > 0.3 and nose_sneer > 0.3:
            label = "Kho chiu"
        elif jaw_open > 0.5:
            label = "Mieng mo to"
        elif mouth_pucker > 0.4:
            label = "Chum moi"
        elif cheek_puff > 0.4:
            label = "Phong ma"
        elif brow_inner_up > 0.35:
            label = "Nhuon may"

        return dict(
            mouth_open=mouth_open,
            mouth_smile=mouth_smile,
            mouth_frown=mouth_frown,
            left_eye_open=left_eye_open,
            right_eye_open=right_eye_open,
            brow_up=brow_up,
            brow_down=brow_down,
            eye_wide=eye_wide,
            eye_squint=eye_squint,
            nose_sneer=nose_sneer,
            cheek_puff=cheek_puff,
            pucker=pucker,
            expression_label=label,
            source='blendshapes',
        )

    @staticmethod
    def analyze_landmarks(face_lms, w, h):
        """Fallback: tính từ landmark distances khi không có blendshapes"""
        if face_lms is None or len(face_lms) < 468:
            return None

        def d(i, j):
            return np.sqrt((face_lms[i].x - face_lms[j].x)**2 +
                           (face_lms[i].y - face_lms[j].y)**2)

        eye_ref = d(133, 362)
        if eye_ref < 1e-6:
            eye_ref = 0.1

        mouth_open = np.clip(d(13, 14) / eye_ref * 3.0, 0, 1)
        mouth_smile = np.clip(d(291, 61) / eye_ref * 1.5 - 0.8, 0, 1)
        l_eye = np.clip(d(159, 145) / eye_ref * 6.0, 0, 1)
        r_eye = np.clip(d(386, 374) / eye_ref * 6.0, 0, 1)
        l_brow_h = face_lms[159].y - face_lms[107].y
        r_brow_h = face_lms[386].y - face_lms[336].y
        brow = (l_brow_h + r_brow_h) / 2 / eye_ref * 5.0

        label = "Binh thuong"
        if mouth_open > 0.4:
            label = "Mieng mo"
        if mouth_smile > 0.5 and brow > 0.3:
            label = "Vui / Cuoi"
        elif brow < -0.1:
            label = "Nhiu may"
        elif brow > 0.5:
            label = "Ngac nhien"
        if l_eye < 0.2 and r_eye < 0.2:
            label = "Nham mat"

        return dict(
            mouth_open=round(mouth_open, 2),
            mouth_smile=round(mouth_smile, 2),
            mouth_frown=0.0,
            left_eye_open=round(l_eye, 2),
            right_eye_open=round(r_eye, 2),
            brow_up=round(max(brow, 0), 2),
            brow_down=round(max(-brow, 0), 2),
            eye_wide=0.0,
            eye_squint=0.0,
            nose_sneer=0.0,
            cheek_puff=0.0,
            pucker=0.0,
            expression_label=label,
            source='landmarks',
        )


# ═══════════════════════════════════════════════════════════
# InteractionVisualizer – tay ↔ vùng cơ thể
# ═══════════════════════════════════════════════════════════

class InteractionVisualizer:
    TOUCH_THR = 0.08
    NEAR_THR = 0.15

    @staticmethod
    def draw(frame, pose_lms, face_lms, left_hand_lms, right_hand_lms, w, h):
        if pose_lms is None:
            return frame, []

        interactions = []

        # ── Vùng cơ thể từ POSE (normalized) ──
        regions = {}
        regions['Dau'] = (pose_lms[0].x, pose_lms[0].y)

        for idx, name in [(7, 'Tai trai'), (8, 'Tai phai')]:
            if pose_lms[idx].visibility > 0.5:
                regions[name] = (pose_lms[idx].x, pose_lms[idx].y)

        if pose_lms[11].visibility > 0.5 and pose_lms[12].visibility > 0.5:
            cx = (pose_lms[11].x + pose_lms[12].x) / 2
            cy = (pose_lms[11].y + pose_lms[12].y) / 2
            regions['Nguc'] = (cx, cy)
            # Sau flip: idx 12 = vai trái thực tế = bên tim
            regions['Tim'] = ((pose_lms[12].x + cx) / 2,
                              (pose_lms[12].y + cy) / 2)

        if all(pose_lms[i].visibility > 0.5 for i in [11, 12, 23, 24]):
            bx = sum(pose_lms[i].x for i in [11, 12, 23, 24]) / 4
            by = sum(pose_lms[i].y for i in [11, 12, 23, 24]) / 4
            regions['Bung'] = (bx, by)

        for idx, name in [(11, 'Vai trai'), (12, 'Vai phai')]:
            if pose_lms[idx].visibility > 0.5:
                regions[name] = (pose_lms[idx].x, pose_lms[idx].y)

        # ── Vùng khuôn mặt từ FACE LANDMARKS (chính xác hơn pose) ──
        if face_lms is not None and len(face_lms) >= 468:
            # Trán (index 10 = giữa trán)
            regions['Tran'] = (face_lms[10].x, face_lms[10].y)

            # Cằm (index 152 = đáy cằm)
            regions['Cam'] = (face_lms[152].x, face_lms[152].y)

            # Má trái (index 234 vùng gò má trái – sau flip = má phải màn hình)
            # Má phải (index 454 vùng gò má phải – sau flip = má trái màn hình)
            # Dùng index 50 (gò má trái) và 280 (gò má phải) chính xác hơn
            regions['Ma trai'] = (face_lms[50].x, face_lms[50].y)
            regions['Ma phai'] = (face_lms[280].x, face_lms[280].y)

        # ── Màu vùng ──
        rcol = {
            'Dau': (255, 200, 0),
            'Tran': (255, 220, 100),
            'Cam': (255, 180, 50),
            'Ma trai': (220, 170, 255),
            'Ma phai': (220, 170, 255),
            'Tai trai': (255, 150, 0),
            'Tai phai': (255, 150, 0),
            'Nguc': (0, 200, 255),
            'Tim': (0, 0, 255),
            'Bung': (0, 255, 200),
            'Vai trai': (200, 200, 0),
            'Vai phai': (200, 200, 0),
        }

        for name, (rx, ry) in regions.items():
            px, py = int(rx * w), int(ry * h)
            c = rcol.get(name, (200, 200, 200))
            cv2.circle(frame, (px, py), 8, c, 2)
            cv2.putText(frame, name, (px + 10, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

        # Kiểm tra tương tác
        hands = []
        if right_hand_lms is not None:
            hands.append(('T.Phai', right_hand_lms[0]))
        if left_hand_lms is not None:
            hands.append(('T.Trai', left_hand_lms[0]))

        for hname, wrist in hands:
            hx, hy = wrist.x, wrist.y
            hp = (int(hx * w), int(hy * h))

            for rname, (rx, ry) in regions.items():
                dist = np.sqrt((hx - rx)**2 + (hy - ry)**2)
                rp = (int(rx * w), int(ry * h))

                if dist < InteractionVisualizer.TOUCH_THR:
                    cv2.line(frame, hp, rp, (0, 0, 255), 3)
                    cv2.circle(frame, rp, 15, (0, 0, 255), 3)
                    interactions.append(f"CHAM: {hname}->{rname}")
                    ov = frame.copy()
                    cv2.circle(ov, rp, 25, (0, 0, 255), -1)
                    cv2.addWeighted(ov, 0.2, frame, 0.8, 0, frame)

                elif dist < InteractionVisualizer.NEAR_THR:
                    for t in range(0, 100, 10):
                        r1 = t / 100.0
                        r2 = (t + 5) / 100.0
                        p1 = (int(hp[0] + r1*(rp[0]-hp[0])),
                              int(hp[1] + r1*(rp[1]-hp[1])))
                        p2 = (int(hp[0] + r2*(rp[0]-hp[0])),
                              int(hp[1] + r2*(rp[1]-hp[1])))
                        cv2.line(frame, p1, p2, (0, 255, 255), 1)

        return frame, interactions


# ═══════════════════════════════════════════════════════════
# FullBodyDrawer – Vẽ keypoints
# ═══════════════════════════════════════════════════════════

class FullBodyDrawer:

    POSE_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,7),  (0,4),(4,5),(5,6),(6,8), (9,10),
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

    # Màu theo vùng trên pose
    POSE_COLORS = {
        **{i: (255,200,0)   for i in range(7)},       # mắt, mũi
        **{i: (255,100,0)   for i in [7, 8]},          # tai
        **{i: (255,0,100)   for i in [9, 10]},         # miệng
        **{i: (0,255,0)     for i in [11, 12]},        # vai
        **{i: (0,200,255)   for i in [13, 14]},        # khuỷu tay
        **{i: (0,100,255)   for i in [15, 16]},        # cổ tay
        **{i: (200,200,200) for i in range(17, 23)},   # bàn tay (pose)
        **{i: (200,0,200)   for i in [23, 24]},        # hông
    }

    @staticmethod
    def draw_pose(frame, pose_lms, w, h):
        if pose_lms is None:
            return frame

        # Connections
        for i, j in FullBodyDrawer.POSE_CONNECTIONS:
            if (i < len(pose_lms) and j < len(pose_lms) and
                    pose_lms[i].visibility > 0.5 and pose_lms[j].visibility > 0.5):
                p1 = lm_to_px(pose_lms[i], w, h)
                p2 = lm_to_px(pose_lms[j], w, h)
                cv2.line(frame, p1, p2, (0, 200, 200), 2)

        # Keypoints (upper body 0-24)
        for idx in range(min(25, len(pose_lms))):
            if pose_lms[idx].visibility > 0.5:
                px, py = lm_to_px(pose_lms[idx], w, h)
                c = FullBodyDrawer.POSE_COLORS.get(idx, (200, 200, 200))
                cv2.circle(frame, (px, py), 5, c, -1)
                cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)

        # Labels cho tai
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

        # Viền mặt
        oval = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,
                379,378,400,377,152,148,176,149,150,136,172,58,132,93,
                234,127,162,21,54,103,67,109,10]
        for k in range(len(oval)-1):
            p1 = lm_to_px(face_lms[oval[k]], w, h)
            p2 = lm_to_px(face_lms[oval[k+1]], w, h)
            cv2.line(frame, p1, p2, (180,180,180), 1)

        # Mắt trái
        le = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33]
        for k in range(len(le)-1):
            cv2.line(frame, lm_to_px(face_lms[le[k]],w,h),
                     lm_to_px(face_lms[le[k+1]],w,h), (0,255,0), 1)

        # Mắt phải
        re = [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466,263]
        for k in range(len(re)-1):
            cv2.line(frame, lm_to_px(face_lms[re[k]],w,h),
                     lm_to_px(face_lms[re[k+1]],w,h), (0,255,0), 1)

        # Lông mày
        for brow in [[70,63,105,66,107], [300,293,334,296,336]]:
            for k in range(len(brow)-1):
                cv2.line(frame, lm_to_px(face_lms[brow[k]],w,h),
                         lm_to_px(face_lms[brow[k+1]],w,h), (0,200,255), 2)

        # Môi
        lips = [61,146,91,181,84,17,314,405,321,375,291,
                409,270,269,267,0,37,39,40,185,61]
        for k in range(len(lips)-1):
            cv2.line(frame, lm_to_px(face_lms[lips[k]],w,h),
                     lm_to_px(face_lms[lips[k+1]],w,h), (0,0,255), 1)

        # Mũi
        nose = [168,6,197,195,5,4,1]
        for k in range(len(nose)-1):
            cv2.line(frame, lm_to_px(face_lms[nose[k]],w,h),
                     lm_to_px(face_lms[nose[k+1]],w,h), (200,200,0), 1)

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
        txt = "Tay P" if label == 'R' else "Tay T"
        cv2.putText(frame, txt, (wx-15, wy+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame


# ═══════════════════════════════════════════════════════════
# LỚP CHÍNH: WebcamVideoCollectorFull
# ═══════════════════════════════════════════════════════════

class WebcamVideoCollectorFull:
    def __init__(self, output_dir='data/videos'):
        self.output_dir = output_dir
        self.metadata_path = os.path.join(output_dir, 'metadata.json')
        os.makedirs(output_dir, exist_ok=True)
        self.metadata = self._load_meta()

        print("\n" + "="*60)
        print(" KHOI TAO HE THONG (MediaPipe Tasks API) ".center(60))
        print("="*60)

        # ── Tải models ──
        hand_model = download_model('hand_landmarker.task')
        pose_model = download_model('pose_landmarker_heavy.task')
        face_model = download_model('face_landmarker.task')

        # ── LIVE_STREAM callbacks ──
        # Kết quả mới nhất được lưu vào dict, main loop đọc ra
        self._latest = dict(pose=None, face=None, hands=None, blendshapes=None)
        self._ts = 0  # timestamp tăng dần (ms)

        # --- PoseLandmarker (LIVE_STREAM) ---
        print("  Khoi tao PoseLandmarker ...")
        pose_opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=pose_model),
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._on_pose,
        )
        self.pose_detector = mp_vision.PoseLandmarker.create_from_options(pose_opts)

        # --- HandLandmarker (LIVE_STREAM) ---
        print("  Khoi tao HandLandmarker ...")
        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=hand_model),
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._on_hand,
        )
        self.hand_detector = mp_vision.HandLandmarker.create_from_options(hand_opts)

        # --- FaceLandmarker (LIVE_STREAM) + BLENDSHAPES ---
        print("  Khoi tao FaceLandmarker (+ Blendshapes) ...")
        face_opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=face_model),
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,   # <<< BẬT BLENDSHAPES
            result_callback=self._on_face,
        )
        self.face_detector = mp_vision.FaceLandmarker.create_from_options(face_opts)

        print("  Tat ca detector da san sang!")
        print("="*60 + "\n")

    # ── Callbacks ──
    def _on_pose(self, result, image, timestamp_ms):
        if result.pose_landmarks:
            self._latest['pose'] = result.pose_landmarks[0]
        else:
            self._latest['pose'] = None

    def _on_hand(self, result, image, timestamp_ms):
        left, right = None, None
        if result.hand_landmarks and result.handedness:
            for i, hand_lms in enumerate(result.hand_landmarks):
                # MediaPipe Tasks trả handedness[i][0].category_name
                cat = result.handedness[i][0].category_name
                # Vì camera flip: "Left" thực ra là tay phải người dùng
                if cat == 'Left':
                    right = hand_lms
                else:
                    left = hand_lms
        self._latest['hands'] = (left, right)

    def _on_face(self, result, image, timestamp_ms):
        if result.face_landmarks:
            self._latest['face'] = result.face_landmarks[0]
        else:
            self._latest['face'] = None
        # Lưu blendshapes (52 scores)
        if result.face_blendshapes:
            self._latest['blendshapes'] = result.face_blendshapes[0]
        else:
            self._latest['blendshapes'] = None

    # ── Metadata ──
    def _load_meta(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return dict(labels={}, total_videos=0,
                    created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def _save_meta(self):
        self.metadata['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def show_statistics(self):
        print("\n" + "="*60)
        print(" THONG KE VIDEO DA THU ".center(60))
        print("="*60)
        if not self.metadata['labels']:
            print("\n     Chua co video nao duoc thu thap")
        else:
            total = 0
            print(f"\n{'Nhan':<30} {'So video':<15} {'Duong dan'}")
            print("-"*70)
            for lb, info in sorted(self.metadata['labels'].items()):
                n = info.get('num_videos', 0)
                print(f"  {lb:<28} {n:<15} {info.get('path','')}")
                total += n
            print("-"*70)
            print(f"  {'TONG CONG':<28} {total:<15}")
        print(f"\n Cap nhat lan cuoi: {self.metadata.get('updated_at','N/A')}")
        print("="*60)

    # ── UI helpers ──
    def _draw_warnings(self, frame, fr, w, h):
        if fr['ok']:
            cv2.rectangle(frame, (2,2), (w-2,h-2), (0,255,0), 3)
            draw_text_bg(frame, "GOC QUAY: OK", (10,h-60),
                         scale=0.6, color=(0,255,0), bg=(0,50,0))
        else:
            cv2.rectangle(frame, (2,2), (w-2,h-2), (0,0,255), 4)
            y = h - 60 - (len(fr['warnings'])-1)*30
            for warn in fr['warnings']:
                draw_text_bg(frame, f"! {warn}", (10,y),
                             scale=0.55, color=(0,0,255), bg=(50,0,0))
                y += 30

        det = fr['details']
        items = [('Mat', det['face_visible']),
                 ('Than', det['upper_body_visible']),
                 ('Tay T', det['left_arm_visible']),
                 ('Tay P', det['right_arm_visible']),
                 ('Ban tay T', det['left_hand_visible']),
                 ('Ban tay P', det['right_hand_visible'])]
        x0 = w - 130
        for i, (nm, ok) in enumerate(items):
            c = (0,255,0) if ok else (0,0,255)
            s = "[OK]" if ok else "[X] "
            cv2.putText(frame, f"{s} {nm}", (x0, 70+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        return frame

    def _draw_expression(self, frame, expr, w, h):
        if expr is None:
            return frame
        px, py = 10, 70

        # Label + nguồn dữ liệu
        src = expr.get('source', '?')
        draw_text_bg(frame, f"Bieu cam: {expr['expression_label']} [{src}]",
                     (px, py), scale=0.55, color=(255,255,0), bg=(40,40,40))

        lines = [
            f"Mieng: {'Mo' if expr['mouth_open']>0.3 else 'Dong'} ({expr['mouth_open']:.2f})"
            f"  Cuoi:{expr['mouth_smile']:.2f}  Biu:{expr.get('mouth_frown',0):.2f}",
            f"Mat T:{expr['left_eye_open']:.2f}  Mat P:{expr['right_eye_open']:.2f}"
            f"  Mo to:{expr.get('eye_wide',0):.2f}  Nheo:{expr.get('eye_squint',0):.2f}",
            f"May len:{expr.get('brow_up',0):.2f}  May xuong:{expr.get('brow_down',0):.2f}",
            f"Chum moi:{expr.get('pucker',0):.2f}  Phong ma:{expr.get('cheek_puff',0):.2f}"
            f"  Nhan mui:{expr.get('nose_sneer',0):.2f}",
        ]
        for i, t in enumerate(lines):
            cv2.putText(frame, t, (px, py+25+i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, (200,200,200), 1)
        return frame

    def _draw_interactions(self, frame, interactions, w, h):
        if not interactions:
            return frame
        y = 200
        draw_text_bg(frame, "TUONG TAC:", (10,y),
                     scale=0.55, color=(0,255,255), bg=(40,40,40))
        for i, txt in enumerate(interactions):
            cv2.putText(frame, f">> {txt}", (10, y+22+i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,255), 1)
        return frame

    # ══════════════════════════════════════════════════════
    # HELPER: Kiểm tra tay thả lỏng / bỏ xuống
    # ══════════════════════════════════════════════════════
    @staticmethod
    def _hands_are_relaxed(pose_lms, left_hand_lms, right_hand_lms):
        """
        Trả về True nếu KHÔNG thấy bàn tay HOẶC 2 tay đều thả lỏng
        (cổ tay nằm dưới hông hoặc gần hông).

        Logic:
        - Không detect được bàn tay nào → True
        - Cổ tay (pose 15,16) nằm DƯỚI hoặc NGANG hông (pose 23,24) → True
        """
        # Trường hợp 1: không thấy bàn tay nào
        if left_hand_lms is None and right_hand_lms is None:
            return True

        if pose_lms is None:
            return False

        # Lấy y-coordinate của hông (càng lớn = càng thấp trong frame)
        hip_y = None
        if pose_lms[23].visibility > 0.4 and pose_lms[24].visibility > 0.4:
            hip_y = (pose_lms[23].y + pose_lms[24].y) / 2
        elif pose_lms[23].visibility > 0.4:
            hip_y = pose_lms[23].y
        elif pose_lms[24].visibility > 0.4:
            hip_y = pose_lms[24].y
        else:
            # Không thấy hông → dùng vai + offset
            if pose_lms[11].visibility > 0.4 and pose_lms[12].visibility > 0.4:
                shoulder_y = (pose_lms[11].y + pose_lms[12].y) / 2
                hip_y = shoulder_y + 0.25  # ước lượng hông
            else:
                return False

        # Ngưỡng: cổ tay phải nằm dưới (hip_y - margin) mới tính là thả lỏng
        margin = 0.03  # cho phép hơi trên hông 1 chút

        # Kiểm tra cổ tay trái (pose idx 15)
        left_wrist_relaxed = True
        if pose_lms[15].visibility > 0.4:
            left_wrist_relaxed = pose_lms[15].y > (hip_y - margin)

        # Kiểm tra cổ tay phải (pose idx 16)
        right_wrist_relaxed = True
        if pose_lms[16].visibility > 0.4:
            right_wrist_relaxed = pose_lms[16].y > (hip_y - margin)

        return left_wrist_relaxed and right_wrist_relaxed

    # ══════════════════════════════════════════════════════
    # THU THẬP VIDEO
    # ══════════════════════════════════════════════════════
    def collect_label(self, label_name):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(" Khong the mo webcam!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        label_dir = os.path.join(self.output_dir, label_name)
        os.makedirs(label_dir, exist_ok=True)
        existing = len([f for f in os.listdir(label_dir) if f.endswith('.mp4')])
        video_count = existing

        # ── Trạng thái máy trạng thái ──
        # States: 'idle' → 'countdown' → 'recording' → 'idle'
        state = 'idle'
        video_writer = None
        frame_count = 0
        start_time = 0
        show_mesh = True
        auto_mode = True   # Bật/tắt chế độ auto
        fp = None          # Đường dẫn file video đang quay

        # Countdown
        COUNTDOWN_SECS = 5
        countdown_start = 0

        # Auto-stop: cần bỏ tay liên tục N frames mới dừng (tránh flicker)
        RELAXED_FRAMES_TO_STOP = 15  # ~0.5s @ 30fps
        relaxed_frame_count = 0

        # Sau khi dừng quay, chờ 1 chút trước khi bắt đầu countdown lại
        COOLDOWN_SECS = 2.0
        last_stop_time = 0

        print("\n" + "="*60)
        print(f" THU THAP VIDEO: {label_name.upper()} ".center(60))
        print("="*60)
        print(f"\n So video hien co: {existing}")
        print(f" Webcam: {width}x{height} @ {fps}FPS")
        print("\nHuong dan phim:")
        print("   [SPACE] - Bat dau / Dung quay (thu cong)")
        print("   [A]     - Bat/Tat che do AUTO (dem nguoc + tu dong dung)")
        print("   [M]     - Bat/Tat face mesh")
        print("   [Q]     - Thoat va luu")
        print("\n  CHE DO AUTO:")
        print(f"   - Khung hinh OK → dem nguoc {COUNTDOWN_SECS}s → tu dong quay")
        print("   - Bo tay xuong / tha long → tu dong dung quay")
        print("="*60 + "\n")

        self._ts = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Lưu frame gốc sạch TRƯỚC khi vẽ overlay
            clean_frame = frame.copy()

            # ── Gửi frame đến các detector (LIVE_STREAM) ──
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._ts += 33

            try:
                self.pose_detector.detect_async(mp_image, self._ts)
            except Exception:
                pass
            try:
                self.hand_detector.detect_async(mp_image, self._ts)
            except Exception:
                pass
            try:
                self.face_detector.detect_async(mp_image, self._ts)
            except Exception:
                pass

            # ── Đọc kết quả mới nhất ──
            pose_lms = self._latest['pose']
            face_lms = self._latest['face']
            blendshapes = self._latest['blendshapes']
            hands = self._latest['hands'] or (None, None)
            left_hand_lms, right_hand_lms = hands

            # ── Vẽ keypoints ──
            FullBodyDrawer.draw_pose(frame, pose_lms, w, h)
            if show_mesh:
                FullBodyDrawer.draw_face_mesh(frame, face_lms, w, h)
            FullBodyDrawer.draw_hand(frame, left_hand_lms, w, h, 'L')
            FullBodyDrawer.draw_hand(frame, right_hand_lms, w, h, 'R')

            # ── Kiểm tra góc quay ──
            framing = FramingChecker.check(pose_lms, face_lms,
                                           (left_hand_lms, right_hand_lms), w, h)
            self._draw_warnings(frame, framing, w, h)

            # ── Biểu cảm ──
            if blendshapes is not None:
                expr = FacialExpressionAnalyzer.analyze_blendshapes(blendshapes)
            else:
                expr = FacialExpressionAnalyzer.analyze_landmarks(face_lms, w, h)
            self._draw_expression(frame, expr, w, h)

            # ── Tương tác tay ↔ cơ thể ──
            frame, interactions = InteractionVisualizer.draw(
                frame, pose_lms, face_lms, left_hand_lms, right_hand_lms, w, h)
            self._draw_interactions(frame, interactions, w, h)

            # ── Kiểm tra tay thả lỏng ──
            hands_relaxed = self._hands_are_relaxed(
                pose_lms, left_hand_lms, right_hand_lms)

            # ══════════════════════════════════════════════
            # MÁY TRẠNG THÁI (AUTO MODE)
            # ══════════════════════════════════════════════
            now = time.time()

            if auto_mode:
                if state == 'idle':
                    # Chờ: khung hình OK + tay KHÔNG thả lỏng (đang giơ tay sẵn sàng)
                    in_cooldown = (now - last_stop_time) < COOLDOWN_SECS
                    if (framing['ok'] and not hands_relaxed and not in_cooldown):
                        state = 'countdown'
                        countdown_start = now
                        relaxed_frame_count = 0
                        print(f" Auto: Khung hinh OK → Dem nguoc {COUNTDOWN_SECS}s ...")

                elif state == 'countdown':
                    elapsed_cd = now - countdown_start
                    remaining = COUNTDOWN_SECS - elapsed_cd

                    if remaining <= 0:
                        # Hết countdown → bắt đầu quay
                        state = 'recording'
                        fn = f'{label_name}_{video_count:04d}.mp4'
                        fp = os.path.join(label_dir, fn)
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(
                            fp, fourcc, fps, (width, height))
                        frame_count = 0
                        start_time = now
                        relaxed_frame_count = 0
                        print(f" Auto: BAT DAU QUAY video {video_count+1}: {fn}")

                    elif not framing['ok'] or hands_relaxed:
                        # Khung hình hỏng hoặc bỏ tay → hủy countdown
                        state = 'idle'
                        print(" Auto: Huy dem nguoc (khung hinh thay doi)")

                elif state == 'recording':
                    # Kiểm tra tay thả lỏng liên tục
                    if hands_relaxed:
                        relaxed_frame_count += 1
                    else:
                        relaxed_frame_count = 0

                    if relaxed_frame_count >= RELAXED_FRAMES_TO_STOP:
                        # Dừng quay
                        state = 'idle'
                        video_writer.release()
                        video_writer = None
                        dur = now - start_time
                        video_count += 1
                        last_stop_time = now
                        relaxed_frame_count = 0
                        print(f" Auto: TU DONG DUNG video {video_count} "
                              f"({frame_count} frames, {dur:.1f}s)")
                        # Upload lên HuggingFace
                        upload_to_hf(fp, label_name)

            # ── Lưu video (frame GỐC sạch, không overlay) ──
            if state == 'recording' and video_writer is not None:
                video_writer.write(clean_frame)
                frame_count += 1

            # ══════════════════════════════════════════════
            # UI OVERLAY
            # ══════════════════════════════════════════════

            # Header
            cv2.rectangle(frame, (0,0), (w,55), (30,30,30), -1)
            cv2.putText(frame, f"Nhan: {label_name.upper()}", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Video: {video_count}", (10,48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            # Auto mode indicator
            auto_txt = "[A] Auto: ON" if auto_mode else "[A] Auto: OFF"
            auto_col = (0,255,0) if auto_mode else (100,100,100)
            cv2.putText(frame, auto_txt, (w-250, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, auto_col, 1)

            # Mesh toggle
            mesh_txt = "[M] Mesh: ON" if show_mesh else "[M] Mesh: OFF"
            cv2.putText(frame, mesh_txt, (w-250, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

            # State display
            if state == 'countdown':
                elapsed_cd = now - countdown_start
                remaining = max(0, COUNTDOWN_SECS - elapsed_cd)
                count_num = int(remaining) + 1

                # Số đếm ngược lớn ở giữa màn hình
                count_text = str(count_num)
                font_scale = 4.0
                (ctw, cth), _ = cv2.getTextSize(
                    count_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 6)
                cx = (w - ctw) // 2
                cy = (h + cth) // 2

                # Nền bán trong suốt
                overlay = frame.copy()
                cv2.rectangle(overlay, (cx-40, cy-cth-30),
                              (cx+ctw+40, cy+30), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

                # Số đếm (màu thay đổi theo countdown)
                if count_num <= 2:
                    num_color = (0, 0, 255)   # đỏ
                elif count_num <= 3:
                    num_color = (0, 165, 255)  # cam
                else:
                    num_color = (0, 255, 0)    # xanh
                cv2.putText(frame, count_text, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, num_color, 6)

                # Text phụ
                cv2.putText(frame, "CHUAN BI...", (w//2-80, cy+50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                # Thanh progress
                progress = elapsed_cd / COUNTDOWN_SECS
                bar_w = int(w * 0.6)
                bar_x = (w - bar_w) // 2
                bar_y = cy + 70
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + bar_w, bar_y + 12), (80,80,80), -1)
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + int(bar_w * progress), bar_y + 12),
                              num_color, -1)

            elif state == 'recording':
                elapsed = now - start_time
                rec_text = f"REC {elapsed:.1f}s | {frame_count}f"
                cv2.putText(frame, rec_text, (w//2-80, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                if int(elapsed*2) % 2 == 0:
                    cv2.circle(frame, (w//2-100, 20), 8, (0,0,255), -1)

                # Hiện cảnh báo nhẹ khi tay bắt đầu thả lỏng
                if relaxed_frame_count > 3:
                    ratio = relaxed_frame_count / RELAXED_FRAMES_TO_STOP
                    stop_text = f"Tha tay... dung sau {RELAXED_FRAMES_TO_STOP - relaxed_frame_count} frames"
                    cv2.putText(frame, stop_text, (w//2-150, h//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
                    # Thanh progress dừng
                    bar_w = 300
                    bar_x = (w - bar_w) // 2
                    cv2.rectangle(frame, (bar_x, h//2+15),
                                  (bar_x+bar_w, h//2+25), (80,80,80), -1)
                    cv2.rectangle(frame, (bar_x, h//2+15),
                                  (bar_x+int(bar_w*ratio), h//2+25),
                                  (0,200,255), -1)

            elif state == 'idle':
                # Cooldown indicator
                in_cooldown = (now - last_stop_time) < COOLDOWN_SECS
                if in_cooldown and auto_mode:
                    cd_remain = COOLDOWN_SECS - (now - last_stop_time)
                    cv2.putText(frame, f"Cho {cd_remain:.1f}s ...",
                                (w//2-60, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (200,200,200), 1)
                elif auto_mode:
                    if hands_relaxed:
                        cv2.putText(frame, "Gio tay len de bat dau",
                                    (w//2-130, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (0,200,255), 1)
                    elif not framing['ok']:
                        cv2.putText(frame, "Dieu chinh khung hinh...",
                                    (w//2-130, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (0,100,255), 1)
                else:
                    cv2.putText(frame, "[SPACE] de bat dau quay",
                                (w//2-120, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0,255,0), 1)

            # Relaxed indicator (nhỏ, luôn hiện)
            relax_txt = "Tay: THA LONG" if hands_relaxed else "Tay: GIO LEN"
            relax_col = (100,100,255) if hands_relaxed else (0,255,100)
            cv2.putText(frame, relax_txt, (w-160, h-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, relax_col, 1)

            # Footer
            cv2.rectangle(frame, (0,h-30), (w,h), (30,30,30), -1)
            ft = "[SPACE] Thu cong  |  [A] Auto  |  [M] Mesh  |  [Q] Thoat"
            cv2.putText(frame, ft, (10, h-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

            cv2.imshow('VSL Collector - Tasks API', frame)

            # ══════════════════════════════════════════════
            # PHÍM BẤM
            # ══════════════════════════════════════════════
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                # SPACE: thủ công bật/tắt quay (hoạt động cả khi auto ON)
                if state != 'recording':
                    # Hủy countdown nếu đang đếm
                    state = 'recording'
                    fn = f'{label_name}_{video_count:04d}.mp4'
                    fp = os.path.join(label_dir, fn)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        fp, fourcc, fps, (width, height))
                    frame_count = 0
                    start_time = time.time()
                    relaxed_frame_count = 0
                    print(f" Thu cong: BAT DAU QUAY video {video_count+1}: {fn}")
                else:
                    state = 'idle'
                    video_writer.release()
                    video_writer = None
                    dur = time.time() - start_time
                    video_count += 1
                    last_stop_time = time.time()
                    relaxed_frame_count = 0
                    print(f" Thu cong: DUNG video {video_count} "
                          f"({frame_count} frames, {dur:.1f}s)")
                    # Upload lên HuggingFace
                    upload_to_hf(fp, label_name)
                    time.sleep(0.3)

            elif key in (ord('a'), ord('A')):
                auto_mode = not auto_mode
                if not auto_mode and state == 'countdown':
                    state = 'idle'  # hủy countdown khi tắt auto
                print(f" Auto mode: {'ON' if auto_mode else 'OFF'}")

            elif key in (ord('m'), ord('M')):
                show_mesh = not show_mesh
                print(f" Face Mesh: {'ON' if show_mesh else 'OFF'}")

            elif key in (ord('q'), ord('Q')):
                if state == 'recording' and video_writer:
                    video_writer.release()
                    video_count += 1
                print("\n Da dung thu thap")
                break

        cap.release()
        cv2.destroyAllWindows()

        # Cleanup
        self._ts = 0

        self.metadata['labels'][label_name] = dict(
            num_videos=video_count, path=label_dir)
        self.metadata['total_videos'] = sum(
            v['num_videos'] for v in self.metadata['labels'].values())
        self._save_meta()
        print(f"\n Hoan thanh: {label_name} - {video_count} video")

    # ══════════════════════════════════════════════════════
    # MENU
    # ══════════════════════════════════════════════════════
    def interactive_menu(self):
        while True:
            print("\n" + "="*60)
            print(" VSL COLLECTOR - MediaPipe Tasks API ".center(60, "="))
            print("="*60)
            print("\n  1. Xem thong ke video")
            print("  2. Tao nhan moi va thu video")
            print("  3. Tiep tuc thu video cho nhan co san")
            print("  4. Luu va thoat")
            print("\n" + "="*60)

            ch = input("\n Chon chuc nang (1-4): ").strip()

            if ch == "1":
                self.show_statistics()
            elif ch == "2":
                lb = input("\n Nhap ten nhan moi: ").strip()
                if not lb:
                    print(" Ten nhan khong duoc de trong!")
                    continue
                lb = lb.lower().replace(" ", "_")
                if lb in self.metadata['labels']:
                    print(f" Nhan '{lb}' da ton tai!")
                    if input("  Tiep tuc thu them? (y/n): ").strip().lower() != 'y':
                        continue
                self.collect_label(lb)
            elif ch == "3":
                if not self.metadata['labels']:
                    print("\n Chua co nhan nao!")
                    continue
                labels = list(self.metadata['labels'].keys())
                print("\nDanh sach nhan:")
                for i, l in enumerate(labels, 1):
                    n = self.metadata['labels'][l]['num_videos']
                    print(f"  {i}. {l} ({n} video)")
                try:
                    idx = int(input("\n Chon nhan (so thu tu): ").strip())
                    if 1 <= idx <= len(labels):
                        self.collect_label(labels[idx-1])
                    else:
                        print(" Lua chon khong hop le!")
                except ValueError:
                    print(" Vui long nhap so!")
            elif ch == "4":
                self._save_meta()
                print("\n Da luu metadata")
                self.show_statistics()
                print("\n Tam biet!\n")
                break
            else:
                print(" Lua chon khong hop le!")

    def close(self):
        """Giải phóng tài nguyên"""
        self.pose_detector.close()
        self.hand_detector.close()
        self.face_detector.close()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    collector = WebcamVideoCollectorFull(output_dir='data/videos')
    try:
        collector.interactive_menu()
    finally:
        collector.close()


if __name__ == "__main__":
    main()