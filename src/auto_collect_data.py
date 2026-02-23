"""
VSL Auto Collector Holistic from JSON - MediaPipe Task API
Super Augmentation Holistic v3

THAY ĐỔI CHÍNH:
- [FIX CRITICAL] normalize_keypoints: Hierarchical normalization
    * POSE  → tâm = midpoint 2 vai, scale = khoảng cách vai
              Encode vị trí/tư thế cơ thể, bao gồm vị trí cổ tay (wrist position)
    * HANDS → tâm = wrist của chính tay đó, scale = wrist→middle_mcp
              Encode HÌNH DẠNG bàn tay độc lập với vị trí
    Tại sao đúng hơn:
    - Trước đây normalize tất cả về shoulder → hình dạng bàn tay thay đổi theo vị trí tay
    - Cùng 1 nắm tay ở gần mặt vs gần ngực → 2 vector khác nhau → model học gấp đôi pattern
    - Sau khi fix: hình dạng tay luôn nhất quán, vị trí tay encode qua Pose wrist landmarks

AUGMENTATION v2 (giữ nguyên):
    Speed, Time Warp, Temporal Crop, Shear, Joint Dropout, Flip+Speed
"""

import cv2
import numpy as np
import os
import json
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from scipy.interpolate import CubicSpline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy chưa cài. Time Warping bị bỏ qua. Cài: pip install scipy")

# ==========================================
# INDEX MAPPING trong vector flatten (553, 3)
# Pose:  kps[0:33]
# Face:  kps[33:511]   (478 điểm)
# LHand: kps[511:532]  (21 điểm)
# RHand: kps[532:553]  (21 điểm)
# ==========================================
POSE_START  = 0
POSE_END    = 33
FACE_START  = 33
FACE_END    = 511
LHAND_START = 511
LHAND_END   = 532
RHAND_START = 532
RHAND_END   = 553


class VSLAutoCollector:
    def __init__(self, json_path, output_dir='../data/raw'):
        self.output_dir      = output_dir
        self.json_path       = json_path
        self.sequence_length = 30

        os.makedirs(output_dir, exist_ok=True)

        print("Initializing MediaPipe Holistic (Auto Collector v3)...")
        self._setup_models()
        self._init_detectors()

    def _setup_models(self):
        models = {
            'hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            'face_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            'pose_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
        }
        import urllib.request
        for name, url in models.items():
            if not os.path.exists(name):
                print(f"  Downloading {name}...")
                try:
                    urllib.request.urlretrieve(url, name)
                except Exception as e:
                    print(f"  ❌ Failed: {e}")

    def _init_detectors(self):
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        self.hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.3))

        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        self.face_detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1))

        base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
        self.pose_detector = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=base_options))

    # ============================================================
    # NORMALIZE — Hierarchical (pose theo vai, tay theo wrist)
    # ============================================================
    def normalize_keypoints(self, keypoints):
        """
        Hierarchical normalization:

        POSE (kps[0:33]):
            Tâm  = midpoint(vai_trái[11], vai_phải[12])
            Scale = ||vai_trái - vai_phải||
            → Encode vị trí tư thế, GIỮ vị trí cổ tay (wrist)
              vì wrist trong Pose đã normalize = vị trí tay tương đối cơ thể

        LEFT HAND (kps[511:532]):
            Tâm  = wrist của tay trái (index 0 trong hand = kps[511])
            Scale = ||wrist - middle_finger_mcp (index 9 = kps[511+9])||
            → Encode HÌNH DẠNG bàn tay, bất biến với vị trí & khoảng cách camera

        RIGHT HAND (kps[532:553]):
            Tương tự Left Hand

        Thông tin VỊ TRÍ tay được giữ qua Pose landmarks 15 (L_wrist), 16 (R_wrist)
        đã được normalize theo vai → model biết tay ở đâu trên cơ thể.
        """
        kps = np.array(keypoints, dtype=np.float32).reshape(-1, 3)  # (553, 3)

        # ---- 1. POSE: normalize theo midpoint 2 vai ----
        left_shoulder  = kps[11].copy()   # Pose landmark 11
        right_shoulder = kps[12].copy()   # Pose landmark 12

        shoulder_ok   = np.any(left_shoulder != 0) and np.any(right_shoulder != 0)
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        shoulder_dist   = np.linalg.norm(left_shoulder - right_shoulder)

        if shoulder_ok and shoulder_dist > 1e-6:
            detected_pose = np.any(kps[POSE_START:POSE_END] != 0, axis=1)
            idxs = np.where(detected_pose)[0] + POSE_START
            kps[idxs] = (kps[idxs] - shoulder_center) / shoulder_dist
        
        # Fallback scale nếu không detect được vai
        fallback_scale = shoulder_dist if (shoulder_ok and shoulder_dist > 1e-6) else 0.3

        # ---- 2. LEFT HAND: normalize theo wrist trái ----
        left_wrist = kps[LHAND_START].copy()   # index 0 của hand = wrist
        if np.any(left_wrist != 0):
            left_mid_mcp = kps[LHAND_START + 9].copy()   # index 9 = middle finger MCP
            hand_scale   = np.linalg.norm(left_wrist - left_mid_mcp)
            hand_scale   = hand_scale if hand_scale > 1e-6 else fallback_scale * 0.3

            detected_lh = np.any(kps[LHAND_START:LHAND_END] != 0, axis=1)
            idxs = np.where(detected_lh)[0] + LHAND_START
            kps[idxs] = (kps[idxs] - left_wrist) / hand_scale

        # ---- 3. RIGHT HAND: normalize theo wrist phải ----
        right_wrist = kps[RHAND_START].copy()
        if np.any(right_wrist != 0):
            right_mid_mcp = kps[RHAND_START + 9].copy()
            hand_scale    = np.linalg.norm(right_wrist - right_mid_mcp)
            hand_scale    = hand_scale if hand_scale > 1e-6 else fallback_scale * 0.3

            detected_rh = np.any(kps[RHAND_START:RHAND_END] != 0, axis=1)
            idxs = np.where(detected_rh)[0] + RHAND_START
            kps[idxs] = (kps[idxs] - right_wrist) / hand_scale

        return kps.flatten()

    def extract_keypoints(self, hand_result, face_result, pose_result):
        """
        Extract: Pose(99) + Face(1434) + LHand(63) + RHand(63) = 1659
        Tay được sắp xếp Left/Right nhất quán qua handedness.
        """
        keypoints = []

        # 1. Pose (33 * 3 = 99)
        if pose_result.pose_landmarks:
            for lm in pose_result.pose_landmarks[0]:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 99)

        # 2. Face (478 * 3 = 1434)
        if face_result.face_landmarks:
            for lm in face_result.face_landmarks[0]:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 1434)

        # 3. Hands: LEFT slot trước (63), RIGHT slot sau (63)
        left_hand  = [0.0] * 63
        right_hand = [0.0] * 63

        if hand_result.hand_landmarks and hand_result.handedness:
            for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                hand_kps = []
                for lm in hand_landmarks:
                    hand_kps.extend([lm.x, lm.y, lm.z])
                label = hand_result.handedness[i][0].category_name  # "Left"/"Right"
                if label == "Left":
                    left_hand = hand_kps
                else:
                    right_hand = hand_kps

        keypoints.extend(left_hand)
        keypoints.extend(right_hand)

        return keypoints

    # ==========================================
    # RESAMPLE
    # ==========================================
    def resample_sequence(self, sequence, target_len):
        sequence = np.array(sequence, dtype=np.float32)
        if len(sequence) == target_len:
            return sequence

        length  = len(sequence)
        indices = np.linspace(0, length - 1, target_len)
        result  = []
        for i in indices:
            low    = int(math.floor(i))
            high   = min(int(math.ceil(i)), length - 1)
            weight = i - low
            result.append(sequence[low] * (1 - weight) + sequence[high] * weight)
        return np.array(result, dtype=np.float32)

    # ==========================================
    # SPATIAL AUGMENTATION
    # ==========================================
    def apply_rotation(self, data_reshaped, angle):
        rad   = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rot   = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        result = data_reshaped.copy()
        result[:, :, :2] = np.dot(result[:, :, :2], rot)
        return result.reshape(self.sequence_length, -1)

    def apply_shear(self, data_reshaped, shear_x=0.0, shear_y=0.0):
        result = data_reshaped.copy()
        ox = data_reshaped[:, :, 0].copy()
        oy = data_reshaped[:, :, 1].copy()
        result[:, :, 0] = ox + shear_x * oy
        result[:, :, 1] = oy + shear_y * ox
        return result.reshape(self.sequence_length, -1)

    def apply_joint_dropout(self, data_reshaped, dropout_rate=0.08, seed=None):
        if seed is not None:
            np.random.seed(seed)
        result   = data_reshaped.copy()
        n_nodes  = result.shape[1]
        n_drop   = max(1, int(n_nodes * dropout_rate))
        drop_idx = np.random.choice(n_nodes, size=n_drop, replace=False)
        result[:, drop_idx, :] = 0.0
        return result.reshape(self.sequence_length, -1)

    # ==========================================
    # TEMPORAL AUGMENTATION
    # ==========================================
    def apply_speed(self, base_data, speed_factor):
        base_data = np.array(base_data, dtype=np.float32)
        T         = len(base_data)
        n_frames  = max(10, min(int(T * speed_factor), T))
        indices   = np.linspace(0, T - 1, n_frames).astype(int)
        return self.resample_sequence(base_data[indices], self.sequence_length)

    def apply_time_warp(self, base_data, sigma=0.15, seed=None):
        if not SCIPY_AVAILABLE:
            return None
        if seed is not None:
            np.random.seed(seed)

        base_data = np.array(base_data, dtype=np.float32)
        T         = len(base_data)
        n_knots   = 6
        knots     = np.linspace(0, T - 1, n_knots)

        warped    = knots + np.random.normal(0, sigma * T, size=n_knots)
        warped    = np.clip(warped, 0, T - 1)
        warped[0]  = 0
        warped[-1] = T - 1
        for k in range(1, len(warped)):
            warped[k] = max(warped[k], warped[k - 1] + 0.5)
        warped = np.clip(warped, 0, T - 1)

        cs_warp = CubicSpline(knots, warped)
        new_idx = np.clip(cs_warp(np.arange(T)), 0, T - 1)

        result = []
        for i in new_idx:
            low    = int(np.floor(i))
            high   = min(int(np.ceil(i)), T - 1)
            weight = i - low
            result.append(base_data[low] * (1 - weight) + base_data[high] * weight)
        return self.resample_sequence(np.array(result, dtype=np.float32), self.sequence_length)

    def apply_temporal_crops(self, raw_sequence, n_crops=3):
        crops = []
        T     = len(raw_sequence)
        if T <= self.sequence_length:
            return crops
        starts = np.linspace(0, T - self.sequence_length, n_crops, dtype=int)
        for start in starts:
            crops.append(np.array(raw_sequence[start: start + self.sequence_length], dtype=np.float32))
        return crops

    # ==========================================
    # PROCESS
    # ==========================================
    def process_json(self, target_list=None, limit=5):
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            video_list      = data if isinstance(data, list) else data.get('words', [])
            data_to_process = []

            if target_list and len(target_list) > 0:
                print(f"🎯 Tìm kiếm: {target_list}")
                targets_lower = [t.lower().strip() for t in target_list]
                for item in video_list:
                    gloss = item.get('gross', '').strip()
                    if gloss.lower() in targets_lower:
                        data_to_process.append(item)
                if not data_to_process:
                    print("⚠️ Không tìm thấy từ nào!")
                    return
            else:
                data_to_process = video_list[:limit]

            print(f"✅ Tìm thấy {len(data_to_process)} video. Bắt đầu xử lý...")

            for index, item in enumerate(data_to_process):
                gloss = item.get('gross')
                url   = item.get('url')
                if gloss and url:
                    safe_name = gloss.replace(" ", "_").lower()
                    print(f"\n[{index+1}/{len(data_to_process)}] '{gloss}'...")
                    self.process_single_video(safe_name, url)

        except Exception as e:
            print(f"Lỗi JSON: {e}")
            import traceback; traceback.print_exc()

    def process_single_video(self, sign_name, video_url):
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            print(f"❌ Không thể mở: {video_url}")
            return

        raw_sequence = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            hand_res = self.hand_detector.detect(mp_image)
            face_res = self.face_detector.detect(mp_image)
            pose_res = self.pose_detector.detect(mp_image)

            kps      = self.extract_keypoints(hand_res, face_res, pose_res)
            norm_kps = self.normalize_keypoints(kps)
            raw_sequence.append(norm_kps)

        cap.release()

        if len(raw_sequence) < 10:
            print(f"⚠️ Video quá ngắn ({len(raw_sequence)} frames). Bỏ qua.")
            return

        print(f"   Video gốc: {len(raw_sequence)} frames")
        self.generate_augmentations(sign_name, raw_sequence)

    # ==========================================
    # AUGMENTATION ENGINE v3
    # ==========================================
    def generate_augmentations(self, sign_name, raw_sequence):
        save_path = os.path.join(self.output_dir, sign_name)
        os.makedirs(save_path, exist_ok=True)

        base_data    = self.resample_sequence(raw_sequence, self.sequence_length)
        n_landmarks  = base_data.shape[1] // 3   # = 553
        base_reshape = base_data.reshape(self.sequence_length, n_landmarks, 3)

        augmentations = []

        # NHÓM 1: GỐC & NOISE (2)
        augmentations.append(("org",   base_data))
        augmentations.append(("noise", base_data + np.random.normal(0, 0.002, base_data.shape)))

        # NHÓM 2: ROTATION (4)
        for angle in [-10, -5, 5, 10]:
            augmentations.append((f"rot{angle}", self.apply_rotation(base_reshape, angle)))

        # NHÓM 3: SCALE (4)
        for scale in [0.88, 0.94, 1.06, 1.12]:
            augmentations.append((f"scale{scale}", base_data * scale))

        # NHÓM 4: SHIFT (4)
        for idx, (sx, sy) in enumerate([(0.05, 0), (-0.05, 0), (0, 0.05), (0, -0.05)]):
            temp = base_reshape.copy()
            temp[:, :, 0] += sx
            temp[:, :, 1] += sy
            augmentations.append((f"shift{idx}", temp.reshape(self.sequence_length, -1)))

        # NHÓM 5: SHEAR (4)
        for sx, sy, name in [(0.10, 0, "shx+"), (-0.10, 0, "shx-"),
                              (0, 0.10, "shy+"), (0, -0.10, "shy-")]:
            augmentations.append((f"shear_{name}", self.apply_shear(base_reshape, sx, sy)))

        # NHÓM 6: JOINT DROPOUT (3)
        for i in range(3):
            augmentations.append((f"dropout{i}",
                self.apply_joint_dropout(base_reshape, 0.08, seed=i * 7)))

        # NHÓM 7: SPEED (4)
        for speed in [0.70, 0.85, 1.15, 1.30]:
            augmentations.append((f"speed{speed}", self.apply_speed(base_data, speed)))

        # NHÓM 8: TIME WARP (3)
        if SCIPY_AVAILABLE:
            for i in range(3):
                aug = self.apply_time_warp(base_data, sigma=0.15, seed=i * 13)
                if aug is not None:
                    augmentations.append((f"timewarp{i}", aug))

        # NHÓM 9: FLIP (1)
        mirror          = base_reshape.copy()
        mirror[:, :, 0] = -mirror[:, :, 0]
        mirror_flat     = mirror.reshape(self.sequence_length, -1)
        augmentations.append(("flip_org", mirror_flat))

        # NHÓM 10: FLIP + ROTATION (4)
        for angle in [-10, -5, 5, 10]:
            augmentations.append((f"flip_rot{angle}", self.apply_rotation(mirror, -angle)))

        # NHÓM 11: FLIP + SPEED (2)
        for speed in [0.80, 1.20]:
            augmentations.append((f"flip_speed{speed}", self.apply_speed(mirror_flat, speed)))

        # NHÓM 12: TEMPORAL CROP (tùy video)
        crops = self.apply_temporal_crops(raw_sequence, n_crops=3)
        for ci, crop in enumerate(crops):
            augmentations.append((f"crop{ci}",       crop))
            augmentations.append((f"crop{ci}_noise", crop + np.random.normal(0, 0.002, crop.shape)))

        # LƯU
        count = 0
        for suffix, data in augmentations:
            np.save(
                os.path.join(save_path, f"{sign_name}_{suffix}.npy"),
                np.array(data, dtype=np.float32)
            )
            count += 1

        print(f"   ✅ Đã tạo {count} file tại: {save_path}")
        return count


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path   = os.path.join(current_dir, 'data.json')
    output_dir  = os.path.join(current_dir, '../data/raw')

    words_to_learn = [
        "vui mừng",
        "buổi sáng",
        "cảm ơn",
        "địa chỉ",
        "xin lỗi",
        "tạm biệt"
    ]

    if os.path.exists(json_path):
        collector = VSLAutoCollector(json_path=json_path, output_dir=output_dir)
        collector.process_json(target_list=words_to_learn)
    else:
        print(f"❌ Không tìm thấy: {json_path}")
