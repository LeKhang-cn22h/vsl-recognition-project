"""
VIDEO TO NPY CONVERTER - FULL BODY VERSION (MediaPipe Tasks API)
Chuyển đổi video VSL thành file .npy để huấn luyện model

Trích xuất ĐẦY ĐỦ:
  - PoseLandmarker  : 25 upper-body keypoints × 3 (x,y,z)  = 75
  - FaceLandmarker  : 30 keypoints quan trọng × 3           = 90
  - HandLandmarker  : 21 × 2 tay × 3                        = 126
  - Blendshapes     : 17 biểu cảm quan trọng nhất           = 17
  - Interactions    : tay↔vùng cơ thể (dist + relative)     = 19   [FIX: 37→19]
  ─────────────────────────────────────────────────────────────
  TỔNG mỗi frame  :                                          = 327 features [FIX: 345→327]

Cài đặt:
    pip install mediapipe opencv-python numpy --break-system-packages

CHANGELOG (so với version cũ):
  [FIX-1] Handedness flip nhất quán với webcam_collector (camera đã flip → "Left"=tay phải)
  [FIX-2] interact_dim = 19 (không phải 37), total_dim = 327 (không phải 345)
  [FIX-3] Thêm assert kiểm tra shape output sau mỗi frame
  [FIX-4] _save_label_map tự động gọi khi dùng option 2/3 (process_folder, process_video)
  [FIX-5] show_statistics hiển thị gốc vs augmented riêng biệt
  [FIX-6] _speed_change: thêm guard khi new_len < 2
"""

import cv2
import numpy as np
import os
import math
import json
import urllib.request
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ═══════════════════════════════════════════════════════════
# CẤU HÌNH
# ═══════════════════════════════════════════════════════════

SEQUENCE_LENGTH = 30

# 30 face landmark indices quan trọng cho VSL
FACE_KEY_INDICES = [
    # Lông mày (10 điểm)
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    # Mắt (8 điểm)
    33, 159, 145, 133,
    263, 386, 374, 362,
    # Miệng (8 điểm)
    13, 14, 61, 291, 0, 17, 78, 308,
    # Tham chiếu (4 điểm)
    1, 4, 10, 152,
]

# 17 blendshapes quan trọng nhất cho VSL
KEY_BLENDSHAPES = [
    'jawOpen',
    'mouthSmileLeft', 'mouthSmileRight',
    'mouthFrownLeft', 'mouthFrownRight',
    'mouthPucker',
    'cheekPuff',
    'eyeBlinkLeft', 'eyeBlinkRight',
    'eyeWideLeft', 'eyeWideRight',
    'eyeSquintLeft', 'eyeSquintRight',
    'browInnerUp',
    'browDownLeft', 'browDownRight',
    'noseSneerLeft',
]

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
    if os.path.exists(filename):
        return filename
    url = MODEL_URLS[filename]
    print(f"  Dang tai {filename} ...")
    urllib.request.urlretrieve(url, filename)
    print(f"  Da tai xong {filename}")
    return filename


# ═══════════════════════════════════════════════════════════
# FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════

class FullBodyExtractor:
    """Trích xuất đầy đủ features từ 1 frame ảnh"""

    def __init__(self):
        print("\n" + "="*60)
        print(" KHOI TAO FULL BODY EXTRACTOR ".center(60))
        print("="*60)

        hand_model = download_model('hand_landmarker.task')
        pose_model = download_model('pose_landmarker_heavy.task')
        face_model = download_model('face_landmarker.task')

        print("  Khoi tao PoseLandmarker ...")
        self.pose_detector = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=pose_model),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            ))

        print("  Khoi tao HandLandmarker ...")
        self.hand_detector = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=hand_model),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            ))

        print("  Khoi tao FaceLandmarker (+ Blendshapes) ...")
        self.face_detector = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=face_model),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3,
                min_tracking_confidence=0.3,
                output_face_blendshapes=True,
            ))

        print("  Tat ca detector da san sang!")

        # ── Dimensions ──
        self.pose_dim    = 25 * 3                      # 75
        self.face_dim    = len(FACE_KEY_INDICES) * 3   # 90
        self.hand_dim    = 21 * 2 * 3                  # 126
        self.blend_dim   = len(KEY_BLENDSHAPES)        # 17
        # interact = 7 dist × 2 tay + 2 rel × 2 tay + 1 two-hand + 6 vùng mặt * 2 tay= 31
        self.interact_dim = 31
        self.total_dim   = (self.pose_dim + self.face_dim + self.hand_dim +
                            self.blend_dim + self.interact_dim)
        # = 75 + 90 + 126 + 17 + 31 = 339

        print(f"  total_dim = {self.total_dim} "
              f"(pose:{self.pose_dim} face:{self.face_dim} "
              f"hand:{self.hand_dim} blend:{self.blend_dim} "
              f"interact:{self.interact_dim})")
        print("="*60 + "\n")

    def get_feature_names(self):
        names = []
        for i in range(25):
            for c in ['x', 'y', 'z']:
                names.append(f'pose_{i}_{c}')
        for idx in FACE_KEY_INDICES:
            for c in ['x', 'y', 'z']:
                names.append(f'face_{idx}_{c}')
        for hand in ['left', 'right']:
            for i in range(21):
                for c in ['x', 'y', 'z']:
                    names.append(f'{hand}_hand_{i}_{c}')
        for bs in KEY_BLENDSHAPES:
            names.append(f'bs_{bs}')
        for hand in ['right', 'left']:
            for region in ['head', 'left_ear', 'right_ear', 'chest', 'belly',
                           'left_shoulder', 'right_shoulder']:
                names.append(f'{hand}_{region}_dist')
            names.append(f'{hand}_chest_rel_x')
            names.append(f'{hand}_chest_rel_y')
            for face_region in ['right_cheek', 'left_cheek',
                                 'right_eye', 'left_eye',
                                 'nose', 'mouth']:
                names.append(f'{hand}_index_{face_region}_dist')
        names.append('two_hand_dist')
        # Kiểm tra tổng
        assert len(names) == self.total_dim, (
            f"Feature name count {len(names)} != total_dim {self.total_dim}")
        return names

    def extract_frame(self, rgb_frame):
        """
        Trích xuất features từ 1 frame RGB.
        Trả về numpy array shape (total_dim,)
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        pose_result = self.pose_detector.detect(mp_image)
        hand_result = self.hand_detector.detect(mp_image)
        face_result = self.face_detector.detect(mp_image)

        # ── Pose (25 upper body) ──
        pose_arr = np.zeros(self.pose_dim, dtype=np.float32)
        pose_lms = None
        if pose_result.pose_landmarks:
            pose_lms = pose_result.pose_landmarks[0]
            for i in range(min(25, len(pose_lms))):
                pose_arr[i*3]   = pose_lms[i].x
                pose_arr[i*3+1] = pose_lms[i].y
                pose_arr[i*3+2] = pose_lms[i].z

        # ── Face (30 key indices) ──
        face_arr = np.zeros(self.face_dim, dtype=np.float32)
        face_lms = None
        if face_result.face_landmarks:
            face_lms = face_result.face_landmarks[0]
            for j, idx in enumerate(FACE_KEY_INDICES):
                if idx < len(face_lms):
                    face_arr[j*3]   = face_lms[idx].x
                    face_arr[j*3+1] = face_lms[idx].y
                    face_arr[j*3+2] = face_lms[idx].z

        hand_arr = np.zeros(self.hand_dim, dtype=np.float32)
        left_hand_lms  = None
        right_hand_lms = None

        if hand_result.hand_landmarks and hand_result.handedness:
            for i, hlms in enumerate(hand_result.hand_landmarks):
                cat = hand_result.handedness[i][0].category_name
                # "Left" trong frame đã flip = tay phải thực tế
                if cat == 'Left':
                    right_hand_lms = hlms
                    offset = 21 * 3   # right hand ở slot sau
                else:
                    left_hand_lms = hlms
                    offset = 0        # left hand ở slot trước
                for k, lm in enumerate(hlms):
                    hand_arr[offset + k*3]   = lm.x
                    hand_arr[offset + k*3+1] = lm.y
                    hand_arr[offset + k*3+2] = lm.z

        # ── Blendshapes (17 key) ──
        blend_arr = np.zeros(self.blend_dim, dtype=np.float32)
        if face_result.face_blendshapes:
            bs_dict = {c.category_name: c.score
                       for c in face_result.face_blendshapes[0]}
            for j, name in enumerate(KEY_BLENDSHAPES):
                blend_arr[j] = bs_dict.get(name, 0.0)

        # ── Interactions ──
        interact_arr = self._compute_interactions(
            pose_lms, left_hand_lms, right_hand_lms)

        # Gộp tất cả
        features = np.concatenate([
            pose_arr, face_arr, hand_arr, blend_arr, interact_arr
        ])

        #  Kiểm tra shape
        assert features.shape[0] == self.total_dim, (
            f"Shape mismatch: {features.shape[0]} != {self.total_dim}")

        return features.astype(np.float32)

    def _compute_interactions(self, pose_lms, left_hand_lms, right_hand_lms,
                              face_lms=None):
        """
        Tính features tương tác tay ↔ cơ thể + ngón trỏ ↔ mặt.

        Output: 31 features
          Mỗi tay × 2:
            7 dist cổ tay → vùng cơ thể     = 14
            2 relative (rel_x, rel_y)        =  4
            6 dist ngón trỏ → vùng mặt      = 12  [MỚI]
          Khoảng cách 2 tay                  =  1
          Tổng: (7+2+6) × 2 + 1 = 31
        """
        result = np.zeros(31, dtype=np.float32)

        if pose_lms is None:
            return result

        def get_xy(lm):
            return np.array([lm.x, lm.y], dtype=np.float32)

        # ── Vùng cơ thể từ pose ──
        head       = get_xy(pose_lms[0])
        l_ear      = get_xy(pose_lms[7])  if pose_lms[7].visibility  > 0.3 else head.copy()
        r_ear      = get_xy(pose_lms[8])  if pose_lms[8].visibility  > 0.3 else head.copy()
        l_shoulder = get_xy(pose_lms[11]) if pose_lms[11].visibility > 0.3 else np.zeros(2)
        r_shoulder = get_xy(pose_lms[12]) if pose_lms[12].visibility > 0.3 else np.zeros(2)
        chest      = (l_shoulder + r_shoulder) / 2

        if pose_lms[23].visibility > 0.3 and pose_lms[24].visibility > 0.3:
            belly = (l_shoulder + r_shoulder +
                     get_xy(pose_lms[23]) + get_xy(pose_lms[24])) / 4
        else:
            belly = chest + np.array([0.0, 0.15], dtype=np.float32)

        body_regions = [head, l_ear, r_ear, chest, belly, l_shoulder, r_shoulder]

        # ── Vùng mặt từ face landmarks [MỚI] ──
        # Dùng face_lms nếu có, fallback về pose head nếu không có
        if face_lms is not None and len(face_lms) >= 468:
            # má phải (index 50), má trái (index 280)
            # mắt phải (giữa: 159), mắt trái (giữa: 386)
            # mũi tip (index 4), môi trên (index 13)
            face_right_cheek = get_xy(face_lms[50])
            face_left_cheek  = get_xy(face_lms[280])
            face_right_eye   = get_xy(face_lms[159])
            face_left_eye    = get_xy(face_lms[386])
            face_nose        = get_xy(face_lms[4])
            face_mouth       = get_xy(face_lms[13])
        else:
            # Fallback: ước lượng từ pose head (ít chính xác hơn)
            face_right_cheek = head + np.array([ 0.06,  0.02], dtype=np.float32)
            face_left_cheek  = head + np.array([-0.06,  0.02], dtype=np.float32)
            face_right_eye   = head + np.array([ 0.03, -0.03], dtype=np.float32)
            face_left_eye    = head + np.array([-0.03, -0.03], dtype=np.float32)
            face_nose        = head + np.array([ 0.00,  0.02], dtype=np.float32)
            face_mouth       = head + np.array([ 0.00,  0.05], dtype=np.float32)

        face_regions = [face_right_cheek, face_left_cheek,
                        face_right_eye,   face_left_eye,
                        face_nose,        face_mouth]

        # ── Tính features ──
        idx = 0
        for hand_lms in [right_hand_lms, left_hand_lms]:
            # Cổ tay (index 0)
            wrist = (get_xy(hand_lms[0])
                     if hand_lms is not None else np.zeros(2, dtype=np.float32))

            # Ngón trỏ tip (index 8) [MỚI]
            index_tip = (get_xy(hand_lms[8])
                         if hand_lms is not None else np.zeros(2, dtype=np.float32))

            # 7 khoảng cách cổ tay → vùng cơ thể
            for region in body_regions:
                result[idx] = float(np.linalg.norm(wrist - region))
                idx += 1

            # 2 relative cổ tay so với ngực
            result[idx] = float(wrist[0] - chest[0]);  idx += 1
            result[idx] = float(wrist[1] - chest[1]);  idx += 1

            # 6 khoảng cách ngón trỏ → vùng mặt [MỚI]
            for face_reg in face_regions:
                result[idx] = float(np.linalg.norm(index_tip - face_reg))
                idx += 1

        # Khoảng cách 2 tay (cổ tay - cổ tay)
        if right_hand_lms is not None and left_hand_lms is not None:
            rw = get_xy(right_hand_lms[0])
            lw = get_xy(left_hand_lms[0])
            result[idx] = float(np.linalg.norm(rw - lw))
        idx += 1

        assert idx == 31, f"interact idx={idx}, expected 31"
        return result

    def close(self):
        self.pose_detector.close()
        self.hand_detector.close()
        self.face_detector.close()


# ═══════════════════════════════════════════════════════════
# NORMALIZER
# ═══════════════════════════════════════════════════════════

class KeypointNormalizer:
    @staticmethod
    def normalize_frame(features, pose_dim=75, face_dim=90, hand_dim=126):
        """Normalize tọa độ theo shoulder center (pose idx 11, 12)"""
        f = features.copy()

        ls = f[11*3:11*3+3]
        rs = f[12*3:12*3+3]
        center = (ls + rs) / 2

        if np.sum(np.abs(center)) < 1e-6:
            return f

        # Pose
        for i in range(25):
            f[i*3]   -= center[0]
            f[i*3+1] -= center[1]

        # Face
        face_start = pose_dim
        for j in range(len(FACE_KEY_INDICES)):
            f[face_start + j*3]   -= center[0]
            f[face_start + j*3+1] -= center[1]

        # Hands
        hand_start = pose_dim + face_dim
        for k in range(42):
            f[hand_start + k*3]   -= center[0]
            f[hand_start + k*3+1] -= center[1]

        return f


# ═══════════════════════════════════════════════════════════
# RESAMPLER
# ═══════════════════════════════════════════════════════════

def resample_sequence(sequence, target_len):
    """Chuẩn hóa độ dài chuỗi frames về target_len (nội suy tuyến tính)"""
    sequence = np.array(sequence)
    n = len(sequence)
    if n == target_len:
        return sequence

    indices = np.linspace(0, n - 1, target_len)
    resampled = []
    for i in indices:
        lo = int(math.floor(i))
        hi = min(int(math.ceil(i)), n - 1)
        w = i - lo
        frame = sequence[lo] * (1 - w) + sequence[hi] * w
        resampled.append(frame)
    return np.array(resampled, dtype=np.float32)


# ═══════════════════════════════════════════════════════════
# AUGMENTATION
# ═══════════════════════════════════════════════════════════

class Augmenter:
    def __init__(self, seq_len=SEQUENCE_LENGTH, total_dim=327,
                 pose_dim=75, face_dim=90, hand_dim=126):
        self.seq_len   = seq_len
        self.total_dim = total_dim
        self.pose_dim  = pose_dim
        self.face_dim  = face_dim
        self.hand_dim  = hand_dim
        self.coord_end = pose_dim + face_dim + hand_dim  # 291

    def _rotate_coords(self, data, angle_deg):
        out = data.copy()
        rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        for t in range(self.seq_len):
            for i in range(0, self.coord_end, 3):
                x, y = out[t, i], out[t, i+1]
                out[t, i]   = x * cos_a - y * sin_a
                out[t, i+1] = x * sin_a + y * cos_a
        return out

    def _scale_coords(self, data, factor):
        out = data.copy()
        out[:, :self.coord_end] *= factor
        return out

    def _shift_coords(self, data, sx, sy):
        out = data.copy()
        for t in range(self.seq_len):
            for i in range(0, self.coord_end, 3):
                out[t, i]   += sx
                out[t, i+1] += sy
        return out

    def _add_noise(self, data, sigma=0.003):
        out = data.copy()
        noise = np.random.normal(0, sigma, (self.seq_len, self.coord_end))
        out[:, :self.coord_end] += noise.astype(np.float32)
        return out

    def _mirror_x(self, data):
        out = data.copy()
        for t in range(self.seq_len):
            for i in range(0, self.coord_end, 3):
                out[t, i] = -out[t, i]
            # Swap left ↔ right hand
            hand_start = self.pose_dim + self.face_dim
            left_hand  = out[t, hand_start:hand_start+63].copy()
            right_hand = out[t, hand_start+63:hand_start+126].copy()
            out[t, hand_start:hand_start+63]    = right_hand
            out[t, hand_start+63:hand_start+126] = left_hand
        return out

    def _speed_change(self, data, factor):
        """ Thêm guard new_len < 2"""
        n = len(data)
        new_len = int(n * factor)
        if new_len < 2:
            print(f"    CANH BAO: speed_change factor={factor} tao new_len={new_len} < 2, bo qua.")
            return data.copy()
        resampled = resample_sequence(data, new_len)
        return resample_sequence(resampled, self.seq_len)

    def generate(self, base_data):
        """Tạo augmentations. Trả về list[(suffix, data)]"""
        augs = []

        augs.append(('org',    base_data))
        augs.append(('noise1', self._add_noise(base_data, 0.002)))
        augs.append(('noise2', self._add_noise(base_data, 0.004)))

        for angle in [-10, -5, 5, 10]:
            augs.append((f'rot{angle:+d}', self._rotate_coords(base_data, angle)))

        for s in [0.9, 0.95, 1.05, 1.1]:
            augs.append((f'scl{s:.2f}', self._scale_coords(base_data, s)))

        for i, (sx, sy) in enumerate([(0.02,0), (-0.02,0), (0,0.02), (0,-0.02)]):
            augs.append((f'sht{i}', self._shift_coords(base_data, sx, sy)))

        for spd in [0.8, 1.2]:
            augs.append((f'spd{spd:.1f}', self._speed_change(base_data, spd)))

        mirror = self._mirror_x(base_data)
        augs.append(('flip', mirror))
        augs.append(('flip_noise', self._add_noise(mirror, 0.003)))
        for angle in [-8, 8]:
            augs.append((f'flip_rot{angle:+d}', self._rotate_coords(mirror, angle)))
        for s in [0.9, 1.1]:
            augs.append((f'flip_scl{s:.1f}', self._scale_coords(mirror, s)))

        return augs


# ═══════════════════════════════════════════════════════════
# MAIN CONVERTER
# ═══════════════════════════════════════════════════════════

class VideoToNPY:
    def __init__(self, output_dir='data/processed', sequence_length=SEQUENCE_LENGTH):
        self.output_dir = output_dir
        self.seq_len    = sequence_length
        os.makedirs(output_dir, exist_ok=True)

        self.extractor = FullBodyExtractor()
        self.augmenter = Augmenter(
            seq_len=sequence_length,
            total_dim=self.extractor.total_dim,
            pose_dim=self.extractor.pose_dim,
            face_dim=self.extractor.face_dim,
            hand_dim=self.extractor.hand_dim,
        )
        self._save_feature_meta()

    def _save_feature_meta(self):
        meta = {
            'sequence_length': self.seq_len,
            'total_features_per_frame': self.extractor.total_dim,
            'breakdown': {
                'pose (25 upper-body x 3)': self.extractor.pose_dim,
                'face (30 key landmarks x 3)': self.extractor.face_dim,
                'hands (21 x 2 x 3)': self.extractor.hand_dim,
                'blendshapes (17 key)': self.extractor.blend_dim,
                'interactions (19)': self.extractor.interact_dim,
            },
            'face_landmark_indices': FACE_KEY_INDICES,
            'key_blendshapes': KEY_BLENDSHAPES,
            'feature_names': self.extractor.get_feature_names(),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        path = os.path.join(self.output_dir, 'feature_metadata.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  Da luu feature metadata: {path}")

    def process_video(self, video_path, label_name, video_id=None,
                      enable_augmentation=True):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  LOI: Khong the mo video: {video_path}")
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  Dang xu ly: {os.path.basename(video_path)} ({total_frames} frames)")

        raw_sequence = []
        hand_detected_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            features = self.extractor.extract_frame(rgb)
            features = KeypointNormalizer.normalize_frame(
                features,
                self.extractor.pose_dim,
                self.extractor.face_dim,
                self.extractor.hand_dim,
            )
            raw_sequence.append(features)

            hand_start = self.extractor.pose_dim + self.extractor.face_dim
            hand_data  = features[hand_start:hand_start + self.extractor.hand_dim]
            if np.sum(np.abs(hand_data)) > 0.01:
                hand_detected_count += 1

        cap.release()

        if len(raw_sequence) < 5:
            print(f"  CANH BAO: Video qua ngan ({len(raw_sequence)} frames). Bo qua.")
            return False

        hand_ratio = hand_detected_count / len(raw_sequence)
        if hand_ratio < 0.2:
            print(f"  CANH BAO: Chi detect tay trong {hand_ratio*100:.0f}% frames.")

        normalized = resample_sequence(raw_sequence, self.seq_len)
        print(f"    {len(raw_sequence)} frames -> {self.seq_len} frames (resampled)")

        save_dir = os.path.join(self.output_dir, label_name)
        os.makedirs(save_dir, exist_ok=True)
        vid_id = video_id or os.path.splitext(os.path.basename(video_path))[0]

        if enable_augmentation:
            augs = self.augmenter.generate(normalized)
            for suffix, data in augs:
                fn = f"{vid_id}_{suffix}.npy"
                np.save(os.path.join(save_dir, fn), data.astype(np.float32))
            print(f"    Da tao {len(augs)} file augmentation")
        else:
            fn = f"{vid_id}_org.npy"
            np.save(os.path.join(save_dir, fn), normalized.astype(np.float32))
            print(f"    Da luu: {fn}")

        return True

    def process_folder(self, input_folder, label_name, enable_augmentation=True):
        exts   = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        videos = sorted([f for f in os.listdir(input_folder)
                         if os.path.splitext(f)[1].lower() in exts])
        if not videos:
            print(f"  LOI: Khong tim thay video trong: {input_folder}")
            return

        print(f"\n  Tim thay {len(videos)} video trong {input_folder}")
        success = 0
        for i, vf in enumerate(videos, 1):
            vpath  = os.path.join(input_folder, vf)
            vid_id = f"{label_name}_{i-1:04d}"
            print(f"\n[{i}/{len(videos)}]", end=" ")
            if self.process_video(vpath, label_name, vid_id, enable_augmentation):
                success += 1

        print(f"\n  Hoan thanh: {success}/{len(videos)} video da xu ly")

        # Tự động cập nhật label map sau khi xử lý folder
        self._update_label_map(label_name)

    def process_collector_output(self, collector_dir='data/videos',
                                 enable_augmentation=True):
        meta_path = os.path.join(collector_dir, 'metadata.json')
        if not os.path.exists(meta_path):
            print(f"  LOI: Khong tim thay metadata tai {meta_path}")
            return

        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        labels = meta.get('labels', {})
        if not labels:
            print("  Chua co label nao trong metadata")
            return

        print(f"\n  Tim thay {len(labels)} labels trong collector output:")
        for lb, info in labels.items():
            print(f"    - {lb}: {info['num_videos']} video")

        for lb, info in labels.items():
            label_dir = info['path']
            if os.path.isdir(label_dir):
                print(f"\n{'='*60}")
                print(f"  Dang xu ly label: {lb.upper()}")
                print(f"{'='*60}")
                self.process_folder(label_dir, lb, enable_augmentation)
            else:
                print(f"  CANH BAO: Thu muc khong ton tai: {label_dir}")

        self._save_label_map(labels.keys())

    def _update_label_map(self, new_label):
        """[FIX-4] Cập nhật label map khi thêm label mới từ process_folder/video"""
        path = os.path.join(self.output_dir, 'label_map.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
        else:
            label_map = {}

        if new_label not in label_map:
            label_map[new_label] = len(label_map)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(label_map, f, indent=2, ensure_ascii=False)
            print(f"  Da cap nhat label map: '{new_label}' -> {label_map[new_label]}")

    def _save_label_map(self, label_names):
        labels    = sorted(label_names)
        label_map = {name: idx for idx, name in enumerate(labels)}
        path      = os.path.join(self.output_dir, 'label_map.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(label_map, f, indent=2, ensure_ascii=False)
        print(f"\n  Da luu label map ({len(labels)} labels): {path}")

    def show_statistics(self):
        """[FIX-5] Hiển thị gốc vs augmented riêng biệt"""
        print("\n" + "="*60)
        print(" THONG KE DU LIEU DA XU LY ".center(60))
        print("="*60)

        if not os.path.isdir(self.output_dir):
            print("  Chua co du lieu")
            return

        total_org = 0
        total_aug = 0
        print(f"\n{'Label':<25} {'Goc':>8} {'Augmented':>12} {'Tong':>8}")
        print("-"*56)

        for label_dir in sorted(os.listdir(self.output_dir)):
            label_path = os.path.join(self.output_dir, label_dir)
            if not os.path.isdir(label_path):
                continue
            npy_files = [f for f in os.listdir(label_path) if f.endswith('.npy')]
            org_files = [f for f in npy_files if f.endswith('_org.npy')]
            aug_files = [f for f in npy_files if not f.endswith('_org.npy')]
            total_org += len(org_files)
            total_aug += len(aug_files)
            print(f"  {label_dir:<23} {len(org_files):>8} {len(aug_files):>12} {len(npy_files):>8}")

        print("-"*56)
        print(f"  {'TONG CONG':<23} {total_org:>8} {total_aug:>12} {total_org+total_aug:>8}")
        print(f"\n  Features/frame : {self.extractor.total_dim}")
        print(f"  Sequence length: {self.seq_len}")
        print(f"  Shape moi file : ({self.seq_len}, {self.extractor.total_dim})")
        print("="*60)

    def close(self):
        self.extractor.close()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    converter = VideoToNPY(output_dir='data/processed')

    while True:
        print("\n" + "="*60)
        print(" VIDEO TO NPY - FULL BODY ".center(60, "="))
        print("="*60)
        print("\n  1. Xu ly tu dong tu Webcam Collector output")
        print("  2. Xu ly 1 thu muc video")
        print("  3. Xu ly 1 video don le")
        print("  4. Xem thong ke")
        print("  5. Thoat")
        print("\n" + "="*60)

        ch = input("\n Chon (1-5): ").strip()

        if ch == '1':
            folder = input("  Thu muc collector (mac dinh: data/videos): ").strip()
            folder = folder or 'data/videos'
            aug = input("  Bat augmentation? (y/n, mac dinh y): ").strip().lower()
            aug = aug != 'n'
            converter.process_collector_output(folder, aug)

        elif ch == '2':
            folder = input("  Duong dan thu muc video: ").strip()
            label  = input("  Ten nhan (label): ").strip()
            if not folder or not label:
                print("  LOI: Nhap day du thong tin!")
                continue
            aug = input("  Bat augmentation? (y/n, mac dinh y): ").strip().lower()
            aug = aug != 'n'
            if os.path.isdir(folder):
                converter.process_folder(folder, label, aug)
            else:
                print(f"  LOI: Thu muc khong ton tai: {folder}")

        elif ch == '3':
            vpath = input("  Duong dan video: ").strip()
            label = input("  Ten nhan (label): ").strip()
            if not vpath or not label:
                print("  LOI: Nhap day du thong tin!")
                continue
            aug = input("  Bat augmentation? (y/n, mac dinh y): ").strip().lower()
            aug = aug != 'n'
            if os.path.exists(vpath):
                ok = converter.process_video(vpath, label, enable_augmentation=aug)
                if ok:
                    # Cập nhật label map sau video đơn lẻ
                    converter._update_label_map(label)
            else:
                print(f"  LOI: File khong ton tai: {vpath}")

        elif ch == '4':
            converter.show_statistics()

        elif ch == '5':
            converter.close()
            print("\n  Tam biet!\n")
            break
        else:
            print("  Lua chon khong hop le!")


if __name__ == "__main__":
    main()