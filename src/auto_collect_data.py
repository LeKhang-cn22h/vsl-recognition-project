"""
VSL Auto Collector from JSON - Combined Face + Hand Features
Thu thập dữ liệu KẾT HỢP từ video URL trong file JSON:
  - Tên ký hiệu lấy từ trường "gross"
  - Trích xuất biểu cảm khuôn mặt (9 features) + hành động tay (126 features)
  - Lưu mỗi sample: shape (30, 135)

Features:
  - 126 hand features  (2 tay × 21 điểm × 3 toạ độ, chuẩn hoá theo cổ tay)
  -   9 face features  (EAR×2, avgEAR, MAR, BROW×2, avgBROW, nose_chin, eye_dist)
"""

import cv2
import numpy as np
import os
import json
import math
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ──────────────────────────────────────────────────────────────
# CẤU HÌNH
# ──────────────────────────────────────────────────────────────
SEQUENCE_LENGTH  = 30    # frames mỗi sample
N_HAND_FEATURES  = 126   # 2 tay × 21 × 3
N_FACE_FEATURES  = 9     # EAR×2, avgEAR, MAR, BROW×2, avgBROW, nose_chin, eye_dist
N_TOTAL          = N_HAND_FEATURES + N_FACE_FEATURES  # 135


class VSLCombinedAutoCollector:
    """
    Đọc file JSON (danh sách {"gross": ..., "url": ...}),
    tải từng video, trích xuất face + hand features,
    áp dụng Super Augmentation và lưu vào thư mục output.
    """

    # ── MediaPipe face landmark indices ──
    LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
    MOUTH_IDX     = [61, 291, 0, 17, 269, 405]
    L_BROW_IDX    = [70, 63, 105, 66, 107]
    R_BROW_IDX    = [300, 293, 334, 296, 336]

    def __init__(self, json_path: str, output_dir: str = '../data/combined_raw'):
        self.json_path   = json_path
        self.output_dir  = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._init_models()
        print("✓ VSLCombinedAutoCollector khởi tạo xong\n")

    # ──────────────────────────────────────────────────────────
    # KHỞI TẠO MODELS
    # ──────────────────────────────────────────────────────────
    def _download_if_missing(self, filename: str, url: str) -> str:
        base = os.path.dirname(os.path.abspath(__file__))
        for folder in [base, os.path.join(base, '..'), os.getcwd()]:
            p = os.path.join(folder, filename)
            if os.path.exists(p):
                return os.path.abspath(p)

        save_path = os.path.join(base, filename)
        print(f"  Đang tải {filename} ...")

        def _prog(count, block_size, total):
            if total > 0:
                pct = min(int(count * block_size * 100 / total), 100)
                print(f"\r  {pct}%", end='', flush=True)

        urllib.request.urlretrieve(url, save_path, reporthook=_prog)
        print(f"\n  ✓ Đã lưu: {save_path}")
        return save_path

    def _init_models(self):
        print("[Init] Hand Landmarker ...")
        hand_path = self._download_if_missing(
            'hand_landmarker.task',
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
            'hand_landmarker/float16/1/hand_landmarker.task'
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=hand_path),
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        )

        print("[Init] Face Landmarker ...")
        face_path = self._download_if_missing(
            'face_landmarker.task',
            'https://storage.googleapis.com/mediapipe-models/face_landmarker/'
            'face_landmarker/float16/1/face_landmarker.task'
        )
        self.face_detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=face_path),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        )
        print("[Init] ✓ Tất cả models đã sẵn sàng\n")

    # ──────────────────────────────────────────────────────────
    # FEATURE EXTRACTION – FACE
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _dist(p1, p2) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _ear(self, pts) -> float:
        """Eye Aspect Ratio"""
        v1 = self._dist(pts[1], pts[5])
        v2 = self._dist(pts[2], pts[4])
        h  = self._dist(pts[0], pts[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

    def _mar(self, pts) -> float:
        """Mouth Aspect Ratio"""
        v1 = self._dist(pts[1], pts[5])
        v2 = self._dist(pts[2], pts[4])
        h  = self._dist(pts[0], pts[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

    def _brow_pos(self, brow_pts, eye_pts) -> float:
        return (np.mean([p[1] for p in eye_pts])
                - np.mean([p[1] for p in brow_pts]))

    def extract_face_features(self, face_landmarks) -> np.ndarray:
        """Trích 9 facial features → np.array(9,)"""
        if not face_landmarks:
            return np.zeros(N_FACE_FEATURES, dtype=np.float32)

        lm  = face_landmarks[0]
        xy  = lambda i: (lm[i].x, lm[i].y)

        le  = [xy(i) for i in self.LEFT_EYE_IDX]
        re  = [xy(i) for i in self.RIGHT_EYE_IDX]
        mo  = [xy(i) for i in self.MOUTH_IDX]
        lb  = [xy(i) for i in self.L_BROW_IDX]
        rb  = [xy(i) for i in self.R_BROW_IDX]

        left_ear    = self._ear(le)
        right_ear   = self._ear(re)
        avg_ear     = (left_ear + right_ear) / 2.0
        mar         = self._mar(mo)
        left_brow   = self._brow_pos(lb, le)
        right_brow  = self._brow_pos(rb, re)
        avg_brow    = (left_brow + right_brow) / 2.0

        nose        = xy(1);   chin = xy(152)
        leye        = xy(33);  reye = xy(263)
        eye_dist    = self._dist(leye, reye)
        nose_chin   = self._dist(nose, chin) / (eye_dist + 1e-6)

        return np.array([left_ear, right_ear, avg_ear, mar,
                         left_brow, right_brow, avg_brow,
                         nose_chin, eye_dist], dtype=np.float32)

    # ──────────────────────────────────────────────────────────
    # FEATURE EXTRACTION – HAND
    # ──────────────────────────────────────────────────────────
    def extract_hand_features(self, hand_result) -> np.ndarray:
        """Trích 126 hand features, chuẩn hoá theo cổ tay → np.array(126,)"""
        kps = []
        if hand_result.hand_landmarks:
            for hand_lm in hand_result.hand_landmarks:
                for lm in hand_lm:
                    kps.extend([lm.x, lm.y, lm.z])

        # Pad đến đúng 126 giá trị
        while len(kps) < N_HAND_FEATURES:
            kps.extend([0.0] * 63)
        kps = kps[:N_HAND_FEATURES]

        arr = np.array(kps, dtype=np.float32).reshape(-1, 3)
        for i in range(2):
            s, e = i * 21, i * 21 + 21
            h_kps = arr[s:e]
            if np.any(h_kps != 0):
                h_kps -= h_kps[0].copy()
                arr[s:e] = h_kps

        return arr.flatten()

    # ──────────────────────────────────────────────────────────
    # COMBINED FRAME  (135 features)
    # ──────────────────────────────────────────────────────────
    def extract_combined_frame(self, hand_result, face_landmarks) -> np.ndarray:
        hand = self.extract_hand_features(hand_result)              # (126,)
        face = self.extract_face_features(face_landmarks)           # (9,)
        return np.concatenate([hand, face])                          # (135,)

    # ──────────────────────────────────────────────────────────
    # RESAMPLE
    # ──────────────────────────────────────────────────────────
    def resample_sequence(self, sequence: list, target_len: int) -> np.ndarray:
        seq = np.array(sequence, dtype=np.float32)
        if len(seq) == target_len:
            return seq
        length  = len(seq)
        indices = np.linspace(0, length - 1, target_len)
        result  = []
        for i in indices:
            lo = int(math.floor(i))
            hi = int(math.ceil(i))
            w  = i - lo
            if hi >= length:
                result.append(seq[length - 1])
            else:
                result.append(seq[lo] * (1 - w) + seq[hi] * w)
        return np.array(result, dtype=np.float32)

    # ──────────────────────────────────────────────────────────
    # SUPER AUGMENTATION  (≥35 file / video)
    # ──────────────────────────────────────────────────────────
    def _apply_rotation(self, data_3d: np.ndarray, angle: float) -> np.ndarray:
        """data_3d: (T, N_points, 3)  →  (T, 135) phẳng"""
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        result = data_3d.copy()
        # Chỉ xoay phần hand (126 features = 42 điểm × 3), giữ face nguyên
        n_hand_pts = N_HAND_FEATURES // 3  # 42
        xy = result[:, :n_hand_pts, :2]
        result[:, :n_hand_pts, :2] = xy @ rot.T
        return result.reshape(SEQUENCE_LENGTH, -1)

    def generate_augmentations(self, sign_name: str, raw_sequence: list):
        save_dir = os.path.join(self.output_dir, sign_name)
        os.makedirs(save_dir, exist_ok=True)

        base_flat = self.resample_sequence(raw_sequence, SEQUENCE_LENGTH)
        # (T, 135)  →  (T, 45, 3)  để dễ biến đổi hình học
        # 45 = 42 hand_pts + 3 "face pseudo-points" (face 9 feat / 3)
        base_3d = base_flat.reshape(SEQUENCE_LENGTH, 45, 3)

        augmentations = []

        # ── 1. Gốc + Noise ──
        augmentations.append(("org",   base_flat))
        noise = np.random.normal(0, 0.002, base_flat.shape).astype(np.float32)
        augmentations.append(("noise", base_flat + noise))

        angles = [-12, -8, -4, 4, 8, 12]
        scales = [0.85, 0.90, 0.95, 1.05, 1.10, 1.15]
        shifts = [(0.03, 0), (-0.03, 0), (0, 0.03), (0, -0.03)]

        # ── 2. Xoay (6) ──
        for a in angles:
            augmentations.append((f"rot{a}", self._apply_rotation(base_3d, a)))

        # ── 3. Scale (6) ──
        for s in scales:
            augmentations.append((f"scale{s}", base_flat * s))

        # ── 4. Shift (4) ──
        for idx, (sx, sy) in enumerate(shifts):
            tmp = base_3d.copy()
            tmp[:, :42, 0] += sx   # chỉ dịch hand x
            tmp[:, :42, 1] += sy   # chỉ dịch hand y
            augmentations.append((f"shift{idx}", tmp.reshape(SEQUENCE_LENGTH, -1)))

        # ── 5. Lật gương ──
        mirror_3d = base_3d.copy()
        mirror_3d[:, :42, 0] = -mirror_3d[:, :42, 0]
        mirror_flat = mirror_3d.reshape(SEQUENCE_LENGTH, -1)
        augmentations.append(("flip_org", mirror_flat))

        # ── 6. Flip + Xoay (6) ──
        for a in angles:
            augmentations.append((f"flip_rot{a}", self._apply_rotation(mirror_3d, -a)))

        # ── 7. Flip + Scale (6) ──
        for s in scales:
            augmentations.append((f"flip_scale{s}", mirror_flat * s))

        # ── 8. Flip + Shift (4) ──
        for idx, (sx, sy) in enumerate(shifts):
            tmp = mirror_3d.copy()
            tmp[:, :42, 0] += sx
            tmp[:, :42, 1] += sy
            augmentations.append((f"flip_shift{idx}", tmp.reshape(SEQUENCE_LENGTH, -1)))

        # Tổng: 2 + 6 + 6 + 4 + 1 + 6 + 6 + 4 = 35 file
        count = 0
        for suffix, data in augmentations:
            fp = os.path.join(save_dir, f"{sign_name}_{suffix}.npy")
            np.save(fp, data.astype(np.float32))
            count += 1

        print(f"   → Đã tạo {count} file augmentation cho '{sign_name}' tại: {save_dir}")

    # ──────────────────────────────────────────────────────────
    # XỬ LÝ 1 VIDEO
    # ──────────────────────────────────────────────────────────
    def process_single_video(self, sign_name: str, video_url: str):
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            print(f"  ✗ Không thể mở video: {video_url}")
            return

        raw_sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            hand_result  = self.hand_detector.detect(mp_img)
            face_result  = self.face_detector.detect(mp_img)

            combined = self.extract_combined_frame(
                hand_result, face_result.face_landmarks
            )
            raw_sequence.append(combined)

        cap.release()

        if len(raw_sequence) < 10:
            print(f"  ⚠ Video quá ngắn ({len(raw_sequence)} frames), bỏ qua.")
            return

        self.generate_augmentations(sign_name, raw_sequence)

    # ──────────────────────────────────────────────────────────
    # XỬ LÝ TOÀN BỘ JSON
    # ──────────────────────────────────────────────────────────
    def process_json(self, target_list: list = None, limit: int = 5):
        """
        target_list : danh sách tên gross cần lọc (None = lấy `limit` đầu tiên).
        limit       : số lượng lấy khi không có target_list.
        """
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if target_list:
            targets_lower = [t.lower().strip() for t in target_list]
            items = [it for it in data
                     if it.get('gross', '').strip().lower() in targets_lower]
            if not items:
                print(f"⚠ Không tìm thấy từ nào trong: {target_list}")
                return
        else:
            items = data[:limit]

        print(f"✅ Tìm thấy {len(items)} mục. Bắt đầu xử lý...\n")

        for idx, item in enumerate(items, 1):
            gross = item.get('gross', '').strip()
            url   = item.get('url', '').strip()

            if not gross or not url:
                print(f"  [{idx}] Bỏ qua mục thiếu dữ liệu: {item}")
                continue

            safe_name = gross.replace(' ', '_').lower()
            print(f"[{idx}/{len(items)}] Đang xử lý: '{gross}'  →  {safe_name}/")
            self.process_single_video(safe_name, url)

        print(f"\n✓ Hoàn thành! Dữ liệu tại: {os.path.abspath(self.output_dir)}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path   = os.path.join(current_dir, 'data.json')
    output_dir  = os.path.join(current_dir, '../data/combined_raw')

    # ══════════════════════════════════════════════════
    # 📝 DANH SÁCH TỪ CẦN THU THẬP  (để trống = lấy 5 đầu)
    # ══════════════════════════════════════════════════
    words_to_learn = [
        "nhập khẩu",
        "nổi da gà",
    ]

    if not os.path.exists(json_path):
        print(f"✗ Không tìm thấy file JSON: {json_path}")
    else:
        collector = VSLCombinedAutoCollector(
            json_path=json_path,
            output_dir=output_dir,
        )
        collector.process_json(target_list=words_to_learn)
        # Để xử lý TOÀN BỘ file JSON:
        # collector.process_json(limit=len(open(json_path).read().count('"gross"')))