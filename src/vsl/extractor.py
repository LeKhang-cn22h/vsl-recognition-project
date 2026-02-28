"""
vsl/extractor.py - Trích xuất features từ frame ảnh
=====================================================
Dùng chung cho realtime_inference.py và video_inference.py.

    # Realtime (LIVE_STREAM)
    from vsl.extractor import RealtimeExtractor
    ext = RealtimeExtractor()
    ext.send_frame(rgb)
    feats = ext.extract_features()

    # Video file (IMAGE mode)
    from vsl.extractor import VideoExtractor
    ext = VideoExtractor()
    feats, landmarks = ext.extract_frame(rgb)

Thay đổi v2:
    - Bỏ blendshapes khỏi feature vector
    - compute_interactions: 31 → 55 dims
      + 14 vùng mặt (fingertip thay cổ tay)
      + 8 vùng body (fingertip thay cổ tay)
      + min_dist 5 ngón × 2 tay
      + binary flag chạm mặt/body × 2 tay
    - Thêm detect_touch() để hiển thị UI
    - FEAT_DIM: 339 → 346
"""

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from vsl.config import (
    cfg, FACE_KEY_INDICES,
    FACE_REGION_INDICES, FACE_REGION_NAMES,
    BODY_REGION_INDICES, BODY_REGION_NAMES,
    TOUCH_THRESHOLD,
)
from vsl.utils import download_model

# Hand landmark indices cho fingertips
FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # cái, trỏ, giữa, áp út, út


# ═══════════════════════════════════════════════════════════
# NORMALIZE
# ═══════════════════════════════════════════════════════════

def normalize_features(feats: np.ndarray) -> np.ndarray:
    """Normalize tọa độ theo shoulder center (pose idx 11, 12)."""
    f  = feats.copy()
    ls = f[33:36]
    rs = f[36:39]
    center = (ls + rs) / 2
    if np.sum(np.abs(center)) < 1e-6:
        return f
    # Pose (25 landmarks)
    for i in range(25):
        f[i*3]   -= center[0]
        f[i*3+1] -= center[1]
    # Face (30 landmarks)
    for j in range(30):
        f[75 + j*3]   -= center[0]
        f[75 + j*3+1] -= center[1]
    # Hands (42 landmarks)
    for k in range(42):
        f[165 + k*3]   -= center[0]
        f[165 + k*3+1] -= center[1]
    return f


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def _xy(lm) -> np.ndarray:
    return np.array([lm.x, lm.y], dtype=np.float32)


def _get_fingertips(hlms) -> list[np.ndarray]:
    """Lấy tọa độ 5 fingertips của 1 bàn tay."""
    if hlms is None:
        return [np.zeros(2)] * 5
    return [_xy(hlms[i]) for i in FINGERTIP_INDICES]


def _min_dist_to_point(fingertips: list, target: np.ndarray) -> float:
    """Khoảng cách nhỏ nhất từ các fingertip đến 1 điểm."""
    dists = [float(np.linalg.norm(ft - target)) for ft in fingertips]
    return min(dists)


def _region_center(lms, indices: list) -> np.ndarray:
    """Tính tâm của 1 vùng từ list landmark indices."""
    pts = [_xy(lms[i]) for i in indices if i < len(lms)]
    if not pts:
        return np.zeros(2)
    return np.mean(pts, axis=0)


# ═══════════════════════════════════════════════════════════
# COMPUTE INTERACTIONS — 55 dims
# ═══════════════════════════════════════════════════════════

def compute_interactions(pose_lms, left_hand, right_hand,
                          face_lms=None) -> np.ndarray:
    """
    55 interaction features:

    Mỗi tay (× 2 = 54):
      dist min_fingertip → 14 vùng mặt  = 14
      dist min_fingertip → 8 vùng body  =  8
      min_dist trong 5 ngón → mặt       =  1  (gần nhất)
      binary flag chạm mặt              =  1  (< threshold)
      binary flag chạm body             =  1  (< threshold)
      relative chest x, y               =  2
      ── subtotal                        = 27

    dist 2 tay với nhau                 =  1
    TỔNG                                = 55
    """
    N_INTERACT = cfg.INTERACT_END - cfg.INTERACT_START  # 55
    result = np.zeros(N_INTERACT, dtype=np.float32)
    if pose_lms is None:
        return result

    # ── Tính tâm các vùng body từ pose landmarks ──
    ls    = _xy(pose_lms[11]) if pose_lms[11].visibility > 0.3 else np.zeros(2)
    rs    = _xy(pose_lms[12]) if pose_lms[12].visibility > 0.3 else np.zeros(2)
    chest = (ls + rs) / 2

    body_centers = {}
    for name, indices in BODY_REGION_INDICES.items():
        pts = []
        for idx in indices:
            if idx < len(pose_lms) and pose_lms[idx].visibility > 0.3:
                pts.append(_xy(pose_lms[idx]))
        body_centers[name] = np.mean(pts, axis=0) if pts else chest.copy()

    # ── Tính tâm các vùng mặt ──
    face_centers = {}
    if face_lms is not None and len(face_lms) >= 468:
        for name, indices in FACE_REGION_INDICES.items():
            face_centers[name] = _region_center(face_lms, indices)
    else:
        # Fallback: ước tính từ pose head landmark
        head = _xy(pose_lms[0])
        offsets = {
            'tran':         np.array([ 0.00, -0.08]),
            'thai_duong_T': np.array([ 0.08, -0.03]),
            'thai_duong_P': np.array([-0.08, -0.03]),
            'chan_may_T':   np.array([ 0.04, -0.04]),
            'chan_may_P':   np.array([-0.04, -0.04]),
            'mat_T':        np.array([ 0.03, -0.02]),
            'mat_P':        np.array([-0.03, -0.02]),
            'mui':          np.array([ 0.00,  0.02]),
            'ma_T':         np.array([ 0.06,  0.02]),
            'ma_P':         np.array([-0.06,  0.02]),
            'mieng':        np.array([ 0.00,  0.05]),
            'cam':          np.array([ 0.00,  0.08]),
            'lo_tai_T':     np.array([ 0.10,  0.00]),
            'lo_tai_P':     np.array([-0.10,  0.00]),
        }
        for name in FACE_REGION_NAMES:
            face_centers[name] = head + offsets.get(name, np.zeros(2))

    # ── Tính features cho mỗi tay ──
    idx = 0
    for hlms in [right_hand, left_hand]:
        tips = _get_fingertips(hlms)

        # Lấy fingertip gần nhất (trỏ nếu có, fallback zeros)
        if hlms is not None:
            index_tip = _xy(hlms[8])
        else:
            index_tip = np.zeros(2)

        # 14 dist fingertip → vùng mặt
        min_face_dist = float('inf')
        for name in FACE_REGION_NAMES:
            d = _min_dist_to_point(tips, face_centers[name])
            result[idx] = d
            idx += 1
            if d < min_face_dist:
                min_face_dist = d

        # 8 dist fingertip → vùng body
        min_body_dist = float('inf')
        for name in BODY_REGION_NAMES:
            d = _min_dist_to_point(tips, body_centers[name])
            result[idx] = d
            idx += 1
            if d < min_body_dist:
                min_body_dist = d

        # min_dist trong 5 ngón → mặt (ngón nào gần nhất)
        result[idx] = min_face_dist if min_face_dist != float('inf') else 1.0
        idx += 1

        # binary flag: có chạm mặt không
        result[idx] = 1.0 if min_face_dist < TOUCH_THRESHOLD else 0.0
        idx += 1

        # binary flag: có chạm body không
        result[idx] = 1.0 if min_body_dist < TOUCH_THRESHOLD else 0.0
        idx += 1

        # relative so với ngực (x, y)
        wrist = _xy(hlms[0]) if hlms else np.zeros(2)
        result[idx] = float(wrist[0] - chest[0])
        idx += 1
        result[idx] = float(wrist[1] - chest[1])
        idx += 1

    # dist 2 tay với nhau
    if right_hand is not None and left_hand is not None:
        result[idx] = float(np.linalg.norm(
            _xy(right_hand[0]) - _xy(left_hand[0])))
    idx += 1

    assert idx == N_INTERACT, f"interact dim mismatch: {idx} != {N_INTERACT}"
    return result


# ═══════════════════════════════════════════════════════════
# DETECT TOUCH — dùng để hiển thị UI (không đưa vào model)
# ═══════════════════════════════════════════════════════════

def detect_touch(pose_lms, left_hand, right_hand,
                  face_lms=None) -> dict:
    """
    Phát hiện tay đang chạm vùng nào trên mặt/body.
    Dùng để hiển thị UI realtime, KHÔNG đưa vào model.

    Trả về:
    {
        'face_zone': 'cam' | 'ma_T' | ... | None,
        'body_zone': 'nguc' | 'vai_T' | ... | None,
        'hand':      'right' | 'left' | 'both' | None,
    }
    """
    ZONE_VN = {
        'tran':         'Trán',
        'thai_duong_T': 'Thái dương trái',
        'thai_duong_P': 'Thái dương phải',
        'chan_may_T':   'Chân mày trái',
        'chan_may_P':   'Chân mày phải',
        'mat_T':        'Mắt trái',
        'mat_P':        'Mắt phải',
        'mui':          'Mũi',
        'ma_T':         'Má trái',
        'ma_P':         'Má phải',
        'mieng':        'Miệng',
        'cam':          'Cằm',
        'lo_tai_T':     'Lỗ tai trái',
        'lo_tai_P':     'Lỗ tai phải',
        'dau':          'Đầu',
        'vai_T':        'Vai trái',
        'vai_P':        'Vai phải',
        'nguc':         'Ngực',
        'khuyu_T':      'Khuỷu trái',
        'khuyu_P':      'Khuỷu phải',
        'hong_T':       'Hông trái',
        'hong_P':       'Hông phải',
    }

    result = {'face_zone': None, 'body_zone': None,
              'hand': None, 'face_zone_vn': None, 'body_zone_vn': None}

    if pose_lms is None:
        return result

    # Tính tâm face regions
    face_centers = {}
    if face_lms is not None and len(face_lms) >= 468:
        for name, indices in FACE_REGION_INDICES.items():
            face_centers[name] = _region_center(face_lms, indices)
    else:
        head = _xy(pose_lms[0])
        offsets = {
            'tran':         np.array([ 0.00, -0.08]),
            'thai_duong_T': np.array([ 0.08, -0.03]),
            'thai_duong_P': np.array([-0.08, -0.03]),
            'chan_may_T':   np.array([ 0.04, -0.04]),
            'chan_may_P':   np.array([-0.04, -0.04]),
            'mat_T':        np.array([ 0.03, -0.02]),
            'mat_P':        np.array([-0.03, -0.02]),
            'mui':          np.array([ 0.00,  0.02]),
            'ma_T':         np.array([ 0.06,  0.02]),
            'ma_P':         np.array([-0.06,  0.02]),
            'mieng':        np.array([ 0.00,  0.05]),
            'cam':          np.array([ 0.00,  0.08]),
            'lo_tai_T':     np.array([ 0.10,  0.00]),
            'lo_tai_P':     np.array([-0.10,  0.00]),
        }
        for name in FACE_REGION_NAMES:
            face_centers[name] = head + offsets.get(name, np.zeros(2))

    # Tính tâm body regions
    body_centers = {}
    ls = _xy(pose_lms[11]) if pose_lms[11].visibility > 0.3 else np.zeros(2)
    rs = _xy(pose_lms[12]) if pose_lms[12].visibility > 0.3 else np.zeros(2)
    chest = (ls + rs) / 2
    for name, indices in BODY_REGION_INDICES.items():
        pts = [_xy(pose_lms[i]) for i in indices
               if i < len(pose_lms) and pose_lms[i].visibility > 0.3]
        body_centers[name] = np.mean(pts, axis=0) if pts else chest.copy()

    # Kiểm tra từng tay
    touch_info = {}
    for hand_name, hlms in [('right', right_hand), ('left', left_hand)]:
        if hlms is None:
            continue
        tips = _get_fingertips(hlms)

        # Kiểm tra face zones
        for zone_name, center in face_centers.items():
            d = _min_dist_to_point(tips, center)
            if d < TOUCH_THRESHOLD:
                touch_info[hand_name] = ('face', zone_name)
                break

        # Kiểm tra body zones nếu chưa chạm mặt
        if hand_name not in touch_info:
            for zone_name, center in body_centers.items():
                d = _min_dist_to_point(tips, center)
                if d < TOUCH_THRESHOLD:
                    touch_info[hand_name] = ('body', zone_name)
                    break

    # Tổng hợp kết quả
    if len(touch_info) == 2:
        result['hand'] = 'both'
    elif 'right' in touch_info:
        result['hand'] = 'right'
    elif 'left' in touch_info:
        result['hand'] = 'left'

    for hand_name, (zone_type, zone_name) in touch_info.items():
        if zone_type == 'face' and result['face_zone'] is None:
            result['face_zone']    = zone_name
            result['face_zone_vn'] = ZONE_VN.get(zone_name, zone_name)
        elif zone_type == 'body' and result['body_zone'] is None:
            result['body_zone']    = zone_name
            result['body_zone_vn'] = ZONE_VN.get(zone_name, zone_name)

    return result


# ═══════════════════════════════════════════════════════════
# BUILD FEATURE VECTOR
# ═══════════════════════════════════════════════════════════

def build_feature_vector(pose_lms, face_lms, blendshapes,
                          left_hand, right_hand) -> np.ndarray:
    """
    Gộp landmarks → vector 346-dim + normalize.
    Không còn blendshapes.
    Layout: pose(75) + face(90) + hand(126) + interact(55)
    """
    # Pose (75)
    pose_arr = np.zeros(75, dtype=np.float32)
    if pose_lms:
        for i in range(min(25, len(pose_lms))):
            pose_arr[i*3:i*3+3] = [pose_lms[i].x,
                                    pose_lms[i].y,
                                    pose_lms[i].z]

    # Face landmarks (90) — 30 key points × 3
    face_arr = np.zeros(90, dtype=np.float32)
    if face_lms:
        for j, idx in enumerate(FACE_KEY_INDICES):
            if idx < len(face_lms):
                face_arr[j*3:j*3+3] = [face_lms[idx].x,
                                        face_lms[idx].y,
                                        face_lms[idx].z]

    # Hands (126) — left=slot 0:63, right=slot 63:126
    hand_arr = np.zeros(126, dtype=np.float32)
    for hlms, offset in [(left_hand, 0), (right_hand, 63)]:
        if hlms:
            for k, lm in enumerate(hlms):
                hand_arr[offset + k*3:offset + k*3+3] = [lm.x, lm.y, lm.z]

    # Interactions (55) — bỏ blendshapes, mở rộng interact
    interact_arr = compute_interactions(
        pose_lms, left_hand, right_hand, face_lms)

    feats = np.concatenate([pose_arr, face_arr, hand_arr, interact_arr])
    assert len(feats) == cfg.FEAT_DIM, \
        f"FEAT_DIM mismatch: {len(feats)} != {cfg.FEAT_DIM}"
    return normalize_features(feats)


# ═══════════════════════════════════════════════════════════
# REALTIME EXTRACTOR (LIVE_STREAM mode)
# ═══════════════════════════════════════════════════════════

class RealtimeExtractor:
    """Dùng cho webcam realtime_inference.py."""

    def __init__(self):
        self._latest = dict(pose=None, face=None,
                            hands=None, blendshapes=None)
        self._ts = 0

        hand_m = download_model('hand_landmarker.task')
        pose_m = download_model('pose_landmarker_heavy.task')
        face_m = download_model('face_landmarker.task')

        def _on_pose(r, img, ts):
            self._latest['pose'] = (r.pose_landmarks[0]
                                    if r.pose_landmarks else None)

        def _on_hand(r, img, ts):
            left = right = None
            if r.hand_landmarks and r.handedness:
                for i, hlms in enumerate(r.hand_landmarks):
                    cat = r.handedness[i][0].category_name
                    if cat == 'Left': right = hlms
                    else:             left  = hlms
            self._latest['hands'] = (left, right)

        def _on_face(r, img, ts):
            self._latest['face'] = (r.face_landmarks[0]
                                    if r.face_landmarks else None)
            self._latest['blendshapes'] = (r.face_blendshapes[0]
                                           if r.face_blendshapes else None)

        self.pose_det = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=pose_m),
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_poses=1,
                min_pose_detection_confidence=0.4,
                min_pose_presence_confidence=0.4,
                min_tracking_confidence=0.4,
                result_callback=_on_pose))

        self.hand_det = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=hand_m),
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_hands=2,
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4,
                result_callback=_on_hand))

        self.face_det = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=face_m),
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_faces=1,
                min_face_detection_confidence=0.4,
                min_face_presence_confidence=0.4,
                min_tracking_confidence=0.4,
                output_face_blendshapes=False,   # bỏ blendshapes
                result_callback=_on_face))

    def send_frame(self, rgb_frame: np.ndarray) -> None:
        self._ts += 33
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        try: self.pose_det.detect_async(mp_img, self._ts)
        except Exception: pass
        try: self.hand_det.detect_async(mp_img, self._ts)
        except Exception: pass
        try: self.face_det.detect_async(mp_img, self._ts)
        except Exception: pass

    def extract_features(self) -> np.ndarray:
        """Đọc kết quả mới nhất → vector 346-dim."""
        pose_lms  = self._latest['pose']
        face_lms  = self._latest['face']
        left_hand, right_hand = self._latest['hands'] or (None, None)
        return build_feature_vector(
            pose_lms, face_lms, None, left_hand, right_hand)

    def get_touch_info(self) -> dict:
        """Lấy thông tin vùng tay đang chạm (cho UI)."""
        pose_lms  = self._latest['pose']
        face_lms  = self._latest['face']
        left_hand, right_hand = self._latest['hands'] or (None, None)
        return detect_touch(pose_lms, left_hand, right_hand, face_lms)

    def get_latest(self) -> dict:
        return self._latest

    def close(self) -> None:
        self.pose_det.close()
        self.hand_det.close()
        self.face_det.close()


# ═══════════════════════════════════════════════════════════
# VIDEO EXTRACTOR (IMAGE mode)
# ═══════════════════════════════════════════════════════════

class VideoExtractor:
    """Dùng cho video_to_npy.py (xử lý từng frame riêng lẻ)."""

    def __init__(self):
        hand_m = download_model('hand_landmarker.task')
        pose_m = download_model('pose_landmarker_heavy.task')
        face_m = download_model('face_landmarker.task')

        self.pose_det = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=pose_m),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
                min_tracking_confidence=0.3))

        self.hand_det = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=hand_m),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3))

        self.face_det = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=face_m),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3,
                min_tracking_confidence=0.3,
                output_face_blendshapes=False))   # bỏ blendshapes

    def extract_frame(self, rgb_frame: np.ndarray):
        """
        Trích xuất đồng bộ từ 1 frame RGB.
        Trả về (features[346], landmarks_dict)
        """
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        pose_r = self.pose_det.detect(mp_img)
        hand_r = self.hand_det.detect(mp_img)
        face_r = self.face_det.detect(mp_img)

        pose_lms = pose_r.pose_landmarks[0] if pose_r.pose_landmarks else None
        face_lms = face_r.face_landmarks[0] if face_r.face_landmarks else None

        left_hand = right_hand = None
        if hand_r.hand_landmarks and hand_r.handedness:
            for i, hlms in enumerate(hand_r.hand_landmarks):
                cat = hand_r.handedness[i][0].category_name
                if cat == 'Left': right_hand = hlms
                else:             left_hand  = hlms

        feats = build_feature_vector(
            pose_lms, face_lms, None, left_hand, right_hand)

        landmarks = dict(pose=pose_lms, face=face_lms,
                         left_hand=left_hand, right_hand=right_hand)
        return feats, landmarks

    def close(self) -> None:
        self.pose_det.close()
        self.hand_det.close()
        self.face_det.close()