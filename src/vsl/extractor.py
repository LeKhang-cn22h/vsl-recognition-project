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
"""

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from vsl.config import cfg, FACE_KEY_INDICES, KEY_BLENDSHAPES
from vsl.utils  import download_model


# ═══════════════════════════════════════════════════════════
# NORMALIZE (dùng chung)
# ═══════════════════════════════════════════════════════════

def normalize_features(feats: np.ndarray) -> np.ndarray:
    """Normalize tọa độ theo shoulder center (pose idx 11, 12)."""
    f  = feats.copy()
    ls = f[33:36]; rs = f[36:39]
    center = (ls + rs) / 2
    if np.sum(np.abs(center)) < 1e-6:
        return f
    # Pose
    for i in range(25):
        f[i*3]   -= center[0]
        f[i*3+1] -= center[1]
    # Face
    for j in range(30):
        f[75 + j*3]   -= center[0]
        f[75 + j*3+1] -= center[1]
    # Hands
    for k in range(42):
        f[165 + k*3]   -= center[0]
        f[165 + k*3+1] -= center[1]
    return f


# ═══════════════════════════════════════════════════════════
# COMPUTE INTERACTIONS (dùng chung)
# ═══════════════════════════════════════════════════════════

def compute_interactions(pose_lms, left_hand, right_hand,
                          face_lms=None) -> np.ndarray:
    """
    31 interaction features:
      Mỗi tay × 2:
        7 dist cổ tay → vùng cơ thể  = 14
        2 relative so với ngực        =  4
        6 dist ngón trỏ → vùng mặt   = 12
      Khoảng cách 2 tay               =  1
    Tổng: 31
    """
    result = np.zeros(31, dtype=np.float32)
    if pose_lms is None:
        return result

    def xy(lm): return np.array([lm.x, lm.y], dtype=np.float32)

    head  = xy(pose_lms[0])
    l_ear = xy(pose_lms[7])  if pose_lms[7].visibility  > 0.3 else head.copy()
    r_ear = xy(pose_lms[8])  if pose_lms[8].visibility  > 0.3 else head.copy()
    ls    = xy(pose_lms[11]) if pose_lms[11].visibility > 0.3 else np.zeros(2)
    rs    = xy(pose_lms[12]) if pose_lms[12].visibility > 0.3 else np.zeros(2)
    chest = (ls + rs) / 2
    belly = (
        (ls + rs + xy(pose_lms[23]) + xy(pose_lms[24])) / 4
        if pose_lms[23].visibility > 0.3 and pose_lms[24].visibility > 0.3
        else chest + np.array([0.0, 0.15])
    )
    body_regions = [head, l_ear, r_ear, chest, belly, ls, rs]

    if face_lms is not None and len(face_lms) >= 468:
        face_regions = [
            xy(face_lms[50]),   # má phải
            xy(face_lms[280]),  # má trái
            xy(face_lms[159]),  # mắt phải
            xy(face_lms[386]),  # mắt trái
            xy(face_lms[4]),    # mũi tip
            xy(face_lms[13]),   # môi trên
        ]
    else:
        face_regions = [
            head + np.array([ 0.06,  0.02]),
            head + np.array([-0.06,  0.02]),
            head + np.array([ 0.03, -0.03]),
            head + np.array([-0.03, -0.03]),
            head + np.array([ 0.00,  0.02]),
            head + np.array([ 0.00,  0.05]),
        ]

    idx = 0
    for hlms in [right_hand, left_hand]:
        wrist     = xy(hlms[0]) if hlms else np.zeros(2)
        index_tip = xy(hlms[8]) if hlms else np.zeros(2)
        for reg in body_regions:
            result[idx] = float(np.linalg.norm(wrist - reg)); idx += 1
        result[idx] = float(wrist[0] - chest[0]); idx += 1
        result[idx] = float(wrist[1] - chest[1]); idx += 1
        for freg in face_regions:
            result[idx] = float(np.linalg.norm(index_tip - freg)); idx += 1

    if right_hand and left_hand:
        result[idx] = float(np.linalg.norm(xy(right_hand[0]) - xy(left_hand[0])))
    idx += 1
    return result


# ═══════════════════════════════════════════════════════════
# BASE: xây dựng feature vector từ các landmarks
# ═══════════════════════════════════════════════════════════

def build_feature_vector(pose_lms, face_lms, blendshapes,
                          left_hand, right_hand) -> np.ndarray:
    """Gộp tất cả landmarks → vector 339-dim + normalize."""
    # Pose (75)
    pose_arr = np.zeros(75, dtype=np.float32)
    if pose_lms:
        for i in range(min(25, len(pose_lms))):
            pose_arr[i*3:i*3+3] = [pose_lms[i].x,
                                    pose_lms[i].y,
                                    pose_lms[i].z]

    # Face (90)
    face_arr = np.zeros(90, dtype=np.float32)
    if face_lms:
        for j, idx in enumerate(FACE_KEY_INDICES):
            if idx < len(face_lms):
                face_arr[j*3:j*3+3] = [face_lms[idx].x,
                                        face_lms[idx].y,
                                        face_lms[idx].z]

    # Hands (126) — left=slot 0, right=slot 63
    hand_arr = np.zeros(126, dtype=np.float32)
    for hlms, offset in [(left_hand, 0), (right_hand, 63)]:
        if hlms:
            for k, lm in enumerate(hlms):
                hand_arr[offset + k*3:offset + k*3+3] = [lm.x, lm.y, lm.z]

    # Blendshapes (17)
    blend_arr = np.zeros(17, dtype=np.float32)
    if blendshapes:
        bs = {c.category_name: c.score for c in blendshapes}
        for j, name in enumerate(KEY_BLENDSHAPES):
            blend_arr[j] = bs.get(name, 0.0)

    # Interactions (31)
    interact_arr = compute_interactions(pose_lms, left_hand, right_hand, face_lms)

    feats = np.concatenate([pose_arr, face_arr, hand_arr, blend_arr, interact_arr])
    return normalize_features(feats)


# ═══════════════════════════════════════════════════════════
# REALTIME EXTRACTOR (LIVE_STREAM mode)
# ═══════════════════════════════════════════════════════════

class RealtimeExtractor:
    """
    Dùng cho webcam realtime_inference.py.
    Gửi frame async, đọc kết quả mới nhất bất kỳ lúc nào.
    """
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
                output_face_blendshapes=True,
                result_callback=_on_face))

    def send_frame(self, rgb_frame: np.ndarray) -> None:
        """Gửi frame đến tất cả detector async."""
        self._ts += 33
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        try: self.pose_det.detect_async(mp_img, self._ts)
        except Exception: pass
        try: self.hand_det.detect_async(mp_img, self._ts)
        except Exception: pass
        try: self.face_det.detect_async(mp_img, self._ts)
        except Exception: pass

    def extract_features(self) -> np.ndarray:
        """Đọc kết quả mới nhất → vector 339-dim."""
        pose_lms    = self._latest['pose']
        face_lms    = self._latest['face']
        blendshapes = self._latest['blendshapes']
        left_hand, right_hand = self._latest['hands'] or (None, None)
        return build_feature_vector(
            pose_lms, face_lms, blendshapes, left_hand, right_hand)

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
    """
    Dùng cho video_inference.py (xử lý từng frame riêng lẻ).
    Không có callback, chạy đồng bộ.
    """
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
                output_face_blendshapes=True))

    def extract_frame(self, rgb_frame: np.ndarray):
        """
        Trích xuất đồng bộ từ 1 frame RGB.
        Trả về (features[339], landmarks_dict)
        """
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        pose_r = self.pose_det.detect(mp_img)
        hand_r = self.hand_det.detect(mp_img)
        face_r = self.face_det.detect(mp_img)

        pose_lms    = pose_r.pose_landmarks[0] if pose_r.pose_landmarks else None
        face_lms    = face_r.face_landmarks[0] if face_r.face_landmarks else None
        blendshapes = face_r.face_blendshapes[0] if face_r.face_blendshapes else None

        left_hand = right_hand = None
        if hand_r.hand_landmarks and hand_r.handedness:
            for i, hlms in enumerate(hand_r.hand_landmarks):
                cat = hand_r.handedness[i][0].category_name
                if cat == 'Left': right_hand = hlms
                else:             left_hand  = hlms

        feats = build_feature_vector(
            pose_lms, face_lms, blendshapes, left_hand, right_hand)

        landmarks = dict(pose=pose_lms, face=face_lms,
                         left_hand=left_hand, right_hand=right_hand)
        return feats, landmarks

    def close(self) -> None:
        self.pose_det.close()
        self.hand_det.close()
        self.face_det.close()