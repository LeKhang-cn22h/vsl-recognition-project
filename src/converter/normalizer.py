"""
converter/normalizer.py - Normalize + Resample chuỗi frame
=============================================================
    from converter.normalizer import KeypointNormalizer, resample_sequence
"""

import math
import numpy as np

from vsl.config import cfg, FACE_KEY_INDICES


class KeypointNormalizer:
    """Normalize tọa độ keypoints theo shoulder center (pose idx 11, 12)."""

    @staticmethod
    def normalize_frame(features: np.ndarray,
                         pose_dim: int = cfg.POSE_END - cfg.POSE_START,
                         face_dim: int = cfg.FACE_END - cfg.FACE_START,
                         hand_dim: int = cfg.HAND_END - cfg.HAND_START
                         ) -> np.ndarray:
        """
        Dịch chuyển toàn bộ tọa độ sao cho shoulder center = (0, 0).
        Không đụng đến blendshapes và interaction features.
        """
        f  = features.copy()
        ls = f[33:36]   # pose idx 11: left shoulder
        rs = f[36:39]   # pose idx 12: right shoulder
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


def resample_sequence(sequence: np.ndarray, target_len: int) -> np.ndarray:
    """
    Chuẩn hóa độ dài chuỗi về target_len bằng nội suy tuyến tính.
    Input shape : (N, feat_dim)
    Output shape: (target_len, feat_dim)
    """
    sequence = np.array(sequence)
    n = len(sequence)
    if n == target_len:
        return sequence

    indices   = np.linspace(0, n - 1, target_len)
    resampled = []
    for i in indices:
        lo = int(math.floor(i))
        hi = min(int(math.ceil(i)), n - 1)
        w  = i - lo
        resampled.append(sequence[lo] * (1 - w) + sequence[hi] * w)
    return np.array(resampled, dtype=np.float32)