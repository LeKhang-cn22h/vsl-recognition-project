"""
converter/augmenter.py - Data augmentation cho chuỗi keypoints
================================================================
    from converter.augmenter import Augmenter

Tạo ~25 biến thể từ 1 chuỗi gốc:
  org, noise×2, rotate×4, scale×4, shift×4, speed×2,
  flip, flip_noise, flip_rot×2, flip_scale×2, timewarp×3
"""

import numpy as np
from converter.normalizer import resample_sequence
from vsl.config import cfg


class Augmenter:
    """
    Sinh augmentation từ 1 chuỗi (seq_len, feat_dim).
    Chỉ biến đổi tọa độ (x,y,z) — không đụng blendshapes & interactions.
    """

    def __init__(self,
                 seq_len:   int = cfg.SEQ_LEN,
                 total_dim: int = cfg.FEAT_DIM,
                 pose_dim:  int = cfg.POSE_END   - cfg.POSE_START,
                 face_dim:  int = cfg.FACE_END   - cfg.FACE_START,
                 hand_dim:  int = cfg.HAND_END   - cfg.HAND_START):

        self.seq_len   = seq_len
        self.total_dim = total_dim
        self.pose_dim  = pose_dim
        self.face_dim  = face_dim
        self.hand_dim  = hand_dim
        self.coord_end = pose_dim + face_dim + hand_dim  # 291

    # ── Transform primitives ─────────────────────────────────

    def _rotate_coords(self, data: np.ndarray, angle_deg: float) -> np.ndarray:
        out = data.copy()
        rad = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        for t in range(self.seq_len):
            for i in range(0, self.coord_end, 3):
                x, y = out[t, i], out[t, i+1]
                out[t, i]   = x*c - y*s
                out[t, i+1] = x*s + y*c
        return out

    def _scale_coords(self, data: np.ndarray, factor: float) -> np.ndarray:
        out = data.copy()
        out[:, :self.coord_end] *= factor
        return out

    def _shift_coords(self, data: np.ndarray,
                       sx: float, sy: float) -> np.ndarray:
        out = data.copy()
        for t in range(self.seq_len):
            for i in range(0, self.coord_end, 3):
                out[t, i]   += sx
                out[t, i+1] += sy
        return out

    def _add_noise(self, data: np.ndarray, sigma: float = 0.003) -> np.ndarray:
        out = data.copy()
        noise = np.random.normal(0, sigma, (self.seq_len, self.coord_end))
        out[:, :self.coord_end] += noise.astype(np.float32)
        return out

    def _mirror_x(self, data: np.ndarray) -> np.ndarray:
        """Lật trái/phải + swap left↔right hand."""
        out = data.copy()
        hs  = self.pose_dim + self.face_dim   # hand start = 165
        for t in range(self.seq_len):
            # Lật tọa độ x
            for i in range(0, self.coord_end, 3):
                out[t, i] = -out[t, i]
            # Swap left ↔ right hand (mỗi tay 63 dims)
            lh = out[t, hs:hs+63].copy()
            rh = out[t, hs+63:hs+126].copy()
            out[t, hs:hs+63]    = rh
            out[t, hs+63:hs+126] = lh
        return out

    def _speed_change(self, data: np.ndarray, factor: float) -> np.ndarray:
        """Tăng/giảm tốc độ ký hiệu bằng cách resample."""
        n       = len(data)
        new_len = int(n * factor)
        if new_len < 2:
            print(f"    [Aug] CANH BAO: speed_change factor={factor} "
                  f"tao new_len={new_len} < 2, bo qua.")
            return data.copy()
        resampled = resample_sequence(data, new_len)
        return resample_sequence(resampled, self.seq_len)

    def _time_warp(self, data: np.ndarray,
                   sigma: float = 0.15, seed: int = None) -> np.ndarray:
        """Biến dạng trục thời gian theo đường cong ngẫu nhiên."""
        try:
            from scipy.interpolate import CubicSpline
        except ImportError:
            return data.copy()

        if seed is not None:
            np.random.seed(seed)

        T       = len(data)
        n_knots = 6
        knots   = np.linspace(0, T-1, n_knots)
        warped  = knots + np.random.normal(0, sigma*T, n_knots)
        warped  = np.clip(warped, 0, T-1)
        warped[0] = 0; warped[-1] = T-1

        # Đảm bảo monotonic tăng
        for k in range(1, len(warped)):
            warped[k] = max(warped[k], warped[k-1] + 0.5)
        warped = np.clip(warped, 0, T-1)

        cs      = CubicSpline(knots, warped)
        new_idx = np.clip(cs(np.arange(T)), 0, T-1)

        result = []
        for i in new_idx:
            lo = int(np.floor(i))
            hi = min(int(np.ceil(i)), T-1)
            w  = i - lo
            result.append(data[lo]*(1-w) + data[hi]*w)
        return resample_sequence(np.array(result, dtype=np.float32), self.seq_len)

    # ── Main generate ────────────────────────────────────────

    def generate(self, base_data: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """
        Tạo tất cả augmentations từ 1 chuỗi gốc.
        Trả về list[(suffix, data)] — suffix dùng làm phần đuôi tên file .npy.

        Tổng ~25 biến thể:
          org, noise×2, rotate×4, scale×4, shift×4, speed×2,
          flip, flip_noise, flip_rot×2, flip_scale×2, timewarp×3
        """
        augs = []

        # ── Gốc ──
        augs.append(('org',    base_data))

        # ── Nhiễu ──
        augs.append(('noise1', self._add_noise(base_data, sigma=0.002)))
        augs.append(('noise2', self._add_noise(base_data, sigma=0.004)))

        # ── Xoay ──
        for angle in [-10, -5, 5, 10]:
            augs.append((f'rot{angle:+d}', self._rotate_coords(base_data, angle)))

        # ── Co dãn ──
        for s in [0.9, 0.95, 1.05, 1.1]:
            augs.append((f'scl{s:.2f}', self._scale_coords(base_data, s)))

        # ── Dịch chuyển ──
        for i, (sx, sy) in enumerate([(0.02,0), (-0.02,0), (0,0.02), (0,-0.02)]):
            augs.append((f'sht{i}', self._shift_coords(base_data, sx, sy)))

        # ── Tốc độ ──
        for spd in [0.8, 1.2]:
            augs.append((f'spd{spd:.1f}', self._speed_change(base_data, spd)))

        # ── Mirror + mirror combos ──
        mirror = self._mirror_x(base_data)
        augs.append(('flip', mirror))
        augs.append(('flip_noise', self._add_noise(mirror, sigma=0.003)))
        for angle in [-8, 8]:
            augs.append((f'flip_rot{angle:+d}', self._rotate_coords(mirror, angle)))
        for s in [0.9, 1.1]:
            augs.append((f'flip_scl{s:.1f}', self._scale_coords(mirror, s)))

        # ── Time Warp ──
        for i in range(3):
            augs.append((f'twarp{i}', self._time_warp(base_data, seed=i*13)))

        return augs