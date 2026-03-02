"""
vsl/inference_engine.py - 5-layer inference pipeline
======================================================
    from vsl.inference_engine import InferenceEngine

Pipeline:
  Frame → [1] Motion Gate → [2] Model Predict → [3] Confidence Check
       → [3.5] IDLE Filter → [4] Consecutive Check → [5] Confirm
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import deque

from vsl.config import cfg
from vsl.utils  import is_idle_label


class InferenceEngine:
    """
    Nhận feature vector từng frame, chạy qua 5-layer pipeline,
    trả về top-K predictions và signal _just_confirmed khi xác nhận 1 label.
    """

    def __init__(self, model, label_map: dict,
                 device: str        = cfg.DEVICE,
                 seq_len: int       = cfg.SEQ_LEN,
                 confidence_thr: float = cfg.CONFIDENCE_THR,
                 motion_thr: float  = cfg.MOTION_THR,
                 consec_thr: int    = cfg.CONSEC_THR,
                 top_k: int         = cfg.TOP_K,
                 smooth_window: int = cfg.SMOOTH_WINDOW):

        self.model          = model
        self.idx2label      = {v: k for k, v in label_map.items()}
        self.device         = device
        self.seq_len        = seq_len
        self.confidence_thr = confidence_thr
        self.MOTION_THR     = motion_thr
        self.CONSEC_THR     = consec_thr
        self.top_k          = top_k

        # ── Buffers ──
        self.frame_buffer  = deque(maxlen=seq_len)
        self.prob_history  = deque(maxlen=smooth_window)

        # ── State ──
        self.prev_features      = None
        self.consecutive_count  = 0
        self.consecutive_label  = None
        self.last_confirmed     = None
        self._just_confirmed    = None   # signal cho bên ngoài đọc
        self.last_result        = None   # top_preds lần cuối (cho pause mode)

    # ── Buffer ───────────────────────────────────────────────────

    def push_frame(self, feature_vec: np.ndarray) -> None:
        self.frame_buffer.append(feature_vec)

    @property
    def buffer_ready(self) -> bool:
        return len(self.frame_buffer) >= self.seq_len

    # ── Motion detection ────────────────────────────────────────

    def compute_motion(self, curr: np.ndarray) -> float:
        """Tính vận tốc cổ tay so với frame trước."""
        if self.prev_features is None:
            self.prev_features = curr.copy()
            return 0.0
        # Cổ tay phải: feat[165:167], trái: feat[228:230]
        v_r = np.linalg.norm(curr[165:167] - self.prev_features[165:167])
        v_l = np.linalg.norm(curr[228:230] - self.prev_features[228:230])
        self.prev_features = curr.copy()
        return float(max(v_r, v_l))

    # ── Reset ───────────────────────────────────────────────────

    def reset(self) -> None:
        """Xóa toàn bộ state (dùng khi nhấn [C])."""
        self.frame_buffer.clear()
        self.prob_history.clear()
        self.prev_features      = None
        self.consecutive_count  = 0
        self.consecutive_label  = None
        self.last_confirmed     = None
        self._just_confirmed    = None
        self.last_result        = None

    # ── Main predict pipeline ────────────────────────────────────

    def predict(self, curr_features: np.ndarray = None):
        """
        5-layer pipeline.
        Trả về list[(label, prob)] hoặc None nếu IDLE/chưa đủ buffer.
        Đặt self._just_confirmed = label khi xác nhận, None nếu không.
        """
        self._just_confirmed = None

        if not self.buffer_ready:
            return None

        # ── Tầng 1: Motion Gate ──
        if curr_features is not None:
            motion = self.compute_motion(curr_features)
            if motion < self.MOTION_THR:
                self.consecutive_count  = 0
                self.consecutive_label  = None
                self.last_confirmed     = None
                return None

        # ── Tầng 2: Model Predict ──
        seq = np.stack(list(self.frame_buffer)[-self.seq_len:])
        x   = torch.from_numpy(seq).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()[0]

        self.prob_history.append(probs)
        smooth = np.mean(self.prob_history, axis=0)

        top_idx   = np.argsort(smooth)[::-1][:self.top_k]
        top_preds = [(self.idx2label.get(i, f'cls_{i}'), float(smooth[i]))
                     for i in top_idx]
        self.last_result = top_preds

        top_label, top_conf = top_preds[0]

        # ── Tầng 3: Confidence Check ──
        if top_conf < self.confidence_thr:
            self.consecutive_count = 0
            self.consecutive_label = None
            return top_preds   # hiển thị UI nhưng chưa xác nhận

        # ── Tầng 3.5: IDLE Label Filter ──
        if is_idle_label(top_label):
            self.consecutive_count  = 0
            self.consecutive_label  = None
            self.last_confirmed     = None   # cho phép ký lại từ cũ
            return None

        # ── Tầng 4: Consecutive Check ──
        if top_label == self.consecutive_label:
            self.consecutive_count += 1
        else:
            self.consecutive_label = top_label
            self.consecutive_count = 1

        # ── Tầng 5: Confirm ──
        if (self.consecutive_count >= self.CONSEC_THR and
                top_label != self.last_confirmed):
            self.last_confirmed    = top_label
            self.consecutive_count = 0
            self._just_confirmed   = top_label

        return top_preds