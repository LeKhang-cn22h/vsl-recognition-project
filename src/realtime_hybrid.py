"""
realtime_hybrid_v5.py
=====================
VSL (Vietnamese Sign Language) — Nhận dạng ngôn ngữ ký hiệu thời gian thực.

KIẾN TRÚC: Dual-Stream (2 model chạy song song mỗi frame)
──────────────────────────────────────────────────────────
  Stream S — StaticMLP:
    Input : 96 features tay (góc khớp, khoảng cách đầu ngón, v.v.)
    Dùng để nhận dạng ký hiệu TĨNH (không cần chuyển động).

  Stream D — BiLSTM:
    Input : ring buffer 64 frames × 346 features (full pose + hand)
    Dùng để nhận dạng ký hiệu ĐỘNG (cần chuỗi thời gian).

QUYẾT ĐỊNH OUTPUT:
    Nếu conf_D >= conf_S + effective_bonus  →  dùng kết quả Dynamic
    Else                                    →  dùng kết quả Static
    effective_bonus = base_bonus × 0.60 (khi đang chuyển động)
                    = base_bonus × 1.20 (khi đứng yên)

FIXES so với v4:
  #1  Key S không còn conflict — S=screenshot, 0/1/2=force mode
  #2  Ring buffer flush khi mất tay ≥ 8 frames liên tiếp
  #3  feat_346=None guard → không crash khi extractor chưa ready
  #4  Force mode chỉ chạy đúng 1 model, không lãng phí
  #5  smooth_s giảm 9→5, smooth_d giảm 5→3 (giảm lag)
  #6  Weighted confidence voting thay vì Counter majority
  #7  Adaptive bonus theo motion state
  #8  torch.inference_mode() thay no_grad()
  #9  Ghost prediction bị xóa khi tay biến mất ≥ 5 frames

Phím điều khiển:
    Q/ESC  – Thoát
    G      – Bật/tắt debug panel
    F      – Fullscreen
    C      – Xóa log
    S      – Screenshot
    +/-    – Tăng/giảm confidence threshold
    [/]    – Tăng/giảm dynamic bonus
    0      – AUTO mode (tắt force)
    1      – Force STATIC
    2      – Force DYNAMIC
"""

import sys, os, glob, time, datetime, math, argparse
from pathlib import Path
from collections import deque, Counter, defaultdict
from itertools import combinations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Thêm project root vào sys.path để import được vsl.* ──────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / 'src')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vsl.extractor import RealtimeExtractor   # MediaPipe extractor bất đồng bộ
from vsl.config import cfg as vsl_cfg          # SEQ_LEN, FEAT_DIM, v.v.


# ══════════════════════════════════════════════════════════════════
# HẰNG SỐ TOÀN CỤC
# ══════════════════════════════════════════════════════════════════

# Số frames không thấy tay liên tiếp trước khi flush 50% ring buffer
# (xóa temporal context cũ để Dynamic không bị nhiễm từ phiên trước)
GAP_RESET_THRESH = 8

# Số frames không thấy tay trước khi xóa ghost prediction trên màn hình
HAND_GONE_FRAMES = 5

# Tỷ lệ ring buffer bị fill bằng zeros khi flush (50% = xóa nửa sau)
ZERO_FILL_RATIO = 0.5


# ══════════════════════════════════════════════════════════════════
# MODEL: StaticMLP — nhận dạng ký hiệu tĩnh
# ══════════════════════════════════════════════════════════════════

class StaticMLP(nn.Module):
    """
    MLP đơn giản nhận 96-dim hand features → xác suất từng lớp ký hiệu tĩnh.

    Kiến trúc: LayerNorm → Linear(96→256) → GELU → Linear(256→128)
               → GELU → Linear(128→64) → GELU → Linear(64→num_classes)
    """
    def __init__(self, feat_dim, num_classes, h1=256, h2=128, h3=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, h1), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(h1, h2),       nn.GELU(), nn.Dropout(0.0),
            nn.Linear(h2, h3),       nn.GELU(), nn.Dropout(0.0),
            nn.Linear(h3, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════
# MODEL: BiLSTMClassifier — nhận dạng ký hiệu động
# ══════════════════════════════════════════════════════════════════

class AttentionLayer(nn.Module):
    """
    Self-attention đơn giản (1 chiều) trên output sequence của LSTM.
    Trả về vector context (weighted sum) và attention weights.
    """
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Linear(d, 1)   # cho mỗi timestep 1 scalar score

    def forward(self, x):
        # x: (batch, seq_len, d)
        w = torch.softmax(self.attn(x).squeeze(-1), dim=-1)  # (batch, seq_len)
        ctx = (x * w.unsqueeze(-1)).sum(1)                   # (batch, d)
        return ctx, w


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM + Attention để phân loại chuỗi thời gian.

    Pipeline:
      input (seq_len, feat_dim)
        → input_proj: Linear + LayerNorm + ReLU     → (seq_len, hidden_dim)
        → BiLSTM                                    → (seq_len, hidden_dim×2)
        → Attention context + last hidden state     → (hidden_dim×4,)
        → classifier: LayerNorm → Linear → GELU × 2 → num_classes
    """
    def __init__(self, feat_dim, hidden_dim, num_layers, num_classes,
                 dropout_lstm=0.0, dropout_fc=0.0,
                 bidirectional=True, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        dirs = 2 if bidirectional else 1

        # Chiếu input về hidden_dim trước khi đưa vào LSTM
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout_fc))

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout_lstm if num_layers > 1 else 0.0)

        out_dim = hidden_dim * dirs   # 512 nếu bidirectional

        self.attention = AttentionLayer(out_dim) if use_attention else None

        # fc_in = attention_ctx + last_hidden = out_dim × 2 (nếu có attention)
        fc_in = out_dim * 2 if use_attention else out_dim
        mid = max(num_classes * 4, 128)

        self.classifier = nn.Sequential(
            nn.LayerNorm(fc_in),
            nn.Linear(fc_in, mid),    nn.GELU(), nn.Dropout(dropout_fc),
            nn.Linear(mid, mid // 2), nn.GELU(), nn.Dropout(dropout_fc / 2),
            nn.Linear(mid // 2, num_classes))

        self._dirs = dirs

    def forward(self, x):
        x = self.input_proj(x)
        out, (hn, _) = self.lstm(x)

        # Ghép hidden state của layer cuối (forward + backward)
        last = torch.cat([hn[-2], hn[-1]], -1) if self._dirs == 2 else hn[-1]

        if self.use_attention and self.attention:
            ctx, _ = self.attention(out)
            feat = torch.cat([ctx, last], -1)   # context + last hidden
        else:
            feat = last

        return self.classifier(feat)


# ══════════════════════════════════════════════════════════════════
# TRÍCH XUẤT FEATURES CHO STATIC MLP
# ══════════════════════════════════════════════════════════════════

# Các điểm đặc trưng trên bàn tay (MediaPipe index)
FINGERTIPS    = [4, 8, 12, 16, 20]        # 5 đầu ngón tay
FINGER_BASES  = [2, 5, 9, 13, 17]         # gốc 5 ngón tay
FINGER_CHAINS = [                          # chuỗi khớp theo từng ngón
    [0, 1, 2, 3, 4],   # ngón cái
    [0, 5, 6, 7, 8],   # ngón trỏ
    [0, 9,10,11,12],   # ngón giữa
    [0,13,14,15,16],   # ngón áp út
    [0,17,18,19,20],   # ngón út
]


def _angle(v1, v2):
    """Tính góc (radian) giữa 2 vector 3D. Trả về 0 nếu vector quá ngắn."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))


def extract_static_features(hand_lms):
    """
    Tạo vector 96-dim từ 21 landmarks bàn tay MediaPipe.

    Gồm:
      - 63 dim: tọa độ xyz của 21 điểm (đã normalize)
      - 20 dim: góc tại mỗi khớp ngón tay (4 khớp × 5 ngón)
      -  5 dim: chiều dài đoạn base→tip mỗi ngón
      - 10 dim: khoảng cách giữa tất cả cặp đầu ngón (C(5,2)=10)
      -  3 dim: vector pháp tuyến mặt phẳng lòng bàn tay

    Normalize: trừ wrist (lm[0]) rồi chia độ dài lm[9] để scale-invariant.
    Trả về None nếu input không hợp lệ.
    """
    if hand_lms is None or len(hand_lms) < 21:
        return None

    # Chuyển landmarks thành mảng (21, 3)
    lm = np.array([[h.x, h.y, h.z] for h in hand_lms], dtype=np.float32)

    # Normalize: đưa wrist về gốc tọa độ, scale theo khoảng cách wrist→MCP giữa
    lm -= lm[0]
    s = np.linalg.norm(lm[9])
    if s > 1e-6:
        lm /= s

    coords  = lm.flatten()   # 21×3 = 63 dim

    # Góc tại mỗi khớp: dùng 2 vector từ khớp đó đến 2 khớp liền kề
    angles  = np.array([
        _angle(lm[c[i-1]] - lm[c[i]], lm[c[i+1]] - lm[c[i]])
        for c in FINGER_CHAINS for i in range(1, 4)
    ])  # 5 ngón × 3 khớp = 15... thực tế chain dài 5 nên range(1,4) = 3 khớp giữa

    # Chiều dài base→tip mỗi ngón
    lengths = np.array([
        np.linalg.norm(lm[t] - lm[b])
        for b, t in zip(FINGER_BASES, FINGERTIPS)
    ])  # 5 dim

    # Khoảng cách giữa tất cả cặp đầu ngón tay
    tips  = lm[FINGERTIPS]
    dists = np.array([
        np.linalg.norm(tips[i] - tips[j])
        for i, j in combinations(range(5), 2)
    ])  # C(5,2) = 10 dim

    # Pháp tuyến lòng bàn tay (cross product của 2 cạnh lòng bàn tay)
    v1, v2 = lm[5] - lm[0], lm[17] - lm[0]
    n = np.cross(v1, v2)
    nn_ = np.linalg.norm(n)
    palm_n = (n / nn_ if nn_ > 1e-6 else n).astype(np.float32)   # 3 dim

    return np.concatenate([coords, angles, lengths, dists, palm_n]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# WEIGHTED CONFIDENCE VOTING  [FIX #6]
# ══════════════════════════════════════════════════════════════════

def weighted_vote(history, decay=0.80):
    """
    Bầu chọn nhãn từ lịch sử dự đoán, ưu tiên frame GẦN ĐÂY hơn.

    Args:
        history : deque of (label: str, conf: float)
                  phần tử đầu = xa nhất, phần tử cuối = gần nhất
        decay   : hệ số giảm weight theo khoảng cách thời gian
                  decay=0.80 → frame cách 1 bước = weight × 0.80

    Returns:
        (best_label, weighted_avg_conf)

    Cách hoạt động:
        Frame i có weight = decay^(n-1-i) → frame cuối (mới nhất) = 1.0
        Score của mỗi label = Σ(conf_i × weight_i)
        Label thắng = label có score cao nhất.
        Confidence trả về = score_thắng / tổng_weight_thắng
    """
    if not history:
        return '', 0.0

    scores = defaultdict(float)   # tổng weighted conf mỗi label
    total  = defaultdict(float)   # tổng weight mỗi label (để tính avg)
    n = len(history)

    for i, (lb, cf) in enumerate(history):
        w = decay ** (n - 1 - i)   # weight giảm dần về quá khứ
        scores[lb] += cf * w
        total[lb]  += w

    best = max(scores, key=scores.get)
    avg_conf = scores[best] / total[best] if total[best] > 0 else 0.0
    return best, avg_conf


# ══════════════════════════════════════════════════════════════════
# PHÁT HIỆN CHUYỂN ĐỘNG (Motion Onset Detector)
# ══════════════════════════════════════════════════════════════════

class MotionDetector:
    """
    Phát hiện chuyển động tay dựa trên velocity của wrist (cổ tay).
    Chỉ dùng để hiển thị indicator và điều chỉnh adaptive bonus.
    KHÔNG gate/block model nào.

    Attributes:
        velocity  : vận tốc trung bình wrist trong window gần nhất
        moving    : True nếu velocity >= threshold
        wrist_ok  : True nếu MediaPipe nhìn thấy cổ tay
    """
    def __init__(self, threshold=0.003, window=5):
        self.buf       = deque(maxlen=window)   # lịch sử vị trí wrist
        self.threshold = threshold
        self.velocity  = 0.0
        self.moving    = False
        self.wrist_ok  = False

    def update(self, pose_lms):
        """
        Cập nhật mỗi frame.
        pose_lms: danh sách landmarks pose từ MediaPipe (hoặc None).
        Index 16 = right wrist, 15 = left wrist.
        """
        wrist = None
        if pose_lms:
            try:
                wr, wl = pose_lms[16], pose_lms[15]
                if wr.visibility > 0.25:
                    wrist = np.array([wr.x, wr.y], dtype=np.float32)
                elif wl.visibility > 0.25:
                    wrist = np.array([wl.x, wl.y], dtype=np.float32)
            except (IndexError, AttributeError):
                pass

        self.wrist_ok = wrist is not None
        if wrist is None:
            return   # không có wrist → không update velocity

        self.buf.append(wrist)
        if len(self.buf) < 2:
            return

        # Tính vận tốc trung bình trên toàn bộ window
        pts = list(self.buf)
        v = sum(float(np.linalg.norm(pts[i] - pts[i-1]))
                for i in range(1, len(pts))) / len(pts)
        self.velocity = v
        self.moving   = v >= self.threshold


# ══════════════════════════════════════════════════════════════════
# DUAL STREAM PREDICTOR  (v5 — all fixes applied)
# ══════════════════════════════════════════════════════════════════

class DualStream:
    """
    Bộ não chính của hệ thống — chạy cả 2 model song song mỗi frame.

    Cơ chế hoạt động:
        1. Mỗi frame gọi step_auto() (hoặc step_force_*).
        2. Stream S chạy StaticMLP trên hand landmarks.
        3. Stream D append feat vào ring buffer rồi chạy BiLSTM.
        4. Quyết định dùng kết quả nào dựa vào confidence + effective bonus.

    Ring buffer (64 frames):
        - KHÔNG bao giờ reset hoàn toàn → Dynamic luôn có đủ context.
        - Khi mất tay ≥ GAP_RESET_THRESH frames → flush 50% bằng zeros
          để xóa temporal context cũ, tránh predict nhầm phiên sau.

    Smoothing:
        - sm_s và sm_d lưu (label, conf) thay vì chỉ label.
        - Bầu chọn bằng weighted_vote() → frame gần hơn có trọng số cao hơn.
    """
    def __init__(self, smodel, s_i2l, dmodel, d_i2l,
                 device, seq_len, feat_dim,
                 dynamic_bonus=0.15,
                 smooth_s=5, smooth_d=3,
                 conf_thr=0.60):
        # ── Models và label maps ──────────────────────────────────
        self.smodel        = smodel    # StaticMLP đã load
        self.s_i2l         = s_i2l    # {int index → label string}
        self.dmodel        = dmodel   # BiLSTMClassifier đã load
        self.d_i2l         = d_i2l
        self.device        = device
        self.seq_len       = seq_len   # độ dài sequence BiLSTM (vd: 64)
        self.feat_dim      = feat_dim  # số chiều mỗi frame (vd: 346)
        self.dynamic_bonus = dynamic_bonus
        self.conf_thr      = conf_thr

        # ── Ring buffer cho Dynamic [FIX #2] ─────────────────────
        self.ring_buf    = deque(maxlen=seq_len)
        self._zero_frame = np.zeros(feat_dim, dtype=np.float32)  # template để flush
        self._gap_count  = 0   # đếm frames liên tiếp không thấy tay

        # ── Smoothing buffers [FIX #5, #6] ───────────────────────
        # Lưu (label, conf) thay vì chỉ label để weighted_vote() dùng conf
        self.sm_s       = deque(maxlen=smooth_s)   # static smoothing
        self.sm_d       = deque(maxlen=smooth_d)   # dynamic smoothing
        self._hand_gone = 0   # đếm frames mất tay để clear ghost [FIX #9]

        # ── Output từng stream ────────────────────────────────────
        self.s_label  = ''
        self.s_conf   = 0.0
        self.d_label  = ''
        self.d_conf   = 0.0
        self.d_ready  = False   # True khi ring_buf đã đủ seq_len frames

        # ── Output cuối cùng (sau khi quyết định dùng stream nào) ─
        self.label      = ''
        self.conf       = 0.0
        self.model_type = ''    # 'static' | 'dynamic' | ''

        # Top-3 predictions của Dynamic (để hiển thị debug)
        self.d_topk = []

    # ─────────────────────────────────────────────────────────────
    # HELPER METHODS
    # ─────────────────────────────────────────────────────────────

    def _flush_ring_buffer(self, ratio=ZERO_FILL_RATIO):
        """
        Ghi đè (ratio × seq_len) frames cuối trong ring buffer bằng zeros.
        Mục đích: "xóa" temporal context cũ khi tay vừa quay lại sau khoảng trống dài.
        Dùng partial reset (50%) thay vì clear hoàn toàn để không mất warmup.
        [FIX #2]
        """
        fill_n = int(self.seq_len * ratio)
        for _ in range(fill_n):
            self.ring_buf.append(self._zero_frame.copy())

    def _clear_ghost(self):
        """
        Xóa tất cả smoothing buffers và reset output về rỗng.
        Gọi khi tay biến mất đủ lâu để tránh hiển thị prediction cũ (ghost).
        [FIX #9]
        """
        self.sm_s.clear()
        self.sm_d.clear()
        self.s_label    = ''
        self.s_conf     = 0.0
        self.d_label    = ''
        self.d_conf     = 0.0
        self.d_topk     = []
        self.label      = ''
        self.conf       = 0.0
        self.model_type = ''

    # ─────────────────────────────────────────────────────────────
    # MAIN STEP — chạy mỗi frame
    # ─────────────────────────────────────────────────────────────

    def step(self, feat_346, hand_lms,
             run_static=True, run_dynamic=True,
             motion_active=False):
        """
        Xử lý 1 frame, cập nhật self.label / self.conf / self.model_type.

        Args:
            feat_346      : np.ndarray (feat_dim,) từ extractor, hoặc None [FIX #3]
            hand_lms      : danh sách landmarks bàn tay MediaPipe, hoặc None
            run_static    : có chạy StaticMLP không [FIX #4]
            run_dynamic   : có chạy BiLSTM không    [FIX #4]
            motion_active : wrist đang chuyển động → điều chỉnh bonus [FIX #7]

        Returns:
            (label, conf, model_type)
        """

        # ── Guard: feat_346 = None [FIX #3] ──────────────────────
        # Extractor bất đồng bộ có thể chưa có frame → dùng zero thay crash
        if feat_346 is None:
            feat_346 = self._zero_frame.copy()

        # ── Kiểm tra tay có hiển thị không [FIX #2, #9] ──────────
        hand_present = (hand_lms is not None and len(hand_lms) >= 21)

        if not hand_present:
            # Tay không hiện → tăng counters
            self._gap_count += 1
            self._hand_gone += 1

            # Đủ ngưỡng → flush ring buffer để xóa context cũ
            if self._gap_count == GAP_RESET_THRESH:
                self._flush_ring_buffer()

            # Đủ ngưỡng → xóa ghost prediction trên màn hình
            if self._hand_gone >= HAND_GONE_FRAMES:
                self._clear_ghost()

            # Vẫn append zero vào ring buf để giữ đúng timing sequence
            self.ring_buf.append(self._zero_frame.copy())
            return self.label, self.conf, self.model_type

        else:
            # Tay vừa quay lại sau khoảng trống dài → clear vote bias
            if self._gap_count >= GAP_RESET_THRESH:
                self.sm_s.clear()
                self.sm_d.clear()
            # Reset cả 2 counter
            self._gap_count = 0
            self._hand_gone = 0

        # ── Stream S: Static MLP [FIX #4, #6, #8] ───────────────
        if run_static:
            s_feat = extract_static_features(hand_lms)

            if s_feat is not None:
                with torch.inference_mode():  # nhanh hơn no_grad() [FIX #8]
                    p = F.softmax(
                        self.smodel(
                            torch.from_numpy(s_feat).unsqueeze(0).to(self.device)
                        ), dim=-1
                    )[0].cpu().numpy()

                idx       = int(np.argmax(p))
                raw_conf  = float(p[idx])
                raw_label = self.s_i2l.get(idx, str(idx))

                # Lưu vào smoothing buffer dưới dạng (label, conf) [FIX #6]
                self.sm_s.append((raw_label, raw_conf))

                # Bầu chọn có trọng số → frame gần nhất có weight cao nhất
                self.s_label, self.s_conf = weighted_vote(self.sm_s)

            else:
                # extract_static_features thất bại (ít hơn 21 điểm)
                self.s_conf  = 0.0
                self.s_label = ''
                self.sm_s.clear()

        # ── Stream D: BiLSTM + ring buffer [FIX #2, #4, #6, #8] ──
        # Luôn append vào ring buffer dù có run_dynamic hay không
        # để ring buf không bị lệch timing khi force static
        self.ring_buf.append(feat_346)
        self.d_ready = len(self.ring_buf) >= self.seq_len

        if run_dynamic and self.d_ready:
            # Lấy toàn bộ ring buffer → (seq_len, feat_dim)
            seq = np.stack(list(self.ring_buf), axis=0)

            with torch.inference_mode():  # [FIX #8]
                p = F.softmax(
                    self.dmodel(
                        torch.from_numpy(seq).unsqueeze(0).to(self.device)
                    ), dim=-1
                )[0].cpu().numpy()

            idx       = int(np.argmax(p))
            raw_conf  = float(p[idx])
            raw_label = self.d_i2l.get(idx, str(idx))

            # Lưu vào smoothing buffer [FIX #6]
            self.sm_d.append((raw_label, raw_conf))
            self.d_label, self.d_conf = weighted_vote(self.sm_d)

            # Lưu top-3 để hiển thị debug (lấy trực tiếp từ softmax frame này)
            top3_idx    = np.argsort(p)[::-1][:3]
            self.d_topk = [(self.d_i2l.get(i, str(i)), float(p[i]))
                           for i in top3_idx]

        elif not run_dynamic:
            # Force static mode → giữ nguyên d_conf/d_label từ frame trước
            pass
        else:
            # Buffer chưa đủ → dynamic chưa ready
            self.d_conf  = 0.0
            self.d_label = ''
            self.d_topk  = []

        # ── Tính effective bonus (adaptive theo motion) [FIX #7] ─
        # Khi đang chuyển động: dynamic dễ "thắng" hơn (bonus thấp hơn)
        # Khi đứng yên: static được ưu tiên hơn (bonus cao hơn)
        if motion_active:
            eff_bonus = self.dynamic_bonus * 0.60
        else:
            eff_bonus = self.dynamic_bonus * 1.20

        # ── Quyết định dùng stream nào ────────────────────────────
        use_dynamic = (
            run_dynamic
            and self.d_ready
            and self.d_conf >= self.conf_thr
            and self.d_conf >= self.s_conf + eff_bonus   # dynamic phải "thắng" đủ xa
        )

        if use_dynamic:
            self.label      = self.d_label
            self.conf       = self.d_conf
            self.model_type = 'dynamic'
        elif run_static and self.s_conf >= self.conf_thr and self.s_label:
            self.label      = self.s_label
            self.conf       = self.s_conf
            self.model_type = 'static'
        else:
            # Cả 2 stream đều dưới threshold → không hiển thị gì
            self.label      = ''
            self.conf       = 0.0
            self.model_type = ''

        return self.label, self.conf, self.model_type

    # ─────────────────────────────────────────────────────────────
    # FORCE MODES [FIX #4] — chỉ chạy đúng 1 model
    # ─────────────────────────────────────────────────────────────

    def step_force_static(self, feat_346, hand_lms, motion_active=False):
        """
        Chỉ infer StaticMLP. Ring buffer vẫn được feed để giữ sync timing.
        Gọi khi người dùng nhấn phím '1'.
        """
        # Đảm bảo feat không phải None trước khi feed ring buf
        if feat_346 is None:
            feat_346 = self._zero_frame.copy()
        # Feed ring buf thủ công (step() với run_dynamic=False vẫn feed)
        self.ring_buf.append(feat_346)

        # Chạy step nhưng tắt dynamic
        self.step(feat_346, hand_lms,
                  run_static=True, run_dynamic=False,
                  motion_active=motion_active)

        # Override output → luôn dùng static
        if self.s_conf >= self.conf_thr and self.s_label:
            self.label      = self.s_label
            self.conf       = self.s_conf
            self.model_type = 'static'
        else:
            self.label = ''; self.conf = 0.0; self.model_type = ''
        return self.label, self.conf, self.model_type

    def step_force_dynamic(self, feat_346, hand_lms, motion_active=False):
        """
        Chỉ infer BiLSTM. StaticMLP không chạy → tiết kiệm compute.
        Gọi khi người dùng nhấn phím '2'.
        """
        # Chạy step nhưng tắt static
        self.step(feat_346, hand_lms,
                  run_static=False, run_dynamic=True,
                  motion_active=motion_active)

        # Override output → luôn dùng dynamic
        if self.d_ready and self.d_conf >= self.conf_thr and self.d_label:
            self.label      = self.d_label
            self.conf       = self.d_conf
            self.model_type = 'dynamic'
        else:
            self.label = ''; self.conf = 0.0; self.model_type = ''
        return self.label, self.conf, self.model_type

    def step_auto(self, feat_346, hand_lms, motion_active=False):
        """Normal dual-stream step — cả 2 model chạy, hệ thống tự quyết định."""
        return self.step(feat_346, hand_lms,
                         run_static=True, run_dynamic=True,
                         motion_active=motion_active)

    # ─────────────────────────────────────────────────────────────
    # DEBUG PROPERTIES
    # ─────────────────────────────────────────────────────────────

    @property
    def gap_count(self):
        """Số frames liên tiếp không thấy tay (dùng cho debug UI)."""
        return self._gap_count

    @property
    def hand_gone(self):
        """Số frames liên tiếp mất tay (dùng cho ghost clear logic)."""
        return self._hand_gone


# ══════════════════════════════════════════════════════════════════
# LOGGER — Ghi log prediction ra file text
# ══════════════════════════════════════════════════════════════════

class HybridLogger:
    """
    Ghi mỗi prediction mới (khi label thay đổi) ra file log text.
    Tránh log trùng bằng cách so sánh "model_type:label" với lần trước.
    """
    def __init__(self):
        log_dir = _PROJECT_ROOT / 'logs'
        log_dir.mkdir(exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.path     = str(log_dir / f'hybrid_{ts}.txt')
        self.entries  = []   # list of dict để tính summary
        self.last_key = ''   # tránh log entry trùng lặp

        # Tạo file với header
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write(f"Hybrid DualStream Log v5 — {ts}\n{'='*65}\n")
            f.write(f"{'Time':<10}{'Model':<10}{'Label':<30}{'Conf':>7}\n{'-'*65}\n")
        print(f"  [Log] {self.path}")

    def log(self, model_type, label, conf):
        """Ghi 1 entry nếu (model_type, label) khác lần trước."""
        key = f"{model_type}:{label}"
        if key == self.last_key:
            return   # bỏ qua nếu prediction không đổi
        self.last_key = key
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        self.entries.append({
            'time': ts, 'model': model_type,
            'label': label, 'conf': round(conf, 4)
        })
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(f"{ts:<10}{model_type:<10}{label:<30}{conf*100:>6.1f}%\n")
        print(f"  [LOG] {ts}  [{model_type:>7}]  {label:<24}  {conf*100:.1f}%")

    def clear(self):
        """Xóa entries trong memory và đánh dấu trong file."""
        self.entries = []
        self.last_key = ''
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- CLEARED {datetime.datetime.now().strftime('%H:%M:%S')} ---\n")
        print("  [Log cleared]")

    def summary(self):
        """In thống kê các ký hiệu đã nhận được khi thoát chương trình."""
        if not self.entries:
            return
        cnt = Counter(f"[{e['model']}] {e['label']}" for e in self.entries)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write('\n' + '='*65 + '\nSUMMARY\n')
            f.write(f"Total: {len(self.entries)}\n")
            for k, v in cnt.most_common():
                f.write(f"  {k:<42} {v}\n")
        print(f"\n  [Log] {self.path}  (total={len(self.entries)})")


# ══════════════════════════════════════════════════════════════════
# DRAWING — Các hàm vẽ UI lên frame OpenCV
# ══════════════════════════════════════════════════════════════════

FONT = cv2.FONT_HERSHEY_DUPLEX

# Bảng màu BGR (OpenCV dùng BGR không phải RGB)
C = dict(
    bg0=(10,11,16),    # nền tối nhất
    bg1=(16,18,26),    # header/footer
    bg2=(24,26,38),    # panel nền
    border=(45,48,65), # viền
    white=(232,232,238),
    dim=(95,98,118),   # text mờ
    green=(55,208,75),   green_d=(16,65,25),    # static stream
    orange=(38,160,252), orange_d=(13,50,88),   # dynamic stream
    yellow=(28,218,228),  # motion indicator / toast
    red=(55,55,215),      # cảnh báo
    teal=(8,195,180),     # debug title
    purple=(195,85,225),
    black=(4,4,7),
)

# Màu main và màu background cho từng model type
_MC  = {'static': C['green'],    'dynamic': C['orange']}
_MBG = {'static': C['green_d'],  'dynamic': C['orange_d']}


def _ar(img, x1, y1, x2, y2, col, a=0.82):
    """Vẽ hình chữ nhật bán trong suốt (alpha blend) lên img."""
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), col, -1)
    cv2.addWeighted(ov, a, img, 1-a, 0, img)


def _rr(img, x1, y1, x2, y2, col, a=0.85, r=8):
    """Vẽ hình chữ nhật góc bo tròn bán trong suốt."""
    ov = img.copy()
    # Phần giữa
    cv2.rectangle(ov, (x1+r, y1), (x2-r, y2), col, -1)
    cv2.rectangle(ov, (x1, y1+r), (x2, y2-r), col, -1)
    # 4 góc tròn
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(ov, (cx, cy), r, col, -1)
    cv2.addWeighted(ov, a, img, 1-a, 0, img)


def _t(img, txt, x, y, sc=0.52, col=None, th=1):
    """Vẽ text với shadow đen (để dễ đọc trên mọi nền)."""
    col = col or C['white']
    cv2.putText(img, txt, (x+1, y+1), FONT, sc, C['black'], th+1, cv2.LINE_AA)  # shadow
    cv2.putText(img, txt, (x,   y),   FONT, sc, col,         th,   cv2.LINE_AA)  # text


def _bar(img, x, y, w, h, ratio, fg, bg=None):
    """Vẽ progress bar ngang (ratio trong [0,1])."""
    bg = bg or C['bg2']
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
    fw = max(0, int(w * min(max(ratio, 0), 1)))
    if fw:
        cv2.rectangle(img, (x, y), (x+fw, y+h), fg, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), C['border'], 1)


def draw_dual_stream_debug(frame, stream, motion, W, py_start=68):
    """
    Vẽ panel debug nhỏ ở góc trái dưới header.
    Hiển thị: conf/label từng stream, buffer fill, gap counter, motion velocity,
              effective bonus, và top-3 dynamic predictions.
    """
    px, py = 10, py_start
    pw, ph = 380, 185
    _ar(frame, px, py, px+pw, py+ph, C['bg2'], 0.88)
    cv2.rectangle(frame, (px, py), (px+pw, py+ph), C['border'], 1)

    _t(frame, "DUAL STREAM DEBUG v5", px+8, py+16, 0.38, C['teal'])

    # Stream S — xanh lá nếu đủ threshold
    sc = C['green'] if stream.s_conf >= stream.conf_thr else C['dim']
    _t(frame, f"[S] static  {stream.s_conf*100:5.1f}%  {stream.s_label:<16}", px+8, py+38, 0.40, sc)
    _bar(frame, px+8, py+42, pw-20, 6, stream.s_conf, C['green'])

    # Stream D — cam nếu ready và đủ threshold
    dc  = C['orange'] if (stream.d_ready and stream.d_conf >= stream.conf_thr) else C['dim']
    rdy = f"buf {len(stream.ring_buf)}/{stream.seq_len}"
    _t(frame, f"[D] dynamic {stream.d_conf*100:5.1f}%  {stream.d_label:<16}  {rdy}", px+8, py+68, 0.40, dc)
    _bar(frame, px+8, py+72, pw-20, 6, stream.d_conf, C['orange'])

    # Buffer fill progress bar
    _t(frame, "buffer:", px+8, py+92, 0.36, C['dim'])
    _bar(frame, px+60, py+80, pw-75, 8, len(stream.ring_buf)/stream.seq_len, C['orange'], C['bg2'])

    # Gap / gone counters — đỏ khi đạt ngưỡng flush
    gap_c = C['red'] if stream.gap_count >= GAP_RESET_THRESH else C['dim']
    _t(frame, f"gap={stream.gap_count}/{GAP_RESET_THRESH}  gone={stream.hand_gone}/{HAND_GONE_FRAMES}",
       px+8, py+108, 0.36, gap_c)

    # Motion indicator
    mc = C['yellow'] if motion.moving else C['dim']
    mv_str = f"MOVING v={motion.velocity:.4f}" if motion.moving else f"still  v={motion.velocity:.4f}"
    _t(frame, f"motion: {mv_str}", px+8, py+130, 0.38, mc)
    if motion.moving:
        cv2.circle(frame, (px+pw-20, py+126), 7, C['yellow'], -1)

    # Effective bonus = base × multiplier
    eff = stream.dynamic_bonus * (0.60 if motion.moving else 1.20)
    bonus_str = f"base={stream.dynamic_bonus:.2f}  eff={eff:.2f}  need d>{stream.s_conf+eff:.2f}"
    _t(frame, bonus_str, px+8, py+152, 0.34, C['dim'])

    # Top-3 predictions của dynamic stream
    if stream.d_topk:
        tk_str = "  ".join(f"{l}:{c*100:.0f}%" for l, c in stream.d_topk[:3])
        _t(frame, f"d-top3: {tk_str}", px+8, py+170, 0.32, C['dim'])


def draw_ui(frame, stream, motion, fps, entries, conf_thr,
            notif, notif_ts, show_debug, force_mode, dynamic_bonus):
    """
    Vẽ toàn bộ UI lên frame:
      - Header: tên app, badge S/D với conf%, motion indicator, FPS
      - Debug panel (nếu show_debug=True)
      - Prediction block ở giữa dưới (label lớn + confidence bar)
      - Top-3 dynamic predictions (mờ)
      - Recent log panel bên phải
      - Footer: gợi ý phím tắt
      - Toast notification (tạm thời)
    """
    H, W = frame.shape[:2]
    cx = W // 2   # tâm ngang của frame

    label      = stream.label
    conf       = stream.conf
    model_type = stream.model_type

    # ── Header (trên) & Footer bg (dưới) ─────────────────────────
    _ar(frame, 0, 0, W, 64, C['bg1'], 0.88)
    _ar(frame, 0, H-72, W, H, C['bg1'], 0.88)

    # Tên app
    _t(frame, "VSL", 14, 42, 1.0, C['white'], 2)
    _t(frame, "DUAL",   76, 26, 0.48, C['dim'])
    _t(frame, "STREAM", 76, 44, 0.40, C['dim'])

    # Badge Stream S (xanh lá)
    sc_s = C['green']  if stream.s_conf >= conf_thr else C['dim']
    _rr(frame, 155, 8, 268, 56, C['green_d'], 0.85, r=7)
    cv2.rectangle(frame, (155, 8), (268, 56), sc_s, 1)
    _t(frame, "S",  168, 28, 0.40, sc_s)
    _t(frame, f"{stream.s_conf*100:.0f}%", 162, 50, 0.38, sc_s)
    _t(frame, (stream.s_label or "--")[:10], 188, 38, 0.42, sc_s)

    # Badge Stream D (cam) + buffer fill mini
    sc_d    = C['orange'] if (stream.d_ready and stream.d_conf >= conf_thr) else C['dim']
    buf_pct = len(stream.ring_buf) / stream.seq_len
    _rr(frame, 275, 8, 410, 56, C['orange_d'], 0.85, r=7)
    cv2.rectangle(frame, (275, 8), (410, 56), sc_d, 1)
    _t(frame, "D",  288, 28, 0.40, sc_d)
    _t(frame, f"{stream.d_conf*100:.0f}%", 282, 50, 0.38, sc_d)
    _t(frame, (stream.d_label or "--")[:12], 308, 38, 0.42, sc_d)
    _bar(frame, 275, 50, 130, 5, buf_pct, sc_d)   # buffer warmup progress

    # Gap indicator — hiện khi đang đếm frames mất tay
    if stream.gap_count > 0:
        gc = C['red'] if stream.gap_count >= GAP_RESET_THRESH else C['yellow']
        _t(frame, f"gap {stream.gap_count}", 415, 36, 0.34, gc)

    # Motion indicator
    if motion.moving:
        cv2.circle(frame, (490, 32), 8, C['yellow'], -1)
        _t(frame, "MOTION", 505, 36, 0.38, C['yellow'])

    # Force mode badge (góc phải header)
    if force_mode:
        fm_c = C['green'] if force_mode == 'static' else C['orange']
        _rr(frame, W-200, 8, W-8, 56, _MBG.get(force_mode, C['bg2']), 0.88, r=6)
        cv2.rectangle(frame, (W-200, 8), (W-8, 56), fm_c, 1)
        _t(frame, f"FORCE: {force_mode.upper()}", W-192, 36, 0.44, fm_c)

    # FPS (xanh nếu ≥24, vàng nếu ≥12, đỏ nếu chậm)
    fc = C['green'] if fps >= 24 else (C['yellow'] if fps >= 12 else C['red'])
    _t(frame, f"{fps:.0f}fps", W-90, 28, 0.50, fc)
    _t(frame, f"thr {conf_thr:.2f}", W-90, 50, 0.36, C['dim'])

    # ── Debug panel (bật/tắt bằng phím G) ────────────────────────
    if show_debug:
        draw_dual_stream_debug(frame, stream, motion, W, py_start=68)

    # ── Prediction block (label lớn ở giữa dưới) ─────────────────
    above = conf >= conf_thr and bool(label)

    if above and model_type:
        mc  = _MC.get(model_type, C['white'])
        mbg = _MBG.get(model_type, C['bg2'])

        # Scale chữ theo chiều rộng frame
        scale_lbl = min(1.8, max(0.9, W / 480))
        (tw, _), _ = cv2.getTextSize(label, FONT, scale_lbl, 2)

        # Hộp chứa label
        bx1, bx2 = cx - tw//2 - 50, cx + tw//2 + 50
        by1, by2 = H - 215, H - 90

        _rr(frame, bx1, by1, bx2, by2, mbg, 0.90, r=18)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), mc, 2)
        cv2.rectangle(frame, (bx1, by1), (bx2, by1+5), mc, -1)   # viền trên màu solid

        # Label text
        _t(frame, label, cx - tw//2, by2-35, scale_lbl, C['white'], 2)

        # Confidence bar
        bw = bx2 - bx1 - 24
        _bar(frame, bx1+12, by2-22, bw, 11, conf, mc)
        _t(frame, f"{conf*100:.1f}%", bx2+14, by2-14, 0.52, mc)

        # Badge "[ STATIC ]" hoặc "[ DYNAMIC ]" phía trên hộp
        badge = f"[ {'STATIC' if model_type=='static' else 'DYNAMIC'} ]"
        (bw2, _), _ = cv2.getTextSize(badge, FONT, 0.40, 1)
        _rr(frame, bx1, by1-32, bx1+bw2+18, by1-4, mbg, 0.90, r=5)
        cv2.rectangle(frame, (bx1, by1-32), (bx1+bw2+18, by1-4), mc, 1)
        _t(frame, badge, bx1+9, by1-13, 0.40, mc)

        # Hiện margin dynamic vs static (chỉ khi dynamic thắng)
        if model_type == 'dynamic':
            margin = stream.d_conf - stream.s_conf
            _t(frame, f"margin +{margin*100:.1f}% vs static", bx1, by1-50, 0.34, C['dim'])

    elif not label:
        # Hint khi không có prediction
        if not motion.wrist_ok:
            hint = "Dung truoc camera..."
        elif stream.s_conf < conf_thr and stream.d_conf < conf_thr:
            hint = "Thuc hien ky hieu..."
        else:
            hint = ""
        if hint:
            _t(frame, hint, cx-100, H//2, 0.60, C['dim'])

    # ── Top-3 dynamic (mờ dần, chỉ khi dynamic không phải kết quả chính) ─
    if stream.d_ready and stream.d_topk and (not above or model_type != 'dynamic'):
        ty = H - 80 if not above else H - 240
        for i, (lb, cf) in enumerate(stream.d_topk[:3]):
            alpha = 1.0 - i * 0.25   # mờ dần theo rank
            col   = tuple(int(x * alpha) for x in C['orange'])
            _t(frame, f"{lb} {cf*100:.0f}%", cx - 60 + i*120, ty, 0.36, col)

    # ── Recent log panel (bên phải) ───────────────────────────────
    if entries:
        rx = W - 295
        rh = min(len(entries), 9) * 30 + 34
        _ar(frame, rx-8, 70, W-4, 70+rh, C['bg2'], 0.78)
        cv2.rectangle(frame, (rx-8, 70), (W-4, 70+rh), C['border'], 1)
        _t(frame, f"RECENT ({len(entries)})", rx, 88, 0.36, C['dim'])

        # Hiện tối đa 9 entry gần nhất, mờ dần theo thời gian
        for i, e in enumerate(reversed(entries[-9:])):
            ey   = 108 + i * 30
            mc   = _MC.get(e['model'], C['dim'])
            fade = max(0.25, 1.0 - i * 0.09)   # entry cũ hơn mờ hơn
            icon = "S" if e['model'] == 'static' else "D"

            _rr(frame, rx, ey-14, rx+24, ey+10, _MBG.get(e['model'], C['bg2']), 0.85, r=4)
            _t(frame, icon,      rx+7,  ey+6, 0.34, mc)
            _t(frame, e['label'], rx+30, ey+6, 0.42, tuple(int(x*fade) for x in C['white']))
            _t(frame, f"{e['conf']*100:.0f}%", W-48, ey+6, 0.38,
               tuple(int(x*fade) for x in mc))
            _bar(frame, rx+30, ey+10, 150, 4, e['conf'],
                 tuple(int(x*fade) for x in mc), C['bg2'])

    # ── Footer: gợi ý phím tắt [FIX #1: 0/1/2 thay D/S] ─────────
    hints = [
        ("Q",   "Thoat"),
        ("G",   "Debug"),
        ("F",   "Full"),
        ("C",   "Clear"),
        ("S",   "Shot"),
        ("+/-", "Thr"),
        ("[/]", "Bonus"),
        ("0",   "Auto"),
        ("1",   "ForceS"),
        ("2",   "ForceD"),
    ]
    hx = 12
    for k, v in hints:
        kw = cv2.getTextSize(k, FONT, 0.34, 1)[0][0]
        vw = cv2.getTextSize(v, FONT, 0.30, 1)[0][0]
        _rr(frame, hx, H-56, hx+kw+10, H-34, C['bg2'], 0.88, r=4)
        cv2.rectangle(frame, (hx, H-56), (hx+kw+10, H-34), C['border'], 1)
        _t(frame, k, hx+5, H-38, 0.34, C['yellow'])
        hx += kw + 14
        _t(frame, v, hx, H-38, 0.30, C['dim'])
        hx += vw + 18
    _t(frame, f"bonus={dynamic_bonus:.2f}", W-130, H-42, 0.34, C['dim'])

    # ── Toast notification (2.5 giây, fade in/out) ────────────────
    if notif and (time.time() - notif_ts < 2.5):
        elapsed = time.time() - notif_ts
        # Fade in 0.15s, fade out 0.3s cuối
        fade = min(1.0, min(elapsed / 0.15, (2.5 - elapsed) / 0.3))
        (nw, _), _ = cv2.getTextSize(notif, FONT, 0.58, 1)
        nx, ny = cx - nw//2, H//2 + 50
        _rr(frame, nx-20, ny-30, nx+nw+20, ny+16, C['bg2'], min(0.92, fade*0.92), r=10)
        cv2.rectangle(frame, (nx-20, ny-30), (nx+nw+20, ny-26), C['yellow'], -1)
        _t(frame, notif, nx, ny+8, 0.58, tuple(int(x*fade) for x in C['yellow']))


# ══════════════════════════════════════════════════════════════════
# LOAD CHECKPOINT
# ══════════════════════════════════════════════════════════════════

def _latest(pat):
    """Tìm file checkpoint mới nhất khớp pattern glob."""
    files = glob.glob(str(_PROJECT_ROOT / 'checkpoints' / pat))
    return max(files, key=os.path.getmtime) if files else None


def load_static(path, device):
    """Load StaticMLP từ checkpoint. Trả về (model, label_map)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    lm = ck['label_map']
    cf = ck.get('cfg', {})
    m  = StaticMLP(cf.get('FEAT_DIM', 96), len(lm),
                   cf.get('HIDDEN_1', 256), cf.get('HIDDEN_2', 128), cf.get('HIDDEN_3', 64))
    m.load_state_dict(ck['model_state'])
    m.to(device).eval()
    print(f"  [Static]  {Path(path).name} | classes={len(lm)} | acc={ck.get('val_acc',0)*100:.1f}%")
    return m, lm


def load_dynamic(path, device):
    """Load BiLSTMClassifier từ checkpoint. Trả về (model, label_map)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    lm = ck['label_map']
    cf = ck.get('cfg', {})
    m  = BiLSTMClassifier(
        cf.get('FEAT_DIM', vsl_cfg.FEAT_DIM),
        cf.get('HIDDEN_DIM', 256),
        cf.get('NUM_LAYERS', 3),
        len(lm)
    )
    m.load_state_dict(ck['model_state'])
    m.to(device).eval()
    print(f"  [Dynamic] {Path(path).name} | classes={len(lm)} | acc={ck.get('val_acc',0)*100:.1f}%")
    return m, lm


# ══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="VSL Hybrid DualStream v5 — Static + Dynamic chay song song")
    ap.add_argument('--static_ckpt',  default=None,  help='Đường dẫn checkpoint StaticMLP')
    ap.add_argument('--dynamic_ckpt', default=None,  help='Đường dẫn checkpoint BiLSTM')
    ap.add_argument('--conf',  type=float, default=0.60, help='Confidence threshold')
    ap.add_argument('--bonus', type=float, default=0.15, help='Dynamic bonus (base)')
    ap.add_argument('--source', default='0', help='Camera index hoặc video path')
    ap.add_argument('--debug',  action='store_true',  help='Bật debug panel ngay khi khởi động')
    args = ap.parse_args()

    device   = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEQ_LEN  = vsl_cfg.SEQ_LEN
    FEAT_DIM = vsl_cfg.FEAT_DIM

    print(f"\n{'='*62}")
    print("  VSL Hybrid DualStream v5  |  Static + Dynamic song song")
    print(f"{'='*62}")
    print(f"  Device      : {device}")
    print(f"  Conf thr    : {args.conf}")
    print(f"  D-bonus     : {args.bonus}  (adaptive theo motion)")
    print(f"  SEQ_LEN     : {SEQ_LEN}")
    print(f"  GAP_RESET   : {GAP_RESET_THRESH} frames  (~{GAP_RESET_THRESH/30:.2f}s @ 30fps)")
    print(f"  HAND_GONE   : {HAND_GONE_FRAMES} frames  (~{HAND_GONE_FRAMES/30:.2f}s @ 30fps)")
    print()

    # ── Load models ───────────────────────────────────────────────
    s_ckpt = args.static_ckpt or _latest('static_mlp_best.pt')
    if not s_ckpt or not os.path.exists(s_ckpt):
        print("  [ERROR] Khong tim thay static checkpoint")
        sys.exit(1)
    smodel, smap = load_static(s_ckpt, device)
    s_i2l = {v: k for k, v in smap.items()}   # đảo label_map: index → string

    d_ckpt = args.dynamic_ckpt or _latest('bilstm_best.pt')
    if not d_ckpt or not os.path.exists(d_ckpt):
        print("  [ERROR] Khong tim thay bilstm checkpoint")
        sys.exit(1)
    dmodel, dmap = load_dynamic(d_ckpt, device)
    d_i2l = {v: k for k, v in dmap.items()}

    # ── Khởi tạo MediaPipe extractor ─────────────────────────────
    print("\n  Khoi tao MediaPipe...")
    ext = RealtimeExtractor()   # chạy trong thread riêng

    # ── Khởi tạo DualStream và MotionDetector ────────────────────
    stream = DualStream(
        smodel, s_i2l, dmodel, d_i2l,
        device=device, seq_len=SEQ_LEN, feat_dim=FEAT_DIM,
        dynamic_bonus=args.bonus,
        conf_thr=args.conf
    )
    motion = MotionDetector(threshold=0.003)

    # ── Mở camera hoặc video file ─────────────────────────────────
    src = args.source
    try:
        src = int(src)   # thử parse thành camera index
    except ValueError:
        pass   # giữ nguyên string (video path)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"  [ERROR] Khong mo duoc camera: {src}")
        sys.exit(1)

    CW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    CH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera      : {CW}x{CH}")

    # ── Khởi tạo biến trạng thái ─────────────────────────────────
    logger        = HybridLogger()
    fps_buf       = deque(maxlen=30)   # rolling average FPS
    t_prev        = time.time()
    entries       = []                 # list entry cho recent log panel

    notif, notif_ts = '', 0.0          # toast notification
    show_debug    = args.debug
    conf_thr      = args.conf
    dynamic_bonus = args.bonus
    force_mode    = None               # None | 'static' | 'dynamic'
    fullscreen    = False
    tick          = 0                  # đếm frame để log debug thưa hơn

    ss_dir = _PROJECT_ROOT / 'screenshots'
    ss_dir.mkdir(exist_ok=True)

    WIN = "VSL Hybrid DualStream v5"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, CW, CH)

    print(f"\n  Phim: Q=Thoat  G=Debug  F=Full  S=Screenshot")
    print(f"        0=Auto  1=ForceStatic  2=ForceDynamic")
    print(f"        +/-=Thr  [/]=Bonus")
    print(f"  {'─'*55}\n")

    # ══════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ══════════════════════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            break   # hết video hoặc camera mất kết nối
        frame = cv2.flip(frame, 1)   # mirror để tự nhiên hơn

        # ── Tính FPS (rolling average 30 frames) ─────────────────
        t_now = time.time()
        fps_buf.append(1.0 / max(t_now - t_prev, 1e-9))
        t_prev = t_now
        fps = float(np.mean(fps_buf))
        tick += 1

        # ── Trích xuất features từ MediaPipe ─────────────────────
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ext.send_frame(rgb)                          # gửi frame vào extractor async
        feat_346 = ext.extract_features()            # None nếu chưa có kết quả
        latest   = ext.get_latest()                  # dict: hands, pose, face
        left_h   = latest['hands'][0] if latest['hands'] else None
        right_h  = latest['hands'][1] if latest['hands'] else None
        pose_lms = latest['pose']

        # Ưu tiên tay phải, fallback tay trái
        hand_lms = right_h if right_h else left_h

        # ── Cập nhật models ───────────────────────────────────────
        stream.conf_thr      = conf_thr      # cho phép thay đổi runtime
        stream.dynamic_bonus = dynamic_bonus
        motion.update(pose_lms)

        # Dispatch đến đúng mode [FIX #4]
        if force_mode == 'static':
            stream.step_force_static(feat_346, hand_lms, motion_active=motion.moving)
        elif force_mode == 'dynamic':
            stream.step_force_dynamic(feat_346, hand_lms, motion_active=motion.moving)
        else:
            stream.step_auto(feat_346, hand_lms, motion_active=motion.moving)

        label, conf, mtype = stream.label, stream.conf, stream.model_type

        # ── Log prediction (chỉ khi label thay đổi) ──────────────
        if label and conf >= conf_thr and mtype:
            logger.log(mtype, label, conf)
            # Thêm vào recent entries nếu label mới
            if not entries or entries[-1]['label'] != label:
                entries.append({'model': mtype, 'label': label, 'conf': conf})
                if len(entries) > 20:
                    entries.pop(0)   # giữ tối đa 20 entries

        # ── Debug terminal (mỗi 30 frames = ~1s) ─────────────────
        if args.debug and tick % 30 == 0:
            print(
                f"  [DBG {tick:05d}]"
                f"  S={stream.s_label}({stream.s_conf*100:.0f}%)"
                f"  D={stream.d_label}({stream.d_conf*100:.0f}%)"
                f"  d_ready={stream.d_ready}"
                f"  buf={len(stream.ring_buf)}/{SEQ_LEN}"
                f"  gap={stream.gap_count}  gone={stream.hand_gone}"
                f"  motion={motion.moving}(v={motion.velocity:.4f})"
                f"  OUT=[{mtype}]{label}({conf*100:.0f}%)"
            )

        # ── Vẽ UI lên frame ───────────────────────────────────────
        draw_ui(frame, stream, motion, fps, entries, conf_thr,
                notif, notif_ts, show_debug, force_mode, dynamic_bonus)

        cv2.imshow(WIN, frame)

        # ── Xử lý phím [FIX #1] ──────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):      # Q / ESC → thoát
            break

        elif key in (ord('g'), ord('G')):         # G → toggle debug panel
            show_debug = not show_debug
            notif, notif_ts = f"Debug: {'ON' if show_debug else 'OFF'}", time.time()

        elif key in (ord('f'), ord('F')):         # F → toggle fullscreen
            fullscreen = not fullscreen
            p = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, p)
            notif, notif_ts = f"Fullscreen: {'ON' if fullscreen else 'OFF'}", time.time()

        elif key in (ord('c'), ord('C')):         # C → xóa log
            logger.clear()
            entries.clear()
            notif, notif_ts = "Log cleared", time.time()

        elif key in (ord('s'), ord('S')):         # S → screenshot [FIX #1: luôn hoạt động]
            ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            path = str(ss_dir / f'dual_{ts}.png')
            cv2.imwrite(path, frame)
            notif, notif_ts = f"Shot: dual_{ts}.png", time.time()

        elif key == ord('0'):                      # 0 → AUTO mode [FIX #1]
            force_mode = None
            notif, notif_ts = "Force: AUTO", time.time()

        elif key == ord('1'):                      # 1 → Force STATIC [FIX #1]
            force_mode = None if force_mode == 'static' else 'static'
            notif, notif_ts = f"Force: {force_mode or 'AUTO'}", time.time()

        elif key == ord('2'):                      # 2 → Force DYNAMIC [FIX #1]
            force_mode = None if force_mode == 'dynamic' else 'dynamic'
            notif, notif_ts = f"Force: {force_mode or 'AUTO'}", time.time()

        elif key in (ord('+'), ord('=')):          # + → tăng confidence threshold
            conf_thr = min(0.99, round(conf_thr + 0.05, 2))
            notif, notif_ts = f"Conf thr -> {conf_thr:.2f}", time.time()

        elif key == ord('-'):                      # - → giảm confidence threshold
            conf_thr = max(0.05, round(conf_thr - 0.05, 2))
            notif, notif_ts = f"Conf thr -> {conf_thr:.2f}", time.time()

        elif key == ord('['):                      # [ → giảm dynamic bonus
            dynamic_bonus = max(0.0, round(dynamic_bonus - 0.05, 2))
            stream.dynamic_bonus = dynamic_bonus
            notif, notif_ts = f"D-bonus -> {dynamic_bonus:.2f}", time.time()

        elif key == ord(']'):                      # ] → tăng dynamic bonus
            dynamic_bonus = min(0.5, round(dynamic_bonus + 0.05, 2))
            stream.dynamic_bonus = dynamic_bonus
            notif, notif_ts = f"D-bonus -> {dynamic_bonus:.2f}", time.time()

    # ── Cleanup ───────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    ext.close()
    logger.summary()
    print("  Thoat.")


if __name__ == '__main__':
    main()