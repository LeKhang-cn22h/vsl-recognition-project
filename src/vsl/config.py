"""
vsl/config.py - Cấu hình toàn bộ hệ thống VSL
================================================
Import ở bất kỳ đâu:
    from vsl.config import cfg, FACE_KEY_INDICES, KEY_BLENDSHAPES
"""

import torch

# ── MediaPipe model download URLs ──
MODEL_URLS = {
    'hand_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'),
    'pose_landmarker_heavy.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'),
    'face_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'face_landmarker/face_landmarker/float16/1/face_landmarker.task'),
}

# ── 30 face landmark indices quan trọng cho VSL ──
FACE_KEY_INDICES = [
    # Lông mày (10)
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    # Mắt (8)
    33, 159, 145, 133,
    263, 386, 374, 362,
    # Miệng (8)
    13, 14, 61, 291, 0, 17, 78, 308,
    # Tham chiếu (4)
    1, 4, 10, 152,
]

# ── 17 blendshapes quan trọng cho VSL ──
KEY_BLENDSHAPES = [
    'jawOpen',
    'mouthSmileLeft', 'mouthSmileRight',
    'mouthFrownLeft', 'mouthFrownRight',
    'mouthPucker', 'cheekPuff',
    'eyeBlinkLeft', 'eyeBlinkRight',
    'eyeWideLeft', 'eyeWideRight',
    'eyeSquintLeft', 'eyeSquintRight',
    'browInnerUp',
    'browDownLeft', 'browDownRight',
    'noseSneerLeft',
]
# 14 vùng mặt — face landmark indices
# Dùng trong compute_interactions()
FACE_REGION_INDICES = {
    'tran':          [10, 67, 297],       # trán
    'thai_duong_T':  [162, 21],           # thái dương trái
    'thai_duong_P':  [389, 251],          # thái dương phải
    'chan_may_T':    [70, 63, 105],       # chân mày trái
    'chan_may_P':    [300, 293, 334],     # chân mày phải
    'mat_T':         [159, 145],          # mắt trái
    'mat_P':         [386, 374],          # mắt phải
    'mui':           [4, 6],              # mũi
    'ma_T':          [50, 117],           # má trái
    'ma_P':          [280, 346],          # má phải
    'mieng':         [13, 14],            # miệng
    'cam':           [152, 175],          # cằm
    'lo_tai_T':      [234, 93],           # lỗ tai trái
    'lo_tai_P':      [454, 323],          # lỗ tai phải
}
FACE_REGION_NAMES = list(FACE_REGION_INDICES.keys())  # 14 vùng

# 8 vùng body — pose landmark indices
# Dùng trong compute_interactions()
BODY_REGION_INDICES = {
    'dau':       [0],           # đầu
    'vai_T':     [11],          # vai trái
    'vai_P':     [12],          # vai phải
    'nguc':      [11, 12],      # ngực (giữa 2 vai)
    'khuyu_T':   [13],          # khuỷu tay trái
    'khuyu_P':   [14],          # khuỷu tay phải
    'hong_T':    [23],          # hông trái
    'hong_P':    [24],          # hông phải
}
BODY_REGION_NAMES = list(BODY_REGION_INDICES.keys())  # 8 vùng

# Ngưỡng "chạm" — dùng cho binary flag
TOUCH_THRESHOLD = 0.06

class Config:
    # ── Sequence ──
    SEQ_LEN  = 64
    FEAT_DIM = 346  # 75 + 90 + 126 + 0 (bỏ blend) + 55

    # ── Feature layout (start:end index trong vector 346) ──
    POSE_START,     POSE_END     = 0,   75
    FACE_START,     FACE_END     = 75,  165
    HAND_START,     HAND_END     = 165, 291
    BLEND_START,    BLEND_END    = 291, 291
    INTERACT_START, INTERACT_END = 291, 346

    # ── Model architecture ──
    D_MODEL           = 256
    SPATIAL_HEADS     = 8
    SPATIAL_LAYERS    = 3
    SPATIAL_FF_DIM    = 512
    SPATIAL_DROPOUT   = 0.1
    TEMPORAL_HEADS    = 8
    TEMPORAL_LAYERS   = 4
    TEMPORAL_FF_DIM   = 512
    TEMPORAL_DROPOUT  = 0.1
    CLASSIFIER_HIDDEN = 256
    DROPOUT_FINAL     = 0.3

    # ── Device ──
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Inference defaults ──
    CONFIDENCE_THR  = 0.60
    MOTION_THR      = 0.015
    CONSEC_THR      = 3
    SMOOTH_WINDOW   = 5
    TOP_K           = 5


cfg = Config()