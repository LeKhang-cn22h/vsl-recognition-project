"""
vsl_config_v3.py - Config chung cho toàn bộ pipeline VSL
=========================================================
FEATURES: 178 dim
  - pose:    45 dim (15 điểm × 3)
  - hands:  126 dim (21 × 2 × 3)
  - emotion:  7 dim (one-hot, do người dùng chọn khi quay)

Emotions: angry, disgust, fear, happy, sad, surprise, neutral
"""

import numpy as np


class VSLConfig:
    # ─── Sequence ───
    SEQ_LEN = 64

    # ─── Pose: 15 điểm quan trọng ───
    POSE_KEY_INDICES = [
        0,          # nose
        11, 12,     # shoulders
        13, 14,     # elbows
        15, 16,     # wrists
        17, 18,     # pinky
        19, 20,     # index
        21, 22,     # thumb
        23, 24,     # hips
    ]
    POSE_DIM = 15 * 3  # 45

    # ─── Hands ───
    HAND_LANDMARKS = 21
    NUM_HANDS = 2
    HAND_DIM = HAND_LANDMARKS * NUM_HANDS * 3  # 126

    # ─── Emotion ───
    EMOTIONS = {
        "angry":    0,
        "disgust":  1,
        "fear":     2,
        "happy":    3,
        "sad":      4,
        "surprise": 5,
        "neutral":  6,
    }
    EMOTION_DIM = len(EMOTIONS)  # 7

    # ─── Tổng features ───
    FEAT_DIM = POSE_DIM + HAND_DIM + EMOTION_DIM  # 45 + 126 + 7 = 178

    # ─── Index ranges ───
    POSE_START, POSE_END = 0, 45
    HAND_START, HAND_END = 45, 171
    LEFT_HAND_START,  LEFT_HAND_END  = 45,  108
    RIGHT_HAND_START, RIGHT_HAND_END = 108, 171
    EMOTION_START, EMOTION_END = 171, 178

    # ─── Paths ───
    VIDEO_DIR      = "videos"
    DATA_DIR       = "data/processed"
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR        = "logs"

    # ─── Training ───
    BATCH_SIZE = 32
    EPOCHS     = 100
    LR         = 1e-3
    PATIENCE   = 15

    # ─── Model ───
    HIDDEN_DIM = 256
    NUM_LAYERS = 3
    DROPOUT    = 0.3

    @classmethod
    def print_info(cls):
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║                   VSL CONFIG v3                           ║
╠═══════════════════════════════════════════════════════════╣
║  FEATURES: {cls.FEAT_DIM} dim                                       ║
║    ├── pose:    {cls.POSE_DIM} dim  (0:{cls.POSE_END})                       ║
║    ├── hands:  {cls.HAND_DIM} dim ({cls.HAND_START}:{cls.HAND_END})                     ║
║    └── emotion:  {cls.EMOTION_DIM} dim ({cls.EMOTION_START}:{cls.EMOTION_END})                     ║
║                                                           ║
║  SEQUENCE: {cls.SEQ_LEN} frames                                     ║
║  EMOTIONS: {list(cls.EMOTIONS.keys())}   ║
╚═══════════════════════════════════════════════════════════╝
        """)


cfg = VSLConfig()


# ═══════════════════════════════════════════════════════════════════
# EMOTION ENCODING / DECODING
# ═══════════════════════════════════════════════════════════════════

def encode_emotion(emotion_name: str) -> np.ndarray:
    """Emotion name → one-hot vector (7 dim)."""
    vec = np.zeros(cfg.EMOTION_DIM, dtype=np.float32)
    vec[cfg.EMOTIONS.get(emotion_name, cfg.EMOTIONS["neutral"])] = 1.0
    return vec


def decode_emotion(emotion_vec: np.ndarray) -> str:
    """One-hot vector → emotion name."""
    idx = int(np.argmax(emotion_vec))
    for name, i in cfg.EMOTIONS.items():
        if i == idx:
            return name
    return "neutral"


def get_emotion_from_filename(filename: str) -> str:
    """Parse emotion từ tên file: label_emotion_timestamp.mp4"""
    import os
    name = os.path.splitext(os.path.basename(filename))[0]
    for emotion in cfg.EMOTIONS:
        if f"_{emotion}_" in name or name.endswith(f"_{emotion}"):
            return emotion
    return "neutral"


# ═══════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg.print_info()
    print("  Testing emotion encoding:")
    for emo in cfg.EMOTIONS:
        vec = encode_emotion(emo)
        decoded = decode_emotion(vec)
        print(f"    {emo:10s} → idx={cfg.EMOTIONS[emo]}  decoded={decoded}")