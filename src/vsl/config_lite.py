"""
vsl/config_lite.py - Lightweight config (171 dim)
"""

class ConfigLite:
    SEQ_LEN = 64
    
    # POSE: 15 điểm × 3 = 45 dim
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
    
    # HANDS: 21 × 2 × 3 = 126 dim
    HAND_LANDMARKS = 21
    NUM_HANDS = 2
    HAND_DIM = 126
    
    # BỎ face và interactions
    FACE_DIM = 0
    FACE_KEY_INDICES = []
    INTERACT_DIM = 0
    
    # TỔNG
    FEAT_DIM = 171  # 45 + 126
    
    # INDEX RANGES
    POSE_START = 0
    POSE_END = 45
    HAND_START = 45
    HAND_END = 171
    LEFT_HAND_START = 45
    LEFT_HAND_END = 108
    RIGHT_HAND_START = 108
    RIGHT_HAND_END = 171
    FACE_START = FACE_END = 171
    INTERACT_START = INTERACT_END = 171

cfg = ConfigLite()