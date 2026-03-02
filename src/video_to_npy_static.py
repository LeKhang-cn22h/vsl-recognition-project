"""
video_to_npy_static.py - Trích xuất đặc trưng bàn tay cho Static MLP
======================================================================
Chỉ dùng 21 hand landmarks từ MediaPipe → vector 96 dims

Feature vector (96 dims):
  [0 :63] - 21 landmarks × 3 (x,y,z) đã normalize về wrist=origin
  [63:78] - 15 joint angles  (góc từng đốt ngón tay)
  [78:83] - 5 finger lengths (độ dài từng ngón)
  [83:93] - 10 fingertip distances (khoảng cách đầu ngón với nhau)
  [93:96] - 3 palm normal vector (hướng lòng bàn tay)

Cấu trúc thư mục input:
  datamlp/raw_static/
  ├── a/
  │   ├── video1.mp4
  │   ├── video2.mp4
  │   └── ...
  ├── b/
  └── ...

Cấu trúc thư mục output:
  datamlp/static/
  ├── label_map.json
  ├── train/
  │   ├── a/  *.npy  (shape: 96,)
  │   ├── b/  *.npy
  │   └── ...
  ├── val/
  └── test/

Chạy:
  python src/static/video_to_npy_static.py
  python src/static/video_to_npy_static.py --raw_dir datamlp/raw_static --out_dir datamlp/static
"""

import os
import sys
import json
import argparse
import math
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from pathlib import Path
from itertools import combinations

# ── Fix sys.path ──────────────────────────────────────────────────
# File ở src/video_to_npy_static.py → parents[1] = vsl-recognition-project/
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / 'src')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request

# ── Download hand_landmarker.task nếu chưa có ────────────────────
_HAND_MODEL_URL  = (
    'https://storage.googleapis.com/mediapipe-models/'
    'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
)
_HAND_MODEL_PATH = str(_PROJECT_ROOT / 'hand_landmarker.task')

def _ensure_hand_model():
    if not os.path.exists(_HAND_MODEL_PATH):
        print(f"  Dang tai hand_landmarker.task ...")
        urllib.request.urlretrieve(_HAND_MODEL_URL, _HAND_MODEL_PATH)
        print(f"  Da tai xong: {_HAND_MODEL_PATH}")
    return _HAND_MODEL_PATH

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════

FEAT_DIM = 96   # tổng số đặc trưng

# 21 landmarks của MediaPipe Hand
# Kết nối giải phẫu học để tính góc
FINGER_JOINTS = {
    'thumb':  [1, 2, 3, 4],      # CMC→MCP→IP→TIP
    'index':  [5, 6, 7, 8],      # MCP→PIP→DIP→TIP
    'middle': [9, 10, 11, 12],
    'ring':   [13, 14, 15, 16],
    'pinky':  [17, 18, 19, 20],
}

# Đầu ngón tay (để tính fingertip distances)
FINGERTIPS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky

# Gốc mỗi ngón (để tính finger length)
FINGER_BASES = [2, 5, 9, 13, 17]  # thumb_MCP, index_MCP, ...


# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize 21 landmarks về wrist (landmark 0) = origin.
    landmarks: (21, 3)
    → dịch chuyển để wrist = (0,0,0)
    → chia cho scale = khoảng cách wrist đến middle_MCP (landmark 9)
      → bất biến với khoảng cách camera
    """
    lm = landmarks.copy()

    # Dịch về gốc wrist
    wrist = lm[0].copy()
    lm    = lm - wrist

    # Scale bất biến với khoảng cách
    scale = np.linalg.norm(lm[9])   # wrist → middle MCP
    if scale > 1e-6:
        lm = lm / scale

    return lm   # (21, 3)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Góc (radian) giữa 2 vector."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(math.acos(cos_a))


def compute_joint_angles(lm: np.ndarray) -> np.ndarray:
    """
    Tính 15 góc khớp ngón tay.
    Mỗi ngón 3 góc: góc tại MCP, PIP, DIP
    lm: (21, 3) đã normalize
    → return (15,) radian
    """
    angles = []
    for finger_name, joints in FINGER_JOINTS.items():
        # joints = [A, B, C, D] → tính góc tại B và C
        # Góc tại B = angle(A→B, B→C)
        # Góc tại C = angle(B→C, C→D)
        for i in range(len(joints) - 2):
            a = lm[joints[i]]
            b = lm[joints[i+1]]
            c = lm[joints[i+2]]
            v1 = a - b
            v2 = c - b
            angles.append(angle_between(v1, v2))
    # Thumb: 3 góc (joints có 4 điểm → 2 góc bên trong)
    # Index/Middle/Ring/Pinky: mỗi cái 2 góc
    # Tổng: thumb=2, index=2, middle=2, ring=2, pinky=2 → chỉ 10?
    # → thêm góc giữa các phalanx liền kề để đủ 15
    # Cách đơn giản hơn: lấy 3 góc mỗi ngón (MCP flex, PIP flex, DIP flex)
    return np.array(angles, dtype=np.float32)


def compute_joint_angles_v2(lm: np.ndarray) -> np.ndarray:
    """
    Tính 15 góc theo cách rõ ràng hơn:
    Mỗi ngón 3 đoạn → 2 góc uốn cong
    Thêm góc giữa các ngón = 5 góc dạng khác

    Cụ thể:
      5 ngón × 2 góc uốn (PIP + DIP) = 10
      5 góc mở rộng (abduction giữa các ngón liền kề) = 5
      Tổng = 15
    """
    angles = []

    finger_chains = [
        [0, 1, 2, 3, 4],    # thumb
        [0, 5, 6, 7, 8],    # index  (từ wrist)
        [0, 9, 10, 11, 12], # middle
        [0, 13, 14, 15, 16],# ring
        [0, 17, 18, 19, 20],# pinky
    ]

    # 5 ngón × 3 góc = 15
    # Mỗi chain có 5 điểm [A,B,C,D,E] → 3 góc tại B, C, D
    for chain in finger_chains:
        for i in range(1, 4):   # i=1,2,3 → góc tại chain[1], chain[2], chain[3]
            a = lm[chain[i-1]]
            b = lm[chain[i]]
            c = lm[chain[i+1]]
            angles.append(angle_between(a - b, c - b))

    return np.array(angles, dtype=np.float32)   # (15,)


def compute_finger_lengths(lm: np.ndarray) -> np.ndarray:
    """
    Tính 5 độ dài ngón tay (từ MCP đến fingertip).
    lm: (21, 3) đã normalize
    → return (5,)
    """
    lengths = []
    for base, tip in zip(FINGER_BASES, FINGERTIPS):
        lengths.append(float(np.linalg.norm(lm[tip] - lm[base])))
    return np.array(lengths, dtype=np.float32)   # (5,)


def compute_fingertip_distances(lm: np.ndarray) -> np.ndarray:
    """
    Tính 10 khoảng cách giữa các đầu ngón tay (C(5,2) = 10 pairs).
    lm: (21, 3) đã normalize
    → return (10,)
    """
    tips = lm[FINGERTIPS]   # (5, 3)
    dists = []
    for i, j in combinations(range(5), 2):
        dists.append(float(np.linalg.norm(tips[i] - tips[j])))
    return np.array(dists, dtype=np.float32)   # (10,)


def compute_palm_normal(lm: np.ndarray) -> np.ndarray:
    """
    Tính vector pháp tuyến lòng bàn tay (hướng lòng tay).
    Dùng 3 điểm: wrist(0), index_MCP(5), pinky_MCP(17)
    lm: (21, 3) đã normalize
    → return (3,) unit vector
    """
    v1 = lm[5]  - lm[0]   # wrist → index MCP
    v2 = lm[17] - lm[0]   # wrist → pinky MCP
    normal = np.cross(v1, v2)
    norm   = np.linalg.norm(normal)
    if norm > 1e-6:
        normal = normal / norm
    return normal.astype(np.float32)   # (3,)


def extract_hand_features(landmarks_21: np.ndarray) -> np.ndarray:
    """
    Pipeline đầy đủ: 21 landmarks → vector 96 dims.

    landmarks_21: (21, 3) tọa độ thô từ MediaPipe

    Returns: (96,) float32
      [0 :63] normalized xyz
      [63:78] 15 joint angles
      [78:83] 5 finger lengths
      [83:93] 10 fingertip distances
      [93:96] 3 palm normal
    """
    # 1. Normalize
    lm = normalize_landmarks(landmarks_21)   # (21, 3)

    # 2. Flatten normalized coords
    coords = lm.flatten()                    # (63,)

    # 3. Joint angles
    angles = compute_joint_angles_v2(lm)     # (15,)

    # 4. Finger lengths
    lengths = compute_finger_lengths(lm)     # (5,)

    # 5. Fingertip distances
    tip_dists = compute_fingertip_distances(lm)  # (10,)

    # 6. Palm normal
    palm_n = compute_palm_normal(lm)         # (3,)

    # 7. Concat
    feat = np.concatenate([coords, angles, lengths, tip_dists, palm_n])
    assert feat.shape == (96,), f"Expected (96,), got {feat.shape}"
    return feat.astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# MEDIAPIPE HAND DETECTOR
# ══════════════════════════════════════════════════════════════════

class HandExtractor:
    """
    Dùng MediaPipe Tasks API (mediapipe >= 0.10)
    Giống cách dùng trong webcam_collector.py
    """
    def __init__(self, max_hands=1, min_detection_confidence=0.5):
        model_path = _ensure_hand_model()

        # Static image mode: RUNNING_MODE = IMAGE (xử lý từng frame độc lập)
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

    def extract(self, frame_bgr: np.ndarray):
        """
        frame_bgr: OpenCV BGR frame
        Returns: (21, 3) numpy array hoặc None nếu không thấy tay
        """
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)

        if not result.hand_landmarks:
            return None

        # Lấy tay đầu tiên
        hand_lm = result.hand_landmarks[0]   # list of 21 NormalizedLandmark

        landmarks = np.array([
            [lm.x, lm.y, lm.z]
            for lm in hand_lm
        ], dtype=np.float32)   # (21, 3)

        return landmarks

    def close(self):
        self.detector.close()


# ══════════════════════════════════════════════════════════════════
# VIDEO PROCESSOR
# ══════════════════════════════════════════════════════════════════

def get_representative_frames(cap, n_frames=10):
    """
    Lấy n_frames frames đại diện từ video (trải đều).
    Tránh lấy frame đầu/cuối (có thể bị nhòe).
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Bỏ 20% đầu và 20% cuối
    start = max(0, int(total * 0.2))
    end   = min(total - 1, int(total * 0.8))

    if end <= start:
        indices = [total // 2]
    else:
        indices = np.linspace(start, end, n_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    return frames


def process_video(video_path: str, extractor: HandExtractor,
                  n_sample_frames: int = 10) -> np.ndarray | None:
    """
    Xử lý 1 video → 1 feature vector (96,).

    Chiến lược:
    1. Sample n_sample_frames frames từ phần giữa video
    2. Extract hand features từ mỗi frame
    3. Lấy TRUNG BÌNH các frames có tay → robust hơn lấy 1 frame
    4. Nếu không có frame nào thấy tay → return None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    [WARN] Khong mo duoc: {video_path}")
        return None

    frames = get_representative_frames(cap, n_sample_frames)
    cap.release()

    if not frames:
        print(f"    [WARN] Khong doc duoc frame: {video_path}")
        return None

    # Extract features từ từng frame
    feat_list = []
    for frame in frames:
        lm = extractor.extract(frame)
        if lm is not None:
            feat = extract_hand_features(lm)
            feat_list.append(feat)

    if not feat_list:
        print(f"    [WARN] Khong phat hien tay: {Path(video_path).name}")
        return None

    # Trung bình các frame → (96,)
    feat_avg = np.mean(feat_list, axis=0).astype(np.float32)

    detection_rate = len(feat_list) / len(frames)
    return feat_avg, detection_rate


# ══════════════════════════════════════════════════════════════════
# DATASET BUILDER
# ══════════════════════════════════════════════════════════════════

def build_dataset(videos_dir: str, out_dir: str, n_sample: int = 10):
    """
    Đọc từ cấu trúc đã có sẵn split:
        videos_dir/
        ├── train/<label>/*.mp4
        ├── val/<label>/*.mp4
        └── test/<label>/*.mp4

    → Extract features → lưu .npy vào:
        out_dir/
        ├── train/<label>/*.npy  (shape: 96,)
        ├── val/<label>/*.npy
        └── test/<label>/*.npy
    """
    videos_path = Path(videos_dir)
    out_path    = Path(out_dir)

    # Kiểm tra cấu trúc
    splits_available = [s for s in ['train', 'val', 'test']
                        if (videos_path / s).is_dir()]
    if not splits_available:
        print(f"[ERROR] Khong tim thay thu muc train/val/test trong: {videos_dir}")
        print(f"        Kiem tra lai duong dan!")
        sys.exit(1)

    # Thu thập tất cả labels từ split train (hoặc bất kỳ split nào có)
    primary_split = 'train' if 'train' in splits_available else splits_available[0]
    label_dirs    = sorted([
        d for d in (videos_path / primary_split).iterdir() if d.is_dir()
    ])
    if not label_dirs:
        print(f"[ERROR] Khong tim thay label nao trong: {videos_path / primary_split}")
        sys.exit(1)

    label_map = {d.name: i for i, d in enumerate(label_dirs)}

    print(f"\n{'='*60}")
    print(f"  STATIC FEATURE EXTRACTION")
    print(f"{'='*60}")
    print(f"  Videos dir : {videos_dir}")
    print(f"  Out dir    : {out_dir}")
    print(f"  Splits     : {splits_available}")
    print(f"  Labels     : {len(label_map)} → {list(label_map.keys())}")
    print(f"  Sample     : {n_sample} frames/video\n")

    extractor    = HandExtractor()
    total_ok     = 0
    total_fail   = 0
    split_counts = {name: {s: 0 for s in splits_available}
                    for name in label_map}

    for split in splits_available:
        print(f"\n  ── Split: {split.upper()} ──")
        split_in  = videos_path / split
        split_out = out_path    / split

        for label, label_idx in label_map.items():
            label_in  = split_in  / label
            label_out = split_out / label
            label_out.mkdir(parents=True, exist_ok=True)

            if not label_in.is_dir():
                print(f"    [{label}] SKIP (khong co trong {split})")
                continue

            videos = sorted(
                list(label_in.glob('*.mp4')) +
                list(label_in.glob('*.avi')) +
                list(label_in.glob('*.mov'))
            )

            if not videos:
                print(f"    [{label}] SKIP (khong co video)")
                continue

            ok, fail = 0, 0
            for vp in videos:
                result = process_video(str(vp), extractor, n_sample)
                if result is None:
                    fail += 1
                    total_fail += 1
                    continue

                feat, _ = result
                out_npy = label_out / f"{vp.stem}.npy"
                np.save(str(out_npy), feat)
                ok += 1
                total_ok += 1
                split_counts[label][split] += 1

            print(f"    [{label:20s}] {len(videos):3d} videos → "
                  f"OK={ok} Fail={fail}")

    extractor.close()

    # Lưu label_map
    lmap_path = out_path / 'label_map.json'
    with open(lmap_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # Lưu metadata
    meta = {
        'feat_dim'    : FEAT_DIM,
        'num_classes' : len(label_map),
        'label_map'   : label_map,
        'split_counts': split_counts,
        'total_ok'    : total_ok,
        'total_fail'  : total_fail,
        'feature_layout': {
            'normalized_xyz'     : '0:63',
            'joint_angles'       : '63:78',
            'finger_lengths'     : '78:83',
            'fingertip_distances': '83:93',
            'palm_normal'        : '93:96',
        }
    }
    with open(out_path / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  HOAN THANH!")
    print(f"  Tong OK   : {total_ok}")
    print(f"  Tong Fail : {total_fail}")
    print(f"  label_map : {lmap_path}")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════
# VERIFY FEATURE (debug)
# ══════════════════════════════════════════════════════════════════

def verify_feature_vector():
    """Test nhanh feature engineering với landmarks ngẫu nhiên."""
    print("  Verifying feature extraction...")
    dummy_lm = np.random.rand(21, 3).astype(np.float32)
    feat     = extract_hand_features(dummy_lm)
    print(f"  Input : (21, 3) landmarks")
    print(f"  Output: {feat.shape} → {feat.dtype}")
    print(f"  Range : [{feat.min():.3f}, {feat.max():.3f}]")

    # Kiểm tra từng phần
    print(f"  Coords  (0:63) : shape={feat[0:63].shape}")
    print(f"  Angles (63:78) : shape={feat[63:78].shape} "
          f"range=[{feat[63:78].min():.2f}, {feat[63:78].max():.2f}] rad")
    print(f"  Lengths(78:83) : shape={feat[78:83].shape}")
    print(f"  TipDist(83:93) : shape={feat[83:93].shape}")
    print(f"  PalmN  (93:96) : shape={feat[93:96].shape} "
          f"norm={np.linalg.norm(feat[93:96]):.3f}")
    print("  OK!\n")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Extract static hand features từ video')
    ap.add_argument('--videos_dir', default='datamlp/videos',
                    help='Thư mục chứa train/val/test videos '
                         '(default: datamlp/videos)')
    ap.add_argument('--out_dir',    default='datamlp/static',
                    help='Thư mục output .npy (default: datamlp/static)')
    ap.add_argument('--n_sample',   type=int, default=10,
                    help='Số frames sample mỗi video (default: 10)')
    ap.add_argument('--seed',       type=int, default=42)
    ap.add_argument('--verify',     action='store_true',
                    help='Chỉ verify feature vector, không process video')
    args = ap.parse_args()

    np.random.seed(args.seed)

    if args.verify:
        verify_feature_vector()
    else:
        videos_dir = str(_PROJECT_ROOT / args.videos_dir) \
                     if not os.path.isabs(args.videos_dir) else args.videos_dir
        out_dir    = str(_PROJECT_ROOT / args.out_dir) \
                     if not os.path.isabs(args.out_dir) else args.out_dir

        build_dataset(
            videos_dir = videos_dir,
            out_dir    = out_dir,
            n_sample   = args.n_sample,
        )