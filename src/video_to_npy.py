"""
video_to_npy.py - Convert videos to 208-dim features (pose + hands + finger curl + emotion)
============================================================================================
BƯỚC 2 trong pipeline

Layout (208 dim):
  [0:45]    pose        (45 dim  — 15 điểm × 3)
  [45:171]  hands xyz   (126 dim — 21 landmark × 2 tay × 3)
  [171:201] finger curl (30 dim  — 5 ngón × 3 features × 2 tay)
            Mỗi ngón: [curl_ratio, bend_angle_pip, tip_dist_norm]
              curl_ratio   : 0 = co hoàn toàn, 1 = duỗi thẳng
              bend_angle   : góc gập tại khớp PIP (0 = thẳng, 1 = co max)
              tip_distance : khoảng cách đầu ngón → wrist (normalized)
  [201:208] emotion     (7 dim   — one-hot)

Input (pre-split):
  videos/train/<label>/*.mp4
  videos/val/<label>/*.mp4
  videos/test/<label>/*.mp4

Output:
  data/processed/train/<label>/*.npy   shape (64, 208)
  data/processed/val/<label>/*.npy
  data/processed/test/<label>/*.npy
  data/processed/label_map.json
  data/processed/feature_metadata.json

Chạy:
    python src/video_to_npy.py                          # xử lý tất cả
  python src/video_to_npy.py --clean                  # xóa data cũ rồi xử lý
  python src/video_to_npy.py --labels khong_thich ghen   # chỉ xử lý 2 labels
  python src/video_to_npy.py --default-emotion neutral
  python src/video_to_npy.py --check-emotion
  python src/video_to_npy.py --assign-emotion
  python src/video_to_npy.py --verify data/processed/train/angry/angry_0000_org.npy
"""

import os
import sys
import json
import shutil
import argparse
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

class Config:
    SEQ_LEN = 64
    SPLITS  = ['train', 'val', 'test']

    # ── Pose ──────────────────────────────────────────────────────
    POSE_KEY_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    POSE_DIM = 45   # 15 điểm × 3

    # ── Hands xyz ─────────────────────────────────────────────────
    HAND_DIM = 126  # 21 landmark × 2 tay × 3

    # ── Finger curl ───────────────────────────────────────────────
    # 5 ngón × 3 features × 2 tay = 30 dim
    FINGER_CURL_DIM = 30

    # Landmark indices từng ngón (MediaPipe Hand)
    # Wrist = 0, mỗi ngón: [MCP, PIP, DIP, TIP]
    FINGER_JOINTS = {
        'thumb':  (1,  2,  3,  4),
        'index':  (5,  6,  7,  8),
        'middle': (9,  10, 11, 12),
        'ring':   (13, 14, 15, 16),
        'pinky':  (17, 18, 19, 20),
    }
    FINGER_ORDER = ['thumb', 'index', 'middle', 'ring', 'pinky']
    WRIST_IDX    = 0
    MIDDLE_MCP   = 9   # dùng làm scale reference

    # ── Emotion ───────────────────────────────────────────────────
    EMOTIONS = {
        "angry": 0, "disgust": 1, "fear": 2,
        "happy": 3, "sad": 4, "surprise": 5, "neutral": 6,
    }
    EMOTION_DIM = 7

    # ── Tổng dim ──────────────────────────────────────────────────
    FEAT_DIM = POSE_DIM + HAND_DIM + FINGER_CURL_DIM + EMOTION_DIM  # 208

    # ── Layout (byte offsets) ─────────────────────────────────────
    POSE_START   = 0
    POSE_END     = 45
    HAND_START   = 45
    HAND_END     = 171
    CURL_START   = 171
    CURL_END     = 201
    EMO_START    = 201
    EMO_END      = 208


cfg = Config()

_PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_URLS = {
    'hand_landmarker.task':
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    'pose_landmarker_heavy.task':
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
}

def ensure_model(name):
    path = _PROJECT_ROOT / name
    if not path.exists():
        print(f"    Downloading {name}...")
        urllib.request.urlretrieve(MODEL_URLS[name], str(path))
    return str(path)


# ═══════════════════════════════════════════════════════════════════
# FINGER CURL CALCULATOR
# ═══════════════════════════════════════════════════════════════════

def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Góc tại đỉnh b, tạo bởi 3 điểm a-b-c (radian)."""
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.arccos(np.clip(cos_val, -1.0, 1.0)))


def compute_finger_curl_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Tính 15 finger curl features cho 1 bàn tay.

    Input:  landmarks (21, 3) — tọa độ normalized xyz
    Output: (15,) — [curl_ratio, bend_angle, tip_dist_norm] × 5 ngón

    Giải thích từng feature:
      curl_ratio   : dist(tip→wrist) / dist(mcp→wrist) / 2
                     → ~0 = ngón co vào lòng bàn tay
                     → ~1 = ngón duỗi thẳng ra ngoài
      bend_angle   : góc PIP chuẩn hóa = (π - angle) / π
                     → 0 = khớp PIP hoàn toàn thẳng
                     → 1 = khớp PIP gập tối đa (co ngón)
      tip_dist_norm: dist(tip→wrist) / dist(middle_mcp→wrist) / 3
                     → khoảng cách tuyệt đối đầu ngón, normalized theo kích thước bàn tay
    """
    curl = np.zeros(15, dtype=np.float32)
    wrist   = landmarks[cfg.WRIST_IDX]
    mid_mcp = landmarks[cfg.MIDDLE_MCP]
    scale   = np.linalg.norm(mid_mcp - wrist) + 1e-8

    for i, fname in enumerate(cfg.FINGER_ORDER):
        mcp_i, pip_i, dip_i, tip_i = cfg.FINGER_JOINTS[fname]
        mcp = landmarks[mcp_i]
        pip = landmarks[pip_i]
        dip = landmarks[dip_i]
        tip = landmarks[tip_i]

        d_tip_wrist = np.linalg.norm(tip - wrist)
        d_mcp_wrist = np.linalg.norm(mcp - wrist) + 1e-8

        curl_ratio    = float(np.clip(d_tip_wrist / d_mcp_wrist / 2.0, 0.0, 1.0))
        angle_pip     = compute_angle(mcp, pip, dip)
        bend_norm     = float(np.clip((np.pi - angle_pip) / np.pi, 0.0, 1.0))
        tip_dist_norm = float(np.clip(d_tip_wrist / scale / 3.0, 0.0, 1.0))

        base = i * 3
        curl[base]     = curl_ratio
        curl[base + 1] = bend_norm
        curl[base + 2] = tip_dist_norm

    return curl


# ═══════════════════════════════════════════════════════════════════
# FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """Extract pose (45) + hands xyz (126) + finger curl (30) = 201 dim."""

    def __init__(self):
        print("    Init FeatureExtractor (208-dim pipeline)...")

        self.hand_detector = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=ensure_model('hand_landmarker.task')),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
            ))

        self.pose_detector = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=ensure_model('pose_landmarker_heavy.task')),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
            ))

        print("    FeatureExtractor ready.")

    def extract_frame(self, rgb: np.ndarray) -> np.ndarray:
        """
        Extract 1 frame → (201,) = pose(45) + hands_xyz(126) + curl(30).
        Emotion CHƯA ghép vào — ghép ở bước sau khi đã biết emotion của video.

        Quy tắc slot tay:
          - 1 tay  → LUÔN vào slot 0 (45:108 + curl 171:186), bất kể trái/phải
          - 2 tay  → sort theo x-center: tay trái màn hình → slot 0, phải → slot 1
          Như vậy khi train với 1 tay dù trái hay phải đều học từ slot 0.
        """
        feat   = np.zeros(cfg.POSE_DIM + cfg.HAND_DIM + cfg.FINGER_CURL_DIM,
                          dtype=np.float32)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # ── Pose (0:45) ───────────────────────────────────────────
        try:
            r = self.pose_detector.detect(mp_img)
            if r.pose_landmarks:
                for i, idx in enumerate(cfg.POSE_KEY_INDICES):
                    lm = r.pose_landmarks[0][idx]
                    feat[i*3:(i+1)*3] = [lm.x, lm.y, lm.z]
        except Exception:
            pass

        # ── Hands xyz (45:171) + Curl (171:201) ───────────────────
        try:
            r = self.hand_detector.detect(mp_img)
            if r.hand_landmarks and r.handedness:
                hands = list(zip(r.hand_landmarks, r.handedness))

                if len(hands) == 1:
                    # Chỉ 1 tay → LUÔN slot 0, không quan tâm trái/phải
                    hand_lms = hands[0][0]
                    for j, lm in enumerate(hand_lms):
                        feat[45 + j*3 : 45 + j*3 + 3] = [lm.x, lm.y, lm.z]
                    lms_arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms],
                                       dtype=np.float32)
                    feat[171 : 186] = compute_finger_curl_features(lms_arr)
                else:
                    # 2 tay → sort theo x để nhất quán
                    hands.sort(key=lambda h: float(np.mean([lm.x for lm in h[0]])))
                    xyz_slots  = [45, 108]
                    curl_slots = [171, 186]
                    for (hand_lms, _), xyz_s, curl_s in zip(
                            hands[:2], xyz_slots, curl_slots):
                        for j, lm in enumerate(hand_lms):
                            feat[xyz_s + j*3 : xyz_s + j*3 + 3] = [lm.x, lm.y, lm.z]
                        lms_arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms],
                                           dtype=np.float32)
                        feat[curl_s : curl_s + 15] = compute_finger_curl_features(lms_arr)

        except Exception:
            pass

        return feat  # (201,)

    def close(self):
        self.hand_detector.close()
        self.pose_detector.close()


# ═══════════════════════════════════════════════════════════════════
# NORMALIZER
# ═══════════════════════════════════════════════════════════════════

def normalize_frame(feat: np.ndarray) -> np.ndarray:
    """
    Normalize pose xyz và hands xyz.
    Finger curl (171:201) KHÔNG normalize vì đã là ratio/angle trong [0,1].

    Pose:  dịch về hip-midpoint, scale theo shoulder width
    Hands: dịch về wrist, scale theo middle-MCP distance
    """
    f = feat.copy()

    # Pose (0:45)
    pose = f[:45].reshape(-1, 3)
    if np.any(pose != 0):
        hip_mid       = (pose[13] + pose[14]) / 2
        pose          = pose - hip_mid
        shoulder_dist = np.linalg.norm(pose[1] - pose[2])
        if shoulder_dist > 1e-6:
            pose = pose / shoulder_dist
        f[:45] = pose.flatten()

    # Left hand xyz (45:108)
    left = f[45:108].reshape(-1, 3)
    if np.any(left != 0):
        left  = left - left[0]         # wrist → origin
        scale = np.linalg.norm(left[9])  # middle MCP distance
        if scale > 1e-6:
            left = left / scale
        f[45:108] = left.flatten()

    # Right hand xyz (108:171)
    right = f[108:171].reshape(-1, 3)
    if np.any(right != 0):
        right = right - right[0]
        scale = np.linalg.norm(right[9])
        if scale > 1e-6:
            right = right / scale
        f[108:171] = right.flatten()

    # Curl (171:201): giữ nguyên
    return f


def resample_sequence(seq, target_len: int) -> np.ndarray:
    """Resample danh sách frames về đúng target_len bằng linear interpolation."""
    n = len(seq)
    if n == target_len:
        return np.array(seq, dtype=np.float32)
    indices = np.linspace(0, n - 1, target_len)
    result  = []
    for i in indices:
        lo = int(np.floor(i))
        hi = min(int(np.ceil(i)), n - 1)
        w  = i - lo
        result.append(seq[lo] * (1 - w) + seq[hi] * w)
    return np.array(result, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# EMOTION UTILS
# ═══════════════════════════════════════════════════════════════════

def encode_emotion(name: str) -> np.ndarray:
    vec = np.zeros(cfg.EMOTION_DIM, dtype=np.float32)
    vec[cfg.EMOTIONS.get(name, 0)] = 1.0
    return vec


def get_emotion_from_file(video_path: str):
    """Tìm emotion từ metadata JSON kế bên video, hoặc từ tên file."""
    meta_path = str(video_path).replace('.mp4', '.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                data = json.load(f)
                if 'emotion' in data:
                    return data['emotion']
        except Exception:
            pass
    # Fallback: parse từ tên file  e.g.  label_angry_001.mp4
    name = Path(video_path).stem
    for emo in cfg.EMOTIONS:
        if f"_{emo}_" in name or name.endswith(f"_{emo}"):
            return emo
    return None


def save_emotion_to_metadata(video_path: str, emotion: str) -> str:
    meta_path = str(video_path).replace('.mp4', '.json')
    data = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                data = json.load(f)
        except Exception:
            pass
    data['emotion']         = emotion
    data['emotion_id']      = cfg.EMOTIONS.get(emotion, 0)
    data['emotion_updated'] = datetime.now().isoformat()
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return meta_path


# ═══════════════════════════════════════════════════════════════════
# SCAN HELPER
# ═══════════════════════════════════════════════════════════════════

def scan_presplit_structure(video_dir, label_filter=None):
    """
    Quét cấu trúc pre-split:
        video_dir/train/<label>/*.mp4
        video_dir/val/<label>/*.mp4
        video_dir/test/<label>/*.mp4

    Returns:
        dict  { split: { label: [video_path, ...] } }
        bool  is_presplit
    """
    video_dir   = Path(video_dir)
    result      = {}
    found_split = False

    for split in cfg.SPLITS:
        split_dir = video_dir / split
        if not split_dir.is_dir():
            continue
        found_split = True
        result[split] = {}
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            if label_filter and label_dir.name not in label_filter:
                continue
            videos = sorted(label_dir.glob('*.mp4'))
            if videos:
                result[split][label_dir.name] = [str(v) for v in videos]

    return result, found_split


# ═══════════════════════════════════════════════════════════════════
# EMOTION CHECKER & ASSIGNER
# ═══════════════════════════════════════════════════════════════════

def check_emotions(video_dir, label_filter=None):
    print(f"\n  Checking emotions in: {video_dir}")
    if label_filter:
        print(f"  Label filter: {label_filter}")
    print("=" * 60)

    split_data, is_presplit = scan_presplit_structure(video_dir, label_filter)

    results = {}
    total   = 0
    missing = 0

    if is_presplit:
        for split, labels in split_data.items():
            for label, videos in labels.items():
                key = f"{split}/{label}"
                results[key] = []
                label_missing = 0
                for vpath in videos:
                    emo = get_emotion_from_file(vpath)
                    results[key].append((vpath, emo))
                    total += 1
                    if emo is None:
                        missing += 1
                        label_missing += 1
                status = "✅" if label_missing == 0 else f"⚠️  {label_missing} missing"
                print(f"  [{split}] {label}: {len(videos)} videos — {status}")
    else:
        for label_dir in sorted(Path(video_dir).iterdir()):
            if not label_dir.is_dir():
                continue
            if label_filter and label_dir.name not in label_filter:
                continue
            videos = sorted(label_dir.glob('*.mp4'))
            if not videos:
                continue
            key = label_dir.name
            results[key] = []
            label_missing = 0
            for vpath in videos:
                emo = get_emotion_from_file(str(vpath))
                results[key].append((str(vpath), emo))
                total += 1
                if emo is None:
                    missing += 1
                    label_missing += 1
            status = "✅" if label_missing == 0 else f"⚠️  {label_missing} missing"
            print(f"  {key}: {len(list(videos))} videos — {status}")

    print("=" * 60)
    print(f"  Total  : {total} videos")
    print(f"  OK     : {total - missing}")
    print(f"  Missing: {missing}")
    return results, missing


def assign_emotions_interactive(video_dir, label_filter=None):
    results, missing = check_emotions(video_dir, label_filter)
    if missing == 0:
        print("\n  ✅ Tất cả videos đã có emotion!")
        return

    emotions_list = list(cfg.EMOTIONS.keys())
    print(f"\n  Emotions:")
    for i, name in enumerate(emotions_list):
        print(f"    {i+1}. {name}")

    for key, videos in results.items():
        missing_videos = [(p, e) for p, e in videos if e is None]
        if not missing_videos:
            continue
        print(f"\n{'='*60}")
        print(f"  {key}  ({len(missing_videos)} missing)")
        choice = input("  Gán emotion cho cả nhóm? (1-7 / n=từng video / s=skip): ").strip().lower()
        if choice == 's':
            continue
        if choice == 'q':
            break
        if choice.isdigit() and 1 <= int(choice) <= 7:
            emo = emotions_list[int(choice) - 1]
            for vpath, _ in missing_videos:
                save_emotion_to_metadata(vpath, emo)
            print(f"    ✅ Đã gán '{emo}' cho {len(missing_videos)} videos")
            continue
        for vpath, _ in missing_videos:
            print(f"\n    {os.path.basename(vpath)}")
            c = input("    Emotion (1-7, s=skip): ").strip().lower()
            if c == 's':
                continue
            if c.isdigit() and 1 <= int(c) <= 7:
                emo = emotions_list[int(c) - 1]
                save_emotion_to_metadata(vpath, emo)
                print(f"    → {emo}")

    print("\n  ✅ Done!")


def assign_default_emotion(video_dir, default_emotion, label_filter=None):
    results, missing = check_emotions(video_dir, label_filter)
    if missing == 0:
        return 0
    print(f"\n  Gán '{default_emotion}' cho {missing} videos...")
    count = 0
    for key, videos in results.items():
        for vpath, emo in videos:
            if emo is None:
                save_emotion_to_metadata(vpath, default_emotion)
                count += 1
    print(f"  ✅ Đã gán {count} videos!")
    return count


# ═══════════════════════════════════════════════════════════════════
# MIRROR AUGMENTATION
# Hoán đổi tay trái ↔ tay phải + flip tọa độ x
# Giúp model nhận dạng được dù biểu diễn tay nào
# ═══════════════════════════════════════════════════════════════════

def mirror_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Mirror toàn bộ sequence (T, 208) theo trục x:
      - Flip x của pose (0:45): x → 1 - x
      - Hoán đổi slot tay trái (45:108) ↔ tay phải (108:171): x → 1 - x
      - Hoán đổi curl trái (171:186) ↔ curl phải (186:201)
        (curl features không có x nên giữ giá trị, chỉ hoán đổi slot)
      - Emotion (201:208): giữ nguyên

    Kết quả: ký hiệu quay bằng tay trái → thành bản tay phải và ngược lại.
    """
    m = seq.copy()   # (T, 208)

    # ── Flip x của pose (0:45): x là dim 0,3,6,... của mỗi landmark ─
    pose = m[:, 0:45].reshape(-1, 15, 3)
    pose[:, :, 0] = 1.0 - pose[:, :, 0]   # flip x
    m[:, 0:45] = pose.reshape(-1, 45)

    # ── Flip x của 2 tay rồi hoán đổi slot ──────────────────────────
    left_xyz  = m[:, 45:108].copy()
    right_xyz = m[:, 108:171].copy()

    # Flip x (dim 0,3,6,... trong mỗi khối 21×3)
    left_xyz_flip  = left_xyz.reshape(-1, 21, 3)
    right_xyz_flip = right_xyz.reshape(-1, 21, 3)
    left_xyz_flip[:, :, 0]  = 1.0 - left_xyz_flip[:, :, 0]
    right_xyz_flip[:, :, 0] = 1.0 - right_xyz_flip[:, :, 0]

    # Hoán đổi: slot 0 ← tay phải đã flip, slot 1 ← tay trái đã flip
    m[:, 45:108]  = right_xyz_flip.reshape(-1, 63)
    m[:, 108:171] = left_xyz_flip.reshape(-1, 63)

    # ── Hoán đổi curl (giá trị giữ nguyên, chỉ đổi slot) ────────────
    curl_left  = m[:, 171:186].copy()
    curl_right = m[:, 186:201].copy()
    m[:, 171:186] = curl_right
    m[:, 186:201] = curl_left

    return m


# ═══════════════════════════════════════════════════════════════════
# VIDEO TO NPY
# ═══════════════════════════════════════════════════════════════════

class VideoToNPY:

    def __init__(self, video_dir="videos", output_dir="data/processed",
                 default_emotion=None, labels=None):
        self.video_dir       = video_dir
        self.output_dir      = output_dir
        self.default_emotion = default_emotion
        self.labels          = labels
        self.skipped_no_emotion = []
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n  [VideoToNPY]")
        print(f"  Input    : {video_dir}")
        print(f"  Output   : {output_dir}")
        print(f"  Feat dim : {cfg.FEAT_DIM}")
        print(f"    [0:45]   pose xyz       (15 điểm × 3)")
        print(f"    [45:171] hands xyz      (21 lm × 2 tay × 3)")
        print(f"    [171:201] finger curl   (5 ngón × 3 feat × 2 tay)")
        print(f"    [201:208] emotion       (7-class one-hot)")
        if default_emotion:
            print(f"  Default emotion: {default_emotion}")
        if labels:
            print(f"  Label filter   : {labels}")

        self.extractor = FeatureExtractor()

    def _expected_files(self, save_dir, video_id, augment):
        files = [f"{save_dir}/{video_id}_org.npy"]
        if augment:
            files += [
                f"{save_dir}/{video_id}_mirror.npy",   # tay đối xứng
                f"{save_dir}/{video_id}_noise0.npy",
                f"{save_dir}/{video_id}_noise1.npy",
                f"{save_dir}/{video_id}_mirror_noise0.npy",
                f"{save_dir}/{video_id}_scale0.npy",
                f"{save_dir}/{video_id}_scale1.npy",
                f"{save_dir}/{video_id}_warp0.npy",
                f"{save_dir}/{video_id}_warp1.npy",
            ]
        return files

    def clean(self):
        if self.labels:
            print(f"  Cleaning labels: {self.labels}")
            for split in cfg.SPLITS:
                for label in self.labels:
                    p = os.path.join(self.output_dir, split, label)
                    if os.path.exists(p):
                        shutil.rmtree(p)
                        print(f"    Removed: {p}")
        else:
            for split in cfg.SPLITS:
                p = os.path.join(self.output_dir, split)
                if os.path.exists(p):
                    shutil.rmtree(p)
            for f in ['label_map.json', 'feature_metadata.json']:
                p = os.path.join(self.output_dir, f)
                if os.path.exists(p):
                    os.remove(p)
        print("  Cleaned!")

    def process_video(self, video_path, label, video_id, split, augment=True):
        save_dir = os.path.join(self.output_dir, split, label)
        expected = self._expected_files(save_dir, video_id, augment)
        if all(os.path.exists(f) for f in expected):
            return len(expected)

        # Lấy emotion
        emotion = get_emotion_from_file(video_path)
        if emotion is None:
            if self.default_emotion:
                emotion = self.default_emotion
            else:
                self.skipped_no_emotion.append(video_path)
                print(f"      [SKIP] No emotion: {os.path.basename(video_path)}")
                return 0

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"      [SKIP] Cannot open: {os.path.basename(video_path)}")
            return 0

        emotion_vec = encode_emotion(emotion)
        raw_seq     = []
        hand_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feat = self.extractor.extract_frame(rgb)  # (201,)
            feat = normalize_frame(feat)
            raw_seq.append(feat)
            if np.sum(np.abs(feat[45:171])) > 0.01:
                hand_frames += 1
        cap.release()

        if len(raw_seq) < 5:
            print(f"      [!] Too short: {len(raw_seq)} frames")
            return 0

        hand_ratio = hand_frames / len(raw_seq)
        if hand_ratio < 0.3:
            print(f"      [!] Low hand detection: {hand_ratio*100:.0f}%")

        # Resample → (64, 201) rồi ghép emotion → (64, 208)
        pose_hands_curl = resample_sequence(raw_seq, cfg.SEQ_LEN)         # (64, 201)
        emotion_seq     = np.tile(emotion_vec, (cfg.SEQ_LEN, 1))          # (64, 7)
        full_seq        = np.concatenate([pose_hands_curl, emotion_seq], axis=1).astype(np.float32)  # (64, 208)

        os.makedirs(save_dir, exist_ok=True)
        files = 0

        np.save(f"{save_dir}/{video_id}_org.npy", full_seq)
        files += 1

        if augment:
            # ── Mirror: flip tay trái ↔ phải ─────────────────────
            # Đây là augment quan trọng nhất để nhận cả 2 tay
            mirrored = mirror_sequence(full_seq)
            np.save(f"{save_dir}/{video_id}_mirror.npy", mirrored)
            files += 1

            # Mirror + noise nhẹ
            aug = mirrored.copy()
            aug[:, 45:171] += np.random.normal(0, 0.003, (cfg.SEQ_LEN, 126)).astype(np.float32)
            np.save(f"{save_dir}/{video_id}_mirror_noise0.npy", aug)
            files += 1

            # Noise: chỉ phần xyz (45:171), KHÔNG augment curl (đã là ratio)
            for i, std in enumerate([0.003, 0.006]):
                aug = full_seq.copy()
                aug[:, 45:171] += np.random.normal(0, std, (cfg.SEQ_LEN, 126)).astype(np.float32)
                np.save(f"{save_dir}/{video_id}_noise{i}.npy", aug)
                files += 1

            # Scale: chỉ xyz
            for i, s in enumerate([0.95, 1.05]):
                aug = full_seq.copy()
                aug[:, 45:171] *= s
                np.save(f"{save_dir}/{video_id}_scale{i}.npy", aug)
                files += 1

            # Time warp
            for i, (start_r, end_r) in enumerate([(0.05, 0.95), (0.1, 0.9)]):
                start = int(cfg.SEQ_LEN * start_r)
                end   = int(cfg.SEQ_LEN * end_r)
                aug   = resample_sequence(list(full_seq[start:end]), cfg.SEQ_LEN)
                np.save(f"{save_dir}/{video_id}_warp{i}.npy", aug)
                files += 1

        return files

    def process_all(self, clean=False):
        if clean:
            self.clean()

        split_data, is_presplit = scan_presplit_structure(self.video_dir, self.labels)
        if not is_presplit or not split_data:
            print("  [ERROR] Không tìm thấy cấu trúc train/val/test trong:", self.video_dir)
            return

        if self.labels:
            found = set()
            for sl in split_data.values():
                found.update(sl.keys())
            not_found = set(self.labels) - found
            if not_found:
                print(f"\n  ⚠️  Labels không tìm thấy: {sorted(not_found)}")
            print(f"\n  Labels sẽ xử lý: {sorted(found)}")

        if not self.default_emotion:
            _, missing = check_emotions(self.video_dir, self.labels)
            if missing > 0:
                print(f"\n  ⚠️  Có {missing} videos chưa có emotion!")
                print("  Dùng --assign-emotion hoặc --default-emotion để xử lý.")
                choice = input("\n  Tiếp tục và SKIP videos thiếu emotion? (y/n): ").strip().lower()
                if choice != 'y':
                    return

        all_labels = set()
        total      = {s: 0 for s in cfg.SPLITS}

        for split in cfg.SPLITS:
            if split not in split_data or not split_data[split]:
                print(f"\n  [WARN] Không có split '{split}' — bỏ qua")
                continue

            labels_dict = split_data[split]
            print(f"\n{'='*60}")
            print(f"  SPLIT: {split}  (augment={'ON' if split == 'train' else 'OFF'})")
            print(f"  Labels: {sorted(labels_dict.keys())}")

            for label, videos in sorted(labels_dict.items()):
                all_labels.add(label)
                print(f"\n  [{split}/{label}] {len(videos)} videos")
                skipped = 0
                for i, vpath in enumerate(videos):
                    video_id     = f"{label}_{i:04d}"
                    augment      = (split == 'train')
                    _save_dir    = os.path.join(self.output_dir, split, label)
                    already_done = all(os.path.exists(f)
                                       for f in self._expected_files(_save_dir, video_id, augment))
                    n = self.process_video(vpath, label, video_id, split, augment)
                    total[split] += n
                    if already_done:
                        skipped += 1
                    elif n:
                        print(f"      ✅ {os.path.basename(vpath)} → {n} file(s)")
                if skipped:
                    print(f"      {skipped}/{len(videos)} videos đã có → skip")
            print(f"\n  → {split}: {total[split]} .npy files")

        # label_map.json
        label_map_path = os.path.join(self.output_dir, 'label_map.json')
        if self.labels and os.path.exists(label_map_path):
            with open(label_map_path) as f:
                existing_map = json.load(f)
            for lbl in sorted(all_labels):
                if lbl not in existing_map:
                    next_idx = max(existing_map.values()) + 1 if existing_map else 0
                    existing_map[lbl] = next_idx
            label_map = existing_map
        else:
            label_map = {name: idx for idx, name in enumerate(sorted(all_labels))}

        with open(label_map_path, 'w') as f:
            json.dump(label_map, f, indent=2)

        # feature_metadata.json
        meta = {
            'feat_dim': cfg.FEAT_DIM,
            'seq_len':  cfg.SEQ_LEN,
            'layout': {
                'pose':        '0:45    (15 điểm pose × xyz)',
                'hands_xyz':   '45:171  (21 lm × 2 tay × xyz)',
                'finger_curl': '171:201 (5 ngón × 3 feat × 2 tay)',
                'emotion':     '201:208 (7-class one-hot)',
            },
            'finger_curl_features': {
                'per_finger': ['curl_ratio', 'bend_angle_pip', 'tip_dist_norm'],
                'order':      cfg.FINGER_ORDER,
                'hands':      ['left_171:186', 'right_186:201'],
                'note':       'curl_ratio: 0=co, 1=duỗi; bend_angle: 0=thẳng, 1=co max',
            },
            'emotions':     cfg.EMOTIONS,
            'splits':       {s: total[s] for s in cfg.SPLITS},
            'label_filter': self.labels,
            'created':      datetime.now().isoformat(),
        }
        with open(os.path.join(self.output_dir, 'feature_metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"\n{'='*60}")
        print("  DONE!")
        print(f"  Labels : {len(all_labels)} → {sorted(all_labels)}")
        print(f"  Files  : train={total['train']}, val={total['val']}, test={total['test']}")
        print(f"  Shape  : ({cfg.SEQ_LEN}, {cfg.FEAT_DIM})")
        if self.skipped_no_emotion:
            print(f"\n  ⚠️  Skipped {len(self.skipped_no_emotion)} videos (no emotion)")
            print("  Run: python video_to_npy.py --assign-emotion")
        print("=" * 60)

    def close(self):
        self.extractor.close()


# ═══════════════════════════════════════════════════════════════════
# VERIFY — debug finger curl
# ═══════════════════════════════════════════════════════════════════

def verify_curl_features(npy_path: str):
    """In curl features để debug / so sánh giữa các ký hiệu."""
    data    = np.load(npy_path)    # (64, 208)
    curl    = data[:, 171:201]     # (64, 30)
    emotion = data[0, 201:208]
    emo_name = list(cfg.EMOTIONS.keys())[int(np.argmax(emotion))]

    print(f"\n  File   : {npy_path}")
    print(f"  Shape  : {data.shape}")
    print(f"  Emotion: {emo_name}")
    print(f"\n  Finger curl (avg over {cfg.SEQ_LEN} frames):")
    print(f"  {'':4} {'Finger':<8} {'curl_ratio':>12} {'bend_angle':>12} {'tip_dist':>10}")
    print("  " + "-"*52)

    for hand_name, offset in [('LEFT', 0), ('RIGHT', 15)]:
        print(f"  [{hand_name}]")
        for i, fname in enumerate(cfg.FINGER_ORDER):
            base = offset + i * 3
            print(f"       {fname:<8} "
                  f"{curl[:, base].mean():>12.3f} "
                  f"{curl[:, base+1].mean():>12.3f} "
                  f"{curl[:, base+2].mean():>10.3f}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Video → NPY (208 dim: pose + hands + finger_curl + emotion)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Ví dụ:
  python src/video_to_npy.py                          # xử lý tất cả
  python src/video_to_npy.py --clean                  # xóa data cũ rồi xử lý
  python src/video_to_npy.py --labels khong_thich ghen   # chỉ xử lý 2 labels
  python src/video_to_npy.py --default-emotion neutral
  python src/video_to_npy.py --check-emotion
  python src/video_to_npy.py --assign-emotion
  python src/video_to_npy.py --verify data/processed/train/angry/angry_0000_org.npy
        """
    )
    parser.add_argument("--video_dir",       default="videos")
    parser.add_argument("--output_dir",      default="data/processed")
    parser.add_argument("--clean",           action="store_true")
    parser.add_argument("--labels",          nargs="+", default=None)
    parser.add_argument("--default-emotion", type=str,  default=None,
                        choices=list(cfg.EMOTIONS.keys()))
    parser.add_argument("--check-emotion",   action="store_true")
    parser.add_argument("--assign-emotion",  action="store_true")
    parser.add_argument("--verify",          type=str,  default=None,
                        help="Path .npy để xem curl features")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" VIDEO TO NPY (208 dim) ".center(60, "="))
    print("=" * 60)
    print(f"  FEAT_DIM = {cfg.FEAT_DIM}  "
          f"(pose=45, hands_xyz=126, finger_curl=30, emotion=7)")

    if args.verify:
        verify_curl_features(args.verify)
        return

    if args.check_emotion:
        check_emotions(args.video_dir, args.labels)
        return

    if args.assign_emotion:
        assign_emotions_interactive(args.video_dir, args.labels)
        return

    converter = VideoToNPY(
        video_dir       = args.video_dir,
        output_dir      = args.output_dir,
        default_emotion = args.default_emotion,
        labels          = args.labels,
    )
    converter.process_all(clean=args.clean)
    converter.close()


if __name__ == "__main__":
    main()