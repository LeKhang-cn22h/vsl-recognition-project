"""
video_to_npy_v2.py - Convert videos to 208-dim features (thêm finger curl)
===========================================================================
BƯỚC 2 trong pipeline — thay thế video_to_npy.py

Thay đổi so với v1 (178 dim):
  + 30 dim finger curl features (5 ngón × 3 dim × 2 tay)
    - curl_ratio   : mức độ co ngón (0=duỗi thẳng, 1=co hoàn toàn)
    - bend_angle   : góc gập tại khớp giữa (MCP-PIP-DIP)
    - tip_distance : khoảng cách đầu ngón đến wrist (normalized)

Layout mới (208 dim):
  [0:45]    pose        (45 dim)
  [45:171]  hands xyz   (126 dim)
  [171:201] finger curl (30 dim) ← MỚI
  [201:208] emotion     (7 dim)

Chạy:
  python video_to_npy_v2.py --clean
  python video_to_npy_v2.py --labels angry disgust
  python video_to_npy_v2.py --default-emotion neutral
  python video_to_npy_v2.py --check-emotion
  python video_to_npy_v2.py --assign-emotion
  python video_to_npy_v2.py --verify data/processed_v2/train/angry/angry_0000_org.npy
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

    # Pose: 15 điểm × 3 = 45 dim
    POSE_KEY_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    POSE_DIM = 45

    # Hands xyz: 21 × 2 × 3 = 126 dim
    HAND_DIM = 126

    # Finger curl: 5 ngón × 3 features × 2 tay = 30 dim
    FINGER_CURL_DIM = 30

    # Emotion: 7 one-hot
    EMOTIONS = {
        "angry": 0, "disgust": 1, "fear": 2,
        "happy": 3, "sad": 4, "surprise": 5, "neutral": 6,
    }
    EMOTION_DIM = 7

    # Total: 45 + 126 + 30 + 7 = 208
    FEAT_DIM = POSE_DIM + HAND_DIM + FINGER_CURL_DIM + EMOTION_DIM  # 208

    # Ranges
    POSE_START,    POSE_END    = 0,   45
    HAND_START,    HAND_END    = 45,  171
    CURL_START,    CURL_END    = 171, 201
    EMOTION_START, EMOTION_END = 201, 208

    # MediaPipe landmark indices cho từng ngón
    FINGER_JOINTS = {
        'thumb':  (1, 2, 3, 4),
        'index':  (5, 6, 7, 8),
        'middle': (9, 10, 11, 12),
        'ring':   (13, 14, 15, 16),
        'pinky':  (17, 18, 19, 20),
    }
    FINGER_ORDER = ['thumb', 'index', 'middle', 'ring', 'pinky']
    WRIST_IDX    = 0
    MIDDLE_MCP   = 9


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
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def compute_finger_curl_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Tính finger curl features cho 1 tay.
    Input:  landmarks (21, 3)
    Output: curl_feats (15,) — 5 ngón × [curl_ratio, bend_angle, tip_dist_norm]
    """
    curl_feats = np.zeros(15, dtype=np.float32)

    wrist   = landmarks[cfg.WRIST_IDX]
    mid_mcp = landmarks[cfg.MIDDLE_MCP]
    scale   = np.linalg.norm(mid_mcp - wrist) + 1e-8

    for i, finger in enumerate(cfg.FINGER_ORDER):
        mcp_idx, pip_idx, dip_idx, tip_idx = cfg.FINGER_JOINTS[finger]

        mcp = landmarks[mcp_idx]
        pip = landmarks[pip_idx]
        dip = landmarks[dip_idx]
        tip = landmarks[tip_idx]

        # Feature 1: curl_ratio
        dist_wrist_tip = np.linalg.norm(tip - wrist)
        dist_wrist_mcp = np.linalg.norm(mcp - wrist) + 1e-8
        curl_ratio = np.clip(dist_wrist_tip / dist_wrist_mcp / 2.0, 0.0, 1.0)

        # Feature 2: bend_angle tại PIP
        bend_angle = compute_angle(mcp, pip, dip)
        bend_norm  = np.clip((np.pi - bend_angle) / np.pi, 0.0, 1.0)

        # Feature 3: tip_distance_normalized
        tip_dist_norm = np.clip(dist_wrist_tip / scale / 3.0, 0.0, 1.0)

        base = i * 3
        curl_feats[base]     = curl_ratio
        curl_feats[base + 1] = bend_norm
        curl_feats[base + 2] = tip_dist_norm

    return curl_feats


# ═══════════════════════════════════════════════════════════════════
# FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """Extract pose + hands xyz + finger curl = 201 dim."""

    def __init__(self):
        print("    Init FeatureExtractor v2 (with finger curl)...")

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

    def extract_frame(self, rgb):
        """Extract pose + hands xyz + finger curl = 201 dim."""
        feat = np.zeros(cfg.POSE_DIM + cfg.HAND_DIM + cfg.FINGER_CURL_DIM,
                        dtype=np.float32)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Pose (0:45)
        try:
            r = self.pose_detector.detect(mp_img)
            if r.pose_landmarks:
                for i, idx in enumerate(cfg.POSE_KEY_INDICES):
                    lm = r.pose_landmarks[0][idx]
                    feat[i*3:(i+1)*3] = [lm.x, lm.y, lm.z]
        except Exception:
            pass

        # Hands xyz + curl (45:201)
        try:
            r = self.hand_detector.detect(mp_img)
            if r.hand_landmarks and r.handedness:
                hands = list(zip(r.hand_landmarks, r.handedness))

                if len(hands) == 1:
                    hand_lms = hands[0][0]
                    for j, lm in enumerate(hand_lms):
                        feat[45 + j*3: 45 + j*3 + 3] = [lm.x, lm.y, lm.z]
                    lms_arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms])
                    feat[171:186] = compute_finger_curl_features(lms_arr)

                elif len(hands) >= 2:
                    def hand_cx(h):
                        return np.mean([lm.x for lm in h[0]])

                    hands_sorted = sorted(hands, key=hand_cx)
                    xyz_slots  = [45, 108]
                    curl_slots = [171, 186]

                    for xyz_s, curl_s, (hand_lms, _) in zip(
                            xyz_slots, curl_slots, hands_sorted):
                        for j, lm in enumerate(hand_lms):
                            feat[xyz_s + j*3: xyz_s + j*3 + 3] = [lm.x, lm.y, lm.z]
                        lms_arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms])
                        feat[curl_s: curl_s + 15] = compute_finger_curl_features(lms_arr)

        except Exception:
            pass

        return feat

    def close(self):
        self.hand_detector.close()
        self.pose_detector.close()


# ═══════════════════════════════════════════════════════════════════
# NORMALIZER
# ═══════════════════════════════════════════════════════════════════

def normalize_frame(feat: np.ndarray) -> np.ndarray:
    """Normalize pose + hands xyz. Curl KHÔNG normalize (đã là ratio/angle)."""
    f = feat.copy()

    pose = f[:45].reshape(-1, 3)
    if np.any(pose != 0):
        hip_mid = (pose[13] + pose[14]) / 2
        pose    = pose - hip_mid
        shoulder_dist = np.linalg.norm(pose[1] - pose[2])
        if shoulder_dist > 1e-6:
            pose = pose / shoulder_dist
        f[:45] = pose.flatten()

    left = f[45:108].reshape(-1, 3)
    if np.any(left != 0):
        left  = left - left[0]
        scale = np.linalg.norm(left[9])
        if scale > 1e-6:
            left = left / scale
        f[45:108] = left.flatten()

    right = f[108:171].reshape(-1, 3)
    if np.any(right != 0):
        right = right - right[0]
        scale = np.linalg.norm(right[9])
        if scale > 1e-6:
            right = right / scale
        f[108:171] = right.flatten()

    # Curl (171:201): không normalize
    return f


def resample_sequence(seq, target_len):
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

def encode_emotion(name):
    vec = np.zeros(cfg.EMOTION_DIM, dtype=np.float32)
    vec[cfg.EMOTIONS.get(name, 0)] = 1.0
    return vec


def get_emotion_from_file(video_path):
    meta_path = str(video_path).replace('.mp4', '.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                data = json.load(f)
                if 'emotion' in data:
                    return data['emotion']
        except Exception:
            pass
    name = Path(video_path).stem
    for emo in cfg.EMOTIONS:
        if f"_{emo}_" in name or name.endswith(f"_{emo}"):
            return emo
    return None


def save_emotion_to_metadata(video_path, emotion):
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
    Quét cấu trúc:
        video_dir/train/<label>/*.mp4
        video_dir/val/<label>/*.mp4
        video_dir/test/<label>/*.mp4
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
# EMOTION CHECKER & ASSIGNER  (port từ v1)
# ═══════════════════════════════════════════════════════════════════

def check_emotions(video_dir, label_filter=None):
    """Kiểm tra tất cả videos có emotion chưa."""
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
        video_dir_p = Path(video_dir)
        for label_dir in sorted(video_dir_p.iterdir()):
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
# VIDEO TO NPY v2
# ═══════════════════════════════════════════════════════════════════

class VideoToNPY:

    def __init__(self, video_dir="videos", output_dir="data/processed_v2",
                 default_emotion=None, labels=None):
        self.video_dir       = video_dir
        self.output_dir      = output_dir
        self.default_emotion = default_emotion
        self.labels          = labels
        self.skipped_no_emotion = []
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n  [VideoToNPY v2]")
        print(f"  Input   : {video_dir}")
        print(f"  Output  : {output_dir}")
        print(f"  Features: {cfg.FEAT_DIM} dim")
        print(f"    pose={cfg.POSE_DIM}, hands_xyz={cfg.HAND_DIM}, "
              f"finger_curl={cfg.FINGER_CURL_DIM}, emotion={cfg.EMOTION_DIM}")
        print(f"  Layout  : [0:45] pose | [45:171] hands | "
              f"[171:201] curl | [201:208] emotion")
        if default_emotion:
            print(f"  Default emotion: {default_emotion}")
        if labels:
            print(f"  Label filter   : {labels}")
        else:
            print(f"  Label filter   : (all labels)")

        self.extractor = FeatureExtractor()

    # ── Skip logic (port từ v1) ────────────────────────────────────
    def _expected_files(self, save_dir, video_id, augment):
        """Danh sách tất cả file .npy sẽ được tạo từ video này."""
        files = [f"{save_dir}/{video_id}_org.npy"]
        if augment:
            files += [
                f"{save_dir}/{video_id}_noise0.npy",
                f"{save_dir}/{video_id}_noise1.npy",
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

        # Skip nếu tất cả file đã tồn tại
        expected     = self._expected_files(save_dir, video_id, augment)
        already_done = all(os.path.exists(f) for f in expected)
        if already_done:
            return len(expected)

        # Check emotion
        emotion = get_emotion_from_file(video_path)
        if emotion is None:
            if self.default_emotion:
                emotion = self.default_emotion
                print(f"      [!] No emotion → default: {emotion}")
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
            feat = self.extractor.extract_frame(rgb)   # (201,)
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

        # Resample (201,) sequences → (64, 201) rồi ghép emotion
        pose_hands_curl = resample_sequence(raw_seq, cfg.SEQ_LEN)            # (64, 201)
        emotion_seq     = np.tile(emotion_vec, (cfg.SEQ_LEN, 1))             # (64,   7)
        full_seq        = np.concatenate([pose_hands_curl, emotion_seq], axis=1)  # (64, 208)

        os.makedirs(save_dir, exist_ok=True)

        files = 0
        np.save(f"{save_dir}/{video_id}_org.npy", full_seq.astype(np.float32))
        files += 1

        if augment:
            # Augment chỉ phần xyz (45:171), KHÔNG augment curl (đã là ratio)
            for i, std in enumerate([0.003, 0.006]):
                aug = full_seq.copy()
                aug[:, 45:171] += np.random.normal(
                    0, std, (cfg.SEQ_LEN, 126)).astype(np.float32)
                np.save(f"{save_dir}/{video_id}_noise{i}.npy", aug)
                files += 1

            for i, s in enumerate([0.95, 1.05]):
                aug = full_seq.copy()
                aug[:, 45:171] *= s
                np.save(f"{save_dir}/{video_id}_scale{i}.npy", aug)
                files += 1

            for i, (start_r, end_r) in enumerate([(0.05, 0.95), (0.1, 0.9)]):
                start = int(cfg.SEQ_LEN * start_r)
                end   = int(cfg.SEQ_LEN * end_r)
                crop  = full_seq[start:end]
                aug   = resample_sequence(list(crop), cfg.SEQ_LEN)
                np.save(f"{save_dir}/{video_id}_warp{i}.npy", aug)
                files += 1

        return files

    def process_all(self, clean=False):
        if clean:
            self.clean()

        split_data, is_presplit = scan_presplit_structure(self.video_dir, self.labels)

        if not is_presplit or not split_data:
            print("  [ERROR] Không tìm thấy cấu trúc train/val/test trong:", self.video_dir)
            print("  Cấu trúc mong đợi:")
            print("    videos/train/<label>/*.mp4")
            print("    videos/val/<label>/*.mp4")
            print("    videos/test/<label>/*.mp4")
            return

        # Kiểm tra label filter hợp lệ
        if self.labels:
            found_labels = set()
            for split_labels in split_data.values():
                found_labels.update(split_labels.keys())
            not_found = set(self.labels) - found_labels
            if not_found:
                print(f"\n  ⚠️  Labels không tìm thấy: {sorted(not_found)}")
            print(f"\n  Labels sẽ xử lý: {sorted(found_labels)}")

        # Kiểm tra emotion trước khi xử lý
        if not self.default_emotion:
            _, missing = check_emotions(self.video_dir, self.labels)
            if missing > 0:
                print(f"\n  ⚠️  Có {missing} videos chưa có emotion!")
                print("  Options:")
                print("    --assign-emotion       : Gán thủ công")
                print("    --default-emotion X    : Gán mặc định")
                print("    Tiếp tục sẽ SKIP các videos này")
                choice = input("\n  Tiếp tục? (y/n): ").strip().lower()
                if choice != 'y':
                    return

        all_labels = set()
        total      = {s: 0 for s in cfg.SPLITS}

        for split in cfg.SPLITS:
            if split not in split_data:
                print(f"\n  [WARN] Không có folder '{split}' — bỏ qua")
                continue

            labels = split_data[split]
            if not labels:
                print(f"\n  [WARN] Split '{split}' không có label — bỏ qua")
                continue

            print(f"\n{'='*60}")
            print(f"  SPLIT: {split}  (augment=ON)")
            print(f"  Labels: {sorted(labels.keys())}")

            for label, videos in sorted(labels.items()):
                all_labels.add(label)
                print(f"\n  [{split}/{label}] {len(videos)} videos")

                skipped = 0
                for i, vpath in enumerate(videos):
                    video_id = f"{label}_{i:04d}"

                    _save_dir    = os.path.join(self.output_dir, split, label)
                    _expected    = self._expected_files(_save_dir, video_id, augment=(split == 'train'))
                    already_done = all(os.path.exists(f) for f in _expected)

                    n_files = self.process_video(vpath, label, video_id, split,
                                                 augment=(split == 'train'))
                    total[split] += n_files

                    if already_done:
                        skipped += 1
                    elif n_files:
                        print(f"      ✅ {os.path.basename(vpath)} → {n_files} file(s)")

                if skipped:
                    print(f"      {skipped}/{len(videos)} videos đã có → skip")

            print(f"\n  → {split}: {total[split]} .npy files")

        # label_map.json — merge với file cũ khi dùng label filter (port từ v1)
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

        meta = {
            'feat_dim': cfg.FEAT_DIM,
            'seq_len':  cfg.SEQ_LEN,
            'version':  'v2_with_finger_curl',
            'layout': {
                'pose':        '0:45',
                'hands_xyz':   '45:171',
                'finger_curl': '171:201',
                'emotion':     '201:208',
            },
            'finger_curl_features': {
                'per_finger': ['curl_ratio', 'bend_angle_pip', 'tip_dist_norm'],
                'fingers':    ['thumb', 'index', 'middle', 'ring', 'pinky'],
                'hands':      ['left_slot_171:186', 'right_slot_186:201'],
                'note':       'curl_ratio: 0=co, 1=duoi; bend_angle: 0=thang, 1=co',
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
        print(f"  Labels : {len(all_labels)}  → {sorted(all_labels)}")
        print(f"  Files  : train={total['train']}, val={total['val']}, test={total['test']}")
        print(f"  Shape  : ({cfg.SEQ_LEN}, {cfg.FEAT_DIM})")
        print(f"  Output : {self.output_dir}/")
        print(f"           ├── train/")
        print(f"           ├── val/")
        print(f"           ├── test/")
        print(f"           ├── label_map.json")
        print(f"           └── feature_metadata.json")

        if self.skipped_no_emotion:
            print(f"\n  ⚠️  Skipped {len(self.skipped_no_emotion)} videos (no emotion)")
            print("  Run: python video_to_npy_v2.py --assign-emotion")

        print("=" * 60)

    def close(self):
        self.extractor.close()


# ═══════════════════════════════════════════════════════════════════
# VERIFY CURL (debug helper)
# ═══════════════════════════════════════════════════════════════════

def verify_curl_features(npy_path: str):
    """In ra curl features để debug phân biệt angry vs disgust."""
    data    = np.load(npy_path)    # (64, 208)
    curl    = data[:, 171:201]     # (64, 30)
    emotion = data[0, 201:208]
    emo_name = list(cfg.EMOTIONS.keys())[np.argmax(emotion)]

    print(f"\n  File   : {npy_path}")
    print(f"  Shape  : {data.shape}")
    print(f"  Emotion: {emo_name}")
    print(f"\n  Finger curl (averaged over {cfg.SEQ_LEN} frames):")
    print(f"  {'Finger':<10} {'curl_ratio':>12} {'bend_angle':>12} {'tip_dist':>12}")
    print("  " + "-" * 48)

    for hand_name, offset in [('LEFT', 0), ('RIGHT', 15)]:
        print(f"  [{hand_name}]")
        for i, fname in enumerate(cfg.FINGER_ORDER):
            base = offset + i * 3
            print(f"  {fname:<10} "
                  f"{curl[:, base].mean():>12.3f} "
                  f"{curl[:, base+1].mean():>12.3f} "
                  f"{curl[:, base+2].mean():>12.3f}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Video to NPY v2 (208 dim — thêm finger curl)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Ví dụ:
  # Xử lý toàn bộ
  python video_to_npy_v2.py

  # Chỉ xử lý angry + disgust
  python video_to_npy_v2.py --labels angry disgust --clean

  # Kiểm tra / gán emotion
  python video_to_npy_v2.py --check-emotion
  python video_to_npy_v2.py --assign-emotion
  python video_to_npy_v2.py --assign-emotion --labels angry disgust

  # Gán emotion mặc định cho video chưa có
  python video_to_npy_v2.py --default-emotion neutral

  # Debug curl features
  python video_to_npy_v2.py --verify data/processed_v2/train/angry/angry_0000_org.npy
        """
    )
    parser.add_argument("--video_dir",       default="data/videos")
    parser.add_argument("--output_dir",      default="data/processed_v2")
    parser.add_argument("--clean",           action="store_true",
                        help="Xóa data cũ (nếu có --labels thì chỉ xóa label đó)")
    parser.add_argument("--labels",          nargs="+", default=None,
                        metavar="LABEL",
                        help="Chỉ xử lý các labels này\n"
                             "Ví dụ: --labels angry disgust")
    parser.add_argument("--default-emotion", type=str, default=None,
                        choices=list(cfg.EMOTIONS.keys()),
                        help="Emotion mặc định cho video chưa có emotion")
    parser.add_argument("--check-emotion",   action="store_true",
                        help="Kiểm tra videos có emotion chưa")
    parser.add_argument("--assign-emotion",  action="store_true",
                        help="Gán emotion thủ công (interactive)")
    parser.add_argument("--verify",          type=str, default=None,
                        help="Path tới file .npy để kiểm tra curl features")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" VIDEO TO NPY v2 (208 dim — with finger curl) ".center(60, "="))
    print("=" * 60)
    print(f"  FEAT_DIM = {cfg.FEAT_DIM}")
    print(f"    pose(45) + hands_xyz(126) + finger_curl(30) + emotion(7)")
    print(f"\n  Finger curl layout (30 dim):")
    print(f"    Left  tay: dim 171-185  (5 ngón × 3)")
    print(f"    Right tay: dim 186-200  (5 ngón × 3)")
    print(f"    Mỗi ngón : [curl_ratio, bend_angle, tip_dist]")

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