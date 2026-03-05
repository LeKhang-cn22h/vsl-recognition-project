"""
video_to_npy_static.py
Extract 96-dim static hand features from videos for MLP training.
"""

import os
import sys
import json
import argparse
import math
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from pathlib import Path
from itertools import combinations

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request


# =========================================================
# PATH FIX
# =========================================================
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# =========================================================
# DOWNLOAD MODEL IF NEEDED
# =========================================================
_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_HAND_MODEL_PATH = str(_PROJECT_ROOT / "hand_landmarker.task")


def _ensure_hand_model():
    if not os.path.exists(_HAND_MODEL_PATH):
        print("Downloading hand_landmarker.task...")
        urllib.request.urlretrieve(_HAND_MODEL_URL, _HAND_MODEL_PATH)
        print("Done.")
    return _HAND_MODEL_PATH


# =========================================================
# CONSTANTS
# =========================================================
FEAT_DIM = 96

FINGERTIPS = [4, 8, 12, 16, 20]
FINGER_BASES = [2, 5, 9, 13, 17]


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def normalize_landmarks(lm):
    lm = lm.copy()
    wrist = lm[0]
    lm -= wrist
    scale = np.linalg.norm(lm[9])
    if scale > 1e-6:
        lm /= scale
    return lm


def angle_between(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(math.acos(cos_a))


def compute_joint_angles(lm):
    finger_chains = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],
    ]
    angles = []
    for chain in finger_chains:
        for i in range(1, 4):
            a = lm[chain[i - 1]]
            b = lm[chain[i]]
            c = lm[chain[i + 1]]
            angles.append(angle_between(a - b, c - b))
    return np.array(angles, dtype=np.float32)


def compute_finger_lengths(lm):
    return np.array(
        [np.linalg.norm(lm[tip] - lm[base])
         for base, tip in zip(FINGER_BASES, FINGERTIPS)],
        dtype=np.float32,
    )


def compute_fingertip_distances(lm):
    tips = lm[FINGERTIPS]
    dists = []
    for i, j in combinations(range(5), 2):
        dists.append(np.linalg.norm(tips[i] - tips[j]))
    return np.array(dists, dtype=np.float32)


def compute_palm_normal(lm):
    v1 = lm[5] - lm[0]
    v2 = lm[17] - lm[0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm > 1e-6:
        normal /= norm
    return normal.astype(np.float32)


def extract_hand_features(lm):
    lm = normalize_landmarks(lm)
    coords = lm.flatten()
    angles = compute_joint_angles(lm)
    lengths = compute_finger_lengths(lm)
    tip_dists = compute_fingertip_distances(lm)
    palm_n = compute_palm_normal(lm)

    feat = np.concatenate([coords, angles, lengths, tip_dists, palm_n])
    assert feat.shape == (96,)
    return feat.astype(np.float32)


def augment_feature(feat, noise_std=0.005):
    aug = feat.copy()
    noise = np.random.normal(0, noise_std, feat.shape).astype(np.float32)
    noise[93:96] = 0
    aug += noise
    return aug


# =========================================================
# MEDIAPIPE HAND EXTRACTOR
# =========================================================
class HandExtractor:
    def __init__(self):
        model_path = _ensure_hand_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

    def extract(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)

        if not result.hand_landmarks:
            return None

        lm = result.hand_landmarks[0]
        return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

    def close(self):
        self.detector.close()


# =========================================================
# VIDEO PROCESSING
# =========================================================
def get_representative_frames(cap, n_frames=10):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []

    start = int(total * 0.2)
    end = int(total * 0.8)
    indices = np.linspace(start, end, n_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    return frames


def process_video(video_path, extractor, n_sample):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = get_representative_frames(cap, n_sample)
    cap.release()

    feats = []
    for f in frames:
        lm = extractor.extract(f)
        if lm is not None:
            feats.append(extract_hand_features(lm))

    return feats if feats else None


# =========================================================
# DATASET BUILDER
# =========================================================
def build_dataset(videos_dir, out_dir, n_sample,
                  aug_train, aug_val, aug_test, noise_std):

    videos_path = Path(videos_dir)
    out_path = Path(out_dir)

    splits = ["train", "val", "test"]
    splits_available = [s for s in splits if (videos_path / s).exists()]

    labels = sorted(
        [d.name for d in (videos_path / splits_available[0]).iterdir()
         if d.is_dir()]
    )
    label_map = {l: i for i, l in enumerate(labels)}

    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    extractor = HandExtractor()

    total_ok = total_fail = 0

    for split in splits_available:
        print(f"\n=== {split.upper()} ===")
        split_in = videos_path / split
        split_out = out_path / split

        for label in labels:
            label_in = split_in / label
            label_out = split_out / label
            label_out.mkdir(parents=True, exist_ok=True)

            videos = list(label_in.glob("*.mp4"))
            if not videos:
                continue

            copies = {
                "train": aug_train,
                "val": aug_val,
                "test": aug_test
            }[split]

            for vp in videos:
                feats = process_video(str(vp), extractor, n_sample)
                if feats is None:
                    total_fail += 1
                    continue

                total_ok += 1
                for i, feat in enumerate(feats):
                    np.save(label_out / f"{vp.stem}_{i}.npy", feat)

                    for n in range(copies):
                        aug = augment_feature(feat, noise_std)
                        np.save(label_out / f"{vp.stem}_{i}_aug{n}.npy", aug)

        print(f"{split} done.")

    extractor.close()
    print("\nDONE.")
    print(f"OK videos: {total_ok}")
    print(f"Fail videos: {total_fail}")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", default="datamlp/videos")
    ap.add_argument("--out_dir", default="datamlp/static")
    ap.add_argument("--n_sample", type=int, default=10)
    ap.add_argument("--aug_train", type=int, default=3)
    ap.add_argument("--aug_val", type=int, default=1)
    ap.add_argument("--aug_test", type=int, default=0)
    ap.add_argument("--noise_std", type=float, default=0.005)

    args = ap.parse_args()

    build_dataset(
        videos_dir=args.videos_dir,
        out_dir=args.out_dir,
        n_sample=args.n_sample,
        aug_train=args.aug_train,
        aug_val=args.aug_val,
        aug_test=args.aug_test,
        noise_std=args.noise_std,
    )