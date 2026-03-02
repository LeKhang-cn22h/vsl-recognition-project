"""
video_to_npy.py - Chuyển video VSL → file .npy để train
=========================================================
Cấu trúc folder INPUT (sau khi chạy organize_dataset.py):
    data/videos/
    ├── train/<label>/*.mp4   → augment (~25 file/video)
    ├── val/<label>/*.mp4     → chỉ _org (1 file/video)
    └── test/<label>/*.mp4    → chỉ _org (1 file/video)

Cấu trúc folder OUTPUT:
    data/processed/
    ├── train/<label>/*.npy   (org + augment)
    ├── val/<label>/*.npy     (chỉ _org)
    ├── test/<label>/*.npy    (chỉ _org)
    └── label_map.json

Thay đổi v2:
    - Bỏ KEY_BLENDSHAPES import (không còn dùng)
    - Cập nhật feature_metadata: bỏ blendshapes, đổi interact 31→55
    - FEAT_DIM: 339 → 346
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from vsl.config    import cfg, FACE_KEY_INDICES
from vsl.extractor import VideoExtractor
from converter     import HFUploader, KeypointNormalizer, resample_sequence, Augmenter

load_dotenv()

SPLITS     = ['train', 'val', 'test']
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


class VideoToNPY:

    def __init__(self, video_base='data/videos',
                 output_base='data/processed',
                 hf_uploader=None):
        self.video_base  = video_base
        self.output_base = output_base
        self.hf          = hf_uploader
        os.makedirs(output_base, exist_ok=True)

        print("  Khoi tao VideoExtractor...")
        self.extractor = VideoExtractor()
        self.augmenter = Augmenter(
            seq_len   = cfg.SEQ_LEN,
            total_dim = cfg.FEAT_DIM,
            pose_dim  = cfg.POSE_END  - cfg.POSE_START,
            face_dim  = cfg.FACE_END  - cfg.FACE_START,
            hand_dim  = cfg.HAND_END  - cfg.HAND_START,
        )
        self._save_feature_meta()

    def _save_feature_meta(self):
        meta = {
            'version': 2,
            'sequence_length': cfg.SEQ_LEN,
            'total_features_per_frame': cfg.FEAT_DIM,
            'breakdown': {
                'pose (25 x 3)':      cfg.POSE_END     - cfg.POSE_START,
                'face (30 x 3)':      cfg.FACE_END     - cfg.FACE_START,
                'hands (21 x 2 x 3)': cfg.HAND_END     - cfg.HAND_START,
                'blendshapes':        0,
                'interactions (55)':  cfg.INTERACT_END - cfg.INTERACT_START,
            },
            'note': 'v2: Blendshapes removed. Interactions expanded 31->55.',
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        path = os.path.join(self.output_base, 'feature_metadata.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _label_map_path(self):
        return os.path.join(self.output_base, 'label_map.json')

    def _load_label_map(self):
        p = self._label_map_path()
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_label_map(self, label_names):
        lm = {name: idx for idx, name in enumerate(sorted(label_names))}
        with open(self._label_map_path(), 'w', encoding='utf-8') as f:
            json.dump(lm, f, indent=2, ensure_ascii=False)
        print(f"  Label map ({len(lm)} labels): {self._label_map_path()}")
        return lm

    def _add_to_label_map(self, label_name):
        lm = self._load_label_map()
        if label_name not in lm:
            lm[label_name] = len(lm)
            with open(self._label_map_path(), 'w', encoding='utf-8') as f:
                json.dump(lm, f, indent=2, ensure_ascii=False)

    def process_video(self, video_path, label_name, video_id,
                      split, augment):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  LOI: Khong mo duoc: {video_path}")
            return False

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  {os.path.basename(video_path)} ({n_frames}f)", end=" ")

        raw_seq    = []
        hand_count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feats, _ = self.extractor.extract_frame(rgb)
            feats    = KeypointNormalizer.normalize_frame(feats)
            raw_seq.append(feats)
            if np.sum(np.abs(feats[cfg.HAND_START:cfg.HAND_END])) > 0.01:
                hand_count += 1
        cap.release()

        if len(raw_seq) < 5:
            print(f"→ QUA NGAN ({len(raw_seq)}f), bo qua")
            return False

        hand_ratio = hand_count / len(raw_seq)
        if hand_ratio < 0.2:
            print(f"→ TAY THAP ({hand_ratio*100:.0f}%)", end=" ")

        normalized = resample_sequence(raw_seq, cfg.SEQ_LEN)
        print(f"→ {len(raw_seq)}f → {cfg.SEQ_LEN}f")

        save_dir = os.path.join(self.output_base, split, label_name)
        os.makedirs(save_dir, exist_ok=True)

        if augment:
            augs        = self.augmenter.generate(normalized)
            upload_list = []
            for suffix, data in augs:
                fn   = f"{video_id}_{suffix}.npy"
                path = os.path.join(save_dir, fn)
                np.save(path, data.astype(np.float32))
                upload_list.append((path, fn))
            print(f"    → {len(augs)} aug files → {split}/{label_name}/")
            if self.hf:
                self.hf.upload_batch(upload_list, label_name, split=split)
        else:
            fn   = f"{video_id}_org.npy"
            path = os.path.join(save_dir, fn)
            np.save(path, normalized.astype(np.float32))
            print(f"    → {fn} → {split}/{label_name}/")
            if self.hf:
                if self.hf.upload_single(path, label_name, fn, split=split):
                    print(f"    [HF] Uploaded")
        return True

    def process_split_label(self, split, label_name):
        augment   = (split == 'train')
        video_dir = os.path.join(self.video_base, split, label_name)
        if not os.path.isdir(video_dir):
            print(f"  CANH BAO: Khong tim thay {video_dir}")
            return 0
        videos = sorted([f for f in os.listdir(video_dir)
                         if Path(f).suffix.lower() in VIDEO_EXTS])
        if not videos:
            return 0
        aug_note = "co aug" if augment else "chi _org"
        print(f"\n  [{split.upper()}] {label_name} ({len(videos)} video, {aug_note})")
        success = 0
        for i, vf in enumerate(videos, 1):
            print(f"  [{i}/{len(videos)}] ", end="")
            if self.process_video(
                    os.path.join(video_dir, vf),
                    label_name, f"{label_name}_{i-1:04d}",
                    split, augment):
                success += 1
        return success

    def process_split(self, split):
        split_dir = os.path.join(self.video_base, split)
        if not os.path.isdir(split_dir):
            print(f"  LOI: Khong tim thay {split_dir}")
            print(f"  Chay organize_dataset.py truoc!")
            return []
        labels = sorted([d for d in os.listdir(split_dir)
                         if os.path.isdir(os.path.join(split_dir, d))])
        if not labels:
            print(f"  [{split}] Khong co label nao"); return []
        print(f"\n{'='*55}")
        print(f"  SPLIT: {split.upper()} ({len(labels)} labels)")
        print(f"{'='*55}")
        done = []
        for lb in labels:
            n = self.process_split_label(split, lb)
            if n > 0:
                done.append(lb)
                self._add_to_label_map(lb)
        return done

    def process_all(self):
        missing = [sp for sp in SPLITS
                   if not os.path.isdir(os.path.join(self.video_base, sp))]
        if missing:
            print(f"\n  LOI: Thieu folder: {missing}")
            print(f"  Chay organize_dataset.py truoc!")
            return
        all_labels = set()
        for split in SPLITS:
            done = self.process_split(split)
            all_labels.update(done)
        if all_labels:
            self._save_label_map(all_labels)
            print(f"\n  Tong: {len(all_labels)} labels da xu ly xong")

    def show_statistics(self):
        print("\n" + "="*65)
        print(" THONG KE FILE .NPY ".center(65))
        print("="*65)
        all_labels = set()
        data = {sp: {} for sp in SPLITS}
        for sp in SPLITS:
            sp_dir = os.path.join(self.output_base, sp)
            if not os.path.isdir(sp_dir): continue
            for lb in os.listdir(sp_dir):
                lp = os.path.join(sp_dir, lb)
                if not os.path.isdir(lp): continue
                npy = list(Path(lp).glob('*.npy'))
                data[sp][lb] = len(npy)
                all_labels.add(lb)
        if not all_labels:
            print("\n  Chua co file .npy nao!")
            return
        print(f"\n  {'Label':<28} {'Train':>8} {'Val':>6} {'Test':>7} {'Tong':>6}")
        print("  " + "-"*58)
        grand = 0
        for lb in sorted(all_labels):
            tr  = data['train'].get(lb, 0)
            va  = data['val'].get(lb, 0)
            te  = data['test'].get(lb, 0)
            tot = tr + va + te; grand += tot
            print(f"  {lb:<28} {tr:>8} {va:>6} {te:>7} {tot:>6}")
        print("  " + "-"*58)
        tr_t = sum(data['train'].values())
        va_t = sum(data['val'].values())
        te_t = sum(data['test'].values())
        print(f"  {'TONG CONG':<28} {tr_t:>8} {va_t:>6} {te_t:>7} {grand:>6}")
        print(f"\n  Shape  : ({cfg.SEQ_LEN}, {cfg.FEAT_DIM})")
        print(f"  Layout : pose(75) + face(90) + hand(126) + interact(55)")
        lm_path = self._label_map_path()
        if os.path.exists(lm_path):
            with open(lm_path) as f:
                lm = json.load(f)
            print(f"  Labels : {len(lm)}")
        else:
            print(f"  CANH BAO: Chua co label_map.json!")
        print("="*65)

    def close(self):
        self.extractor.close()


def _setup_hf():
    hf_repo  = os.getenv("HF_REPO_ID", "").strip()
    hf_token = os.getenv("HF_TOKEN",   "").strip() or None
    print("\n" + "="*55)
    print(" HUGGINGFACE UPLOAD CONFIG ".center(55))
    print("="*55)
    if hf_repo:
        print(f"  HF_REPO_ID = {hf_repo}")
        if input("  Upload .npy len HuggingFace? (y/n, mac dinh y): ").strip().lower() == 'n':
            return None
    else:
        print("  Khong co HF_REPO_ID trong .env")
        hf_repo = input("  Nhap Repo ID (Enter bo qua): ").strip()
        if not hf_repo: return None
        hf_token = input("  HF Token (Enter neu public): ").strip() or None
    return HFUploader(repo_id=hf_repo, token=hf_token)


def main():
    hf        = _setup_hf()
    converter = VideoToNPY(
        video_base  = 'data/videos',
        output_base = 'data/processed',
        hf_uploader = hf,
    )

    while True:
        print("\n" + "="*55)
        print(" VIDEO TO NPY ".center(55, "="))
        print("="*55)
        print("  1. Xu ly tu dong toan bo (train + val + test)")
        print("  2. Xu ly 1 split cu the")
        print("  3. Xu ly 1 video don le")
        print("  4. Xem thong ke")
        print("  5. Thoat")
        print("="*55)
        print(f"  FEAT_DIM = {cfg.FEAT_DIM}  (pose+face+hand+interact)")
        print("="*55)

        ch = input("\n  Chon (1-5): ").strip()

        if ch == '1':
            converter.process_all()
        elif ch == '2':
            print("\n  1. train  2. val  3. test")
            sp_map = {'1': 'train', '2': 'val', '3': 'test',
                      'train': 'train', 'val': 'val', 'test': 'test'}
            split = sp_map.get(input("  > ").strip().lower())
            if not split:
                print("  Khong hop le!"); continue
            done = converter.process_split(split)
            if done:
                all_lm = set(converter._load_label_map().keys()) | set(done)
                converter._save_label_map(all_lm)
        elif ch == '3':
            vpath = input("  Duong dan video: ").strip()
            label = input("  Ten label: ").strip()
            if not vpath or not label:
                print("  Nhap day du!"); continue
            sp = input("  Split (train/val/test, mac dinh train): ").strip() or 'train'
            if sp not in SPLITS:
                print("  Split khong hop le!"); continue
            if not os.path.exists(vpath):
                print(f"  Khong tim thay: {vpath}"); continue
            vid_id = os.path.splitext(os.path.basename(vpath))[0]
            if converter.process_video(vpath, label, vid_id, sp, sp == 'train'):
                converter._add_to_label_map(label)
        elif ch == '4':
            converter.show_statistics()
        elif ch == '5':
            converter.close()
            print("\n  Tam biet!\n")
            break
        else:
            print("  Khong hop le!")


if __name__ == "__main__":
    main()