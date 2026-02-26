"""
video_to_npy.py - Chuyển video VSL → file .npy để train
=========================================================
Cách chạy:
    python video_to_npy.py

Menu:
    1. Xử lý tự động từ Webcam Collector output
    2. Xử lý 1 thư mục video
    3. Xử lý 1 video đơn lẻ
    4. Xem thống kê
    5. Thoát
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

import mediapipe as mp

from vsl.config    import cfg, FACE_KEY_INDICES, KEY_BLENDSHAPES
from vsl.extractor import VideoExtractor
from converter     import HFUploader, KeypointNormalizer, resample_sequence, Augmenter

load_dotenv()


# ══════════════════════════════════════════════════════════
# MAIN CONVERTER
# ══════════════════════════════════════════════════════════

class VideoToNPY:
    """
    Đọc video → trích xuất features (VideoExtractor)
    → normalize → resample → augment → lưu .npy → upload HF.
    """

    def __init__(self, output_dir: str = 'data/processed',
                 sequence_length: int  = cfg.SEQ_LEN,
                 hf_uploader: HFUploader = None):

        self.output_dir  = output_dir
        self.seq_len     = sequence_length
        self.hf          = hf_uploader
        os.makedirs(output_dir, exist_ok=True)

        print("  Khoi tao VideoExtractor...")
        self.extractor = VideoExtractor()
        self.augmenter = Augmenter(
            seq_len   = sequence_length,
            total_dim = cfg.FEAT_DIM,
            pose_dim  = cfg.POSE_END   - cfg.POSE_START,
            face_dim  = cfg.FACE_END   - cfg.FACE_START,
            hand_dim  = cfg.HAND_END   - cfg.HAND_START,
        )
        self._save_feature_meta()

    # ── Metadata ─────────────────────────────────────────

    def _save_feature_meta(self):
        meta = {
            'sequence_length': self.seq_len,
            'total_features_per_frame': cfg.FEAT_DIM,
            'breakdown': {
                'pose (25 upper-body x 3)': cfg.POSE_END - cfg.POSE_START,
                'face (30 key landmarks x 3)': cfg.FACE_END - cfg.FACE_START,
                'hands (21 x 2 x 3)': cfg.HAND_END - cfg.HAND_START,
                'blendshapes (17 key)': cfg.BLEND_END - cfg.BLEND_START,
                'interactions (31)': cfg.INTERACT_END - cfg.INTERACT_START,
            },
            'face_landmark_indices': FACE_KEY_INDICES,
            'key_blendshapes': KEY_BLENDSHAPES,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        path = os.path.join(self.output_dir, 'feature_metadata.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  Feature metadata: {path}")

    # ── Label map ────────────────────────────────────────

    def _load_label_map(self) -> dict:
        path = os.path.join(self.output_dir, 'label_map.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_label_map(self, label_names):
        labels    = sorted(label_names)
        label_map = {name: idx for idx, name in enumerate(labels)}
        path      = os.path.join(self.output_dir, 'label_map.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(label_map, f, indent=2, ensure_ascii=False)
        print(f"  Label map ({len(labels)} labels): {path}")

    def _update_label_map(self, new_label: str):
        """Thêm label mới vào label_map.json + hỏi tên tiếng Việt."""
        # Display names
        dn_path = os.path.join(self.output_dir, 'display_names.json')
        dn = {}
        if os.path.exists(dn_path):
            with open(dn_path, 'r', encoding='utf-8') as f:
                dn = json.load(f)
        if new_label not in dn:
            viet = input(
                f"  Ten tieng Viet cho '{new_label}' (Enter giu nguyen): "
            ).strip()
            dn[new_label] = viet or new_label
            with open(dn_path, 'w', encoding='utf-8') as f:
                json.dump(dn, f, indent=2, ensure_ascii=False)
            print(f"  Da luu: '{new_label}' → '{dn[new_label]}'")

        # Label map
        lm = self._load_label_map()
        if new_label not in lm:
            lm[new_label] = len(lm)
            path = os.path.join(self.output_dir, 'label_map.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(lm, f, indent=2, ensure_ascii=False)
            print(f"  Label map: '{new_label}' → {lm[new_label]}")

    # ── Process single video ─────────────────────────────

    def process_video(self, video_path: str, label_name: str,
                       video_id: str = None,
                       enable_augmentation: bool = True) -> bool:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  LOI: Khong mo duoc: {video_path}")
            return False

        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  Video: {os.path.basename(video_path)} ({n_total} frames)")

        raw_sequence   = []
        hand_detected  = 0

        while True:
            ret, frame = cap.read()
            if not ret: break
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feats, _ = self.extractor.extract_frame(rgb)
            feats    = KeypointNormalizer.normalize_frame(feats)
            raw_sequence.append(feats)

            # Kiểm tra tay có detect không
            hs = cfg.POSE_END + cfg.FACE_END - cfg.FACE_START
            if np.sum(np.abs(feats[cfg.HAND_START:cfg.HAND_END])) > 0.01:
                hand_detected += 1

        cap.release()

        if len(raw_sequence) < 5:
            print(f"  CANH BAO: Qua ngan ({len(raw_sequence)} frames). Bo qua.")
            return False

        hand_ratio = hand_detected / len(raw_sequence)
        if hand_ratio < 0.2:
            print(f"  CANH BAO: Chi detect tay {hand_ratio*100:.0f}% frames.")

        normalized = resample_sequence(raw_sequence, self.seq_len)
        print(f"    {len(raw_sequence)} → {self.seq_len} frames")

        save_dir = os.path.join(self.output_dir, label_name)
        os.makedirs(save_dir, exist_ok=True)
        vid_id = video_id or os.path.splitext(os.path.basename(video_path))[0]

        if enable_augmentation:
            augs        = self.augmenter.generate(normalized)
            upload_list = []
            for suffix, data in augs:
                fn   = f"{vid_id}_{suffix}.npy"
                path = os.path.join(save_dir, fn)
                np.save(path, data.astype(np.float32))
                upload_list.append((path, fn))
            print(f"    {len(augs)} augmented files saved")
            if self.hf:
                self.hf.upload_batch(upload_list, label_name)
        else:
            fn   = f"{vid_id}_org.npy"
            path = os.path.join(save_dir, fn)
            np.save(path, normalized.astype(np.float32))
            print(f"    Saved: {fn}")
            if self.hf:
                if self.hf.upload_single(path, label_name, fn):
                    print(f"    [HF] Uploaded: {fn}")

        return True

    # ── Process folder ────────────────────────────────────

    def process_folder(self, input_folder: str, label_name: str,
                        enable_augmentation: bool = True):
        exts   = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        videos = sorted([f for f in os.listdir(input_folder)
                         if os.path.splitext(f)[1].lower() in exts])
        if not videos:
            print(f"  LOI: Khong tim thay video trong: {input_folder}")
            return

        print(f"\n  {len(videos)} video trong {input_folder}")
        success = 0
        for i, vf in enumerate(videos, 1):
            print(f"\n  [{i}/{len(videos)}]", end=" ")
            if self.process_video(os.path.join(input_folder, vf),
                                   label_name, f"{label_name}_{i-1:04d}",
                                   enable_augmentation):
                success += 1

        print(f"\n  Hoan thanh: {success}/{len(videos)} video")
        self._update_label_map(label_name)

    # ── Process collector output ──────────────────────────

    def process_collector_output(self, collector_dir: str = 'data/videos',
                                  enable_augmentation: bool = True):
        meta_path = os.path.join(collector_dir, 'metadata.json')
        if not os.path.exists(meta_path):
            print(f"  LOI: Khong tim thay metadata: {meta_path}"); return

        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        labels = meta.get('labels', {})
        if not labels:
            print("  Chua co label nao!"); return

        print(f"\n  {len(labels)} labels:")
        for lb, info in labels.items():
            print(f"    - {lb}: {info['num_videos']} video")

        for lb, info in labels.items():
            label_dir = info['path']
            if os.path.isdir(label_dir):
                print(f"\n{'='*55}")
                print(f"  Label: {lb.upper()}")
                print(f"{'='*55}")
                self.process_folder(label_dir, lb, enable_augmentation)
            else:
                print(f"  CANH BAO: Khong tim thay: {label_dir}")

        self._save_label_map(labels.keys())

    # ── Statistics ────────────────────────────────────────

    def show_statistics(self):
        print("\n" + "="*55)
        print(" THONG KE DU LIEU DA XU LY ".center(55))
        print("="*55)
        if not os.path.isdir(self.output_dir):
            print("  Chua co du lieu"); return

        total_org = total_aug = 0
        print(f"\n  {'Label':<25} {'Goc':>6} {'Aug':>10} {'Tong':>7}")
        print("  " + "-"*52)

        for ld in sorted(os.listdir(self.output_dir)):
            lp = os.path.join(self.output_dir, ld)
            if not os.path.isdir(lp): continue
            npy = [f for f in os.listdir(lp) if f.endswith('.npy')]
            org = [f for f in npy if f.endswith('_org.npy')]
            aug = [f for f in npy if not f.endswith('_org.npy')]
            total_org += len(org); total_aug += len(aug)
            print(f"  {ld:<25} {len(org):>6} {len(aug):>10} {len(npy):>7}")

        print("  " + "-"*52)
        print(f"  {'TONG CONG':<25} {total_org:>6} "
              f"{total_aug:>10} {total_org+total_aug:>7}")
        print(f"\n  Feature/frame : {cfg.FEAT_DIM}")
        print(f"  Sequence len  : {self.seq_len}")
        print(f"  Shape moi file: ({self.seq_len}, {cfg.FEAT_DIM})")
        print("="*55)

    def close(self):
        self.extractor.close()


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def _setup_hf() -> HFUploader | None:
    """Hỏi config HuggingFace, trả về HFUploader hoặc None."""
    hf_repo  = os.getenv("HF_REPO_ID", "").strip()
    hf_token = os.getenv("HF_TOKEN",   "").strip() or None

    print("\n" + "="*55)
    print(" HUGGINGFACE UPLOAD CONFIG ".center(55))
    print("="*55)

    if hf_repo:
        print(f"  HF_REPO_ID = {hf_repo}")
        ans = input("  Upload npy len HuggingFace? (y/n, mac dinh y): ").strip().lower()
        if ans == 'n':
            print("  Bo qua HuggingFace.")
            return None
    else:
        print("  Khong co HF_REPO_ID trong .env")
        hf_repo = input("  Nhap Repo ID (Enter bo qua): ").strip()
        if not hf_repo:
            return None
        hf_token = input("  HF Token (Enter neu public): ").strip() or None

    return HFUploader(repo_id=hf_repo, token=hf_token)


def main():
    hf        = _setup_hf()
    converter = VideoToNPY(output_dir='data/processed', hf_uploader=hf)

    while True:
        print("\n" + "="*55)
        print(" VIDEO TO NPY ".center(55, "="))
        print("="*55)
        print("  1. Xu ly tu dong tu Webcam Collector output")
        print("  2. Xu ly 1 thu muc video")
        print("  3. Xu ly 1 video don le")
        print("  4. Xem thong ke")
        print("  5. Thoat")
        print("="*55)

        ch = input("\n  Chon (1-5): ").strip()

        if ch == '1':
            folder = input("  Thu muc collector (mac dinh: data/videos): ").strip()
            folder = folder or 'data/videos'
            aug    = input("  Bat augmentation? (y/n, mac dinh y): ").strip().lower()
            converter.process_collector_output(folder, aug != 'n')

        elif ch == '2':
            folder = input("  Duong dan thu muc video: ").strip()
            label  = input("  Ten label: ").strip()
            if not folder or not label:
                print("  Nhap day du!"); continue
            aug = input("  Bat augmentation? (y/n, mac dinh y): ").strip().lower()
            if os.path.isdir(folder):
                converter.process_folder(folder, label, aug != 'n')
            else:
                print(f"  Khong tim thay: {folder}")

        elif ch == '3':
            vpath = input("  Duong dan video: ").strip()
            label = input("  Ten label: ").strip()
            if not vpath or not label:
                print("  Nhap day du!"); continue
            aug = input("  Bat augmentation? (y/n, mac dinh y): ").strip().lower()
            if os.path.exists(vpath):
                if converter.process_video(vpath, label, enable_augmentation=(aug != 'n')):
                    converter._update_label_map(label)
            else:
                print(f"  Khong tim thay: {vpath}")

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