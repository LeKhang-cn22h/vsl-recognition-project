"""
video_to_npy_lite.py - Convert videos to 171-dim .npy
Chạy: python src/video_to_npy_lite.py --clean
"""
import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path

from vsl.config_lite import cfg
from vsl.extractor_lite import VideoExtractorLite, KeypointNormalizerLite


def resample_sequence(raw_seq, target_len):
    n = len(raw_seq)
    if n == target_len:
        return np.array(raw_seq)
    indices = np.linspace(0, n - 1, target_len)
    result = []
    for i in indices:
        lo, hi = int(np.floor(i)), min(int(np.ceil(i)), n - 1)
        w = i - lo
        result.append(raw_seq[lo] * (1 - w) + raw_seq[hi] * w)
    return np.array(result, dtype=np.float32)


class VideoToNPYLite:
    def __init__(self, video_base='data/videos', output_base='data/processed_lite'):
        self.video_base = video_base
        self.output_base = output_base
        os.makedirs(output_base, exist_ok=True)
        
        print(f"  Init VideoExtractorLite ({cfg.FEAT_DIM} dim)...")
        self.extractor = VideoExtractorLite()
    
    def clean(self):
        for split in ['train', 'val', 'test']:
            p = os.path.join(self.output_base, split)
            if os.path.exists(p):
                shutil.rmtree(p)
        lm = os.path.join(self.output_base, 'label_map.json')
        if os.path.exists(lm):
            os.remove(lm)
        print("  Cleaned!")
    
    def process_video(self, video_path, label, video_id, split, augment):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        raw_seq = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feat, _ = self.extractor.extract_frame(rgb)
            feat = KeypointNormalizerLite.normalize_frame(feat)
            raw_seq.append(feat)
        cap.release()
        
        if len(raw_seq) < 5:
            return False
        
        seq = resample_sequence(raw_seq, cfg.SEQ_LEN)
        
        save_dir = os.path.join(self.output_base, split, label)
        os.makedirs(save_dir, exist_ok=True)
        
        if augment:
            # Original
            np.save(f"{save_dir}/{video_id}_org.npy", seq.astype(np.float32))
            # Noise augmentations
            for i, std in enumerate([0.003, 0.006, 0.009]):
                aug = seq + np.random.normal(0, std, seq.shape).astype(np.float32)
                np.save(f"{save_dir}/{video_id}_noise{i}.npy", aug)
            # Scale augmentations
            for i, s in enumerate([0.95, 1.05]):
                aug = seq.copy()
                aug[:, 45:171] *= s  # Scale hands only
                np.save(f"{save_dir}/{video_id}_scale{i}.npy", aug)
        else:
            np.save(f"{save_dir}/{video_id}_org.npy", seq.astype(np.float32))
        
        return True
    
    def process_all(self, clean=False):
        if clean:
            self.clean()
        
        all_labels = set()
        
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.video_base, split)
            if not os.path.isdir(split_dir):
                continue
            
            augment = (split == 'train')
            print(f"\n=== {split.upper()} ===")
            
            for label in sorted(os.listdir(split_dir)):
                label_dir = os.path.join(split_dir, label)
                if not os.path.isdir(label_dir):
                    continue
                
                videos = [f for f in os.listdir(label_dir) if f.endswith('.mp4')]
                if not videos:
                    continue
                
                print(f"  {label}: {len(videos)} videos")
                all_labels.add(label)
                
                for i, vf in enumerate(sorted(videos)):
                    self.process_video(
                        os.path.join(label_dir, vf),
                        label, f'{label}_{i:04d}',
                        split, augment
                    )
        
        # Save label map
        label_map = {name: idx for idx, name in enumerate(sorted(all_labels))}
        with open(os.path.join(self.output_base, 'label_map.json'), 'w') as f:
            json.dump(label_map, f, indent=2)
        
        print(f"\nDone! {len(label_map)} labels")
        print(f"Output: {self.output_base}")
    
    def close(self):
        self.extractor.close()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean', action='store_true')
    args = ap.parse_args()
    
    converter = VideoToNPYLite()
    converter.process_all(clean=args.clean)
    converter.close()