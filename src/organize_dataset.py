"""
Tổ chức lại dataset vào train/val/test
=============================================================
Cấu trúc HIỆN TẠI:
    data/videos/train/train/<label>/*.mp4

Cấu trúc SAU KHI CHẠY:
    data/videos/train/<label>/*.mp4
    data/videos/val/<label>/*.mp4
    data/videos/test/<label>/*.mp4

Nguyên tắc chia:
    - Test  : video cuối cùng mỗi label
    - Val   : video gần cuối mỗi label
    - Train : tất cả còn lại

Chạy:
    python organize_dataset.py
"""

import os
import shutil
from pathlib import Path

# ══════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════

# Thư mục hiện tại chứa các label  (train/train/)
LABEL_SRC_DIR = r'data\videos\train'

# Thư mục gốc — train/ val/ test/ sẽ nằm ở đây  (data/videos/)
SPLITS_ROOT   = r'data\videos'

MIN_VIDEOS    = 3
VIDEO_EXTS    = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

# ══════════════════════════════════════════════════════════

def scan_labels(src: str) -> dict:
    result = {}
    skip   = {'train', 'val', 'test'}
    for entry in sorted(Path(src).iterdir()):
        if not entry.is_dir(): continue
        if entry.name in skip: continue
        videos = sorted([
            str(p) for p in entry.iterdir()
            if p.suffix.lower() in VIDEO_EXTS
        ])
        if videos:
            result[entry.name] = videos
    return result


def move_file(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)


def organize():
    print("\n" + "="*62)
    print("  ORGANIZE DATASET — CHIA TRAIN / VAL / TEST")
    print("="*62)
    print(f"\n  Source : {LABEL_SRC_DIR}")
    print(f"\n  Output sau khi chia:")
    print(f"    {SPLITS_ROOT}\\train\\<label>\\")
    print(f"    {SPLITS_ROOT}\\val\\<label>\\")
    print(f"    {SPLITS_ROOT}\\test\\<label>\\")

    if not os.path.isdir(LABEL_SRC_DIR):
        print(f"\n❌ Không tìm thấy: {LABEL_SRC_DIR}"); return

    labels = scan_labels(LABEL_SRC_DIR)
    if not labels:
        print(f"\n Không tìm thấy label nào trong {LABEL_SRC_DIR}"); return

    ok, warn = {}, {}
    for lb, vids in labels.items():
        (ok if len(vids) >= MIN_VIDEOS else warn)[lb] = vids

    # ── Preview bảng ──
    print(f"\n  Tìm thấy {len(labels)} labels:\n")
    print(f"  {'Label':<30} {'Tổng':>6} {'Train':>7} {'Val':>5} {'Test':>6}")
    print("  " + "-"*57)
    for lb, vids in sorted(ok.items()):
        n = len(vids)
        print(f"  {lb:<30} {n:>6} {n-2:>7} {'1':>5} {'1':>6}")

    if warn:
        print(f"\n  ⚠️  Bỏ qua (cần ≥ {MIN_VIDEOS} video):")
        for lb, vids in warn.items():
            print(f"     {lb:<30} chỉ có {len(vids)} video")

    if not ok:
        print("\n❌ Không có label nào đủ video."); return

    print()
    confirm = input("  Xác nhận di chuyển file? (y/n): ").strip().lower()
    if confirm != 'y':
        print("  Đã huỷ."); return

    # ── Di chuyển ──
    print("\n  Đang di chuyển...\n")
    count = {'train': 0, 'val': 0, 'test': 0}

    for lb, vids in sorted(ok.items()):
        test_vid   = vids[-1]
        val_vid    = vids[-2]
        train_vids = vids[:-2]

        # val
        dst = os.path.join(SPLITS_ROOT, 'val', lb, os.path.basename(val_vid))
        move_file(val_vid, dst)
        print(f"  [val  ] {lb}/{os.path.basename(val_vid)}")
        count['val'] += 1

        # test
        dst = os.path.join(SPLITS_ROOT, 'test', lb, os.path.basename(test_vid))
        move_file(test_vid, dst)
        print(f"  [test ] {lb}/{os.path.basename(test_vid)}")
        count['test'] += 1

        # train — move lên data/videos/train/<label>/
        for vpath in train_vids:
            dst = os.path.join(SPLITS_ROOT, 'train', lb, os.path.basename(vpath))
            move_file(vpath, dst)
            count['train'] += 1
        print(f"  [train] {lb}/ — {len(train_vids)} video")

        # Xóa label folder cũ nếu rỗng
        old_dir = os.path.join(LABEL_SRC_DIR, lb)
        try:
            if os.path.isdir(old_dir) and not os.listdir(old_dir):
                os.rmdir(old_dir)
        except Exception:
            pass

    # Xóa train/train/ nếu rỗng
    try:
        if os.path.isdir(LABEL_SRC_DIR) and not os.listdir(LABEL_SRC_DIR):
            os.rmdir(LABEL_SRC_DIR)
            # Xóa luôn thư mục cha train/ cũ nếu rỗng
            old_train = os.path.dirname(LABEL_SRC_DIR)
            if os.path.isdir(old_train) and not os.listdir(old_train):
                os.rmdir(old_train)
            print(f"\n  🗑️  Đã dọn folder cũ: {LABEL_SRC_DIR}")
    except Exception:
        pass

    print(f"\n{'='*62}")
    print(f"  ✅ Hoàn thành!")
    print(f"  Train : {count['train']} video")
    print(f"  Val   : {count['val']} video")
    print(f"  Test  : {count['test']} video")
    print(f"\n  Cấu trúc mới:")
    print(f"    data/videos/train/<label>/*.mp4")
    print(f"    data/videos/val/<label>/*.mp4")
    print(f"    data/videos/test/<label>/*.mp4")
    print(f"{'='*62}\n")


if __name__ == '__main__':
    organize()