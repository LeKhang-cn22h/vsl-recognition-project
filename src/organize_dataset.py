import os
import shutil
import random
from pathlib import Path

# ══════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════

LABEL_SRC_DIR = r'data\videos\train\train'
SPLITS_ROOT   = r'data\videos'

MIN_VIDEOS = 3
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

random.seed(42)

# ══════════════════════════════════════════════════════════


def scan_labels(src: str) -> dict:
    """
    Quét các folder label và kiểm tra video đã được gán nhãn chưa
    """
    result = {}

    for entry in sorted(Path(src).iterdir()):

        if not entry.is_dir():
            continue

        videos = [
            str(p) for p in entry.iterdir()
            if p.suffix.lower() in VIDEO_EXTS
        ]

        if not videos:
            continue

        result[entry.name] = videos

    return result


def move_file(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)


def split_ratio(videos):
    """
    Chia đúng tỉ lệ 70 / 15 / 15
    """

    random.shuffle(videos)

    n = len(videos)

    n_train = int(n * 0.7)
    n_val   = int(n * 0.15)

    n_test  = n - n_train - n_val

    # đảm bảo tối thiểu
    if n_val == 0:
        n_val = 1
        n_train -= 1

    if n_test == 0:
        n_test = 1
        n_train -= 1

    train = videos[:n_train]
    val   = videos[n_train:n_train+n_val]
    test  = videos[n_train+n_val:]

    return train, val, test


def organize():

    print("\n" + "="*60)
    print("ORGANIZE DATASET — TRAIN / VAL / TEST")
    print("="*60)

    if not os.path.isdir(LABEL_SRC_DIR):
        print(f"\nKhông tìm thấy: {LABEL_SRC_DIR}")
        return

    labels = scan_labels(LABEL_SRC_DIR)

    if not labels:
        print("Không có label nào.")
        return

    print(f"\nTìm thấy {len(labels)} labels\n")

    print(f"{'Label':<30} {'Total':>6} {'Train':>7} {'Val':>5} {'Test':>6}")
    print("-"*60)

    preview = {}

    for lb, vids in labels.items():

        n = len(vids)

        train, val, test = split_ratio(vids.copy())

        preview[lb] = (train, val, test)

        print(f"{lb:<30} {n:>6} {len(train):>7} {len(val):>5} {len(test):>6}")

    confirm = input("\nXác nhận di chuyển file? (y/n): ").lower()

    if confirm != "y":
        print("Đã huỷ.")
        return

    count = {'train':0,'val':0,'test':0}

    print("\nĐang di chuyển...\n")

    for lb,(train,val,test) in preview.items():

        for v in train:
            dst = os.path.join(SPLITS_ROOT,'train',lb,os.path.basename(v))
            move_file(v,dst)
            count['train']+=1

        for v in val:
            dst = os.path.join(SPLITS_ROOT,'val',lb,os.path.basename(v))
            move_file(v,dst)
            count['val']+=1

        for v in test:
            dst = os.path.join(SPLITS_ROOT,'test',lb,os.path.basename(v))
            move_file(v,dst)
            count['test']+=1

    print("\n"+"="*60)
    print("HOÀN THÀNH")
    print("="*60)

    print(f"Train : {count['train']} video")
    print(f"Val   : {count['val']} video")
    print(f"Test  : {count['test']} video")

    print("\nCấu trúc mới:")

    print("data/videos/train/<label>/*.mp4")
    print("data/videos/val/<label>/*.mp4")
    print("data/videos/test/<label>/*.mp4")

    print("="*60)


if __name__ == '__main__':
    organize()