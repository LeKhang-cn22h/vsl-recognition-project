"""
 Tổ chức lại dataset vào train/val/test
=============================================================
Chuyển từ cấu trúc cũ:
    data/videos/<label>/*.mp4

Sang cấu trúc mới:
    data/videos/train/<label>/*.mp4
    data/videos/val/<label>/*.mp4
    data/videos/test/<label>/*.mp4

Nguyên tắc chia:
    - Train : lấy TẤT CẢ video trừ val/test
    - Val   : 1 video gốc mỗi label (video gần cuối)
    - Test  : 1 video gốc mỗi label (video cuối cùng)
    - Nếu label có < 3 video → cảnh báo, không chia

Chạy:
    python organize_dataset.py
"""

import os
import shutil
import json
from pathlib import Path

# ══════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════

VIDEO_SRC_DIR = 'data/videos'        # thư mục cũ (flat)
VIDEO_DST_DIR = 'data/videos'        # thư mục mới (có train/val/test)
MIN_VIDEOS_PER_LABEL = 3             # tối thiểu để chia được

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════

def scan_labels(src_dir: str) -> dict:
    """
    Quét thư mục cũ, trả về dict:
    { label_name: [sorted list of video paths] }
    Bỏ qua các thư mục train/val/test (đã chia rồi).
    """
    result = {}
    skip   = {'train', 'val', 'test'}

    for entry in sorted(os.scandir(src_dir), key=lambda e: e.name):
        if not entry.is_dir(): continue
        if entry.name in skip:  continue

        videos = sorted([
            str(fp) for fp in Path(entry.path).glob('*')
            if fp.suffix.lower() in VIDEO_EXTS
        ])
        if videos:
            result[entry.name] = videos

    return result


def already_organized(src_dir: str) -> bool:
    """Kiểm tra đã có cấu trúc train/val/test chưa."""
    for split in ['train', 'val', 'test']:
        if os.path.isdir(os.path.join(src_dir, split)):
            return True
    return False


def move_video(src: str, dst: str, dry_run: bool = False) -> bool:
    """Di chuyển 1 file, tạo thư mục đích nếu chưa có."""
    if dry_run:
        print(f"    [DRY] {src} → {dst}")
        return True
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        return True
    except Exception as e:
        print(f"    LOI: {e}")
        return False


# ══════════════════════════════════════════════════════════
# MAIN LOGIC
# ══════════════════════════════════════════════════════════

def organize(src_dir: str = VIDEO_SRC_DIR,
             dry_run: bool = False) -> dict:
    """
    Chia video vào train/val/test.
    dry_run=True: chỉ in ra sẽ làm gì, không thực sự di chuyển.
    Trả về dict thống kê.
    """
    labels = scan_labels(src_dir)

    if not labels:
        print("  Khong tim thay label nao trong:", src_dir)
        return {}

    print(f"\n  Tim thay {len(labels)} labels:")
    for lb, vids in labels.items():
        print(f"    {lb:<30} {len(vids)} video")

    # ── Kiểm tra labels đủ video không ──
    ok_labels   = {}
    warn_labels = {}
    for lb, vids in labels.items():
        if len(vids) >= MIN_VIDEOS_PER_LABEL:
            ok_labels[lb]   = vids
        else:
            warn_labels[lb] = vids

    if warn_labels:
        print(f"\n  CANH BAO: {len(warn_labels)} label CHUA DU {MIN_VIDEOS_PER_LABEL} video:")
        for lb, vids in warn_labels.items():
            print(f"    {lb:<30} chi co {len(vids)} video → BO QUA")
        print(f"  (Can quay them video cho cac label nay truoc khi chia)")

    if not ok_labels:
        print("\n  Khong co label nao du video de chia. Thoat.")
        return {}

    # ── Chia từng label ──
    stats  = {}
    splits = {'train': [], 'val': [], 'test': []}

    for lb, vids in ok_labels.items():
        n        = len(vids)
        # Luôn lấy video cuối cùng cho test, gần cuối cho val
        test_vid = vids[-1]
        val_vid  = vids[-2]
        train_vids = vids[:-2]

        splits['train'].append((lb, train_vids))
        splits['val'].append((lb,  [val_vid]))
        splits['test'].append((lb, [test_vid]))

        stats[lb] = {
            'train': len(train_vids),
            'val':   1,
            'test':  1,
            'total': n,
        }

    # ── In preview ──
    print(f"\n  {'Label':<30} {'Train':>7} {'Val':>5} {'Test':>6} {'Tong':>6}")
    print("  " + "-"*57)
    for lb, s in stats.items():
        print(f"  {lb:<30} {s['train']:>7} {s['val']:>5} {s['test']:>6} {s['total']:>6}")

    if dry_run:
        print("\n  [DRY RUN] Khong thuc su di chuyen file.")
        return stats

    # ── Xác nhận ──
    confirm = input("\n  Xac nhan di chuyen file? (y/n): ").strip().lower()
    if confirm != 'y':
        print("  Da huy.")
        return {}

    # ── Di chuyển file ──
    print("\n  Dang di chuyen...")
    for split_name, label_list in splits.items():
        for lb, vids in label_list:
            dst_dir = os.path.join(src_dir, split_name, lb)
            os.makedirs(dst_dir, exist_ok=True)
            for vpath in vids:
                fname = os.path.basename(vpath)
                dst   = os.path.join(dst_dir, fname)
                ok    = move_video(vpath, dst)
                if ok:
                    print(f"    [{split_name}] {lb}/{fname}")

            # Xóa thư mục label gốc nếu rỗng
            old_dir = os.path.join(src_dir, lb)
            try:
                if os.path.isdir(old_dir) and not os.listdir(old_dir):
                    os.rmdir(old_dir)
            except Exception:
                pass

    print(f"\n  Hoan thanh! Cau truc moi:")
    print(f"  {src_dir}/train/<label>/*.mp4")
    print(f"  {src_dir}/val/<label>/*.mp4")
    print(f"  {src_dir}/test/<label>/*.mp4")

    return stats


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print(" ORGANIZE DATASET - CHIA TRAIN/VAL/TEST ".center(60, "="))
    print("="*60)
    print(f"\n  Thu muc nguon : {VIDEO_SRC_DIR}")
    print(f"  Toi thieu     : {MIN_VIDEOS_PER_LABEL} video/label")
    print(f"\n  Quy tac chia:")
    print(f"    Train : tat ca video tru 2 cai cuoi")
    print(f"    Val   : video thu N-1 (gan cuoi)")
    print(f"    Test  : video thu N   (cuoi cung)")

    if not os.path.isdir(VIDEO_SRC_DIR):
        print(f"\n  LOI: Khong tim thay thu muc: {VIDEO_SRC_DIR}")
        return

    if already_organized(VIDEO_SRC_DIR):
        print(f"\n  CANH BAO: Da tim thay thu muc train/val/test trong {VIDEO_SRC_DIR}")
        print(f"  Co the da chia roi. Kiem tra lai truoc khi chay.")
        ans = input("  Tiep tuc anyway? (y/n): ").strip().lower()
        if ans != 'y':
            return

    # Preview trước
    print("\n  [Preview - chua di chuyen gi]")
    organize(VIDEO_SRC_DIR, dry_run=True)

    # Thực sự làm
    print("\n" + "-"*60)
    organize(VIDEO_SRC_DIR, dry_run=False)


if __name__ == '__main__':
    main()