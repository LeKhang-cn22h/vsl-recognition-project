"""
check_data.py - Kiểm tra số lượng file đã split (train/val/test)
=================================================================
Cấu trúc mong đợi:
    data/videos/train/<label>/*.mp4
    data/videos/val/<label>/*.mp4
    data/videos/test/<label>/*.mp4

    data/processed/train/<label>/*.npy
    data/processed/val/<label>/*.npy
    data/processed/test/<label>/*.npy

Chạy:
    python src/check_data.py
    python src/check_data.py --videos data/videos --processed data/processed
"""

import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ['train', 'val', 'test']


# ══════════════════════════════════════════════════════════════════
# COUNT
# ══════════════════════════════════════════════════════════════════

def count_files(root: Path, ext: str) -> dict:
    """{split: {label: count}}"""
    result = {}
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            result[split] = {}
            continue
        labels = {}
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            n = len(list(label_dir.glob(f'*{ext}')))
            if n > 0:
                labels[label_dir.name] = n
        result[split] = labels
    return result


def all_labels(data: dict) -> list:
    labels = set()
    for sd in data.values():
        labels.update(sd.keys())
    return sorted(labels)


# ══════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ══════════════════════════════════════════════════════════════════

def sep(char='─', n=72): print(char * n)
def header(title): sep('═'); print(f"  {title}"); sep('═')


def print_table(data: dict, ext: str):
    labels = all_labels(data)
    if not labels:
        print(f"  (Không tìm thấy file {ext} nào)\n")
        return {s: 0 for s in SPLITS}, 0

    lw = max(len(lb) for lb in labels) + 2
    cw = 8

    print(f"  {'Label':<{lw}}" + "".join(f"  {s:>{cw}}" for s in SPLITS) + f"  {'TOTAL':>{cw}}")
    sep()

    split_totals = {s: 0 for s in SPLITS}
    grand_total  = 0
    warnings     = []

    for label in labels:
        row = f"  {label:<{lw}}"
        total = 0
        for s in SPLITS:
            n = data[s].get(label, 0)
            split_totals[s] += n
            total += n
            row += f"  {'–':>{cw}}" if n == 0 else f"  {n:>{cw}}"
            if n == 0:
                warnings.append(f"  ⚠  '{label}' KHÔNG có file trong [{s}]")
            elif n < 5:
                warnings.append(f"  ⚠  '{label}' chỉ có {n} file trong [{s}] — quá ít")
        grand_total += total
        row += f"  {total:>{cw}}"
        print(row)

    sep()
    print(f"  {'TOTAL':<{lw}}" + "".join(f"  {split_totals[s]:>{cw}}" for s in SPLITS)
          + f"  {grand_total:>{cw}}")
    print()

    if warnings:
        print("  WARNINGS:")
        for w in warnings: print(w)
        print()

    return split_totals, grand_total


def print_ratio(split_totals: dict, ext: str):
    grand = sum(split_totals.values())
    if grand == 0:
        return
    print(f"  Tỉ lệ split ({ext}):")
    sep('─', 55)
    for s in SPLITS:
        pct = split_totals[s] / grand * 100
        bar = '█' * int(pct / 2)
        print(f"  {s:<6}  {split_totals[s]:>7}  ({pct:5.1f}%)  {bar}")
    print()


def check_balance(train: dict):
    if not train:
        print("  (Không có data train)\n")
        return
    min_c = min(train.values())
    max_c = max(train.values())
    ratio = max_c / min_c if min_c > 0 else float('inf')
    scale = 40 / max_c if max_c > 0 else 1
    lw    = max(len(lb) for lb in train) + 2

    for lb, n in sorted(train.items(), key=lambda x: x[1]):
        bar  = '█' * max(1, int(n * scale))
        flag = '  ← ÍT NHẤT' if n == min_c else ('  ← NHIỀU NHẤT' if n == max_c else '')
        print(f"  {lb:<{lw}} {n:>7}  {bar}{flag}")
    print()

    if ratio > 3:
        print(f"  ⚠  Imbalanced nặng! max/min = {ratio:.1f}x")
        print(f"     → Nên oversample class ít hoặc dùng weighted loss")
    elif ratio > 1.5:
        print(f"  ⚠  Hơi mất cân bằng (max/min = {ratio:.1f}x) — chấp nhận được")
    else:
        print(f"  ✓  Balanced tốt (max/min = {ratio:.1f}x)")
    print()


def check_consistency(mp4_data: dict, npy_data: dict):
    issues = []
    for s in SPLITS:
        m = set(mp4_data[s].keys())
        n = set(npy_data[s].keys())
        for lb in sorted(m - n):
            issues.append(f"  ✗ [{s}] '{lb}': có mp4 nhưng KHÔNG có npy → chưa process?")
        for lb in sorted(n - m):
            issues.append(f"  ✗ [{s}] '{lb}': có npy nhưng KHÔNG có mp4 → mp4 đã xóa?")
        for lb in sorted(m & n):
            nm, nn = mp4_data[s][lb], npy_data[s][lb]
            if nm != nn:
                tag = 'thiếu' if nn < nm else 'thừa'
                issues.append(f"  ✗ [{s}] '{lb}': mp4={nm} ≠ npy={nn} ({tag} {abs(nm-nn)})")
    if issues:
        for i in issues: print(i)
    else:
        print("  ✓ Tất cả mp4 ↔ npy khớp hoàn toàn.")
    print()


def check_recommendations(npy_totals: dict, npy_data: dict):
    grand = sum(npy_totals.values())
    if grand == 0:
        print("  (Không có data)\n")
        return

    val_n  = npy_totals.get('val',  0)
    test_n = npy_totals.get('test', 0)
    tr_n   = npy_totals.get('train',0)
    val_pct  = val_n  / grand * 100
    test_pct = test_n / grand * 100
    n_classes = len(all_labels(npy_data))

    # Val ratio
    if val_pct < 10:
        print(f"  ⚠  Val chỉ {val_pct:.1f}% ({val_n} samples) → QUÁ NHỎ")
        print(f"     Đây là nguyên nhân val loss zigzag trong training chart")
        print(f"     → Cần ít nhất {int(grand * 0.15):,} samples ({int(grand*0.15/n_classes)}/class)")
    else:
        print(f"  ✓  Val {val_pct:.1f}% — OK")

    # Test ratio
    if test_pct < 10:
        print(f"  ⚠  Test chỉ {test_pct:.1f}% ({test_n} samples) → QUÁ NHỎ")
        print(f"     → Cần ít nhất {int(grand * 0.10):,} samples")
    else:
        print(f"  ✓  Test {test_pct:.1f}% — OK")

    # Samples tuyệt đối
    if n_classes > 0:
        avg_val_per_class = val_n // n_classes
        if avg_val_per_class < 20:
            print(f"  ⚠  Trung bình chỉ {avg_val_per_class} samples/class trong val")
            print(f"     → Cần ít nhất 20-30 samples/class để val ổn định")

    print()
    print(f"  Phân phối lý tưởng cho {grand:,} samples:")
    sep('─', 50)
    for s, pct in [('train', 0.75), ('val', 0.15), ('test', 0.10)]:
        ideal = int(grand * pct)
        current = npy_totals.get(s, 0)
        diff = current - ideal
        flag = f"  (+{diff})" if diff > 0 else f"  ({diff})" if diff != 0 else "  ✓"
        print(f"  {s:<6}  ideal={ideal:>7}  current={current:>7}{flag}")
    print()


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos',    default='data/videos',
                    help='Folder mp4 đã split (default: data/videos/)')
    ap.add_argument('--processed', default='data/processed',
                    help='Folder npy đã split (default: data/processed/)')
    args = ap.parse_args()

    videos_root    = _PROJECT_ROOT / args.videos
    processed_root = _PROJECT_ROOT / args.processed

    print()
    header("DATA CHECKER — VSL Recognition Project")
    print(f"  Project  : {_PROJECT_ROOT}")
    print(f"  Videos   : {videos_root}  {'✓' if videos_root.is_dir() else '✗ KHÔNG TỒN TẠI'}")
    print(f"  Processed: {processed_root}  {'✓' if processed_root.is_dir() else '✗ KHÔNG TỒN TẠI'}")
    print()

    mp4_data = count_files(videos_root,    '.mp4') if videos_root.is_dir()    else {s:{} for s in SPLITS}
    npy_data = count_files(processed_root, '.npy') if processed_root.is_dir() else {s:{} for s in SPLITS}

    # ── MP4 ───────────────────────────────────────────────────────
    header("📹  MP4 — Raw Videos")
    mp4_totals, mp4_grand = print_table(mp4_data, '.mp4')
    if mp4_grand > 0:
        print_ratio(mp4_totals, '.mp4')

    # ── NPY ───────────────────────────────────────────────────────
    header("🔢  NPY — Processed Features")
    npy_totals, npy_grand = print_table(npy_data, '.npy')
    if npy_grand > 0:
        print_ratio(npy_totals, '.npy')

    # ── Consistency ───────────────────────────────────────────────
    if mp4_grand > 0 and npy_grand > 0:
        header("🔍  Consistency  mp4 ↔ npy")
        check_consistency(mp4_data, npy_data)

    # ── Class balance ─────────────────────────────────────────────
    src_data = mp4_data if mp4_grand > 0 else npy_data
    src_name = "MP4" if mp4_grand > 0 else "NPY"
    header(f"⚖️   Class Balance — {src_name} Train")
    check_balance(src_data.get('train', {}))

    # ── Recommendations ───────────────────────────────────────────
    header("💡  Recommendations")
    check_recommendations(npy_totals, npy_data)

    sep('═')
    print("  XONG.")
    sep('═')
    print()


if __name__ == '__main__':
    main()