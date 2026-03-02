"""
download_from_hf.py - Tải file từ HuggingFace Dataset về máy
=============================================================
Cách chạy:
    python download_from_hf.py

Cấu hình (.env):
    HF_TOKEN=hf_xxxxxxxxxxxxxx
    HF_REPO_ID=KhangCN/Video_VSL

Cấu trúc repo HuggingFace:
    videos/
    ├── train/<label>/*.mp4
    ├── val/<label>/*.mp4
    └── test/<label>/*.mp4
    processed/
    ├── train/<label>/*.npy
    ├── val/<label>/*.npy
    └── test/<label>/*.npy

Cấu trúc sau khi tải về:
    data/
    ├── videos/train|val|test/<label>/*.mp4
    └── processed/train|val|test/<label>/*.npy
"""

import os
import sys
import shutil
from dotenv import load_dotenv

load_dotenv()

try:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("  LOI: Chua cai huggingface_hub!")
    print("  Chay: pip install huggingface_hub")
    sys.exit(1)

SPLITS = ['train', 'val', 'test']


# ══════════════════════════════════════════════════════════
# DOWNLOADER
# ══════════════════════════════════════════════════════════

class HFDownloader:
    def __init__(self, repo_id: str, token: str = None):
        self.repo_id = repo_id
        self.token   = token
        self.api     = HfApi(token=token)
        print(f"  Repo : {repo_id}")
        print(f"  Token: {'co' if token else 'khong (public repo)'}")

    def list_files(self, prefix: str = "") -> list[str]:
        try:
            files = list(list_repo_files(
                repo_id=self.repo_id, repo_type="dataset", token=self.token))
            if prefix:
                files = [f for f in files if f.startswith(prefix)]
            return sorted(files)
        except Exception as e:
            print(f"  LOI doc danh sach file: {e}")
            return []

    def get_labels(self, file_type: str, split: str) -> list[str]:
        """Lấy danh sách labels trong file_type/split/."""
        files  = self.list_files(prefix=f"{file_type}/{split}/")
        labels = set()
        for f in files:
            parts = f.split("/")
            if len(parts) >= 3:
                labels.add(parts[2])
        return sorted(labels)

    def count_files(self, file_type: str, split: str, label: str) -> int:
        return len(self.list_files(prefix=f"{file_type}/{split}/{label}/"))

    def download_file(self, remote_path: str, local_path: str):
        """Tải 1 file. Trả về True=tải mới, None=bỏ qua, False=lỗi."""
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return None
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            hf_hub_download(
                repo_id=self.repo_id, filename=remote_path,
                repo_type="dataset", token=self.token,
                local_dir=".", local_dir_use_symlinks=False,
            )
            if remote_path != local_path and os.path.exists(remote_path):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                shutil.move(remote_path, local_path)
                # Dọn thư mục rỗng
                parent = os.path.dirname(remote_path)
                while parent and parent != ".":
                    if os.path.isdir(parent) and not os.listdir(parent):
                        os.rmdir(parent); parent = os.path.dirname(parent)
                    else: break
            return True
        except Exception as e:
            print(f"  LOI: {e}"); return False

    def download_split_label(self, file_type: str, split: str,
                              label: str, base_out: str) -> tuple:
        """
        Tải tất cả file của 1 split/label.
        Lưu vào base_out/<split>/<label>/
        """
        ext   = ".mp4" if file_type == "videos" else ".npy"
        files = self.list_files(prefix=f"{file_type}/{split}/{label}/")
        files = [f for f in files if f.endswith(ext)]

        if not files:
            print(f"    Khong co file {ext}")
            return 0, 0, 0

        out_dir = os.path.join(base_out, split, label)
        os.makedirs(out_dir, exist_ok=True)

        downloaded = skipped = failed = 0
        for i, remote_path in enumerate(files, 1):
            fname      = os.path.basename(remote_path)
            local_path = os.path.join(out_dir, fname)
            print(f"    [{i:>3}/{len(files)}] {fname:<45}", end=" ", flush=True)
            result = self.download_file(remote_path, local_path)
            if result is True:
                print("✓"); downloaded += 1
            elif result is None:
                print("(bo qua)"); skipped += 1
            else:
                print("✗ LOI"); failed += 1

        return downloaded, skipped, failed

    def download_splits(self, file_type: str, splits: list[str],
                         labels: list[str], base_out: str) -> dict:
        """Tải nhiều split × label."""
        stats = {}
        for split in splits:
            for label in labels:
                key = f"{split}/{label}"
                print(f"\n  [{split.upper()}] {label}")
                d, s, f = self.download_split_label(
                    file_type, split, label, base_out)
                stats[key] = {'downloaded': d, 'skipped': s, 'failed': f}
                print(f"    → Tai: {d} | Bo qua: {s} | Loi: {f}")
        return stats


# ══════════════════════════════════════════════════════════
# HIỂN THỊ
# ══════════════════════════════════════════════════════════

def show_repo_info(dl: HFDownloader):
    print("\n" + "="*65)
    print(" THONG TIN REPO ".center(65))
    print("="*65)

    for file_type in ["videos", "processed"]:
        ext = ".mp4" if file_type == "videos" else ".npy"
        print(f"\n  [{file_type}/]")
        print(f"  {'Split':<8} {'Label':<30} {'So file':>8}")
        print("  " + "-"*48)
        grand_total = 0
        for split in SPLITS:
            labels = dl.get_labels(file_type, split)
            if not labels:
                print(f"  {split:<8} (trong)")
                continue
            for lb in labels:
                n = dl.count_files(file_type, split, lb)
                print(f"  {split:<8} {lb:<30} {n:>8}")
                grand_total += n
        print("  " + "-"*48)
        print(f"  {'TONG':<39} {grand_total:>8}")


def pick_splits_and_labels(dl: HFDownloader,
                            file_type: str) -> tuple[list, list]:
    """Chọn splits và labels cần tải."""
    # Chọn split
    print(f"\n  Chon split:")
    print(f"   0. Tat ca (train + val + test)")
    for i, sp in enumerate(SPLITS, 1):
        print(f"   {i}. {sp}")
    sp_ans = input("  > ").strip()
    if sp_ans == "0":
        selected_splits = SPLITS
    else:
        sp_map = {'1': 'train', '2': 'val', '3': 'test'}
        selected_splits = [sp_map[c.strip()] for c in sp_ans.split(",")
                           if c.strip() in sp_map]
    if not selected_splits:
        print("  Khong hop le!"); return [], []

    # Lấy tất cả labels từ các split đã chọn
    all_labels = set()
    for sp in selected_splits:
        all_labels.update(dl.get_labels(file_type, sp))
    all_labels = sorted(all_labels)

    if not all_labels:
        print(f"  Khong co label nao trong {selected_splits}")
        return [], []

    print(f"\n  Labels tim thay ({len(all_labels)}):")
    for i, lb in enumerate(all_labels, 1):
        print(f"   {i:>3}. {lb}")
    print(f"    0. Tat ca")

    lb_ans = input("  Chon label (0 / so / vd: 1,3): ").strip()
    if lb_ans == "0":
        selected_labels = all_labels
    else:
        selected_labels = []
        for part in lb_ans.split(","):
            part = part.strip()
            try:
                idx = int(part) - 1
                if 0 <= idx < len(all_labels):
                    selected_labels.append(all_labels[idx])
            except ValueError:
                if part in all_labels:
                    selected_labels.append(part)

    return selected_splits, selected_labels


def print_summary(stats: dict, base_out: str):
    print("\n" + "="*60)
    print(" KET QUA TAI ".center(60))
    print("="*60)
    td = ts = tf = 0
    for key, s in stats.items():
        td += s['downloaded']; ts += s['skipped']; tf += s['failed']
        print(f"  {key:<40} Tai:{s['downloaded']:>4} "
              f"Bo qua:{s['skipped']:>4} Loi:{s['failed']:>3}")
    print("-"*60)
    print(f"  {'TONG':<40} Tai:{td:>4} Bo qua:{ts:>4} Loi:{tf:>3}")
    print(f"\n  Da luu vao: {base_out}/")


# ══════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════

def setup_downloader() -> HFDownloader | None:
    repo_id = os.getenv("HF_REPO_ID", "").strip()
    token   = os.getenv("HF_TOKEN",   "").strip() or None

    print("\n" + "="*60)
    print(" HUGGINGFACE DOWNLOADER ".center(60, "="))
    print("="*60)

    if repo_id:
        print(f"\n  HF_REPO_ID = {repo_id}")
        if input("  Dung repo nay? (y/n, mac dinh y): ").strip().lower() == 'n':
            repo_id = ""

    if not repo_id:
        repo_id = input("\n  Nhap Repo ID: ").strip()
        if not repo_id:
            print("  Thoat."); return None
        t = input("  HF Token (Enter neu public): ").strip()
        if t: token = t

    try:
        return HFDownloader(repo_id=repo_id, token=token)
    except Exception as e:
        print(f"  LOI: {e}"); return None


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    dl = setup_downloader()
    if dl is None: return

    while True:
        print("\n" + "="*60)
        print(" MENU TAI FILE ".center(60, "="))
        print("="*60)
        print("  1. Xem thong tin repo")
        print("  2. Tai video raw (.mp4)  → data/videos/")
        print("  3. Tai file .npy         → data/processed/")
        print("  4. Tai ca 2 (video + npy)")
        print("  5. Doi repo / token")
        print("  6. Thoat")
        print("="*60)
        print("  Cau truc: videos|processed / train|val|test / <label>")
        print("="*60)

        ch = input("\n  Chon (1-6): ").strip()

        if ch == "1":
            show_repo_info(dl)

        elif ch == "2":
            print("\n  === TAI VIDEO RAW (.mp4) ===")
            splits, labels = pick_splits_and_labels(dl, "videos")
            if not splits or not labels: continue
            print(f"\n  Se tai {len(splits)} split × {len(labels)} label → data/videos/")
            if input("  Xac nhan? (y/n): ").strip().lower() != 'y': continue
            stats = dl.download_splits("videos", splits, labels, "data/videos")
            print_summary(stats, "data/videos")

        elif ch == "3":
            print("\n  === TAI FILE .NPY ===")
            splits, labels = pick_splits_and_labels(dl, "processed")
            if not splits or not labels: continue
            print(f"\n  Se tai {len(splits)} split × {len(labels)} label → data/processed/")
            if input("  Xac nhan? (y/n): ").strip().lower() != 'y': continue
            stats = dl.download_splits("processed", splits, labels, "data/processed")
            print_summary(stats, "data/processed")

        elif ch == "4":
            print("\n  === TAI CA 2 (VIDEO + NPY) ===")
            print("\n  [1/2] Chon VIDEO:")
            sp_v, lb_v = pick_splits_and_labels(dl, "videos")
            print("\n  [2/2] Chon NPY:")
            ans = input("  Dung cung split/label voi video? (y/n, mac dinh y): ").strip().lower()
            if ans == 'n':
                sp_n, lb_n = pick_splits_and_labels(dl, "processed")
            else:
                sp_n, lb_n = sp_v, lb_v

            if not sp_v and not sp_n: continue
            print(f"\n  Se tai:")
            if sp_v: print(f"    Video : {sp_v} × {lb_v} → data/videos/")
            if sp_n: print(f"    NPY   : {sp_n} × {lb_n} → data/processed/")
            if input("\n  Xac nhan? (y/n): ").strip().lower() != 'y': continue

            if sp_v:
                print("\n  --- VIDEO ---")
                stats_v = dl.download_splits("videos", sp_v, lb_v, "data/videos")
                print_summary(stats_v, "data/videos")
            if sp_n:
                print("\n  --- NPY ---")
                stats_n = dl.download_splits("processed", sp_n, lb_n, "data/processed")
                print_summary(stats_n, "data/processed")

        elif ch == "5":
            dl_new = setup_downloader()
            if dl_new: dl = dl_new

        elif ch == "6":
            print("\n  Tam biet!\n"); break

        else:
            print("  Khong hop le!")


if __name__ == "__main__":
    main()