"""
 Tải file từ HuggingFace Dataset về máy
=============================================================
Tải video raw (.mp4) và/hoặc file .npy đã xử lý từ HuggingFace.

Cách chạy:
    python download_from_hf.py

Cấu hình (.env):
    HF_TOKEN=hf_xxxxxxxxxxxxxx        (bắt buộc nếu repo private)
    HF_REPO_ID=KhangCN/Video_VSL      (repo chứa video)

Cấu trúc repo trên HuggingFace:
    videos/
    └── <label>/
        └── *.mp4           ← video raw
    processed/
    └── <label>/
        └── *.npy           ← features đã xử lý

Cấu trúc sau khi tải về:
    data/
    ├── videos/
    │   └── <label>/*.mp4
    └── processed/
        └── <label>/*.npy
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════
# KIỂM TRA THƯ VIỆN
# ══════════════════════════════════════════════════════════

try:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("  LOI: Chua cai huggingface_hub!")
    print("  Chay: pip install huggingface_hub")
    sys.exit(1)

# ══════════════════════════════════════════════════════════
# DOWNLOADER
# ══════════════════════════════════════════════════════════

class HFDownloader:
    """Tải file từ HuggingFace Dataset về local."""

    def __init__(self, repo_id: str, token: str = None):
        self.repo_id = repo_id
        self.token   = token
        self.api     = HfApi(token=token)
        print(f"  Repo    : {repo_id}")
        print(f"  Token   : {'co' if token else 'khong (public repo)'}")

    # ── Liệt kê files trên HF ─────────────────────────────

    def list_files(self, prefix: str = "") -> list[str]:
        """Liệt kê tất cả file trong repo, lọc theo prefix."""
        try:
            all_files = list(list_repo_files(
                repo_id   = self.repo_id,
                repo_type = "dataset",
                token     = self.token,
            ))
            if prefix:
                all_files = [f for f in all_files if f.startswith(prefix)]
            return sorted(all_files)
        except Exception as e:
            print(f"  LOI khi doc danh sach file: {e}")
            return []

    def get_labels(self, file_type: str = "videos") -> list[str]:
        """
        Lấy danh sách labels có trong repo.
        file_type: "videos" hoặc "processed"
        """
        files  = self.list_files(prefix=f"{file_type}/")
        labels = set()
        for f in files:
            parts = f.split("/")
            if len(parts) >= 2:
                labels.add(parts[1])
        return sorted(labels)

    def count_files(self, file_type: str, label: str) -> int:
        """Đếm số file của 1 label trong repo."""
        files = self.list_files(prefix=f"{file_type}/{label}/")
        return len(files)

    # ── Download ──────────────────────────────────────────

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """
        Tải 1 file từ HF về local_path.
        Bỏ qua nếu file đã tồn tại và cùng kích thước.
        """
        # Bỏ qua nếu đã có
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return None   # None = skipped

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            hf_hub_download(
                repo_id       = self.repo_id,
                filename      = remote_path,
                repo_type     = "dataset",
                token         = self.token,
                local_dir     = ".",           # download về thư mục hiện tại
                local_dir_use_symlinks = False,
            )
            # hf_hub_download lưu vào ./<remote_path>, move về local_path
            if remote_path != local_path:
                src = remote_path
                if os.path.exists(src):
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    import shutil
                    shutil.move(src, local_path)
                    # Dọn thư mục rỗng
                    try:
                        parent = os.path.dirname(src)
                        while parent and parent != ".":
                            if os.path.isdir(parent) and not os.listdir(parent):
                                os.rmdir(parent)
                                parent = os.path.dirname(parent)
                            else:
                                break
                    except Exception:
                        pass
            return True
        except Exception as e:
            print(f"  LOI: {e}")
            return False

    def download_label(self, file_type: str, label: str,
                        output_dir: str,
                        show_progress: bool = True) -> tuple[int, int, int]:
        """
        Tải tất cả file của 1 label.
        Trả về (downloaded, skipped, failed)
        """
        ext      = ".mp4" if file_type == "videos" else ".npy"
        files    = self.list_files(prefix=f"{file_type}/{label}/")
        files    = [f for f in files if f.endswith(ext)]

        if not files:
            print(f"    Khong co file {ext} cho label '{label}'")
            return 0, 0, 0

        label_dir  = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        downloaded = skipped = failed = 0

        for i, remote_path in enumerate(files, 1):
            fname      = os.path.basename(remote_path)
            local_path = os.path.join(label_dir, fname)

            if show_progress:
                print(f"    [{i:>3}/{len(files)}] {fname:<50}", end=" ", flush=True)

            result = self.download_file(remote_path, local_path)
            if result is True:
                if show_progress: print("✓")
                downloaded += 1
            elif result is None:
                if show_progress: print("(da co, bo qua)")
                skipped += 1
            else:
                if show_progress: print("✗ THAT BAI")
                failed += 1

        return downloaded, skipped, failed

    def download_all_labels(self, file_type: str, labels: list[str],
                             output_dir: str) -> dict:
        """Tải tất cả labels. Trả về dict thống kê."""
        stats = {}
        for label in labels:
            print(f"\n  Label: {label}")
            d, s, f = self.download_label(file_type, label, output_dir)
            stats[label] = {'downloaded': d, 'skipped': s, 'failed': f}
            print(f"    → Tai moi: {d} | Bo qua: {s} | Loi: {f}")
        return stats


# ══════════════════════════════════════════════════════════
# HIỂN THỊ THÔNG TIN REPO
# ══════════════════════════════════════════════════════════

def show_repo_info(dl: HFDownloader):
    """In thống kê tổng quan của repo."""
    print("\n" + "="*60)
    print(" THONG TIN REPO ".center(60))
    print("="*60)

    for file_type, ext in [("videos", ".mp4"), ("processed", ".npy")]:
        labels = dl.get_labels(file_type)
        if not labels:
            print(f"\n  [{file_type}] Chua co du lieu")
            continue

        print(f"\n  [{file_type}/] — {len(labels)} labels:")
        print(f"  {'Label':<35} {'So file':>10}")
        print("  " + "-"*47)
        total = 0
        for lb in labels:
            n = dl.count_files(file_type, lb)
            print(f"  {lb:<35} {n:>10}")
            total += n
        print("  " + "-"*47)
        print(f"  {'TONG':<35} {total:>10}")


# ══════════════════════════════════════════════════════════
# MENU CHỌN LABELS
# ══════════════════════════════════════════════════════════

def pick_labels(dl: HFDownloader, file_type: str) -> list[str]:
    """Hiện danh sách labels, cho người dùng chọn."""
    labels = dl.get_labels(file_type)
    if not labels:
        print(f"  Khong co label nao trong '{file_type}/'")
        return []

    print(f"\n  Labels trong '{file_type}/':")
    for i, lb in enumerate(labels, 1):
        n = dl.count_files(file_type, lb)
        print(f"  {i:>3}. {lb:<40} ({n} file)")

    print(f"\n  Chon:")
    print(f"   0  - Tai tat ca ({len(labels)} labels)")
    print(f"   1-{len(labels)} - Chon 1 label cu the")
    print(f"   vd: 1,3,5 - Chon nhieu label")

    ans = input("\n  > ").strip()

    if ans == "0":
        return labels

    selected = []
    for part in ans.split(","):
        part = part.strip()
        try:
            idx = int(part) - 1
            if 0 <= idx < len(labels):
                selected.append(labels[idx])
            else:
                print(f"  So {part} khong hop le, bo qua")
        except ValueError:
            # Có thể nhập thẳng tên label
            if part in labels:
                selected.append(part)
            else:
                print(f"  Label '{part}' khong ton tai, bo qua")

    return selected


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def setup_downloader() -> HFDownloader | None:
    """Đọc config và khởi tạo HFDownloader."""
    repo_id = os.getenv("HF_REPO_ID", "").strip()
    token   = os.getenv("HF_TOKEN",   "").strip() or None

    print("\n" + "="*60)
    print(" HUGGINGFACE DOWNLOADER ".center(60, "="))
    print("="*60)

    if repo_id:
        print(f"\n  Doc tu .env: HF_REPO_ID = {repo_id}")
        ans = input("  Dung repo nay? (y/n, mac dinh y): ").strip().lower()
        if ans == 'n':
            repo_id = ""

    if not repo_id:
        repo_id = input("\n  Nhap Repo ID (vd: KhangCN/Video_VSL): ").strip()
        if not repo_id:
            print("  Khong co Repo ID. Thoat.")
            return None
        token_input = input("  Nhap HF Token (Enter neu repo public): ").strip()
        if token_input:
            token = token_input

    try:
        dl = HFDownloader(repo_id=repo_id, token=token)
        return dl
    except Exception as e:
        print(f"  LOI khoi tao: {e}")
        return None


def print_summary(stats: dict, file_type: str, output_dir: str):
    """In tổng kết sau khi tải xong."""
    print("\n" + "="*60)
    print(" KET QUA TAI ".center(60))
    print("="*60)
    total_d = total_s = total_f = 0
    for label, s in stats.items():
        total_d += s['downloaded']
        total_s += s['skipped']
        total_f += s['failed']
        print(f"  {label:<35} "
              f"Tai: {s['downloaded']:>4}  "
              f"Bo qua: {s['skipped']:>4}  "
              f"Loi: {s['failed']:>3}")
    print("-"*60)
    print(f"  {'TONG CONG':<35} "
          f"Tai: {total_d:>4}  "
          f"Bo qua: {total_s:>4}  "
          f"Loi: {total_f:>3}")
    print(f"\n  Da luu vao: {output_dir}/")


def main():
    dl = setup_downloader()
    if dl is None:
        return

    while True:
        print("\n" + "="*60)
        print(" MENU TAI FILE ".center(60, "="))
        print("="*60)
        print("  1. Xem thong tin repo (danh sach labels + so file)")
        print("  2. Tai video raw (.mp4)  → data/videos/")
        print("  3. Tai file .npy         → data/processed/")
        print("  4. Tai ca 2 (video + npy)")
        print("  5. Doi repo / token")
        print("  6. Thoat")
        print("="*60)

        ch = input("\n  Chon (1-6): ").strip()

        # ── 1. Xem thông tin ──
        if ch == "1":
            show_repo_info(dl)

        # ── 2. Tải video raw ──
        elif ch == "2":
            print("\n  === TAI VIDEO RAW (.mp4) ===")
            selected = pick_labels(dl, "videos")
            if not selected:
                continue
            out_dir = "data/videos"
            print(f"\n  Se tai {len(selected)} label vao '{out_dir}/'")
            if input("  Xac nhan? (y/n): ").strip().lower() != 'y':
                continue
            stats = dl.download_all_labels("videos", selected, out_dir)
            print_summary(stats, "videos", out_dir)

        # ── 3. Tải .npy ──
        elif ch == "3":
            print("\n  === TAI FILE .NPY ===")
            selected = pick_labels(dl, "processed")
            if not selected:
                continue
            out_dir = "data/processed"
            print(f"\n  Se tai {len(selected)} label vao '{out_dir}/'")
            if input("  Xac nhan? (y/n): ").strip().lower() != 'y':
                continue
            stats = dl.download_all_labels("processed", selected, out_dir)
            print_summary(stats, "processed", out_dir)

        # ── 4. Tải cả 2 ──
        elif ch == "4":
            print("\n  === TAI CA 2 (VIDEO + NPY) ===")

            # Video
            print("\n  [1/2] Chon labels de tai VIDEO:")
            sel_videos = pick_labels(dl, "videos")

            # NPY
            print("\n  [2/2] Chon labels de tai NPY:")
            print("  (Nhan Enter de dung cung labels vua chon)")
            ans = input("  Dung cung labels voi video? (y/n, mac dinh y): ").strip().lower()
            if ans == 'n':
                sel_npy = pick_labels(dl, "processed")
            else:
                sel_npy = sel_videos

            if not sel_videos and not sel_npy:
                continue

            print(f"\n  Se tai:")
            if sel_videos:
                print(f"    Video : {len(sel_videos)} labels → data/videos/")
            if sel_npy:
                print(f"    NPY   : {len(sel_npy)} labels → data/processed/")

            if input("\n  Xac nhan? (y/n): ").strip().lower() != 'y':
                continue

            if sel_videos:
                print("\n  --- Dang tai VIDEO ---")
                stats_v = dl.download_all_labels("videos", sel_videos, "data/videos")
                print_summary(stats_v, "videos", "data/videos")

            if sel_npy:
                print("\n  --- Dang tai NPY ---")
                stats_n = dl.download_all_labels("processed", sel_npy, "data/processed")
                print_summary(stats_n, "processed", "data/processed")

        # ── 5. Đổi repo ──
        elif ch == "5":
            dl_new = setup_downloader()
            if dl_new:
                dl = dl_new
                print("  Da doi repo thanh cong.")

        # ── 6. Thoát ──
        elif ch == "6":
            print("\n  Tam biet!\n")
            break

        else:
            print("  Khong hop le!")


if __name__ == "__main__":
    main()