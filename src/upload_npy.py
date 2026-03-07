"""
upload_npy.py - Upload file .npy len HuggingFace Dataset
=========================================================
Tich hop voi HFUploader (hf_uploader.py):
  - Track file da upload qua .hf_uploaded_npy.json
  - Khong upload lai file cu (check MD5 hash)
  - Batch: nhieu file -> 1 commit / label

Cau truc local:
    data/processed/train/<label>/*.npy
    data/processed/val/<label>/*.npy
    data/processed/test/<label>/*.npy

Cau truc tren HF (giu nguyen):
    processed/train/<label>/*.npy
    processed/val/<label>/*.npy
    processed/test/<label>/*.npy

Chay:
  python upload_npy.py                        # upload tat ca (skip da co)
  python upload_npy.py --split train          # chi train
  python upload_npy.py --split train val      # train + val
  python upload_npy.py --label ai lo_so       # chi 2 labels
  python upload_npy.py --dry-run              # preview khong upload
  python upload_npy.py --mode bulk            # 1 commit toan bo (upload_folder)
  python upload_npy.py --reset-tracking       # xoa cache -> upload lai tat ca
  python upload_npy.py --stats                # xem thong ke tracking
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from huggingface_hub import HfApi, create_repo, upload_folder, CommitOperationAdd
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Chua cai huggingface_hub! Chay: pip install huggingface_hub")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

LOCAL_DIR     = r"data\processed"
TRACKING_FILE = r"hf_uploaded_npy.json"
SPLITS        = ["train", "val", "test"]


class NPYUploader:
    """
    Upload .npy files len HuggingFace voi tracking de khong upload lai.
    Pattern giong het HFUploader — chi khac remote prefix va tracking file.
    """

    def __init__(self, repo_id=None, token=None,
                 tracking_file=TRACKING_FILE):
        self.repo_id       = repo_id  or os.getenv("HF_REPO_ID", "")
        self.token         = token    or os.getenv("HF_TOKEN",    "")  or None
        self.tracking_file = tracking_file

        self._api      = None
        self._queue    = []   # [(local_path, remote_path), ...]
        self._uploaded = self._load_tracking()

        self._init_api()

    # ── Init ─────────────────────────────────────────────────────

    def _init_api(self):
        if not self.token:
            print("  [HF-NPY] KHONG CO TOKEN - Chi dry-run / preview")
            print("  [HF-NPY] Them HF_TOKEN vao .env de upload that")
            return
        if not self.repo_id:
            print("  [HF-NPY] KHONG CO REPO_ID")
            return
        try:
            self._api = HfApi(token=self.token)
            print(f"  [HF-NPY] Ready: {self.repo_id}")
            print(f"  [HF-NPY] Tracking: {self.tracking_file}"
                  f"  ({len(self._uploaded)} files da tracked)")
        except Exception as e:
            print(f"  [HF-NPY] Init error: {e}")

    # ── Tracking (giong het HFUploader) ──────────────────────────

    def _load_tracking(self):
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_tracking(self):
        with open(self.tracking_file, 'w') as f:
            json.dump(self._uploaded, f, indent=2)

    def _file_hash(self, filepath):
        h = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    def _is_uploaded(self, local_path):
        if local_path not in self._uploaded:
            return False
        current_hash = self._file_hash(local_path)
        return self._uploaded[local_path].get('hash') == current_hash

    def _mark_uploaded(self, local_path, remote_path):
        self._uploaded[local_path] = {
            'remote':      remote_path,
            'hash':        self._file_hash(local_path),
            'uploaded_at': datetime.now().isoformat(),
        }

    # ── Queue (giong HFUploader.queue_file, doi remote prefix) ───

    def queue_file(self, local_path, label_name, split='train'):
        """
        Them 1 file .npy vao queue.
        Skip neu da upload va hash khong doi.
        Returns True neu them duoc, False neu skip.
        """
        if not os.path.exists(local_path):
            return False
        if self._is_uploaded(local_path):
            return False

        filename    = os.path.basename(local_path)
        # "processed/" thay vi "videos/" trong HFUploader
        remote_path = f"processed/{split}/{label_name}/{filename}"

        self._queue.append((local_path, remote_path))
        return True

    def queue_folder(self, folder_path, label_name, split='train'):
        """
        Them tat ca .npy trong folder vao queue.
        Returns so files da them.
        """
        if not os.path.isdir(folder_path):
            return 0
        count = 0
        for f in sorted(os.listdir(folder_path)):
            if f.endswith('.npy'):
                local_path = os.path.join(folder_path, f)
                if self.queue_file(local_path, label_name, split):
                    count += 1
        return count

    # ── Flush (y het HFUploader.flush) ───────────────────────────

    def flush(self, commit_message=None, dry_run=False):
        """
        Upload tat ca files trong queue -> 1 commit.
        Returns so files da upload.
        """
        if not self._queue:
            return 0

        n = len(self._queue)

        if dry_run:
            print(f"    [DRY RUN] Se upload {n} files (1 commit)")
            self._queue = []
            return n

        if self._api is None:
            print(f"    [HF-NPY] Skip {n} files (no API)")
            self._queue = []
            return 0

        msg = commit_message or f"Add {n} npy files"
        print(f"    [HF-NPY] Uploading {n} files...")

        try:
            operations = [
                CommitOperationAdd(
                    path_in_repo    = remote_path,
                    path_or_fileobj = local_path,
                )
                for local_path, remote_path in self._queue
            ]

            self._api.create_commit(
                repo_id        = self.repo_id,
                repo_type      = "dataset",
                operations     = operations,
                commit_message = msg,
            )

            for local_path, remote_path in self._queue:
                self._mark_uploaded(local_path, remote_path)
            self._save_tracking()

            self._queue = []
            print(f"    [HF-NPY] OK! {n} files uploaded")
            return n

        except Exception as e:
            print(f"    [HF-NPY] Error: {e}")
            self._queue = []
            return 0

    # ── Smart upload (giong HFUploader.upload_folder_smart) ──────

    def upload_label_smart(self, folder_path, label_name, split='train'):
        """
        Upload folder thong minh: chi upload files moi.
        Returns (uploaded_count, skipped_count).
        """
        if not os.path.isdir(folder_path):
            return 0, 0

        total   = len([f for f in os.listdir(folder_path) if f.endswith('.npy')])
        queued  = self.queue_folder(folder_path, label_name, split)
        skipped = total - queued

        if queued > 0:
            uploaded = self.flush(f"Update processed/{split}/{label_name}")
            return uploaded, skipped
        else:
            print(f"    [HF-NPY] {label_name}: Khong co file moi (skip {skipped})")
            return 0, skipped

    # ── Bulk mode ─────────────────────────────────────────────────

    def upload_bulk(self, local_dir, dry_run=False):
        """Upload toan bo thu muc 1 commit duy nhat bang upload_folder."""
        npy_files = list(Path(local_dir).rglob("*.npy"))
        size_mb   = sum(f.stat().st_size for f in npy_files) / 1024 / 1024
        print(f"\n  [BULK] {len(npy_files)} files  |  {size_mb:.1f} MB")
        print(f"  local: {local_dir}  ->  HF: processed/")

        if dry_run:
            print("  [DRY RUN] Khong upload"); return len(npy_files)
        if self._api is None:
            print("  Khong co API"); return 0

        confirm = input(f"\n  Xac nhan upload {len(npy_files)} files"
                        f" ({size_mb:.1f} MB)? (y/n): ").strip().lower()
        if confirm != 'y':
            print("  Da huy."); return 0

        print("\n  Dang upload... (multi_commits=True)")
        try:
            upload_folder(
                folder_path    = local_dir,
                path_in_repo   = "processed",
                repo_id        = self.repo_id,
                repo_type      = "dataset",
                token          = self.token,
                multi_commits  = True,
                commit_message = f"Bulk upload processed/ ({len(npy_files)} npy)"
                                 f" {datetime.now():%Y-%m-%d}",
                ignore_patterns= ["*.tmp", "*.log"],
            )
            print(f"\n  BULK upload hoan thanh! {len(npy_files)} files")
            return len(npy_files)
        except Exception as e:
            print(f"\n  Loi: {e}"); return 0

    # ── Stats / Reset (giong HFUploader) ─────────────────────────

    def get_stats(self):
        return {
            'queue_size':     len(self._queue),
            'uploaded_count': len(self._uploaded),
            'repo_id':        self.repo_id,
            'api_ready':      self._api is not None,
            'tracking_file':  self.tracking_file,
        }

    def reset_tracking(self):
        self._uploaded = {}
        self._save_tracking()
        print(f"  [HF-NPY] Tracking reset - se upload lai tat ca files")


# ══════════════════════════════════════════════════════════════════
# SCAN LOCAL
# ══════════════════════════════════════════════════════════════════

def scan_local(local_dir, splits, label_filter):
    result = {}
    base   = Path(local_dir)
    for split in splits:
        split_dir = base / split
        if not split_dir.exists():
            continue
        result[split] = {}
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            if label_filter and label_dir.name not in label_filter:
                continue
            npy_files = list(label_dir.glob("*.npy"))
            if npy_files:
                result[split][label_dir.name] = {
                    'dir'  : str(label_dir),
                    'count': len(npy_files),
                }
    return result


def print_scan_table(scan_data):
    total = 0
    print(f"  {'Split':<8} {'Label':<25} {'Files':>7}")
    print("  " + "-"*44)
    for split, labels in scan_data.items():
        for label, info in labels.items():
            print(f"  {split:<8} {label:<25} {info['count']:>7}")
            total += info['count']
    print("  " + "-"*44)
    print(f"  {'TOTAL':<34} {total:>7}\n")


# ══════════════════════════════════════════════════════════════════
# MAIN UPLOAD LOOP
# ══════════════════════════════════════════════════════════════════

def run_upload(uploader, scan_data, dry_run):
    summary = []

    for split, labels in scan_data.items():
        if not labels:
            continue
        print(f"\n{'='*60}")
        print(f"  SPLIT: {split}  ({len(labels)} labels)")
        print(f"{'='*60}")

        for label, info in labels.items():
            label_dir = info['dir']
            total     = info['count']
            print(f"\n  [{split}/{label}]  {total} files")

            # Queue toan bo folder
            total_npy   = len([f for f in os.listdir(label_dir) if f.endswith('.npy')])
            queued_n    = uploader.queue_folder(label_dir, label, split)
            skipped_n   = total_npy - queued_n

            if queued_n == 0:
                print(f"    Skip ({skipped_n} da upload, hash khong doi)")
                summary.append({'split': split, 'label': label, 'total': total,
                                'queued': 0, 'skipped': skipped_n, 'status': 'skipped'})
                continue

            print(f"    Moi: {queued_n}  |  Skip: {skipped_n}")
            uploaded = uploader.flush(
                commit_message = f"Add processed/{split}/{label} ({queued_n} npy files)",
                dry_run        = dry_run,
            )
            status = 'dry_run' if dry_run else ('uploaded' if uploaded > 0 else 'error')
            summary.append({'split': split, 'label': label, 'total': total,
                            'queued': queued_n, 'skipped': skipped_n, 'status': status})

    # ── Summary ───────────────────────────────────────────────────
    icons = {'uploaded': 'OK', 'skipped': 'SKIP', 'dry_run': 'DRY', 'error': 'ERR'}
    print(f"\n{'='*60}")
    print("  TONG KET".center(60))
    print(f"{'='*60}")
    print(f"  {'Split':<8} {'Label':<22} {'Total':>6} {'Moi':>5} {'Skip':>5}  Status")
    print(f"  {'-'*56}")

    total_new = total_skip = 0
    for r in summary:
        icon = icons.get(r['status'], '???')
        print(f"  {r['split']:<8} {r['label']:<22} {r['total']:>6}"
              f" {r['queued']:>5} {r['skipped']:>5}  [{icon}]")
        total_new  += r['queued']
        total_skip += r['skipped']

    print(f"  {'-'*56}")
    print(f"  {len(summary)} labels  |  Moi: {total_new}  |  Skip: {total_skip}")

    if dry_run:
        print(f"\n  [DRY RUN] Chua upload gi! Bo --dry-run de upload that.")
    elif total_new > 0:
        print(f"\n  Hoan thanh!")
        print(f"  Xem: https://huggingface.co/datasets/{uploader.repo_id}")
    else:
        print(f"\n  Tat ca da duoc upload truoc do - khong co gi moi!")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Upload .npy len HuggingFace Dataset",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Vi du:
  python upload_npy.py                        # upload tat ca
  python upload_npy.py --split train          # chi train
  python upload_npy.py --label ai lo_so       # chi 2 labels
  python upload_npy.py --dry-run              # preview
  python upload_npy.py --mode bulk            # 1 commit toan bo
  python upload_npy.py --reset-tracking       # upload lai tat ca
  python upload_npy.py --stats                # xem thong ke
        """
    )
    parser.add_argument('--local-dir',      default=LOCAL_DIR)
    parser.add_argument('--split',          nargs='+', default=SPLITS,
                        choices=SPLITS)
    parser.add_argument('--label',          nargs='+', default=None)
    parser.add_argument('--mode',           default='folder',
                        choices=['folder', 'bulk'],
                        help='folder=1 commit/label | bulk=1 commit toan bo')
    parser.add_argument('--dry-run',        action='store_true')
    parser.add_argument('--reset-tracking', action='store_true')
    parser.add_argument('--stats',          action='store_true')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  UPLOAD NPY -> HUGGINGFACE".center(60))
    print("="*60)

    uploader = NPYUploader(tracking_file=TRACKING_FILE)

    print(f"  Repo     : {uploader.repo_id or '(chua set HF_REPO_ID)'}")
    print(f"  Token    : {'co' if uploader.token else '(chua set HF_TOKEN)'}")
    print(f"  Local    : {args.local_dir}")
    print(f"  Tracking : {TRACKING_FILE}")
    print(f"  Splits   : {args.split}")
    print(f"  Labels   : {args.label or 'all'}")
    print(f"  Mode     : {args.mode}")
    print(f"  Dry run  : {args.dry_run}")

    # Stats
    if args.stats:
        stats = uploader.get_stats()
        print(f"\n  Stats:")
        for k, v in stats.items():
            print(f"    {k:<20}: {v}")
        return

    # Reset tracking
    if args.reset_tracking:
        confirm = input("\n  Xoa toan bo tracking? (y/n): ").strip().lower()
        if confirm == 'y':
            uploader.reset_tracking()
        return

    # Ensure repo
    if not args.dry_run and uploader._api and uploader.repo_id:
        try:
            create_repo(repo_id=uploader.repo_id, repo_type="dataset",
                        exist_ok=True, token=uploader.token)
        except Exception as e:
            print(f"  create_repo: {e}")

    # Scan
    if not Path(args.local_dir).exists():
        print(f"\n  Khong tim thay: {args.local_dir}"); sys.exit(1)

    scan_data = scan_local(args.local_dir, args.split, args.label)
    if not scan_data:
        print(f"\n  Khong co .npy nao trong {args.local_dir}"); sys.exit(1)

    print(f"\n  Tong quan local:")
    print_scan_table(scan_data)

    # Upload
    if args.mode == 'bulk':
        uploader.upload_bulk(args.local_dir, dry_run=args.dry_run)
    else:
        run_upload(uploader, scan_data, dry_run=args.dry_run)


if __name__ == '__main__':
    main()