"""
converter/hf_uploader.py - Upload file .npy lên HuggingFace
=============================================================
    from converter.hf_uploader import HFUploader

Cấu trúc trên HF:
    processed/train/<label>/*.npy
    processed/val/<label>/*.npy
    processed/test/<label>/*.npy

Dùng upload_folder (1 commit / batch) để tránh rate-limit 128 commits/giờ.
"""

import os

try:
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class HFUploader:
    """
    Upload batch .npy files lên HuggingFace Dataset.
    Dùng upload_folder → 1 commit cho cả batch, không bị rate-limit.
    """

    def __init__(self, repo_id: str = None, token: str = None,
                 enabled: bool = True):
        self.enabled = enabled and HF_AVAILABLE
        self.repo_id = repo_id
        self.token   = token
        self.api     = None

        if not HF_AVAILABLE:
            print("  HuggingFace: CHUA CAI huggingface_hub")
            self.enabled = False
            return

        if not repo_id:
            print("  HuggingFace: KHONG CO REPO ID - Chi luu local")
            self.enabled = False
            return

        if self.enabled:
            self.api = HfApi(token=token)
            mode = f"token={'co' if token else 'khong (public repo)'}"
            print(f"  HuggingFace: OK | repo={repo_id} | {mode}")

    def upload_batch(self, file_list: list, label_name: str,
                      split: str = 'train') -> None:
        """
        Upload toàn bộ folder label 1 lần = 1 commit.

        file_list : list[(local_path, filename)]
        label_name: tên label
        split     : 'train' | 'val' | 'test'

        Path trên HF: processed/<split>/<label_name>/
        """
        if not self.enabled or self.api is None or not file_list:
            return

        local_label_dir = os.path.dirname(file_list[0][0])
        repo_path       = f"processed/{split}/{label_name}"
        print(f"    [HF] Uploading '{split}/{label_name}' ({len(file_list)} files)...")

        try:
            self.api.upload_folder(
                folder_path    = local_label_dir,
                path_in_repo   = repo_path,
                repo_id        = self.repo_id,
                repo_type      = "dataset",
                token          = self.token,
                commit_message = f"Add {repo_path} batch",
                ignore_patterns= ["*.tmp", "*.log"],
            )
            print(f"    [HF] Done '{split}/{label_name}' (1 commit)")
        except Exception as ex:
            print(f"    [HF] Upload error: {ex}")

    def upload_single(self, local_path: str, label_name: str,
                       filename: str, split: str = 'train') -> bool:
        """
        Upload 1 file đơn lẻ.
        Path trên HF: processed/<split>/<label_name>/<filename>
        """
        if not self.enabled or self.api is None:
            return False
        try:
            repo_path = f"processed/{split}/{label_name}/{filename}"
            self.api.upload_file(
                path_or_fileobj = local_path,
                path_in_repo    = repo_path,
                repo_id         = self.repo_id,
                repo_type       = "dataset",
                token           = self.token,
                commit_message  = f"Add {repo_path}",
            )
            return True
        except Exception as ex:
            print(f"    [HF] Upload error {filename}: {ex}")
            return False