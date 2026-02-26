"""
converter/hf_uploader.py - Upload file .npy lên HuggingFace
=============================================================
    from converter.hf_uploader import HFUploader

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
    HF tự bỏ qua file đã tồn tại (so sánh hash).
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

    def upload_batch(self, file_list: list, label_name: str) -> None:
        """
        Upload toàn bộ folder label 1 lần = 1 commit.

        file_list: list[(local_path, filename)]
                   Chỉ dùng để lấy đường dẫn thư mục.
        label_name: tên label, dùng làm path trên HF.

        Cấu trúc trên HF: processed/<label_name>/<filename>
        """
        if not self.enabled or self.api is None or not file_list:
            return

        local_label_dir = os.path.dirname(file_list[0][0])
        print(f"    [HF] Uploading '{label_name}' ({len(file_list)} files)...")

        try:
            self.api.upload_folder(
                folder_path       = local_label_dir,
                path_in_repo      = f"processed/{label_name}",
                repo_id           = self.repo_id,
                repo_type         = "dataset",
                token             = self.token,
                commit_message    = f"Add processed/{label_name} batch",
                ignore_patterns   = ["*.tmp", "*.log"],
            )
            print(f"    [HF] Done '{label_name}' (1 commit)")
        except Exception as ex:
            print(f"    [HF] Upload error: {ex}")

    def upload_single(self, local_path: str,
                       label_name: str, filename: str) -> bool:
        """
        Upload 1 file đơn lẻ (dùng khi không augment).
        Trả về True nếu thành công.
        """
        if not self.enabled or self.api is None:
            return False
        try:
            self.api.upload_file(
                path_or_fileobj = local_path,
                path_in_repo    = f"processed/{label_name}/{filename}",
                repo_id         = self.repo_id,
                repo_type       = "dataset",
                token           = self.token,
                commit_message  = f"Add processed/{label_name}/{filename}",
            )
            return True
        except Exception as ex:
            print(f"    [HF] Upload error {filename}: {ex}")
            return False