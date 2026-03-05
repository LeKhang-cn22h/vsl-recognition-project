"""
Upload video lên HuggingFace
Cấu trúc trên HF:
    videos/train/<label>/*.mp4
    videos/val/<label>/*.mp4
    videos/test/<label>/*.mp4
"""

import os
from dotenv import load_dotenv

load_dotenv()

_hf_token  = os.getenv("HF_TOKEN")
_hf_api    = None
HF_REPO_ID = os.getenv("HF_REPO_ID", "KhangCN/Video_VSL")


def init_hf():
    """Khởi tạo HfApi, gọi 1 lần khi start."""
    global _hf_api
    try:
        from huggingface_hub import HfApi
        if _hf_token:
            _hf_api = HfApi(token=_hf_token)
            print(f"  HuggingFace: OK (repo={HF_REPO_ID})")
        else:
            print("  HuggingFace: KHONG CO TOKEN - Video chi luu local")
            print("  (Tao file .env voi HF_TOKEN=hf_xxx de bat upload)")
    except ImportError:
        print("  HuggingFace: CHUA CAI huggingface_hub")


def upload_to_hf(local_path: str, label_name: str, split: str = 'train') -> bool:
    """
    Upload toàn bộ folder label lên HuggingFace dùng upload_folder (1 commit).
    Chỉ upload file mới hơn so với lần trước (dùng ignore_patterns không cần,
    HF Hub tự skip file đã có + hash giống).

    local_path : đường dẫn tới 1 file video bất kỳ trong folder
                 (để giữ tương thích với code cũ gọi hàm này sau mỗi video)
    label_name : tên label / thư mục
    split      : 'train' | 'val' | 'test'

    Upload thực sự xảy ra: folder chứa local_path được upload lên
        videos/<split>/<label_name>/
    """
    if _hf_api is None:
        return False

    # Lấy đường dẫn folder chứa file vừa ghi xong
    folder_path = os.path.dirname(os.path.abspath(local_path))

    if not os.path.isdir(folder_path):
        print(f"  HF Upload LOI: Khong tim thay folder {folder_path}")
        return False

    remote_folder = f"videos/{split}/{label_name}"

    try:
        _hf_api.upload_folder(
            folder_path=folder_path,
            path_in_repo=remote_folder,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            # Gộp tất cả file vào 1 commit duy nhất → tránh rate limit
            commit_message=f"Update {split}/{label_name}",
        )
        n_files = len([f for f in os.listdir(folder_path)
                       if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))])
        print(f"  HF Upload: {remote_folder}/ ({n_files} video) OK")
        return True
    except Exception as e:
        print(f"  HF Upload LOI: {e}")
        return False