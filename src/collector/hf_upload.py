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


def upload_to_hf(local_path: str, label_name: str, split:str ='train') -> bool:
    """Upload 1 video lên HuggingFace.  split: 'train' | 'val' | 'test'
    Path trên HF: videos/<split>/<label>/<filename>."""
    if _hf_api is None:
        return False
    try:
        filename    = os.path.basename(local_path)
        remote_path = f"videos/{split}/{label_name}/{filename}"
        _hf_api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
        )
        print(f"  HF Upload: {remote_path} OK")
        return True
    except Exception as e:
        print(f"  HF Upload LOI: {e}")
        return False