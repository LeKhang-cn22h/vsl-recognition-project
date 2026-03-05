import os
from huggingface_hub import HfApi, create_repo, upload_folder

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")

LOCAL_FOLDER = "data/processed"

api = HfApi(token=HF_TOKEN)

# Tạo repo nếu chưa có
create_repo(
    repo_id=HF_REPO_ID,
    repo_type="dataset",
    exist_ok=True,
    token=HF_TOKEN,
)

# Upload toàn bộ folder (giữ nguyên cấu trúc)
upload_folder(
    folder_path=LOCAL_FOLDER,
    repo_id=HF_REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN,
    multi_commits=True,  # rất quan trọng nếu nhiều file
)

print(" Upload completed!")