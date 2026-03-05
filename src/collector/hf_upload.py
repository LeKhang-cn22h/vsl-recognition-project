"""
hf_uploader.py - Upload video len HuggingFace (toi uu)
======================================================
Tinh nang:
  1. Chi upload files MOI (chua upload lan nao)
  2. Batch upload: gom nhieu files vao 1 commit
  3. Track files da upload trong .hf_uploaded.json
  4. Khong upload lai files cu

Su dung:
    from hf_uploader import HFUploader
    
    uploader = HFUploader()
    uploader.queue_file("videos/train/xin_chao/video1.mp4", "xin_chao", "train")
    uploader.queue_file("videos/train/xin_chao/video2.mp4", "xin_chao", "train")
    uploader.flush()  # Upload tat ca files trong queue
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

HF_REPO_ID = os.getenv("HF_REPO_ID", "KhangCN/Video_VSL")
class HFUploader:
    """
    Upload files len HuggingFace voi tracking de khong upload lai.
    """
    
    def __init__(self, repo_id=None, token=None, tracking_file=".hf_uploaded.json"):
        self.repo_id = repo_id or os.getenv("HF_REPO_ID", "KhangCN/Video_VSL")
        self.token = token or os.getenv("HF_TOKEN")
        self.tracking_file = tracking_file
        
        self._api = None
        self._queue = []  # [(local_path, remote_path), ...]
        self._uploaded = self._load_tracking()
        
        self._init_api()
    
    def _init_api(self):
        """Khoi tao HfApi."""
        if not self.token:
            print("  [HF] KHONG CO TOKEN - Chi luu local")
            print("  [HF] Tao .env voi HF_TOKEN=hf_xxx de upload")
            return
        
        try:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.token)
            print(f"  [HF] Ready: {self.repo_id}")
        except ImportError:
            print("  [HF] Chua cai: pip install huggingface_hub")
    
    def _load_tracking(self):
        """Load danh sach files da upload."""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_tracking(self):
        """Luu danh sach files da upload."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self._uploaded, f, indent=2)
    
    def _file_hash(self, filepath):
        """Tinh hash MD5 cua file (de detect thay doi)."""
        h = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()
    
    def _is_uploaded(self, local_path):
        """Kiem tra file da upload chua (va khong thay doi)."""
        if local_path not in self._uploaded:
            return False
        
        # Check hash
        current_hash = self._file_hash(local_path)
        return self._uploaded[local_path].get('hash') == current_hash
    
    def _mark_uploaded(self, local_path, remote_path):
        """Danh dau file da upload."""
        self._uploaded[local_path] = {
            'remote': remote_path,
            'hash': self._file_hash(local_path),
            'uploaded_at': datetime.now().isoformat(),
        }
    
    # ══════════════════════════════════════════════════════
    # PUBLIC METHODS
    # ══════════════════════════════════════════════════════
    
    def queue_file(self, local_path, label_name, split='train'):
        """
        Them file vao hang doi upload.
        Chi them neu file CHUA upload hoac DA THAY DOI.
        
        Args:
            local_path: duong dan file local
            label_name: ten label
            split: 'train' | 'val' | 'test'
        
        Returns:
            True neu da them vao queue, False neu skip
        """
        if not os.path.exists(local_path):
            return False
        
        # Skip neu da upload va khong thay doi
        if self._is_uploaded(local_path):
            return False
        
        filename = os.path.basename(local_path)
        remote_path = f"videos/{split}/{label_name}/{filename}"
        
        self._queue.append((local_path, remote_path))
        return True
    
    def queue_folder(self, folder_path, label_name, split='train'):
        """
        Them tat ca video trong folder vao queue.
        
        Returns:
            So files da them vao queue
        """
        if not os.path.isdir(folder_path):
            return 0
        
        count = 0
        extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        
        for f in os.listdir(folder_path):
            if f.lower().endswith(extensions):
                local_path = os.path.join(folder_path, f)
                if self.queue_file(local_path, label_name, split):
                    count += 1
        
        return count
    
    def flush(self, commit_message=None):
        """
        Upload tat ca files trong queue (1 commit).
        
        Returns:
            So files da upload thanh cong
        """
        if not self._queue:
            return 0
        
        if self._api is None:
            print(f"  [HF] Skip {len(self._queue)} files (no API)")
            self._queue = []
            return 0
        
        print(f"  [HF] Uploading {len(self._queue)} files...")
        
        try:
            from huggingface_hub import CommitOperationAdd
            
            # Tao operations
            operations = []
            for local_path, remote_path in self._queue:
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=remote_path,
                        path_or_fileobj=local_path,
                    )
                )
            
            # Commit tat ca 1 lan
            msg = commit_message or f"Add {len(operations)} files"
            self._api.create_commit(
                repo_id=self.repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=msg,
            )
            
            # Mark as uploaded
            for local_path, remote_path in self._queue:
                self._mark_uploaded(local_path, remote_path)
            
            self._save_tracking()
            
            count = len(self._queue)
            self._queue = []
            
            print(f"  [HF] OK! {count} files uploaded")
            return count
            
        except Exception as e:
            print(f"  [HF] Error: {e}")
            return 0
    
    def upload_single(self, local_path, label_name, split='train'):
        """
        Upload 1 file ngay lap tuc (khong dung queue).
        Khong khuyen khich - nen dung queue_file + flush.
        """
        if self._is_uploaded(local_path):
            return True  # Da upload roi
        
        if self.queue_file(local_path, label_name, split):
            return self.flush() > 0
        return True
    
    def upload_folder_smart(self, folder_path, label_name, split='train'):
        """
        Upload folder thong minh: chi upload files moi.
        
        Returns:
            (uploaded_count, skipped_count)
        """
        if not os.path.isdir(folder_path):
            return 0, 0
        
        queued = self.queue_folder(folder_path, label_name, split)
        
        # Dem files da skip
        total = len([f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))])
        skipped = total - queued
        
        if queued > 0:
            uploaded = self.flush(f"Update {split}/{label_name}")
            return uploaded, skipped
        else:
            print(f"  [HF] {label_name}: Khong co file moi (skip {skipped})")
            return 0, skipped
    
    def get_queue_size(self):
        """Tra ve so files trong queue."""
        return len(self._queue)
    
    def clear_queue(self):
        """Xoa queue ma khong upload."""
        self._queue = []
    
    def reset_tracking(self):
        """Xoa tracking (se upload lai tat ca)."""
        self._uploaded = {}
        self._save_tracking()
        print("  [HF] Tracking reset - se upload lai tat ca files")
    
    def get_stats(self):
        """Tra ve thong ke."""
        return {
            'queue_size': len(self._queue),
            'uploaded_count': len(self._uploaded),
            'repo_id': self.repo_id,
            'api_ready': self._api is not None,
        }


# ══════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS (tuong thich code cu)
# ══════════════════════════════════════════════════════════

_uploader = None

def init_hf():
    """Khoi tao uploader (goi 1 lan khi start)."""
    global _uploader
    _uploader = HFUploader()

def upload_to_hf(local_path, label_name, split='train'):
    """
    Them file vao queue (KHONG upload ngay).
    Goi flush_hf() de upload tat ca.
    """
    global _uploader
    if _uploader is None:
        init_hf()
    return _uploader.queue_file(local_path, label_name, split)

def flush_hf(commit_message=None):
    """Upload tat ca files trong queue."""
    global _uploader
    if _uploader is None:
        return 0
    return _uploader.flush(commit_message)

def upload_folder_to_hf(folder_path, label_name, split='train'):
    """Upload folder thong minh (chi files moi)."""
    global _uploader
    if _uploader is None:
        init_hf()
    return _uploader.upload_folder_smart(folder_path, label_name, split)


# ══════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*50)
    print(" HF UPLOADER TEST ".center(50, "="))
    print("="*50)
    
    uploader = HFUploader()
    print(f"\n  Stats: {uploader.get_stats()}")
    
    # Test queue
    print("\n  Testing queue...")
    # uploader.queue_file("test.mp4", "test_label", "train")
    # uploader.flush()