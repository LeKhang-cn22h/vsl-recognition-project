"""
vsl/utils.py - Tiện ích dùng chung
====================================
    from vsl.utils import download_model, get_display_name, is_idle_label, resample_sequence
"""

import os
import json
import math
import urllib.request

import numpy as np

from vsl.config import MODEL_URLS


# ── Download MediaPipe models ──────────────────────────────────────

def download_model(filename: str) -> str:
    """Tải model MediaPipe nếu chưa có, trả về đường dẫn."""
    if not os.path.exists(filename):
        url = MODEL_URLS[filename]
        print(f"  Dang tai {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"  Da tai xong: {filename}")
    return filename


# ── Display names (tên tiếng Việt) ───────────────────────────────

_display_names: dict = {}
_display_path  = 'data/processed/display_names.json'

def load_display_names(path: str = _display_path) -> dict:
    global _display_names
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            _display_names = json.load(f)
        print(f"  Da load {len(_display_names)} ten hien thi tieng Viet")
    return _display_names

def get_display_name(label_key: str) -> str:
    """label_key → tên tiếng Việt, fallback về key nếu chưa có."""
    return _display_names.get(label_key, label_key)

def save_display_name(label_key: str, viet_name: str,
                       path: str = _display_path) -> None:
    """Lưu 1 tên mới vào display_names.json."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            dn = json.load(f)
    else:
        dn = {}
    if label_key not in dn:
        dn[label_key] = viet_name
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dn, f, indent=2, ensure_ascii=False)
        _display_names[label_key] = viet_name
        print(f"  Da luu: '{label_key}' → '{viet_name}'")


# ── Label helpers ─────────────────────────────────────────────────

def is_idle_label(label: str) -> bool:
    """Kiểm tra label có phải IDLE không."""
    return label.startswith('__idle__')


# ── Sequence resampling ───────────────────────────────────────────

def resample_sequence(sequence: np.ndarray, target_len: int) -> np.ndarray:
    """Chuẩn hóa độ dài chuỗi frames về target_len (nội suy tuyến tính)."""
    sequence = np.array(sequence)
    n = len(sequence)
    if n == target_len:
        return sequence
    indices = np.linspace(0, n - 1, target_len)
    result  = []
    for i in indices:
        lo = int(math.floor(i))
        hi = min(int(math.ceil(i)), n - 1)
        w  = i - lo
        result.append(sequence[lo] * (1 - w) + sequence[hi] * w)
    return np.array(result, dtype=np.float32)