"""
video_cut.py - Cắt video dài thành nhiều video con
=======================================================
Mở video dài, dùng phím để đánh dấu IN/OUT rồi cắt ra.
Phần đã cắt sẽ bị xóa khỏi video gốc → video gốc tự động rút ngắn.

Chạy:
    python src/video_cut.py --input path/to/video.mp4
    python src/video_cut.py --input path/to/video.mp4 --out_dir data/videos/a

Phím bấm:
    SPACE       – Play / Pause
    ←  / →      – Lùi / Tiến 1 frame
    A  / D      – Lùi / Tiến 1 giây
    I           – Đặt điểm IN (bắt đầu clip)
    O           – Đặt điểm OUT (kết thúc clip)
    ENTER       – Xác nhận cắt đoạn IN→OUT
    Z           – Xóa điểm IN/OUT vừa đặt
    S           – Lưu video gốc đã loại bỏ các đoạn đã cắt
    Q / ESC     – Thoát (hỏi có lưu không)
"""

import sys
import os
import cv2
import argparse
import datetime
import shutil
import subprocess
import tempfile
from pathlib import Path
from collections import deque

import numpy as np

# ── Fix sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════

FONT       = cv2.FONT_HERSHEY_DUPLEX
BLACK      = (8,    8,   8)
WHITE      = (240, 240, 240)
GREEN      = (50,  210,  60)
ORANGE     = (30,  160, 255)
RED        = (50,   50, 220)
YELLOW     = (20,  220, 220)
GRAY       = (100, 100, 110)
TEAL       = (0,   200, 180)
DARK       = (20,   20,  28)


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _put(img, text, pos, scale=0.55, color=WHITE, thick=1):
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), FONT, scale, BLACK,  thick+1, cv2.LINE_AA)
    cv2.putText(img, text, pos,         FONT, scale, color,  thick,   cv2.LINE_AA)


def _pill(img, x1, y1, x2, y2, color, alpha=0.80, r=8):
    ov = img.copy()
    cv2.rectangle(ov, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(ov, (x1, y1+r), (x2, y2-r), color, -1)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(ov, (cx,cy), r, color, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)


def frame_to_time(frame_idx: int, fps: float) -> str:
    total_sec = frame_idx / fps
    h  = int(total_sec // 3600)
    m  = int((total_sec % 3600) // 60)
    s  = int(total_sec % 60)
    ms = int((total_sec - int(total_sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def ts_label(frame_idx: int, fps: float) -> str:
    """Rút gọn timestamp để hiển thị."""
    total_sec = frame_idx / fps
    m  = int(total_sec // 60)
    s  = int(total_sec % 60)
    ms = int((total_sec - int(total_sec)) * 10)
    return f"{m:02d}:{s:02d}.{ms}"


def make_output_name(out_dir: Path, label: str, clip_idx: int) -> str:
    ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return str(out_dir / f"{label}_{clip_idx:04d}_{ts}.mp4")


# ══════════════════════════════════════════════════════════════════
# VIDEO WRITER
# ══════════════════════════════════════════════════════════════════

def write_clip(cap, start_frame: int, end_frame: int,
               out_path: str, fps: float, w: int, h: int) -> bool:
    """Ghi đoạn [start_frame, end_frame) ra file mp4."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw     = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not vw.isOpened():
        print(f"  [ERROR] Khong tao duoc VideoWriter: {out_path}")
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        vw.write(frame)
        written += 1

    vw.release()
    dur = written / fps
    print(f"  [CLIP] {Path(out_path).name}  "
          f"({written}f, {dur:.2f}s)  → {out_path}")
    return True


def rebuild_source(cap, total_frames: int, cut_segments: list,
                   src_path: str, fps: float, w: int, h: int) -> str:
    """
    Xây dựng lại video gốc bằng cách bỏ các đoạn đã cắt.
    cut_segments: list of (start, end) đã được sort + merge.
    Trả về đường dẫn file tạm.
    """
    # Tính các đoạn GIỮ LẠI (complement của cut_segments)
    keep_segments = []
    prev = 0
    for s, e in sorted(cut_segments):
        if s > prev:
            keep_segments.append((prev, s))
        prev = max(prev, e)
    if prev < total_frames:
        keep_segments.append((prev, total_frames))

    if not keep_segments:
        print("  [WARN] Toan bo video da bi cat, khong con gi de luu!")
        return None

    # Ghi file tạm
    tmp_path = src_path + '_trimmed_tmp.mp4'
    fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
    vw       = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

    total_kept = 0
    for seg_s, seg_e in keep_segments:
        cap.set(cv2.CAP_PROP_POS_FRAMES, seg_s)
        for _ in range(seg_e - seg_s):
            ret, frame = cap.read()
            if not ret:
                break
            vw.write(frame)
            total_kept += 1

    vw.release()
    dur = total_kept / fps
    print(f"  [REBUILD] Giu lai {total_kept}f ({dur:.2f}s) "
          f"tu {len(keep_segments)} doan")
    return tmp_path


# ══════════════════════════════════════════════════════════════════
# UI DRAW
# ══════════════════════════════════════════════════════════════════

def draw_ui(frame, cur_frame: int, total_frames: int, fps: float,
            pt_in: int | None, pt_out: int | None,
            clips: list, is_playing: bool,
            w: int, h: int, notif: str, notif_ts: float,
            label: str, clip_count: int):
    import time
    H, W = frame.shape[:2]

    # ── Header bar ───────────────────────────────────────────────
    cv2.rectangle(frame, (0,0), (W,50), DARK, -1)
    _put(frame, f"Videocut  [{label}]", (10, 32), 0.65, WHITE, 2)
    _put(frame, f"Clips: {clip_count}", (W-120, 32), 0.55, GREEN)

    # ── Timeline bar ─────────────────────────────────────────────
    tl_y  = H - 80
    tl_x1 = 20
    tl_x2 = W - 20
    tl_w  = tl_x2 - tl_x1

    # Nền timeline
    cv2.rectangle(frame, (tl_x1, tl_y), (tl_x2, tl_y+18), (40,40,50), -1)

    # Tô các đoạn đã cắt (đỏ)
    for seg_s, seg_e, _ in clips:
        x1 = tl_x1 + int(tl_w * seg_s / total_frames)
        x2 = tl_x1 + int(tl_w * seg_e / total_frames)
        cv2.rectangle(frame, (x1, tl_y), (x2, tl_y+18), (50,50,180), -1)

    # Tô đoạn đang chọn (xanh lá nhạt)
    if pt_in is not None:
        ix = tl_x1 + int(tl_w * pt_in / total_frames)
        ox = (tl_x1 + int(tl_w * pt_out / total_frames)
              if pt_out else ix + 3)
        cv2.rectangle(frame, (ix, tl_y), (ox, tl_y+18), (40,120,40), -1)

    # Viền timeline
    cv2.rectangle(frame, (tl_x1, tl_y), (tl_x2, tl_y+18), (80,80,90), 1)

    # Vị trí hiện tại (thanh trắng)
    cx = tl_x1 + int(tl_w * cur_frame / max(total_frames-1, 1))
    cv2.line(frame, (cx, tl_y-4), (cx, tl_y+22), WHITE, 2)

    # Điểm IN / OUT markers
    if pt_in is not None:
        ix = tl_x1 + int(tl_w * pt_in / total_frames)
        cv2.line(frame, (ix, tl_y-8), (ix, tl_y+22), GREEN, 2)
        _put(frame, "I", (ix-4, tl_y-10), 0.35, GREEN)
    if pt_out is not None:
        ox = tl_x1 + int(tl_w * pt_out / total_frames)
        cv2.line(frame, (ox, tl_y-8), (ox, tl_y+22), ORANGE, 2)
        _put(frame, "O", (ox-4, tl_y-10), 0.35, ORANGE)

    # ── Thông tin thời gian ───────────────────────────────────────
    time_str = ts_label(cur_frame, fps)
    tot_str  = ts_label(total_frames, fps)
    _put(frame, f"{time_str} / {tot_str}", (tl_x1, tl_y+36), 0.45, GRAY)

    pct = cur_frame / max(total_frames-1, 1) * 100
    _put(frame, f"{pct:.1f}%", (W-70, tl_y+36), 0.45, GRAY)

    # ── IN / OUT info ─────────────────────────────────────────────
    info_y = H - 130
    if pt_in is not None:
        _put(frame, f"IN:  {ts_label(pt_in, fps)}", (20, info_y),
             0.50, GREEN)
    else:
        _put(frame, "IN:  --", (20, info_y), 0.50, GRAY)

    if pt_out is not None:
        dur = (pt_out - pt_in) / fps if pt_in is not None else 0
        _put(frame, f"OUT: {ts_label(pt_out, fps)}  ({dur:.2f}s)",
             (20, info_y+22), 0.50, ORANGE)
    else:
        _put(frame, "OUT: --", (20, info_y+22), 0.50, GRAY)

    # Play/Pause badge
    badge = "▶ PLAY" if is_playing else "⏸ PAUSE"
    badge_c = GREEN if is_playing else YELLOW
    _pill(frame, W-140, info_y-8, W-10, info_y+14, (20,20,30), 0.7, r=5)
    _put(frame, badge, (W-132, info_y+8), 0.45, badge_c)

    # ── Hint bar ─────────────────────────────────────────────────
    cv2.rectangle(frame, (0, H-22), (W, H), DARK, -1)
    _put(frame,
         "SPACE:Play  I:In  O:Out  ENTER:Cut  Z:Reset  S:Save  ←→:Frame  A/D:1s  Q:Quit",
         (8, H-6), 0.30, (70,70,80))

    # ── Notification toast ────────────────────────────────────────
    if notif and (time.time() - notif_ts < 2.5):
        (nw,_),_ = cv2.getTextSize(notif, FONT, 0.55, 1)
        nx = W//2 - nw//2
        _pill(frame, nx-14, H//2-26, nx+nw+14, H//2+10,
              (30,30,50), 0.85, r=8)
        _put(frame, notif, (nx, H//2+4), 0.55, YELLOW)

    # ── Danh sách clips đã cắt (bên phải) ────────────────────────
    if clips:
        px = W - 250
        _put(frame, "Da cat:", (px, 60), 0.42, GRAY)
        for i, (s, e, name) in enumerate(clips[-6:]):
            dur_s = (e - s) / fps
            _put(frame, f"#{i+1} {ts_label(s,fps)}→{ts_label(e,fps)} {dur_s:.1f}s",
                 (px, 80+i*18), 0.36, TEAL)


# ══════════════════════════════════════════════════════════════════
# MAIN cut
# ══════════════════════════════════════════════════════════════════

def run_cut(input_path: str, out_dir: str, label: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  [ERROR] Khong mo duoc: {input_path}")
        return

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n  Video  : {Path(input_path).name}")
    print(f"  Size   : {W}×{H}  FPS={fps:.1f}  Frames={total_frames}")
    print(f"  Dur    : {frame_to_time(total_frames, fps)}")
    print(f"  OutDir : {out_dir}\n")

    os.makedirs(out_dir, exist_ok=True)

    # State
    cur_frame  = 0
    pt_in      = None
    pt_out     = None
    is_playing = False
    clips      = []    # list of (start_frame, end_frame, out_path)
    clip_idx   = 0
    notif      = ''
    notif_ts   = 0.0

    WIN = "Videocut"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(W, 1280), min(H + 120, 800))

    import time

    # Click trên timeline để seek
    tl_x1_g = 20
    tl_x2_g = W - 20

    def on_mouse(event, x, y, flags, param):
        nonlocal cur_frame, is_playing
        if event == cv2.EVENT_LBUTTONDOWN:
            if H - 80 <= y <= H - 62:   # click trên timeline
                ratio     = max(0.0, min(1.0, (x - tl_x1_g) / (tl_x2_g - tl_x1_g)))
                cur_frame = int(ratio * (total_frames - 1))
                is_playing = False
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)

    cv2.setMouseCallback(WIN, on_mouse)

    # Đọc frame hiện tại
    def read_frame(idx: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        return f if ret else np.zeros((H, W, 3), dtype=np.uint8)

    frame = read_frame(0)

    print("  Phim: SPACE=Play/Pause  I=In  O=Out  ENTER=Cat  S=Luu  Q=Thoat")
    print("  Click tren timeline de seek nhanh\n")

    last_frame_time = time.time()

    while True:
        # Auto advance khi playing
        if is_playing:
            now = time.time()
            if now - last_frame_time >= 1.0 / fps:
                last_frame_time = now
                cur_frame += 1
                if cur_frame >= total_frames:
                    cur_frame  = total_frames - 1
                    is_playing = False
                ret, frame = cap.read()
                if not ret:
                    is_playing = False

        # Vẽ UI lên frame copy
        display = frame.copy()

        # Tô vùng IN→OUT trực tiếp trên frame
        if pt_in is not None and cur_frame >= pt_in:
            end_mark = pt_out if pt_out else cur_frame
            if cur_frame <= end_mark:
                cv2.rectangle(display, (0,0), (6, H), GREEN, -1)

        draw_ui(display, cur_frame, total_frames, fps,
                pt_in, pt_out, clips, is_playing,
                W, H, notif, notif_ts,
                label, len(clips))

        cv2.imshow(WIN, display)

        wait_ms = 1 if is_playing else 20
        key     = cv2.waitKey(wait_ms) & 0xFF

        # ── Phím bấm ────────────────────────────────────────────

        if key in (ord('q'), ord('Q'), 27):
            # Hỏi có lưu không
            print("\n  Thoat — ban co muon luu video da cat? (y/n): ", end='')
            ans = input().strip().lower()
            if ans == 'y' and clips:
                _save_source(cap, total_frames, clips, input_path, fps, W, H)
            break

        elif key == ord(' '):
            is_playing = not is_playing
            if is_playing:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)

        # Frame step
        elif key == 81 or key == 2:   # ← (Linux/Mac) hoặc key==2 (Windows)
            cur_frame  = max(0, cur_frame - 1)
            is_playing = False
            frame      = read_frame(cur_frame)
        elif key == 83 or key == 3:   # →
            cur_frame  = min(total_frames-1, cur_frame + 1)
            is_playing = False
            frame      = read_frame(cur_frame)

        # Windows arrow keys (special keys)
        elif key == 0xFF:
            pass
        elif key == 75:   # ← Windows
            cur_frame  = max(0, cur_frame - 1)
            is_playing = False
            frame      = read_frame(cur_frame)
        elif key == 77:   # → Windows
            cur_frame  = min(total_frames-1, cur_frame + 1)
            is_playing = False
            frame      = read_frame(cur_frame)

        # 1 giây
        elif key in (ord('a'), ord('A')):
            cur_frame  = max(0, cur_frame - int(fps))
            is_playing = False
            frame      = read_frame(cur_frame)
        elif key in (ord('d'), ord('D')):
            cur_frame  = min(total_frames-1, cur_frame + int(fps))
            is_playing = False
            frame      = read_frame(cur_frame)

        # Đặt IN
        elif key in (ord('i'), ord('I')):
            pt_in = cur_frame
            if pt_out is not None and pt_out <= pt_in:
                pt_out = None
            notif, notif_ts = f"IN set: {ts_label(pt_in, fps)}", time.time()
            print(f"  [IN]  frame {pt_in}  {ts_label(pt_in, fps)}")

        # Đặt OUT
        elif key in (ord('o'), ord('O')):
            if pt_in is None:
                notif, notif_ts = "Dat IN truoc!", time.time()
            elif cur_frame <= pt_in:
                notif, notif_ts = "OUT phai sau IN!", time.time()
            else:
                pt_out = cur_frame
                dur    = (pt_out - pt_in) / fps
                notif, notif_ts = f"OUT set: {ts_label(pt_out, fps)}  ({dur:.2f}s)", time.time()
                print(f"  [OUT] frame {pt_out}  {ts_label(pt_out, fps)}  dur={dur:.2f}s")

        # Xác nhận cắt
        elif key == 13:   # ENTER
            if pt_in is None or pt_out is None:
                notif, notif_ts = "Can dat ca IN va OUT!", time.time()
            elif pt_out <= pt_in:
                notif, notif_ts = "OUT phai sau IN!", time.time()
            else:
                clip_idx += 1
                out_path  = make_output_name(Path(out_dir), label, clip_idx)
                ok = write_clip(cap, pt_in, pt_out, out_path, fps, W, H)
                if ok:
                    clips.append((pt_in, pt_out, out_path))
                    dur = (pt_out - pt_in) / fps
                    notif, notif_ts = (
                        f"Cat xong #{clip_idx}  {dur:.2f}s", time.time())
                    # Reset markers
                    pt_in  = None
                    pt_out = None
                    # Restore cap position
                    cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
                else:
                    notif, notif_ts = "Loi khi cat clip!", time.time()

        # Reset IN/OUT
        elif key in (ord('z'), ord('Z')):
            pt_in  = None
            pt_out = None
            notif, notif_ts = "Reset IN/OUT", time.time()

        # Lưu video gốc đã loại bỏ các đoạn đã cắt
        elif key in (ord('s'), ord('S')):
            pass

        # Click seek (handled by mouse callback)
        # Refresh frame nếu không playing
        if not is_playing and key != ord(' '):
            pass   # frame đã được read_frame() cập nhật ở trên

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n  Tong clips da cat: {len(clips)}")
    for i, (s, e, p) in enumerate(clips, 1):
        dur = (e - s) / fps
        print(f"  #{i:2d}  {ts_label(s,fps)} → {ts_label(e,fps)}  "
              f"{dur:.2f}s  {Path(p).name}")


def _save_source(cap, total_frames: int, clips: list,
                 src_path: str, fps: float, W: int, H: int) -> bool:
    """Lưu lại video gốc đã loại bỏ các đoạn đã cắt."""
    # Chuyển clips → list (start, end)
    cut_segs = [(s, e) for s, e, _ in clips]

    # Merge overlapping segments
    cut_segs.sort()
    merged = []
    for s, e in cut_segs:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append([s, e])

    tmp_path = rebuild_source(cap, total_frames, merged,
                              src_path, fps, W, H)
    if not tmp_path:
        return False

    # Backup video gốc
    backup = src_path + '.backup'
    shutil.copy2(src_path, backup)
    print(f"  [BACKUP] {backup}")

    # Thay thế video gốc
    shutil.move(tmp_path, src_path)
    print(f"  [SAVED]  Video goc moi: {src_path}")
    print(f"           (Backup tai: {backup})")
    return True


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Cắt video dài thành nhiều video con')
    ap.add_argument('--input',   default=None,
                    help='Đường dẫn video đầu vào')
    ap.add_argument('--out_dir', default=None,
                    help='Thư mục lưu clips (default: cùng thư mục với input)')
    ap.add_argument('--label',   default=None,
                    help='Tên nhãn cho clips (default: tên file input)')
    args = ap.parse_args()

    inp = args.input

    # ── Nếu không truyền --input → mở hộp thoại chọn file ──
    if inp is None:
        import tkinter as tk
        from tkinter import filedialog
        _root = tk.Tk()
        _root.withdraw()  # ẩn cửa sổ chính
        inp = filedialog.askopenfilename(
            title="Chọn file video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*")
            ]
        )
        _root.destroy()
        if not inp:
            print("  Không chọn file, thoát.")
            sys.exit(0)

    if not os.path.isabs(inp):
        inp = str(_PROJECT_ROOT / inp)

    if not os.path.exists(inp):
        print(f"  [ERROR] Khong tim thay file: {inp}")
        sys.exit(1)

    label   = args.label or Path(inp).stem
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = str(Path(inp).parent)
    elif not os.path.isabs(out_dir):
        out_dir = str(_PROJECT_ROOT / out_dir)

    run_cut(inp, out_dir, label)

if __name__ == '__main__':
    main()