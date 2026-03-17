"""
video_trimmer_gui.py - Cắt video VSL với giao diện đồ họa tương tác
====================================================================
Tính năng:
  - Chọn file MP4 qua hộp thoại
  - Tự động phát hiện điểm cắt bằng MediaPipe
  - Giao diện kéo thanh để điều chỉnh điểm đầu / cuối
  - Preview video đã cắt ngay trong ứng dụng
  - Thêm / bớt frame tùy ý bằng nút hoặc nhập tay
  - Ưng ý → Save

Cách chạy:
    pip install mediapipe opencv-python pillow numpy
    python video_trimmer_gui.py
"""

import cv2
import os
import sys
import threading
import urllib.request
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
SHOULDER_MARGIN  = 0.04
SMOOTH_WINDOW    = 9
ACTIVE_THR       = 0.35
MIN_ACTIVE_RATIO = 0.03
PADDING_SEC      = 0.25

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

MODEL_URLS = {
    'hand_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'),
    'pose_landmarker_heavy.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'),
}

PREVIEW_W = 640
PREVIEW_H = 360


# ══════════════════════════════════════════════════════════════
# MODEL DOWNLOAD
# ══════════════════════════════════════════════════════════════

def download_model(filename, progress_cb=None):
    if os.path.exists(filename):
        return filename
    url = MODEL_URLS[filename]

    def _hook(count, block_size, total_size):
        if progress_cb and total_size > 0:
            pct = min(int(count * block_size * 100 / total_size), 100)
            progress_cb(f"Đang tải {filename}... {pct}%")

    urllib.request.urlretrieve(url, filename, reporthook=_hook)
    return filename


# ══════════════════════════════════════════════════════════════
# DETECTOR
# ══════════════════════════════════════════════════════════════

class FrameDetector:
    def __init__(self, progress_cb=None):
        if not MEDIAPIPE_OK:
            raise ImportError("Cần cài: pip install mediapipe")

        hand_m = download_model('hand_landmarker.task', progress_cb)
        pose_m = download_model('pose_landmarker_heavy.task', progress_cb)

        self.pose_det = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=pose_m),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.4,
                min_pose_presence_confidence=0.4,
                min_tracking_confidence=0.4))

        self.hand_det = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=hand_m),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4))

    def detect(self, frame_bgr):
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_res = self.pose_det.detect(mp_img)
        hand_res = self.hand_det.detect(mp_img)

        pose_lms = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None
        left_h = right_h = None
        if hand_res.hand_landmarks and hand_res.handedness:
            for i, hlms in enumerate(hand_res.hand_landmarks):
                cat = hand_res.handedness[i][0].category_name
                if cat == 'Left':
                    right_h = hlms
                else:
                    left_h = hlms
        return pose_lms, left_h, right_h

    def close(self):
        self.pose_det.close()
        self.hand_det.close()


# ══════════════════════════════════════════════════════════════
# SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════

def _wrist_ys(pose_lms, left_h, right_h):
    ys = []
    for hlms in [left_h, right_h]:
        if hlms is not None:
            ys.append(hlms[8].y)
            ys.append(hlms[0].y)
    if not ys and pose_lms is not None:
        for idx in [15, 16, 19, 20]:
            if pose_lms[idx].visibility > 0.3:
                ys.append(pose_lms[idx].y)
    return ys


def _shoulder_y(pose_lms):
    if pose_lms is None:
        return None
    ys = []
    for idx in [11, 12]:
        if pose_lms[idx].visibility > 0.3:
            ys.append(pose_lms[idx].y)
    return float(np.mean(ys)) if ys else None


def compute_active_signal(frames_data):
    N   = len(frames_data)
    raw = np.zeros(N, dtype=np.float32)
    for i, (pose, lh, rh) in enumerate(frames_data):
        sh_y = _shoulder_y(pose)
        ys   = _wrist_ys(pose, lh, rh)
        if not ys or sh_y is None:
            continue
        thr = sh_y - SHOULDER_MARGIN
        if any(y < thr for y in ys):
            raw[i] = 1.0
    kernel = np.ones(SMOOTH_WINDOW, dtype=np.float32) / SMOOTH_WINDOW
    return np.convolve(raw, kernel, mode='same')


def find_trim_range(signal, fps):
    N       = len(signal)
    binary  = (signal > ACTIVE_THR).astype(np.int32)
    padding = int(PADDING_SEC * fps)

    segments  = []
    in_seg    = False
    seg_start = 0
    for i in range(N):
        if binary[i] and not in_seg:
            in_seg    = True
            seg_start = i
        elif not binary[i] and in_seg:
            in_seg = False
            segments.append((seg_start, i - 1))
    if in_seg:
        segments.append((seg_start, N - 1))

    if not segments:
        return None

    best   = max(segments, key=lambda s: s[1] - s[0])
    length = best[1] - best[0] + 1
    if length < N * MIN_ACTIVE_RATIO:
        return None

    start = max(0,     best[0] - padding)
    end   = min(N - 1, best[1] + padding)
    return start, end


# ══════════════════════════════════════════════════════════════
# MAIN GUI APPLICATION
# ══════════════════════════════════════════════════════════════

class VideoTrimmerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("✂️ Video Trimmer - VSL")
        self.root.geometry("900x720")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(True, True)

        # State
        self.video_path   = None
        self.all_frames   = []       # list of np.ndarray (BGR)
        self.total_frames = 0
        self.fps          = 30.0
        self.start_frame  = 0
        self.end_frame    = 0
        self.current_frame_idx = 0
        self.is_playing   = False
        self.play_thread  = None
        self.play_mode    = "full"   # "full" | "trim"

        self._build_ui()

    # ── UI BUILD ──────────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton",
                         background="#7c3aed", foreground="white",
                         font=("Segoe UI", 10, "bold"), padding=6)
        style.map("TButton",
                  background=[("active", "#5b21b6")])
        style.configure("Green.TButton",
                         background="#059669", foreground="white",
                         font=("Segoe UI", 10, "bold"), padding=6)
        style.map("Green.TButton",
                  background=[("active", "#047857")])
        style.configure("Red.TButton",
                         background="#dc2626", foreground="white",
                         font=("Segoe UI", 10, "bold"), padding=6)
        style.map("Red.TButton",
                  background=[("active", "#b91c1c")])
        style.configure("TScale",
                         background="#1e1e2e", troughcolor="#374151",
                         slidercolor="#7c3aed")
        style.configure("TLabel",
                         background="#1e1e2e", foreground="#e5e7eb",
                         font=("Segoe UI", 9))
        style.configure("Title.TLabel",
                         background="#1e1e2e", foreground="#a78bfa",
                         font=("Segoe UI", 13, "bold"))
        style.configure("Info.TLabel",
                         background="#2d2d44", foreground="#86efac",
                         font=("Courier New", 9))
        style.configure("TFrame", background="#1e1e2e")

        # ── Top bar ──
        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, padx=12, pady=(10, 4))

        ttk.Label(top, text="✂️ Video Trimmer VSL", style="Title.TLabel").pack(side=tk.LEFT)

        ttk.Button(top, text="📂 Mở video",
                   command=self._open_file).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="🤖 Tự động phát hiện",
                   command=self._auto_detect).pack(side=tk.RIGHT, padx=4)

        # ── Video preview ──
        self.canvas = tk.Canvas(self.root, width=PREVIEW_W, height=PREVIEW_H,
                                bg="#000000", highlightthickness=0)
        self.canvas.pack(pady=6)
        self._show_placeholder()

        # ── Timeline scrubber ──
        scrub_frame = ttk.Frame(self.root)
        scrub_frame.pack(fill=tk.X, padx=20, pady=(0, 2))

        ttk.Label(scrub_frame, text="Frame hiện tại:").pack(side=tk.LEFT)
        self.lbl_current = ttk.Label(scrub_frame, text="0 / 0  (0.00s)")
        self.lbl_current.pack(side=tk.LEFT, padx=8)

        self.scrub_var = tk.IntVar(value=0)
        self.scrub = ttk.Scale(self.root, from_=0, to=100,
                               orient=tk.HORIZONTAL, variable=self.scrub_var,
                               command=self._on_scrub)
        self.scrub.pack(fill=tk.X, padx=20, pady=(0, 4))

        # ── Playback controls ──
        pb = ttk.Frame(self.root)
        pb.pack(pady=2)

        ttk.Button(pb, text="⏮", width=4,
                   command=lambda: self._seek_to(self.start_frame)).pack(side=tk.LEFT, padx=2)
        ttk.Button(pb, text="◀◀", width=4,
                   command=lambda: self._step(-10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(pb, text="◀", width=3,
                   command=lambda: self._step(-1)).pack(side=tk.LEFT, padx=2)

        self.btn_play = ttk.Button(pb, text="▶ Play (Full)",
                                   command=self._toggle_play_full)
        self.btn_play.pack(side=tk.LEFT, padx=6)

        self.btn_play_trim = ttk.Button(pb, text="▶ Preview Cắt",
                                        command=self._toggle_play_trim,
                                        style="Green.TButton")
        self.btn_play_trim.pack(side=tk.LEFT, padx=6)

        ttk.Button(pb, text="▶", width=3,
                   command=lambda: self._step(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(pb, text="▶▶", width=4,
                   command=lambda: self._step(10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(pb, text="⏭", width=4,
                   command=lambda: self._seek_to(self.end_frame)).pack(side=tk.LEFT, padx=2)

        # ── Trim range controls ──
        trim_outer = tk.Frame(self.root, bg="#2d2d44", bd=1, relief=tk.GROOVE)
        trim_outer.pack(fill=tk.X, padx=20, pady=6)

        ttk.Label(trim_outer, text="⚙️  ĐIỀU CHỈNH ĐIỂM CẮT",
                  background="#2d2d44", foreground="#a78bfa",
                  font=("Segoe UI", 10, "bold")).pack(pady=(6, 4))

        # START controls
        sf = tk.Frame(trim_outer, bg="#2d2d44")
        sf.pack(fill=tk.X, padx=10, pady=3)

        tk.Label(sf, text="▶ Điểm BẮT ĐẦU (frame):", bg="#2d2d44",
                 fg="#86efac", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)

        self.start_var = tk.IntVar(value=0)
        self.start_scale = ttk.Scale(sf, from_=0, to=100,
                                     orient=tk.HORIZONTAL,
                                     variable=self.start_var,
                                     command=self._on_start_scale)
        self.start_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        self.start_entry = ttk.Entry(sf, width=7,
                                     textvariable=self.start_var)
        self.start_entry.pack(side=tk.LEFT)
        self.start_entry.bind("<Return>", self._on_start_entry)

        ttk.Button(sf, text="-10", width=4,
                   command=lambda: self._adjust_start(-10)).pack(side=tk.LEFT, padx=1)
        ttk.Button(sf, text="-1", width=3,
                   command=lambda: self._adjust_start(-1)).pack(side=tk.LEFT, padx=1)
        ttk.Button(sf, text="+1", width=3,
                   command=lambda: self._adjust_start(1)).pack(side=tk.LEFT, padx=1)
        ttk.Button(sf, text="+10", width=4,
                   command=lambda: self._adjust_start(10)).pack(side=tk.LEFT, padx=1)
        ttk.Button(sf, text="📍 Frame này",
                   command=self._set_start_to_current).pack(side=tk.LEFT, padx=4)

        self.lbl_start = ttk.Label(sf, text="0.00s", width=7)
        self.lbl_start.pack(side=tk.LEFT, padx=4)

        # END controls
        ef = tk.Frame(trim_outer, bg="#2d2d44")
        ef.pack(fill=tk.X, padx=10, pady=(0, 6))

        tk.Label(ef, text="⏹ Điểm KẾT THÚC (frame):", bg="#2d2d44",
                 fg="#fca5a5", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)

        self.end_var = tk.IntVar(value=0)
        self.end_scale = ttk.Scale(ef, from_=0, to=100,
                                   orient=tk.HORIZONTAL,
                                   variable=self.end_var,
                                   command=self._on_end_scale)
        self.end_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        self.end_entry = ttk.Entry(ef, width=7,
                                   textvariable=self.end_var)
        self.end_entry.pack(side=tk.LEFT)
        self.end_entry.bind("<Return>", self._on_end_entry)

        ttk.Button(ef, text="-10", width=4,
                   command=lambda: self._adjust_end(-10)).pack(side=tk.LEFT, padx=1)
        ttk.Button(ef, text="-1", width=3,
                   command=lambda: self._adjust_end(-1)).pack(side=tk.LEFT, padx=1)
        ttk.Button(ef, text="+1", width=3,
                   command=lambda: self._adjust_end(1)).pack(side=tk.LEFT, padx=1)
        ttk.Button(ef, text="+10", width=4,
                   command=lambda: self._adjust_end(10)).pack(side=tk.LEFT, padx=1)
        ttk.Button(ef, text="📍 Frame này",
                   command=self._set_end_to_current).pack(side=tk.LEFT, padx=4)

        self.lbl_end = ttk.Label(ef, text="0.00s", width=7)
        self.lbl_end.pack(side=tk.LEFT, padx=4)

        # ── Info bar ──
        info_f = tk.Frame(self.root, bg="#2d2d44")
        info_f.pack(fill=tk.X, padx=20, pady=(0, 6))
        self.lbl_info = tk.Label(info_f, text="  Chưa tải video",
                                  bg="#2d2d44", fg="#86efac",
                                  font=("Courier New", 9), anchor="w")
        self.lbl_info.pack(fill=tk.X, padx=6, pady=4)

        # ── Status + Save ──
        bot = ttk.Frame(self.root)
        bot.pack(fill=tk.X, padx=20, pady=(0, 10))

        self.lbl_status = ttk.Label(bot, text="Sẵn sàng", foreground="#9ca3af")
        self.lbl_status.pack(side=tk.LEFT)

        ttk.Button(bot, text="💾 Lưu video đã cắt",
                   command=self._save_video,
                   style="Green.TButton").pack(side=tk.RIGHT, padx=4)
        ttk.Button(bot, text="🔄 Reset về auto",
                   command=self._reset_to_auto).pack(side=tk.RIGHT, padx=4)

    # ── PLACEHOLDER ───────────────────────────────────────────

    def _show_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, PREVIEW_W, PREVIEW_H,
                                     fill="#111827", outline="")
        self.canvas.create_text(PREVIEW_W // 2, PREVIEW_H // 2,
                                text="📂  Mở video để bắt đầu",
                                fill="#6b7280", font=("Segoe UI", 16))

    # ── OPEN FILE ─────────────────────────────────────────────

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Chọn file video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                       ("All files", "*.*")])
        if not path:
            return

        self._stop_play()
        self.video_path = path
        self._set_status("Đang tải video...")
        self.root.update()

        # Load all frames
        self.all_frames = []
        cap = cv2.VideoCapture(path)
        self.fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._set_status(f"Đang đọc {self.total_frames} frames...")
        self.root.update()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.all_frames.append(frame)
        cap.release()

        self.total_frames = len(self.all_frames)
        if self.total_frames == 0:
            messagebox.showerror("Lỗi", "Không đọc được frame nào!")
            return

        # Init trim range = full video
        self.start_frame = 0
        self.end_frame   = self.total_frames - 1
        self._update_scale_ranges()
        self._update_trim_labels()
        self._seek_to(0)
        self._update_info()

        self._set_status(f"Đã tải: {os.path.basename(path)} "
                         f"({self.total_frames}f @ {self.fps:.1f}fps)")

    # ── AUTO DETECT ───────────────────────────────────────────

    def _auto_detect(self):
        if not self.all_frames:
            messagebox.showwarning("Chú ý", "Hãy mở video trước!")
            return
        if not MEDIAPIPE_OK:
            messagebox.showerror("Lỗi",
                "Cần cài mediapipe:\n  pip install mediapipe")
            return

        self._stop_play()
        self._set_status("Đang khởi tạo MediaPipe...")
        self.root.update()

        def _worker():
            try:
                detector = FrameDetector(
                    progress_cb=lambda msg: self._set_status(msg))
                frames_data = []
                total = len(self.all_frames)
                for i, frame in enumerate(self.all_frames):
                    pose, lh, rh = detector.detect(frame)
                    frames_data.append((pose, lh, rh))
                    if i % 30 == 0:
                        self._set_status(
                            f"Phân tích frame {i}/{total}... "
                            f"({i*100//total}%)")
                detector.close()

                signal = compute_active_signal(frames_data)
                result = find_trim_range(signal, self.fps)

                if result:
                    s, e = result
                    self.root.after(0, lambda: self._apply_auto(s, e))
                else:
                    self.root.after(0, lambda: self._auto_no_result())

            except Exception as ex:
                self.root.after(0,
                    lambda: messagebox.showerror("Lỗi", str(ex)))
                self.root.after(0,
                    lambda: self._set_status("Lỗi khi phân tích"))

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_auto(self, s, e):
        self.auto_start = s
        self.auto_end   = e
        self.start_frame = s
        self.end_frame   = e
        self._update_scale_ranges()
        self._update_trim_labels()
        self._seek_to(s)
        self._update_info()
        dur = (e - s) / self.fps
        self._set_status(
            f"✅ Tự động phát hiện: frame {s}~{e}  ({dur:.2f}s)")

    def _auto_no_result(self):
        messagebox.showinfo("Thông báo",
            "Không phát hiện được ký hiệu rõ ràng.\n"
            "Giữ nguyên toàn bộ video.")
        self._set_status("Không phát hiện — giữ nguyên")

    def _reset_to_auto(self):
        if hasattr(self, 'auto_start'):
            self.start_frame = self.auto_start
            self.end_frame   = self.auto_end
            self._update_scale_ranges()
            self._update_trim_labels()
            self._seek_to(self.start_frame)
            self._update_info()
        else:
            messagebox.showinfo("Chú ý", "Chưa chạy Auto Detect lần nào.")

    # ── DISPLAY ───────────────────────────────────────────────

    def _display_frame(self, idx):
        if not self.all_frames or idx < 0 or idx >= len(self.all_frames):
            return
        frame = self.all_frames[idx]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w  = rgb.shape[:2]
        scale = min(PREVIEW_W / w, PREVIEW_H / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb, (nw, nh))

        # Letterbox
        canvas_img = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)
        y0 = (PREVIEW_H - nh) // 2
        x0 = (PREVIEW_W - nw) // 2
        canvas_img[y0:y0+nh, x0:x0+nw] = resized

        # Highlight trim boundaries
        in_trim = (self.start_frame <= idx <= self.end_frame)
        border_color = (0, 200, 80) if in_trim else (200, 60, 60)
        cv2.rectangle(canvas_img, (0, 0),
                      (PREVIEW_W-1, PREVIEW_H-1),
                      border_color, 3)

        # Label
        label = f"Frame {idx}  ({idx/self.fps:.2f}s)"
        if idx == self.start_frame:
            label += "  ◀ START"
        elif idx == self.end_frame:
            label += "  END ▶"
        cv2.putText(canvas_img, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    border_color, 1, cv2.LINE_AA)

        img_pil  = Image.fromarray(canvas_img)
        img_tk   = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas._img = img_tk  # prevent GC

        # Update scrubber
        self.current_frame_idx = idx
        self.scrub_var.set(idx)
        self.lbl_current.config(
            text=f"{idx} / {self.total_frames-1}  ({idx/self.fps:.2f}s)")

    def _seek_to(self, idx):
        idx = max(0, min(idx, self.total_frames - 1))
        self._display_frame(idx)

    def _step(self, delta):
        self._stop_play()
        self._seek_to(self.current_frame_idx + delta)

    # ── SCRUBBER ──────────────────────────────────────────────

    def _on_scrub(self, val):
        idx = int(float(val))
        self._display_frame(idx)

    # ── TRIM SCALE CALLBACKS ──────────────────────────────────

    def _update_scale_ranges(self):
        n = max(self.total_frames - 1, 1)
        self.scrub.configure(to=n)
        self.start_scale.configure(to=n)
        self.end_scale.configure(to=n)
        self.start_var.set(self.start_frame)
        self.end_var.set(self.end_frame)

    def _on_start_scale(self, val):
        v = int(float(val))
        v = max(0, min(v, self.end_frame))
        self.start_frame = v
        self.start_var.set(v)
        self._update_trim_labels()
        self._seek_to(v)
        self._update_info()

    def _on_end_scale(self, val):
        v = int(float(val))
        v = max(self.start_frame, min(v, self.total_frames - 1))
        self.end_frame = v
        self.end_var.set(v)
        self._update_trim_labels()
        self._seek_to(v)
        self._update_info()

    def _on_start_entry(self, event=None):
        try:
            v = int(self.start_var.get())
            v = max(0, min(v, self.end_frame))
            self.start_frame = v
            self.start_var.set(v)
            self._update_trim_labels()
            self._seek_to(v)
            self._update_info()
        except (ValueError, tk.TclError):
            pass

    def _on_end_entry(self, event=None):
        try:
            v = int(self.end_var.get())
            v = max(self.start_frame, min(v, self.total_frames - 1))
            self.end_frame = v
            self.end_var.set(v)
            self._update_trim_labels()
            self._seek_to(v)
            self._update_info()
        except (ValueError, tk.TclError):
            pass

    def _adjust_start(self, delta):
        v = max(0, min(self.start_frame + delta, self.end_frame))
        self.start_frame = v
        self.start_var.set(v)
        self._update_trim_labels()
        self._seek_to(v)
        self._update_info()

    def _adjust_end(self, delta):
        v = max(self.start_frame,
                min(self.end_frame + delta, self.total_frames - 1))
        self.end_frame = v
        self.end_var.set(v)
        self._update_trim_labels()
        self._seek_to(v)
        self._update_info()

    def _set_start_to_current(self):
        self._adjust_start(0)  # snap to valid
        v = max(0, min(self.current_frame_idx, self.end_frame))
        self.start_frame = v
        self.start_var.set(v)
        self._update_trim_labels()
        self._update_info()

    def _set_end_to_current(self):
        v = max(self.start_frame,
                min(self.current_frame_idx, self.total_frames - 1))
        self.end_frame = v
        self.end_var.set(v)
        self._update_trim_labels()
        self._update_info()

    def _update_trim_labels(self):
        self.lbl_start.config(
            text=f"{self.start_frame/self.fps:.2f}s")
        self.lbl_end.config(
            text=f"{self.end_frame/self.fps:.2f}s")

    # ── INFO BAR ──────────────────────────────────────────────

    def _update_info(self):
        if not self.all_frames:
            return
        total_s = self.total_frames / self.fps
        trim_s  = (self.end_frame - self.start_frame + 1) / self.fps
        cut_s   = total_s - trim_s
        pct     = cut_s / total_s * 100 if total_s > 0 else 0
        self.lbl_info.config(
            text=f"  Gốc: {self.total_frames}f ({total_s:.2f}s)  │  "
                 f"Sau cắt: frame {self.start_frame}~{self.end_frame} "
                 f"({self.end_frame-self.start_frame+1}f, {trim_s:.2f}s)  │  "
                 f"Cắt bỏ: {cut_s:.2f}s ({pct:.0f}%)")

    # ── PLAYBACK ──────────────────────────────────────────────

    def _toggle_play_full(self):
        if self.is_playing and self.play_mode == "full":
            self._stop_play()
        else:
            self._stop_play()
            self.play_mode = "full"
            self._start_play(0, self.total_frames - 1)
            self.btn_play.config(text="⏸ Pause")

    def _toggle_play_trim(self):
        if self.is_playing and self.play_mode == "trim":
            self._stop_play()
        else:
            self._stop_play()
            self.play_mode = "trim"
            self._start_play(self.start_frame, self.end_frame)
            self.btn_play_trim.config(text="⏸ Pause Preview")

    def _start_play(self, start, end):
        self.is_playing = True
        delay = max(1, int(1000 / self.fps))

        def _run():
            idx = start
            while self.is_playing and idx <= end:
                self.root.after(0, lambda i=idx: self._display_frame(i))
                import time
                time.sleep(delay / 1000.0)
                idx += 1
            self.root.after(0, self._stop_play)

        self.play_thread = threading.Thread(target=_run, daemon=True)
        self.play_thread.start()

    def _stop_play(self):
        self.is_playing = False
        self.btn_play.config(text="▶ Play (Full)")
        self.btn_play_trim.config(text="▶ Preview Cắt")

    # ── SAVE ──────────────────────────────────────────────────

    def _save_video(self):
        if not self.all_frames:
            messagebox.showwarning("Chú ý", "Chưa tải video!")
            return

        p = Path(self.video_path)
        default_name = p.stem + "_trimmed" + p.suffix
        out_path = filedialog.asksaveasfilename(
            title="Lưu video đã cắt",
            initialdir=str(p.parent),
            initialfile=default_name,
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi"), ("All", "*.*")])
        if not out_path:
            return

        self._stop_play()
        self._set_status("Đang xuất video...")
        self.root.update()

        def _worker():
            try:
                h, w = self.all_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(out_path, fourcc,
                                         self.fps, (w, h))
                for i in range(self.start_frame, self.end_frame + 1):
                    writer.write(self.all_frames[i])
                writer.release()

                size_mb = os.path.getsize(out_path) / 1024 / 1024
                trim_s  = (self.end_frame - self.start_frame + 1) / self.fps
                self.root.after(0, lambda: messagebox.showinfo(
                    "✅ Xong!",
                    f"Đã lưu thành công!\n\n"
                    f"File: {os.path.basename(out_path)}\n"
                    f"Độ dài: {trim_s:.2f}s\n"
                    f"Kích thước: {size_mb:.1f} MB"))
                self.root.after(0, lambda: self._set_status(
                    f"✅ Đã lưu: {os.path.basename(out_path)} ({size_mb:.1f}MB)"))
            except Exception as ex:
                self.root.after(0,
                    lambda: messagebox.showerror("Lỗi", str(ex)))
                self.root.after(0,
                    lambda: self._set_status("Lỗi khi lưu"))

        threading.Thread(target=_worker, daemon=True).start()

    # ── STATUS ────────────────────────────────────────────────

    def _set_status(self, msg):
        try:
            self.lbl_status.config(text=msg)
            self.root.update_idletasks()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    app  = VideoTrimmerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()