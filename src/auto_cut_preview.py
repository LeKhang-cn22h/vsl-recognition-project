"""
auto_cut_preview.py - VSL Video Cutter
Logic: phan tich TOAN BO folder -> duyet clip -> luu 1 lan
Cat doan: tay TREN eo = dang ky hieu; tay XUONG DUOI eo = ket thuc doan.
"""

import os, sys, cv2, json, time, argparse, threading, urllib.request, random
import numpy as np
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk
    HAS_TK = True
except ImportError:
    HAS_TK = False

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    HAS_MP = True
except ImportError:
    HAS_MP = False

PREVIEW_W   = 720
PREVIEW_H   = 405
SPLITS      = ["train", "val", "test"]
SPLIT_RATIO = {"train": 0.7, "val": 0.15, "test": 0.15}
VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".webm",
               ".MP4", ".MOV", ".AVI", ".MKV"}

MODEL_URLS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"),
    "pose_landmarker_heavy.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"),
}

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = (_SCRIPT_DIR.parent
                 if _SCRIPT_DIR.name.lower() == "src"
                 else _SCRIPT_DIR)
_ROOT = _SCRIPT_DIR

BG    = "#0d0d14"
BG2   = "#161622"
PANEL = "#1e1e30"
CARD  = "#252538"
ACC   = "#7c3aed"
ACC2  = "#a78bfa"
GRN   = "#10b981"
GRN2  = "#6ee7b7"
RED   = "#ef4444"
RED2  = "#fca5a5"
YEL   = "#f59e0b"
GRAY  = "#6b7280"
GR2   = "#9ca3af"
WHT   = "#f3f4f6"
MONO  = "Courier New"


# ─────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────

def ensure_model(name):
    p = _ROOT / name
    if not p.exists():
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(MODEL_URLS[name], str(p))
    return str(p)


# ─────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────

def get_existing_labels(data_dir):
    labels = set()
    data_dir = Path(data_dir)
    for split in SPLITS:
        sd = data_dir / split
        if sd.is_dir():
            for d in sd.iterdir():
                if d.is_dir():
                    labels.add(d.name)
    return sorted(labels)


def count_label_clips(data_dir, label):
    counts = {}
    for split in SPLITS:
        d = Path(data_dir) / split / label
        counts[split] = len(list(d.glob("*.mp4"))) if d.is_dir() else 0
    return counts


def next_clip_index(data_dir, label):
    max_idx = 0
    for split in SPLITS:
        d = Path(data_dir) / split / label
        if not d.is_dir():
            continue
        for f in d.glob(f"{label}_*.mp4"):
            try:
                idx = int(f.stem.split("_")[-1])
                max_idx = max(max_idx, idx)
            except ValueError:
                pass
    return max_idx + 1


# ─────────────────────────────────────────────────────────────────────
# MediaPipe detector
# ─────────────────────────────────────────────────────────────────────

class Detector:
    def __init__(self, cb=None):
        if cb:
            cb("Khoi tao MediaPipe...")
        self.pose = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=ensure_model("pose_landmarker_heavy.task")),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.4,
                min_pose_presence_confidence=0.4,
                min_tracking_confidence=0.4))
        self.hand = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=ensure_model("hand_landmarker.task")),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4))

    def detect(self, bgr):
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        pr     = self.pose.detect(mp_img)
        hr     = self.hand.detect(mp_img)
        pose   = pr.pose_landmarks[0] if pr.pose_landmarks else None
        hands  = hr.hand_landmarks    if hr.hand_landmarks else []
        return pose, hands

    def close(self):
        self.pose.close()
        self.hand.close()


# ─────────────────────────────────────────────────────────────────────
# Landmark helpers
# ─────────────────────────────────────────────────────────────────────

def _hip_y(pose):
    """
    Y trung binh cua 2 hong (landmark 23, 24).
    MediaPipe: Y tang tu tren xuong duoi (0=dinh anh, 1=day anh).
    Tay TREN eo khi wrist_y < hip_y.
    """
    if pose is None:
        return None
    ys = [pose[i].y for i in [23, 24] if pose[i].visibility > 0.3]
    return float(np.mean(ys)) if ys else None


def _wrist_min_y(pose, hands):
    """Y nho nhat (cao nhat) cua ban tay / co tay."""
    ys = []
    for h in hands:
        ys += [h[0].y, h[8].y, h[4].y]
    if not ys and pose:
        ys = [pose[i].y for i in [15, 16, 19, 20]
              if pose[i].visibility > 0.3]
    return min(ys) if ys else None


def _finger_curl_active(hands, curl_thresh=0.15):
    if not hands:
        return False
    FINGERS = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
    for hand in hands:
        curls = []
        for mcp_i, pip_i, tip_i in FINGERS:
            mcp = np.array([hand[mcp_i].x, hand[mcp_i].y])
            pip = np.array([hand[pip_i].x, hand[pip_i].y])
            tip = np.array([hand[tip_i].x, hand[tip_i].y])
            v1 = pip - mcp
            v2 = tip - pip
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                curls.append(0.0)
                continue
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            curls.append(1.0 - cos_a)
        if float(np.mean(curls)) > curl_thresh:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────────────────────────────

def analyze_frames(frames, fps,
                   padding_sec=0.25, min_dur=0.4, idle_sec=0.7,
                   waist_offset=0.03, smooth_w=7,
                   use_hip=True, use_finger=False,
                   finger_curl_thresh=0.15,
                   progress_cb=None):
    det = Detector(progress_cb)
    N   = len(frames)
    raw = np.zeros(N, dtype=np.float32)

    for i, frame in enumerate(frames):
        if progress_cb and i % 15 == 0:
            progress_cb(f"Phan tich frame {i}/{N}  ({i*100//N}%)")
        pose, hands = det.detect(frame)

        cond_hip    = True
        cond_finger = True

        if use_hip:
            hy = _hip_y(pose)
            wy = _wrist_min_y(pose, hands)
            cond_hip = (hy is not None and wy is not None
                        and wy < hy - waist_offset)

        if use_finger:
            cond_finger = _finger_curl_active(hands, finger_curl_thresh)

        if use_hip and use_finger:
            active = cond_hip and cond_finger
        elif use_hip:
            active = cond_hip
        elif use_finger:
            active = cond_finger and bool(hands)
        else:
            active = bool(hands)

        if active:
            raw[i] = 1.0

    det.close()

    k      = np.ones(smooth_w, dtype=np.float32) / smooth_w
    signal = np.convolve(raw, k, mode="same")
    binary = (signal > 0.3).astype(np.int32)

    segs, in_s, s0 = [], False, 0
    for i in range(N):
        if binary[i] and not in_s:
            in_s, s0 = True, i
        elif not binary[i] and in_s:
            in_s = False
            segs.append([s0, i - 1])
    if in_s:
        segs.append([s0, N - 1])
    if not segs:
        return []

    gap    = int(idle_sec * fps)
    merged = [segs[0][:]]
    for s, e in segs[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    min_f = int(min_dur * fps)
    pad   = int(padding_sec * fps)
    return [[max(0, s - pad), min(N - 1, e + pad)]
            for s, e in merged if e - s >= min_f]


# ─────────────────────────────────────────────────────────────────────
# Dialogs
# ─────────────────────────────────────────────────────────────────────

class LabelDialog(tk.Toplevel):
    def __init__(self, parent, data_dir, n_clips):
        super().__init__(parent)
        self.data_dir = Path(data_dir)
        self.n_clips  = n_clips
        self.result   = None

        self.title("Chon nhan & split")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.grab_set()
        self.transient(parent)

        existing = get_existing_labels(data_dir)
        self._build(existing)
        self.wait_window()

    def _build(self, existing):
        pad = dict(padx=20)
        tk.Label(self, text="Gan nhan & Split cho clip",
                 bg=BG, fg=ACC2,
                 font=(MONO, 13, "bold")).pack(pady=(16, 2), **pad)
        tk.Label(self, text=f"Se luu {self.n_clips} clip duoc chon",
                 bg=BG, fg=GR2, font=(MONO, 9)).pack(pady=(0, 2))
        tk.Label(self, text=f"Luu vao: {self.data_dir}",
                 bg=BG, fg=YEL, font=(MONO, 8)).pack(pady=(0, 8))

        tk.Label(self, text="Nhan co san  (click de chon):",
                 bg=BG, fg=WHT,
                 font=(MONO, 9, "bold")).pack(anchor="w", **pad)

        lf = tk.Frame(self, bg=PANEL, bd=1, relief=tk.SOLID)
        lf.pack(fill=tk.X, padx=20, pady=(4, 0))
        self._lbl_var = tk.StringVar()

        if existing:
            sb = tk.Scrollbar(lf, orient=tk.VERTICAL)
            self._lb = tk.Listbox(
                lf, listvariable=tk.StringVar(value=existing),
                yscrollcommand=sb.set, selectmode=tk.SINGLE,
                bg=CARD, fg=WHT, selectbackground=ACC,
                font=(MONO, 10), height=min(7, len(existing)),
                relief=tk.FLAT, highlightthickness=0, bd=0)
            sb.config(command=self._lb.yview)
            self._lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            sb.pack(side=tk.RIGHT, fill=tk.Y)

            def _sel(evt):
                sel = self._lb.curselection()
                if sel:
                    self._lbl_var.set(existing[sel[0]])
                    self._new_ent.delete(0, tk.END)
            self._lb.bind("<<ListboxSelect>>", _sel)
        else:
            tk.Label(lf, text="  (chua co nhan nao - hay tao moi)",
                     bg=CARD, fg=GRAY, font=(MONO, 9)).pack(pady=6)
            self._lb = None

        nr = tk.Frame(self, bg=BG)
        nr.pack(fill=tk.X, padx=20, pady=(10, 0))
        tk.Label(nr, text="Tao nhan moi:", bg=BG, fg=GRN2,
                 font=(MONO, 9, "bold")).pack(side=tk.LEFT)
        self._new_ent = tk.Entry(nr, font=(MONO, 11), width=20,
                                 bg=CARD, fg=WHT, insertbackground=WHT,
                                 relief=tk.FLAT, bd=4)
        self._new_ent.pack(side=tk.LEFT, padx=(8, 0))

        def _typed(evt=None):
            txt = self._new_ent.get().strip()
            if txt:
                self._lbl_var.set(txt)
                if self._lb:
                    self._lb.selection_clear(0, tk.END)
        self._new_ent.bind("<KeyRelease>", _typed)

        tk.Label(self, text="Chia split:", bg=BG, fg=WHT,
                 font=(MONO, 9, "bold")).pack(anchor="w", padx=20, pady=(12, 2))

        self._split_mode = tk.StringVar(value="auto")
        modes_f = tk.Frame(self, bg=BG)
        modes_f.pack(anchor="w", padx=28)
        rkw = dict(bg=BG, fg=GR2, selectcolor=PANEL,
                   activebackground=BG, activeforeground=WHT, font=(MONO, 9))
        tk.Radiobutton(modes_f,
                       text="Tu dong  (70% train / 15% val / 15% test)",
                       variable=self._split_mode, value="auto",
                       **rkw).pack(anchor="w")
        for sp in SPLITS:
            tk.Radiobutton(modes_f, text=f"Tat ca vao  {sp}",
                           variable=self._split_mode, value=sp,
                           **rkw).pack(anchor="w")

        self._cnt_lbl = tk.Label(self, text="", bg=BG, fg=YEL,
                                 font=(MONO, 8), justify=tk.LEFT)
        self._cnt_lbl.pack(pady=(6, 0), padx=20, anchor="w")

        def _upd(*_):
            lbl = self._lbl_var.get().strip()
            if not lbl:
                self._cnt_lbl.config(text="")
                return
            counts = count_label_clips(self.data_dir, lbl)
            mode   = self._split_mode.get()
            n      = self.n_clips
            if mode == "auto":
                nt = max(1, round(n * 0.7))
                nv = max(0, round(n * 0.15))
                ne = n - nt - nv
                txt = (f"Hien co  train:{counts['train']}  "
                       f"val:{counts['val']}  test:{counts['test']}\n"
                       f"Se them  train:+{nt}  val:+{nv}  test:+{ne}")
            else:
                txt = (f"Hien co  {mode}:{counts[mode]}\n"
                       f"Se them  {mode}:+{n}")
            self._cnt_lbl.config(text=txt)

        self._lbl_var.trace_add("write", _upd)
        self._split_mode.trace_add("write", _upd)

        br = tk.Frame(self, bg=BG)
        br.pack(pady=(14, 16))
        bkw = dict(font=(MONO, 10, "bold"), relief=tk.FLAT,
                   cursor="hand2", padx=14, pady=7)
        tk.Button(br, text="Xac nhan", bg=GRN, fg="white",
                  command=self._confirm, **bkw).pack(side=tk.LEFT, padx=8)
        tk.Button(br, text="Huy", bg="#374151", fg=WHT,
                  command=self.destroy, **bkw).pack(side=tk.LEFT, padx=8)

    def _confirm(self):
        label = (self._new_ent.get().strip() or self._lbl_var.get().strip())
        if not label:
            messagebox.showwarning("Thieu nhan",
                                   "Vui long chon hoac nhap ten nhan!",
                                   parent=self)
            return
        self.result = (label.replace(" ", "_"), self._split_mode.get())
        self.destroy()


class ProgressWin:
    def __init__(self, root, title="Dang xu ly..."):
        self.win = tk.Toplevel(root)
        self.win.title(title)
        self.win.configure(bg=BG)
        self.win.geometry("540x150")
        self.win.resizable(False, False)

        self._title_lbl = tk.Label(self.win, text=title,
                                   bg=BG, fg=ACC2,
                                   font=(MONO, 11, "bold"))
        self._title_lbl.pack(pady=(16, 2))

        self._lbl = tk.Label(self.win, text="Chuan bi...",
                             bg=BG, fg=GR2, font=(MONO, 9))
        self._lbl.pack()

        self._pb = ttk.Progressbar(self.win, mode="indeterminate", length=480)
        self._pb.pack(pady=8)
        self._pb.start(10)

    def set_title(self, t):
        try:
            self._title_lbl.config(text=t)
            self.win.update_idletasks()
        except Exception:
            pass

    def update(self, msg):
        try:
            self._lbl.config(text=msg)
            self.win.update_idletasks()
        except Exception:
            pass

    def close(self):
        try:
            self.win.destroy()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────

class App:
    def __init__(self, root, data_dir, cfg):
        self.root     = root
        self.data_dir = Path(data_dir)
        self.cfg      = cfg

        self.folder_path = None
        self.video_files = []

        # ── Batch results ──────────────────────────────────────────────
        # all_results: list of dict, one per video
        #   { 'path': Path, 'fps': float,
        #     'clips': [[s, e], ...], 'kept': [bool, ...] }
        # Frames are NOT stored here — loaded on demand for preview.
        self.all_results = []

        # flat_clips[i] = (video_idx, clip_idx_within_that_video)
        self.flat_clips  = []
        self.flat_idx    = -1

        # Currently loaded video frames for preview
        self.cur_vid_idx = -1
        self.frames      = []
        self.fps         = 30.0

        self.play_pos   = 0
        self.is_playing = False
        self._play_job  = None

        self._cfg_vars   = {}
        self._use_hip    = tk.BooleanVar(value=True)
        self._use_finger = tk.BooleanVar(value=False)

        root.title(f"VSL Auto Cut  —  {data_dir}")
        root.configure(bg=BG)
        root.geometry("1120x860")
        root.minsize(900, 700)

        self._build_ui()

    # ─── UI construction ─────────────────────────────────────────────

    def _build_ui(self):
        # Toolbar
        tb = tk.Frame(self.root, bg=PANEL, pady=8)
        tb.pack(fill=tk.X)

        tk.Label(tb, text="VSL Auto Cut",
                 bg=PANEL, fg=ACC2,
                 font=(MONO, 14, "bold")).pack(side=tk.LEFT, padx=16)

        bkw = dict(font=(MONO, 9, "bold"), relief=tk.FLAT,
                   cursor="hand2", padx=10, pady=4)

        tk.Button(tb, text="Mo Folder",
                  bg=ACC, fg=WHT,
                  command=self._open_folder, **bkw).pack(side=tk.LEFT, padx=4)

        tk.Button(tb, text="▶ Phan tich TAT CA",
                  bg=YEL, fg="#1f2937",
                  command=self._analyse_all, **bkw).pack(side=tk.LEFT, padx=4)

        tk.Button(tb, text="Lam moi DS",
                  bg="#374151", fg=WHT,
                  command=self._refresh_list, **bkw).pack(side=tk.LEFT, padx=4)

        self.lbl_folder = tk.Label(tb, text="Chua chon folder",
                                   bg=PANEL, fg=GRAY, font=(MONO, 8))
        self.lbl_folder.pack(side=tk.LEFT, padx=10)

        # Param sliders
        pr = tk.Frame(tb, bg=PANEL)
        pr.pack(side=tk.RIGHT, padx=12)

        def _make_param(parent, lbl, key, lo, hi, res, default):
            f = tk.Frame(parent, bg=PANEL)
            f.pack(side=tk.LEFT, padx=5)
            tk.Label(f, text=lbl, bg=PANEL, fg=GR2,
                     font=(MONO, 7)).pack()
            var = tk.DoubleVar(value=default)
            tk.Scale(f, from_=lo, to=hi, resolution=res,
                     orient=tk.HORIZONTAL, variable=var,
                     bg=PANEL, fg=WHT, troughcolor=CARD,
                     highlightthickness=0, length=80,
                     font=(MONO, 7), showvalue=True).pack()
            self._cfg_vars[key] = var

        _make_param(pr, "padding(s)", "padding",      0.05, 1.0, 0.05,
                    self.cfg.get("padding", 0.25))
        _make_param(pr, "min_dur(s)", "min_dur",      0.1,  3.0, 0.1,
                    self.cfg.get("min_dur", 0.4))
        _make_param(pr, "idle(s)",    "idle",         0.2,  3.0, 0.1,
                    self.cfg.get("idle_sec", 0.7))
        _make_param(pr, "waist_off",  "waist_offset", -0.3, 0.3, 0.01,
                    self.cfg.get("waist_offset", 0.03))

        # Toggle filters
        tg = tk.Frame(tb, bg=PANEL)
        tg.pack(side=tk.RIGHT, padx=4)

        def _make_toggle(parent, text_on, text_off, var, hint):
            bh = [None]

            def _toggle():
                var.set(not var.get())
                _ref()

            def _ref():
                on = var.get()
                bh[0].config(
                    text=f"✔ {text_on}" if on else f"✗ {text_off}",
                    bg=GRN if on else "#374151", fg="white")

            btn = tk.Button(parent, text="", font=(MONO, 8, "bold"),
                            relief=tk.FLAT, cursor="hand2",
                            padx=8, pady=5, command=_toggle)
            btn.pack(pady=2)
            bh[0] = btn
            tk.Label(parent, text=hint, bg=PANEL, fg=GRAY,
                     font=(MONO, 6)).pack()
            _ref()

        tk.Label(tg, text="Bo loc", bg=PANEL, fg=GR2,
                 font=(MONO, 7, "bold")).pack()
        _make_toggle(tg, "Tay > eo",    "Tay > eo OFF",
                     self._use_hip,    "cat khi tay xuong duoi eo")
        _make_toggle(tg, "Finger curl", "Finger OFF",
                     self._use_finger, "ngon tay dang co/ky hieu")

        # Main pane
        pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                              bg=BG, sashwidth=5, sashrelief=tk.FLAT)
        pane.pack(fill=tk.BOTH, expand=True)

        # ─── Left panel ───────────────────────────────────────────────
        left = tk.Frame(pane, bg=BG2, width=290)
        pane.add(left, minsize=220)

        tk.Label(left, text="VIDEO TRONG FOLDER",
                 bg=BG2, fg=GRAY,
                 font=(MONO, 8, "bold")).pack(pady=(8, 2), padx=8, anchor="w")

        vf = tk.Frame(left, bg=CARD)
        vf.pack(fill=tk.X, padx=6, pady=(0, 2))
        vsb = tk.Scrollbar(vf)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.video_lb = tk.Listbox(vf, yscrollcommand=vsb.set,
                                   bg=CARD, fg=WHT, selectbackground=ACC,
                                   font=(MONO, 9), height=6,
                                   relief=tk.FLAT, bd=0,
                                   highlightthickness=0, activestyle="none")
        self.video_lb.pack(fill=tk.BOTH, expand=True)
        vsb.config(command=self.video_lb.yview)

        tk.Frame(left, bg=PANEL, height=1).pack(fill=tk.X, padx=6, pady=3)

        self.lbl_summary = tk.Label(left, text="CLIPS PHAT HIEN",
                                    bg=BG2, fg=GRAY,
                                    font=(MONO, 8, "bold"))
        self.lbl_summary.pack(pady=(2, 2), padx=8, anchor="w")

        cf = tk.Frame(left, bg=CARD)
        cf.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 4))
        csb = tk.Scrollbar(cf)
        csb.pack(side=tk.RIGHT, fill=tk.Y)
        self.clip_lb = tk.Listbox(cf, yscrollcommand=csb.set,
                                  bg=CARD, fg=WHT, selectbackground=ACC,
                                  font=(MONO, 9),
                                  relief=tk.FLAT, bd=0,
                                  highlightthickness=0, activestyle="none")
        self.clip_lb.pack(fill=tk.BOTH, expand=True)
        csb.config(command=self.clip_lb.yview)
        self.clip_lb.bind("<<ListboxSelect>>", self._on_clip_select)

        # ─── Right panel: preview ─────────────────────────────────────
        right = tk.Frame(pane, bg=BG)
        pane.add(right, minsize=600)

        self.canvas = tk.Canvas(right, width=PREVIEW_W, height=PREVIEW_H,
                                bg="#000", highlightthickness=2,
                                highlightbackground=ACC)
        self.canvas.pack(pady=(10, 0), padx=10)
        self._show_placeholder()

        pbr = tk.Frame(right, bg=BG)
        pbr.pack(pady=4)
        pbkw = dict(font=(MONO, 10, "bold"), relief=tk.FLAT,
                    cursor="hand2", padx=8, pady=4)
        self.btn_play = tk.Button(pbr, text="Play", bg=ACC, fg=WHT,
                                  command=self._toggle_play, **pbkw)
        self.btn_play.pack(side=tk.LEFT, padx=4)
        for lbl, fn in [("<<", self._goto_start),
                        ("<",  lambda: self._step(-1)),
                        (">",  lambda: self._step(1)),
                        (">>", self._goto_end)]:
            tk.Button(pbr, text=lbl, bg="#374151", fg=WHT,
                      command=fn, **pbkw).pack(side=tk.LEFT, padx=2)

        self.scrub_var = tk.IntVar()
        self.scrub = tk.Scale(right, from_=0, to=100,
                              orient=tk.HORIZONTAL, variable=self.scrub_var,
                              bg=BG, fg=WHT, troughcolor=CARD,
                              highlightthickness=0, sliderrelief=tk.FLAT,
                              command=self._on_scrub, length=PREVIEW_W)
        self.scrub.pack(padx=10)
        self.lbl_time = tk.Label(right, text="", bg=BG, fg=GRAY,
                                 font=(MONO, 8))
        self.lbl_time.pack()

        # Trim controls
        tc = tk.Frame(right, bg=PANEL, padx=10, pady=6)
        tc.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(tc, text="Chinh diem cat:", bg=PANEL, fg=WHT,
                 font=(MONO, 9, "bold")).grid(row=0, column=0,
                                             columnspan=8, sticky="w")

        def _row(row, label, fg_col, entry_var, adj_fn, here_fn):
            tk.Label(tc, text=label, bg=PANEL, fg=fg_col,
                     font=(MONO, 8)).grid(row=row, column=0, padx=4, pady=2)
            tk.Entry(tc, textvariable=entry_var, width=7,
                     bg=CARD, fg=WHT, insertbackground=WHT,
                     font=(MONO, 9), relief=tk.FLAT).grid(
                row=row, column=1, padx=2)
            for ci, (t, d) in enumerate([("-10",-10),("-1",-1),
                                          ("+1",1),("+10",10)]):
                tk.Button(tc, text=t, bg=CARD, fg=WHT,
                          font=(MONO, 7), relief=tk.FLAT, padx=4,
                          command=lambda dd=d: adj_fn(dd)).grid(
                    row=row, column=2+ci, padx=1)
            tk.Button(tc, text="Frame nay", bg=CARD, fg=WHT,
                      font=(MONO, 7), relief=tk.FLAT, padx=6,
                      command=here_fn).grid(row=row, column=6, padx=6)

        self.start_var = tk.IntVar()
        self.end_var   = tk.IntVar()
        _row(1, "Start:", GRN2, self.start_var, self._adj_start, self._set_start_here)
        _row(2, "End:  ", RED2, self.end_var,   self._adj_end,   self._set_end_here)

        # Keep / Skip / Nav
        ar = tk.Frame(right, bg=BG)
        ar.pack(pady=4)
        akw = dict(font=(MONO, 11, "bold"), relief=tk.FLAT,
                   cursor="hand2", padx=12, pady=6)
        self.btn_keep = tk.Button(ar, text="GIU", bg=GRN, fg="white",
                                  command=self._keep, **akw)
        self.btn_keep.pack(side=tk.LEFT, padx=6)
        self.btn_skip = tk.Button(ar, text="BO", bg=RED, fg="white",
                                  command=self._skip, **akw)
        self.btn_skip.pack(side=tk.LEFT, padx=6)
        tk.Button(ar, text="< Truoc", bg="#374151", fg=WHT,
                  command=self._prev_clip, **akw).pack(side=tk.LEFT, padx=4)
        self.lbl_nav = tk.Label(ar, text="--", bg=BG, fg=WHT,
                                font=(MONO, 10, "bold"), width=16)
        self.lbl_nav.pack(side=tk.LEFT)
        tk.Button(ar, text="Tiep >", bg="#374151", fg=WHT,
                  command=self._next_clip, **akw).pack(side=tk.LEFT, padx=4)

        # Status bar + Save button
        sb2 = tk.Frame(self.root, bg=PANEL, pady=8)
        sb2.pack(fill=tk.X, side=tk.BOTTOM)
        self.lbl_status = tk.Label(
            sb2,
            text="1) Mo Folder   2) Phan tich TAT CA   3) Duyet clip   4) LUU",
            bg=PANEL, fg=GR2, font=(MONO, 8))
        self.lbl_status.pack(side=tk.LEFT, padx=16)

        tk.Button(sb2, text="LUU  -  Chon nhan & Split",
                  bg=GRN, fg="white",
                  font=(MONO, 11, "bold"), relief=tk.FLAT,
                  cursor="hand2", padx=16, pady=6,
                  command=self._save_dialog).pack(side=tk.RIGHT, padx=12)

        # Keyboard shortcuts
        self.root.bind("<space>",  lambda e: self._toggle_play())
        self.root.bind("<Left>",   lambda e: self._step(-1))
        self.root.bind("<Right>",  lambda e: self._step(1))
        self.root.bind("<k>",      lambda e: self._keep())
        self.root.bind("<d>",      lambda e: self._skip())
        self.root.bind("<n>",      lambda e: self._next_clip())
        self.root.bind("<p>",      lambda e: self._prev_clip())
        self.root.bind("<Return>", lambda e: self._save_dialog())

    # ─── Folder / file management ─────────────────────────────────────

    def _open_folder(self):
        path = filedialog.askdirectory(title="Chon folder chua video")
        if not path:
            return
        self.folder_path = Path(path)
        short = str(self.folder_path)
        self.lbl_folder.config(text=short[-65:] if len(short) > 65 else short)
        self._refresh_list()

    def _refresh_list(self):
        if not self.folder_path:
            messagebox.showinfo("Chu y", "Hay chon folder truoc!")
            return
        self.video_files = sorted([
            f for f in self.folder_path.iterdir()
            if f.is_file() and f.suffix in VIDEO_EXTS
        ])
        self.video_lb.delete(0, tk.END)
        for f in self.video_files:
            self.video_lb.insert(tk.END, f"  {f.name}")
        self._set_status(
            f"Tim thay {len(self.video_files)} video. "
            "Nhan  '▶ Phan tich TAT CA'  de bat dau.")

    # ─── Batch analysis ───────────────────────────────────────────────

    def _analyse_all(self):
        """Phan tich toan bo video trong folder, khong giu frames trong RAM."""
        if not self.video_files:
            messagebox.showinfo("Chu y",
                "Chua co video nao. Hay Mo Folder truoc!")
            return

        self._stop_play()
        self.all_results = []
        self.flat_clips  = []
        self.flat_idx    = -1
        self.cur_vid_idx = -1
        self.frames      = []
        self.clip_lb.delete(0, tk.END)
        self._show_placeholder()

        params = dict(
            padding_sec  = self._cfg_vars["padding"].get(),
            min_dur      = self._cfg_vars["min_dur"].get(),
            idle_sec     = self._cfg_vars["idle"].get(),
            waist_offset = self._cfg_vars["waist_offset"].get(),
            use_hip      = self._use_hip.get(),
            use_finger   = self._use_finger.get(),
        )

        n_vids      = len(self.video_files)
        result_list = []

        prog = ProgressWin(self.root, "Phan tich toan bo folder...")

        def worker():
            for vi, path in enumerate(self.video_files):
                try:
                    prog.set_title(
                        f"[{vi+1}/{n_vids}]  {path.name}")
                    prog.update("Doc frames...")

                    cap   = cv2.VideoCapture(str(path))
                    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frames = []
                    fi = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(frame)
                        if fi % 60 == 0:
                            prog.update(
                                f"Doc frame {fi}/{total}...")
                        fi += 1
                    cap.release()

                    clips = analyze_frames(frames, fps,
                                           progress_cb=prog.update,
                                           **params)
                    result_list.append({
                        "path":  path,
                        "fps":   fps,
                        "clips": clips,
                        "kept":  [True] * len(clips),
                    })
                    prog.update(
                        f"OK: {len(clips)} clip phat hien")

                except Exception as ex:
                    prog.update(f"LOI: {ex}")
                    result_list.append({
                        "path": path, "fps": 30.0,
                        "clips": [], "kept": [],
                        "error": str(ex),
                    })

            self.root.after(0, prog.close)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while t.is_alive():
            try:
                self.root.update()
            except Exception:
                break
        t.join()

        self.all_results = result_list
        self._rebuild_flat()

        total_clips = len(self.flat_clips)
        n_ok = sum(1 for r in self.all_results if not r.get("error"))
        self._set_status(
            f"Hoan tat {n_ok}/{n_vids} video  |  "
            f"{total_clips} clip phat hien  |  "
            "Duyet xong nhan LUU")

        if total_clips:
            self._load_flat_clip(0)
        else:
            messagebox.showinfo(
                "Khong tim thay clip",
                "Khong phat hien ky hieu nao trong toan bo folder.\n\n"
                "Thu giam tham so  waist_off  hoac  min_dur.\n"
                "Hoac tat bo loc  'Tay > eo'  neu khong detect duoc hong.")

    # ─── Flat list helpers ────────────────────────────────────────────

    def _rebuild_flat(self):
        """Xay dung lai flat_clips va toan bo clip listbox."""
        self.flat_clips = []
        self.clip_lb.delete(0, tk.END)

        for vi, res in enumerate(self.all_results):
            vname = res["path"].stem
            short = (vname[:18] + "..") if len(vname) > 20 else vname
            for ci, (s, e) in enumerate(res["clips"]):
                dur  = (e - s) / res["fps"]
                mark = "GIU" if res["kept"][ci] else "BO "
                self.flat_clips.append((vi, ci))
                self.clip_lb.insert(
                    tk.END,
                    f"  {short} #{ci+1:02d}  {dur:.1f}s  {mark}")

        n_kept = sum(r["kept"].count(True) for r in self.all_results)
        self.lbl_summary.config(
            text=f"CLIPS  ({n_kept}/{len(self.flat_clips)} GIU)")

    def _update_flat_item(self, fi):
        """Cap nhat mot dong trong clip listbox."""
        if fi < 0 or fi >= len(self.flat_clips):
            return
        vi, ci = self.flat_clips[fi]
        res    = self.all_results[vi]
        vname  = res["path"].stem
        short  = (vname[:18] + "..") if len(vname) > 20 else vname
        s, e   = res["clips"][ci]
        dur    = (e - s) / res["fps"]
        mark   = "GIU" if res["kept"][ci] else "BO "
        self.clip_lb.delete(fi)
        self.clip_lb.insert(
            fi, f"  {short} #{ci+1:02d}  {dur:.1f}s  {mark}")
        self.clip_lb.itemconfig(fi, bg=ACC if fi == self.flat_idx else CARD)

        n_kept = sum(r["kept"].count(True) for r in self.all_results)
        self.lbl_summary.config(
            text=f"CLIPS  ({n_kept}/{len(self.flat_clips)} GIU)")

    # ─── Clip navigation ──────────────────────────────────────────────

    def _on_clip_select(self, event=None):
        sel = self.clip_lb.curselection()
        if sel:
            self._load_flat_clip(sel[0])

    def _load_flat_clip(self, fi):
        if not self.flat_clips or fi < 0 or fi >= len(self.flat_clips):
            return
        self._stop_play()

        vi, ci = self.flat_clips[fi]
        res    = self.all_results[vi]

        # Load video frames only when switching to a different video
        if vi != self.cur_vid_idx:
            self._load_video_frames(vi)

        self.flat_idx = fi
        self.fps      = res["fps"]

        s, e = res["clips"][ci]
        self.play_pos = s
        self.start_var.set(s)
        self.end_var.set(e)
        self.scrub.configure(from_=s, to=e)
        self.scrub_var.set(s)

        n_kept = sum(r["kept"].count(True) for r in self.all_results)
        self.lbl_nav.config(text=f"Clip {fi+1}/{len(self.flat_clips)}")
        self._set_status(
            f"{res['path'].name}  clip#{ci+1}  "
            f"{s}->{e}  ({(e-s)/res['fps']:.2f}s)  |  "
            f"{n_kept}/{len(self.flat_clips)} da chon")

        self._update_btn_style()
        self._show_frame(s)

        # Highlight in clip listbox
        prev = self.flat_idx  # already updated above
        self.clip_lb.selection_clear(0, tk.END)
        self.clip_lb.selection_set(fi)
        self.clip_lb.see(fi)
        for i in range(max(0, fi - 1), min(len(self.flat_clips), fi + 2)):
            self.clip_lb.itemconfig(i, bg=ACC if i == fi else CARD)

        # Highlight in video listbox
        try:
            self.video_lb.selection_clear(0, tk.END)
            self.video_lb.selection_set(vi)
            self.video_lb.see(vi)
        except Exception:
            pass

    def _load_video_frames(self, vi):
        """Read all frames of video[vi] into self.frames (replaces previous)."""
        self.frames = []
        path = self.all_results[vi]["path"]
        self._set_status(f"Dang tai: {path.name}...")
        try:
            cap = cv2.VideoCapture(str(path))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self.frames.append(frame)
            cap.release()
        except Exception as ex:
            messagebox.showerror("Loi tai video", str(ex))
        self.cur_vid_idx = vi

    # ─── Playback ─────────────────────────────────────────────────────

    def _cur_clip(self):
        if self.flat_idx < 0 or not self.flat_clips:
            return None
        vi, ci = self.flat_clips[self.flat_idx]
        return self.all_results[vi]["clips"][ci]

    def _toggle_play(self):
        if self.is_playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        if not self.frames or self._cur_clip() is None:
            return
        self.is_playing = True
        self.btn_play.config(text="Pause")
        self._play_loop()

    def _stop_play(self):
        self.is_playing = False
        self.btn_play.config(text="Play")
        if self._play_job:
            self.root.after_cancel(self._play_job)
            self._play_job = None

    def _play_loop(self):
        if not self.is_playing:
            return
        cl = self._cur_clip()
        if cl is None:
            return
        s, e = cl
        nxt  = self.play_pos + 1
        if nxt > e:
            nxt = s
        self._show_frame(nxt)
        self._play_job = self.root.after(max(1, int(1000 / self.fps)),
                                         self._play_loop)

    def _goto_start(self):
        self._stop_play()
        cl = self._cur_clip()
        if cl:
            self._show_frame(cl[0])

    def _goto_end(self):
        self._stop_play()
        cl = self._cur_clip()
        if cl:
            self._show_frame(cl[1])

    def _step(self, d):
        self._stop_play()
        if self._cur_clip():
            self._show_frame(self.play_pos + d)

    def _on_scrub(self, val):
        self._stop_play()
        if self._cur_clip():
            self._show_frame(int(float(val)))

    # ─── Frame rendering ──────────────────────────────────────────────

    def _show_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, PREVIEW_W, PREVIEW_H,
                                     fill="#080810", outline="")
        self.canvas.create_text(
            PREVIEW_W // 2, PREVIEW_H // 2,
            text="Nhan  '▶ Phan tich TAT CA'  de bat dau",
            fill=GRAY, font=(MONO, 13))

    def _show_frame(self, fidx):
        if not self.frames:
            return
        cl = self._cur_clip()
        if cl is None:
            return
        s, e  = cl
        fidx  = max(0, min(fidx, len(self.frames) - 1))
        self.play_pos = fidx
        self.scrub_var.set(fidx)

        bgr    = self.frames[fidx]
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w   = rgb.shape[:2]
        sc     = min(PREVIEW_W / w, PREVIEW_H / h)
        nw, nh = int(w * sc), int(h * sc)
        img    = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)
        y0     = (PREVIEW_H - nh) // 2
        x0     = (PREVIEW_W - nw) // 2
        img[y0:y0+nh, x0:x0+nw] = cv2.resize(rgb, (nw, nh))

        cv2.putText(img,
                    f"Frame {fidx}  +{(fidx-s)/self.fps:.2f}s"
                    f"  / {(e-s)/self.fps:.2f}s",
                    (8, 24), cv2.FONT_HERSHEY_DUPLEX, 0.52,
                    (200, 200, 255), 1, cv2.LINE_AA)

        if self.flat_idx >= 0 and self.flat_clips:
            vi, ci = self.flat_clips[self.flat_idx]
            vname  = self.all_results[vi]["path"].name
            cv2.putText(img, vname, (8, PREVIEW_H - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.42,
                        (180, 180, 180), 1, cv2.LINE_AA)

        kept = False
        if self.flat_idx >= 0 and self.flat_clips:
            vi, ci = self.flat_clips[self.flat_idx]
            kept   = self.all_results[vi]["kept"][ci]
        bdr = (16, 185, 129) if kept else (239, 68, 68)
        cv2.rectangle(img, (0, 0), (PREVIEW_W-1, PREVIEW_H-1), bdr, 3)

        pil   = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas._img = imgtk

        self.lbl_time.config(
            text=f"Frame {fidx}/{e}    "
                 f"({fidx/self.fps:.2f}s / {e/self.fps:.2f}s)    "
                 f"Do dai: {(e-s)/self.fps:.2f}s")

    # ─── Trim ─────────────────────────────────────────────────────────

    def _apply_trim(self):
        if self.flat_idx < 0:
            return
        vi, ci = self.flat_clips[self.flat_idx]
        res    = self.all_results[vi]
        n      = len(self.frames)
        s = max(0, min(int(self.start_var.get()), n - 2))
        e = max(s + 1, min(int(self.end_var.get()), n - 1))
        res["clips"][ci] = [s, e]
        self.start_var.set(s)
        self.end_var.set(e)
        self.scrub.configure(from_=s, to=e)
        self._show_frame(max(s, min(self.play_pos, e)))
        self._update_flat_item(self.flat_idx)

    def _adj_start(self, d):
        if self.flat_idx < 0:
            return
        vi, ci = self.flat_clips[self.flat_idx]
        s, _   = self.all_results[vi]["clips"][ci]
        self.start_var.set(max(0, s + d))
        self._apply_trim()

    def _adj_end(self, d):
        if self.flat_idx < 0:
            return
        vi, ci = self.flat_clips[self.flat_idx]
        _, e   = self.all_results[vi]["clips"][ci]
        self.end_var.set(min(len(self.frames) - 1, e + d))
        self._apply_trim()

    def _set_start_here(self):
        if self.flat_idx >= 0:
            self.start_var.set(self.play_pos)
            self._apply_trim()

    def _set_end_here(self):
        if self.flat_idx >= 0:
            self.end_var.set(self.play_pos)
            self._apply_trim()

    # ─── Keep / Skip ──────────────────────────────────────────────────

    def _update_btn_style(self):
        if self.flat_idx < 0 or not self.flat_clips:
            return
        vi, ci = self.flat_clips[self.flat_idx]
        kept   = self.all_results[vi]["kept"][ci]
        self.btn_keep.config(bg=GRN       if kept else "#374151")
        self.btn_skip.config(bg="#374151" if kept else RED)

    def _keep(self):
        if self.flat_idx < 0:
            return
        vi, ci = self.flat_clips[self.flat_idx]
        self.all_results[vi]["kept"][ci] = True
        self._update_btn_style()
        self._update_flat_item(self.flat_idx)
        if self.flat_idx < len(self.flat_clips) - 1:
            self.root.after(180, lambda: self._load_flat_clip(self.flat_idx + 1))

    def _skip(self):
        if self.flat_idx < 0:
            return
        vi, ci = self.flat_clips[self.flat_idx]
        self.all_results[vi]["kept"][ci] = False
        self._update_btn_style()
        self._update_flat_item(self.flat_idx)
        if self.flat_idx < len(self.flat_clips) - 1:
            self.root.after(180, lambda: self._load_flat_clip(self.flat_idx + 1))

    def _prev_clip(self):
        self._stop_play()
        if self.flat_idx > 0:
            self._load_flat_clip(self.flat_idx - 1)

    def _next_clip(self):
        self._stop_play()
        if self.flat_idx < len(self.flat_clips) - 1:
            self._load_flat_clip(self.flat_idx + 1)

    # ─── Save ─────────────────────────────────────────────────────────

    def _save_dialog(self):
        if not self.all_results:
            messagebox.showinfo("Chu y",
                "Chua phan tich video nao!\n"
                "Nhan  '▶ Phan tich TAT CA'  truoc.")
            return

        total_kept = sum(r["kept"].count(True) for r in self.all_results)
        if total_kept == 0:
            messagebox.showwarning("Chu y",
                "Khong co clip nao duoc GIU!\n"
                "Nhan GIU (phim K) tren it nhat 1 clip.")
            return

        dlg = LabelDialog(self.root, self.data_dir, total_kept)
        if dlg.result is None:
            return

        label, split_mode = dlg.result
        self._do_save_all(label, split_mode)

    def _do_save_all(self, label, split_mode):
        """
        Luu toan bo clip GIU tu moi video.
        Video duoc doc lai tung cai mot khi luu (khong giu tat ca frames).
        """
        self._stop_play()

        # Build ordered flat list of kept clips
        to_save = []
        for vi, res in enumerate(self.all_results):
            for ci, ((s, e), kept) in enumerate(
                    zip(res["clips"], res["kept"])):
                if kept:
                    to_save.append((vi, ci, s, e))

        n = len(to_save)
        if n == 0:
            return

        # Assign train/val/test splits
        if split_mode == "auto":
            indices = list(range(n))
            random.shuffle(indices)
            n_train = max(1, round(n * SPLIT_RATIO["train"]))
            n_val   = max(0, round(n * SPLIT_RATIO["val"]))
            n_test  = n - n_train - n_val
            if n_test < 0:
                n_val += n_test
                n_test = 0
            split_assign = {}
            for rank, oi in enumerate(indices):
                if rank < n_train:
                    split_assign[rank] = "train"
                elif rank < n_train + n_val:
                    split_assign[rank] = "val"
                else:
                    split_assign[rank] = "test"
        else:
            split_assign = {i: split_mode for i in range(n)}

        for sp in SPLITS:
            (self.data_dir / sp / label).mkdir(parents=True, exist_ok=True)

        start_idx = next_clip_index(self.data_dir, label)
        fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
        saved     = []

        prog = ProgressWin(self.root, "Dang luu clips...")

        def worker():
            cur_vi     = -1
            vid_frames = []
            vid_fps    = 30.0
            vid_w = vid_h = 0

            for rank, (vi, ci, s, e) in enumerate(to_save):
                # Re-read video only when switching to a new one
                if vi != cur_vi:
                    path = self.all_results[vi]["path"]
                    prog.set_title(f"Tai video: {path.name}")
                    prog.update("Doc frames...")
                    vid_frames = []
                    cap = cv2.VideoCapture(str(path))
                    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        vid_frames.append(frame)
                    cap.release()
                    if vid_frames:
                        vid_h, vid_w = vid_frames[0].shape[:2]
                    cur_vi = vi

                sp       = split_assign[rank]
                file_idx = start_idx + rank
                name     = f"{label}_{file_idx:04d}.mp4"
                out_path = self.data_dir / sp / label / name

                prog.set_title(f"Luu {rank+1}/{n}: {name}")
                prog.update(f"-> {sp}/")
                vw = cv2.VideoWriter(
                    str(out_path), fourcc, vid_fps, (vid_w, vid_h))
                for fi in range(s, min(e + 1, len(vid_frames))):
                    vw.write(vid_frames[fi])
                vw.release()
                saved.append((sp, name))

            self.root.after(0, prog.close)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while t.is_alive():
            try:
                self.root.update()
            except Exception:
                break
        t.join()

        by_split = {}
        for sp, _ in saved:
            by_split[sp] = by_split.get(sp, 0) + 1

        msg  = f"Da luu {len(saved)} clip\n\n"
        msg += f"  Nhan   : {label}\n"
        msg += f"  Thu muc: {self.data_dir}\n\n"
        for sp in SPLITS:
            if sp in by_split:
                msg += f"  {sp:5s}: {by_split[sp]} clip\n"

        messagebox.showinfo("Luu thanh cong!", msg)
        self._set_status(
            f"Da luu {len(saved)} clip -> [{label}]  "
            + "  ".join(f"{sp}:{c}" for sp, c in by_split.items()))

        # Mark saved clips as BO and refresh list
        for vi, ci, s, e in to_save:
            self.all_results[vi]["kept"][ci] = False
        self._rebuild_flat()

    # ─── Misc ─────────────────────────────────────────────────────────

    def _set_status(self, msg):
        try:
            self.lbl_status.config(text=msg)
            self.root.update_idletasks()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="VSL Auto Cut - phan tich ca folder roi luu 1 lan")
    ap.add_argument("--data_dir",     default=None)
    ap.add_argument("--padding",      type=float, default=0.25)
    ap.add_argument("--min_dur",      type=float, default=0.4)
    ap.add_argument("--idle_sec",     type=float, default=0.7)
    ap.add_argument("--waist_offset", type=float, default=0.03,
                    help="Bien do tinh chinh eo: + chat hon, - rong hon")
    args = ap.parse_args()

    if not HAS_TK:
        print("[ERROR] Can cai: pip install pillow")
        sys.exit(1)
    if not HAS_MP:
        print("[ERROR] Can cai: pip install mediapipe")
        sys.exit(1)

    data_dir = (Path(args.data_dir) if args.data_dir
                else _PROJECT_ROOT / "data" / "videos")
    data_dir.mkdir(parents=True, exist_ok=True)
    for sp in SPLITS:
        (data_dir / sp).mkdir(exist_ok=True)

    print(f"  Script dir  : {_SCRIPT_DIR}")
    print(f"  Project root: {_PROJECT_ROOT}")
    print(f"  Data dir    : {data_dir}")
    labels = get_existing_labels(data_dir)
    print(f"  Nhan hien co: {labels if labels else '(chua co)'}")

    cfg = dict(padding=args.padding, min_dur=args.min_dur,
               idle_sec=args.idle_sec, waist_offset=args.waist_offset)
    root = tk.Tk()
    App(root, data_dir, cfg)
    root.mainloop()


if __name__ == "__main__":
    main()