import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import os
import sys
import io
import threading
import numpy as np
import glob
import tempfile
import shutil
import subprocess

try:
    import windnd
    HAS_WINDND = True
except ImportError:
    HAS_WINDND = False


def _get_app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _find_model_dir():
    app_dir = _get_app_dir()
    candidates = [
        os.path.join(app_dir, "model"),
    ]
    parent = os.path.dirname(app_dir)
    if parent and parent != app_dir:
        candidates.append(os.path.join(parent, "model"))
        candidates.append(os.path.join(parent, "PictureCut", "model"))
    if not getattr(sys, 'frozen', False):
        candidates.append(os.path.join(app_dir, "dist", "PictureCut", "model"))
    for d in candidates:
        if os.path.isfile(os.path.join(d, "rembg", "__init__.py")):
            return d
    return None


_FOUND_MODEL_DIR = _find_model_dir()
_MODEL_DIR = _FOUND_MODEL_DIR if _FOUND_MODEL_DIR else os.path.join(_get_app_dir(), "model")
_U2NET_HOME = os.path.join(_MODEL_DIR, ".u2net")

HAS_REMBG = False
_REMBG_IMPORT_ERROR = ""
_rembg_checked = False


def _check_model_dir():
    return _FOUND_MODEL_DIR is not None


def _get_missing_models():
    missing = []
    for mid, mlabel, mdesc in AI_MODELS:
        if not _check_model_cached(mid):
            missing.append((mid, mlabel))
    return missing


def _try_import_rembg():
    global HAS_REMBG, _REMBG_IMPORT_ERROR, _rembg_checked
    if _rembg_checked:
        return HAS_REMBG
    _rembg_checked = True
    HAS_REMBG = False
    _REMBG_IMPORT_ERROR = ""

    if not _check_model_dir():
        _REMBG_IMPORT_ERROR = f"未找到 model 文件夹，期望路径: PictureCut\\model"
        return False

    if getattr(sys, 'frozen', False):
        ort_dir = os.path.join(_MODEL_DIR, "onnxruntime")
        if not os.path.isdir(ort_dir):
            _REMBG_IMPORT_ERROR = "model 文件夹中缺少 onnxruntime"
            return False
        HAS_REMBG = True
        return True

    try:
        os.environ["U2NET_HOME"] = _U2NET_HOME
        if _MODEL_DIR not in sys.path:
            sys.path.insert(0, _MODEL_DIR)
        _internal_dir = os.path.join(os.path.dirname(_MODEL_DIR), "_internal")
        if os.path.isdir(_internal_dir) and _internal_dir not in sys.path:
            sys.path.insert(0, _internal_dir)
        ort_capi = os.path.join(_MODEL_DIR, "onnxruntime", "capi")
        if os.path.isdir(ort_capi):
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(ort_capi)
                except OSError:
                    pass
            os.environ["PATH"] = ort_capi + os.pathsep + os.environ.get("PATH", "")
        import importlib.util
        rembg_init = os.path.join(_MODEL_DIR, "rembg", "__init__.py")
        spec = importlib.util.spec_from_file_location(
            "rembg", rembg_init,
            submodule_search_locations=[os.path.join(_MODEL_DIR, "rembg")]
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        global _rembg_remove_fn, _rembg_new_session_fn
        _rembg_remove_fn = mod.remove
        _rembg_new_session_fn = mod.new_session
    except Exception as e:
        _REMBG_IMPORT_ERROR = str(e)[:200]
        return False

    HAS_REMBG = True
    return HAS_REMBG


if not getattr(sys, 'frozen', False):
    try:
        _try_import_rembg()
    except Exception:
        pass

AI_MODELS = [
    ("birefnet-general", "BiRefNet-General - 高精度通用",
     "复杂背景/精细边缘 | 当前最高精度 | 速度较慢，适合最终出图"),
    ("birefnet-general-lite", "BiRefNet-Lite - 轻量通用",
     "通用场景 | 精度与速度平衡 | 比BiRefNet-General快"),
    ("birefnet-portrait", "BiRefNet-Portrait - 人像专用",
     "人物/角色/半身像 | 人像边缘精细 | 不适合物品/场景"),
]

def _check_model_cached(model_name):
    model_onnx_dir = os.path.join(_MODEL_DIR, ".u2net")
    if os.path.isdir(model_onnx_dir):
        pattern = os.path.join(model_onnx_dir, f"*{model_name}*")
        matches = glob.glob(pattern)
        for m in matches:
            if m.lower().endswith('.onnx'):
                return True
    home = os.path.expanduser("~")
    for d in [os.path.join(home, ".u2net"), os.path.join(home, ".rembg")]:
        if os.path.isdir(d):
            pattern = os.path.join(d, f"*{model_name}*")
            matches = glob.glob(pattern)
            for m in matches:
                if m.lower().endswith('.onnx'):
                    return True
    return False


class Selection:
    _counter = 0

    def __init__(self, x1, y1, x2, y2, name=None):
        Selection._counter += 1
        self.id = Selection._counter
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.name = name if name else f"icon_{self.id:03d}"

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2

    def contains(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def hit_handle(self, x, y, margin=6):
        handles = self.get_handles()
        for name, (hx, hy) in handles.items():
            if abs(x - hx) <= margin and abs(y - hy) <= margin:
                return name
        return None

    def get_handles(self):
        mx = (self.x1 + self.x2) / 2
        my = (self.y1 + self.y2) / 2
        return {
            'nw': (self.x1, self.y1), 'n': (mx, self.y1), 'ne': (self.x2, self.y1),
            'e': (self.x2, my), 'se': (self.x2, self.y2), 's': (mx, self.y2),
            'sw': (self.x1, self.y2), 'w': (self.x1, my),
        }

    def snapshot(self):
        return (self.x1, self.y1, self.x2, self.y2, self.name)

    @staticmethod
    def from_snapshot(snap):
        x1, y1, x2, y2, name = snap
        sel = Selection.__new__(Selection)
        sel.id = 0
        sel.x1, sel.y1, sel.x2, sel.y2, sel.name = x1, y1, x2, y2, name
        return sel


class AppleButton(tk.Canvas):
    def __init__(self, parent, text="", command=None, accent=False, width=None, height=32,
                 font_size=12, padding_x=16, **kwargs):
        self._command = command
        self._accent = accent
        self._text = text
        self._height = height
        self._padding_x = padding_x
        self._pressed = False
        self._hovered = False
        self._explicit_width = width

        bg = "#F5F5F7" if not accent else "#007AFF"
        super().__init__(parent, height=height, bg=bg, highlightthickness=0,
                         bd=0, cursor="hand2", **kwargs)

        self._text_color = "#1D1D1F" if not accent else "#FFFFFF"
        self._bg_normal = "#F5F5F7" if not accent else "#007AFF"
        self._bg_hover = "#E8E8ED" if not accent else "#0066D6"
        self._bg_pressed = "#D2D2D7" if not accent else "#0055B3"

        self._font = ("Microsoft YaHei UI", font_size)

        if width:
            self.configure(width=width)
        else:
            text_w = self._measure_text()
            self.configure(width=text_w + 2 * padding_x)

        self._draw()
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Configure>", self._on_configure)

    def _measure_text(self):
        tmp = tk.Label(self, text=self._text, font=self._font)
        w = tmp.winfo_reqwidth()
        tmp.destroy()
        return w

    def _draw(self):
        self.delete("all")
        w = self.winfo_width()
        if w <= 1:
            w = self._measure_text() + 2 * self._padding_x
            if self._explicit_width:
                w = self._explicit_width
        h = self._height

        if self._pressed:
            bg = self._bg_pressed
        elif self._hovered:
            bg = self._bg_hover
        else:
            bg = self._bg_normal

        self.configure(bg=bg)

        r = 8
        self.create_rounded_rect(0, 0, w, h, r, fill=bg, outline="")

        self.create_text(w / 2, h / 2, text=self._text, fill=self._text_color,
                         font=self._font, anchor=tk.CENTER)

    def _on_configure(self, event):
        self._draw()

    def create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1 + r, y1, x2 - r, y1,
            x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2,
            x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r,
            x1, y1 + r, x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _on_enter(self, event):
        self._hovered = True
        self._draw()

    def _on_leave(self, event):
        self._hovered = False
        self._pressed = False
        self._draw()

    def _on_press(self, event):
        self._pressed = True
        self._draw()

    def _on_release(self, event):
        self._pressed = False
        self._draw()
        if self._command:
            self._command()

    def configure_text(self, text):
        self._text = text
        self._draw()

    def configure_state(self, state):
        if state == "disabled":
            self._bg_normal = "#F5F5F7" if not self._accent else "#99CCFF"
            self._text_color = "#C7C7CC" if not self._accent else "#E0E0E0"
            self.configure(cursor="arrow")
        else:
            self._text_color = "#1D1D1F" if not self._accent else "#FFFFFF"
            self._bg_normal = "#F5F5F7" if not self._accent else "#007AFF"
            self.configure(cursor="hand2")
        self._draw()


class AppleSection(tk.Frame):
    def __init__(self, parent, title="", **kwargs):
        super().__init__(parent, bg="#FFFFFF", bd=0, highlightthickness=1,
                         highlightbackground="#E5E5EA", highlightcolor="#E5E5EA",
                         padx=16, pady=12, **kwargs)
        self._title = title
        self._header = None
        if title:
            self._header = tk.Frame(self, bg="#FFFFFF")
            self._header.pack(fill=tk.X, pady=(0, 8))
            tk.Label(self._header, text=title, font=("Microsoft YaHei UI", 11, "bold"),
                     fg="#1D1D1F", bg="#FFFFFF").pack(side=tk.LEFT)

    def add_separator(self):
        sep = tk.Frame(self, bg="#E5E5EA", height=1)
        sep.pack(fill=tk.X, pady=8)
        return sep


class PictureCutApp:
    HANDLE_SIZE = 5
    MIN_SELECTION = 5
    MAX_HISTORY = 50

    BG = "#F5F5F7"
    BG_WHITE = "#FFFFFF"
    TEXT_PRIMARY = "#1D1D1F"
    TEXT_SECONDARY = "#86868B"
    TEXT_TERTIARY = "#AEAEB2"
    ACCENT = "#007AFF"
    ACCENT_LIGHT = "#E3F2FF"
    BORDER = "#E5E5EA"
    CANVAS_BG = "#1D1D1F"
    SEL_COLOR = "#007AFF"
    SEL_COLOR_ALT = "#FF9500"

    def __init__(self, root):
        self.root = root
        self.root.title("Picture Cut")
        self.root.geometry("1280x800")
        self.root.minsize(1024, 680)
        self.root.configure(bg=self.BG)

        icon_path = os.path.join(_get_app_dir(), "PictureCut.ico")
        if not os.path.isfile(icon_path):
            icon_path = os.path.join(_get_app_dir(), "_internal", "PictureCut.ico")
        if os.path.isfile(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except Exception:
                pass

        self.source_image = None
        self.original_image = None
        self.tk_image = None
        self.selections = []
        self.selected_indices = []

        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        self.drag_state = None
        self.drawing = False
        self.draw_start = None

        self.history = []
        self.history_index = -1
        self._history_lock = False

        self.clipboard = []

        self.grid_visible = tk.BooleanVar(value=True)
        self.grid_size = tk.IntVar(value=32)
        self.snap_to_grid = tk.BooleanVar(value=False)
        self.naming_prefix = tk.StringVar(value="icon")
        self.naming_digits = tk.IntVar(value=3)
        self.input_w = tk.IntVar(value=256)
        self.input_h = tk.IntVar(value=256)
        self.bg_threshold = tk.IntVar(value=240)
        self.ai_model_var = tk.StringVar(value="BiRefNet-Lite - 轻量通用")

        self._ai_processing = False
        self._progress_timer = None
        self._progress_value = 0
        self.status_var = tk.StringVar(value="就绪")

        self.crop_padding_var = tk.IntVar(value=10)
        self.crop_min_area_var = tk.IntVar(value=100)
        self.crop_status_var = tk.StringVar(value="")

        self._configure_styles()
        self._build_ui()
        self._bind_events()
        self._push_history()
        self._animate_entrance()

    def _configure_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.style.configure(".", background=self.BG, foreground=self.TEXT_PRIMARY,
                             font=("Microsoft YaHei UI", 10))
        self.style.configure("TFrame", background=self.BG)
        self.style.configure("White.TFrame", background=self.BG_WHITE)
        self.style.configure("TLabel", background=self.BG, foreground=self.TEXT_PRIMARY,
                             font=("Microsoft YaHei UI", 10))
        self.style.configure("Secondary.TLabel", background=self.BG, foreground=self.TEXT_SECONDARY,
                             font=("Microsoft YaHei UI", 9))
        self.style.configure("Tertiary.TLabel", background=self.BG, foreground=self.TEXT_TERTIARY,
                             font=("Microsoft YaHei UI", 8))
        self.style.configure("Accent.TLabel", background=self.BG, foreground=self.ACCENT,
                             font=("Microsoft YaHei UI", 9))

        self.style.configure("TCheckbutton", background=self.BG, foreground=self.TEXT_PRIMARY,
                             font=("Microsoft YaHei UI", 10), focuscolor=self.BG)
        self.style.map("TCheckbutton",
                        background=[("active", self.BG), ("!active", self.BG)])

        self.style.configure("TCombobox", fieldbackground=self.BG_WHITE,
                             background=self.BG_WHITE, foreground=self.TEXT_PRIMARY,
                             bordercolor=self.BORDER, focuscolor=self.ACCENT,
                             arrowcolor=self.TEXT_SECONDARY, padding=(8, 4))
        self.style.map("TCombobox",
                        fieldbackground=[("readonly", self.BG_WHITE)],
                        selectbackground=[("readonly", self.ACCENT_LIGHT)],
                        selectforeground=[("readonly", self.ACCENT)])

        self.style.configure("TSpinbox", fieldbackground=self.BG_WHITE,
                             background=self.BG_WHITE, foreground=self.TEXT_PRIMARY,
                             bordercolor=self.BORDER, arrowcolor=self.TEXT_SECONDARY,
                             padding=(8, 4))

        self.style.configure("TEntry", fieldbackground=self.BG_WHITE,
                             foreground=self.TEXT_PRIMARY, bordercolor=self.BORDER,
                             padding=(8, 4))

        self.style.configure("Horizontal.TProgressbar", troughcolor=self.BORDER,
                             background=self.ACCENT, bordercolor=self.BORDER,
                             lightcolor=self.ACCENT, darkcolor=self.ACCENT)

        self.style.configure("TScrollbar", background=self.BG, troughcolor=self.BG,
                             bordercolor=self.BG, arrowcolor=self.TEXT_SECONDARY,
                             gripcount=0)
        self.style.map("TScrollbar",
                        background=[("active", "#D1D1D6")])

        self.style.configure("TPanedwindow", background=self.BG)

    def _build_ui(self):
        self._build_toolbar()
        self._build_main_area()
        self._build_status_bar()

    def _build_toolbar(self):
        toolbar = tk.Frame(self.root, bg=self.BG_WHITE, height=44,
                           highlightthickness=0, bd=0)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        toolbar.pack_propagate(False)

        inner = tk.Frame(toolbar, bg=self.BG_WHITE, padx=10)
        inner.pack(fill=tk.BOTH, expand=True)

        tk.Label(inner, text="Picture Cut", font=("Microsoft YaHei UI", 12, "bold"),
                 fg=self.TEXT_PRIMARY, bg=self.BG_WHITE).pack(side=tk.LEFT, padx=(0, 12))

        self._make_toolbar_btn(inner, "上传图片", self._upload_image, accent=True)
        self._make_toolbar_btn(inner, "适应窗口", self._fit_to_window)
        self._make_toolbar_btn(inner, "1:1", self._zoom_original, width=36)

        sep1 = tk.Frame(inner, bg=self.BORDER, width=1, height=18)
        sep1.pack(side=tk.LEFT, padx=8, pady=13)

        tk.Label(inner, text="缩放", font=("Microsoft YaHei UI", 9),
                 fg=self.TEXT_SECONDARY, bg=self.BG_WHITE).pack(side=tk.LEFT, padx=(0, 4))
        self.zoom_var = tk.StringVar(value="100%")
        zoom_entry = tk.Entry(inner, textvariable=self.zoom_var, width=5,
                              font=("Microsoft YaHei UI", 9), bg=self.BG,
                              fg=self.TEXT_PRIMARY, relief=tk.FLAT, bd=0,
                              highlightthickness=1, highlightcolor=self.ACCENT,
                              highlightbackground=self.BORDER, insertbackground=self.TEXT_PRIMARY)
        zoom_entry.pack(side=tk.LEFT, padx=(0, 2), ipady=2)
        zoom_entry.bind("<Return>", self._on_zoom_entry)
        self._make_toolbar_btn(inner, "+", lambda: self._zoom_by(1.25), width=28, font_size=12)
        self._make_toolbar_btn(inner, "−", lambda: self._zoom_by(0.8), width=28, font_size=12)

        sep2 = tk.Frame(inner, bg=self.BORDER, width=1, height=18)
        sep2.pack(side=tk.LEFT, padx=8, pady=13)

        grid_cb = ttk.Checkbutton(inner, text="网格", variable=self.grid_visible,
                                   command=self._redraw_canvas, style="TCheckbutton")
        grid_cb.pack(side=tk.LEFT, padx=(0, 4))

        tk.Label(inner, text="大小", font=("Microsoft YaHei UI", 9),
                 fg=self.TEXT_SECONDARY, bg=self.BG_WHITE).pack(side=tk.LEFT, padx=(0, 2))
        gs = ttk.Spinbox(inner, from_=8, to=256, increment=8, textvariable=self.grid_size,
                          width=3, style="TSpinbox")
        gs.pack(side=tk.LEFT, padx=(0, 4))
        gs.bind("<Return>", lambda e: self._redraw_canvas())

        snap_cb = ttk.Checkbutton(inner, text="吸附", variable=self.snap_to_grid,
                                   style="TCheckbutton")
        snap_cb.pack(side=tk.LEFT, padx=(0, 4))

        sep3 = tk.Frame(inner, bg=self.BORDER, width=1, height=18)
        sep3.pack(side=tk.LEFT, padx=8, pady=13)

        self._make_toolbar_btn(inner, "导出", self._export_all, accent=True)

        bottom_line = tk.Frame(self.root, bg=self.BORDER, height=1)
        bottom_line.pack(side=tk.TOP, fill=tk.X)

    def _make_toolbar_btn(self, parent, text, command, accent=False, width=None, font_size=10):
        btn = AppleButton(parent, text=text, command=command, accent=accent,
                          height=28, font_size=font_size, padding_x=8)
        if width:
            btn.configure(width=width)
        btn.pack(side=tk.LEFT, padx=2, pady=8)
        return btn

    def _build_status_bar(self):
        status_frame = tk.Frame(self.root, bg=self.BG_WHITE, height=28,
                                highlightthickness=0, bd=0)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        status_frame.pack_propagate(False)

        top_line = tk.Frame(self.root, bg=self.BORDER, height=1)
        top_line.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Label(status_frame, textvariable=self.status_var,
                 font=("Microsoft YaHei UI", 9), fg=self.TEXT_TERTIARY,
                 bg=self.BG_WHITE).pack(side=tk.LEFT, padx=20, pady=4)

    def _build_main_area(self):
        main = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=self.BG,
                              sashwidth=1, sashrelief=tk.FLAT, sashpad=0,
                              opaqueresize=True)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=0, pady=0)

        canvas_container = tk.Frame(main, bg=self.CANVAS_BG, bd=0)
        main.add(canvas_container, stretch="always", minsize=400)

        canvas_hint = tk.Frame(canvas_container, bg=self.CANVAS_BG)
        canvas_hint.pack(fill=tk.X, padx=0, pady=0)

        self.canvas = tk.Canvas(canvas_container, bg=self.CANVAS_BG, cursor="crosshair",
                                highlightthickness=0, bd=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        right_outer = tk.Frame(main, bg=self.BG, width=300)
        main.add(right_outer, stretch="never", minsize=280)

        r_canvas = tk.Canvas(right_outer, highlightthickness=0, bg=self.BG, width=280)
        r_scroll = ttk.Scrollbar(right_outer, orient=tk.VERTICAL, command=r_canvas.yview,
                                  style="TScrollbar")
        self.right_inner = tk.Frame(r_canvas, bg=self.BG)

        self.right_inner.bind("<Configure>", lambda e: r_canvas.configure(scrollregion=r_canvas.bbox("all")))
        r_canvas.create_window((0, 0), window=self.right_inner, anchor=tk.NW, width=280)
        r_canvas.configure(yscrollcommand=r_scroll.set)

        r_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        r_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def _on_mousewheel_right(event):
            r_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        r_canvas.bind("<MouseWheel>", _on_mousewheel_right)
        self.right_inner.bind("<MouseWheel>", _on_mousewheel_right)

        self._build_selection_panel(self.right_inner)
        self._build_pixel_input_panel(self.right_inner)
        self._build_bg_remove_panel(self.right_inner)
        self._build_ai_crop_panel(self.right_inner)
        self._build_align_panel(self.right_inner)
        self._build_export_panel(self.right_inner)

    def _make_section(self, parent, title):
        section = AppleSection(parent, title=title)
        section.pack(fill=tk.X, padx=12, pady=(8, 0))
        return section

    def _make_row(self, parent):
        row = tk.Frame(parent, bg=self.BG_WHITE)
        row.pack(fill=tk.X, pady=4)
        return row

    def _make_label(self, parent, text, secondary=False, tertiary=False):
        fg = self.TEXT_SECONDARY if secondary else (self.TEXT_TERTIARY if tertiary else self.TEXT_PRIMARY)
        font_size = 9 if secondary or tertiary else 10
        lbl = tk.Label(parent, text=text, font=("Microsoft YaHei UI", font_size),
                       fg=fg, bg=self.BG_WHITE)
        return lbl

    def _make_btn(self, parent, text, command, accent=False, small=False):
        font_size = 10 if small else 11
        height = 28 if small else 32
        btn = AppleButton(parent, text=text, command=command, accent=accent,
                          height=height, font_size=font_size, padding_x=10)
        return btn

    def _make_input(self, parent, var, width=6):
        entry = tk.Entry(parent, textvariable=var, width=width,
                         font=("Microsoft YaHei UI", 10), bg=self.BG,
                         fg=self.TEXT_PRIMARY, relief=tk.FLAT, bd=0,
                         highlightthickness=1, highlightcolor=self.ACCENT,
                         highlightbackground=self.BORDER, insertbackground=self.TEXT_PRIMARY)
        return entry

    def _make_spinbox(self, parent, var, from_, to, increment=1, width=5):
        spin = ttk.Spinbox(parent, from_=from_, to=to, increment=increment,
                            textvariable=var, width=width, style="TSpinbox")
        return spin

    def _show_tooltip(self, event, text):
        self._hide_tooltip()
        x = event.widget.winfo_rootx() + 20
        y = event.widget.winfo_rooty() + event.widget.winfo_height() + 4
        self._tooltip_win = tw = tk.Toplevel(event.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)
        frame = tk.Frame(tw, bg="#1D1D1F", bd=0, highlightthickness=0)
        frame.pack()
        tk.Label(frame, text=text, font=("Microsoft YaHei UI", 9),
                 fg="#FFFFFF", bg="#1D1D1F", justify=tk.LEFT,
                 padx=8, pady=6).pack()

    def _hide_tooltip(self):
        if hasattr(self, '_tooltip_win') and self._tooltip_win:
            try:
                self._tooltip_win.destroy()
            except tk.TclError:
                pass
            self._tooltip_win = None

    def _build_selection_panel(self, parent):
        section = self._make_section(parent, "选区列表")
        self._sel_section = section

        btn_row = self._make_row(section)
        self._make_btn(btn_row, "删除", self._delete_selected, small=True).pack(side=tk.LEFT, padx=(0, 4))
        self._make_btn(btn_row, "清除全部", self._clear_selections, small=True).pack(side=tk.LEFT)

        list_frame = tk.Frame(section, bg=self.BG_WHITE)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, style="TScrollbar")
        self.sel_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED,
                                       font=("SF Mono", 10) if sys.platform == "darwin" else ("Consolas", 10),
                                       yscrollcommand=scrollbar.set, activestyle="none", height=5,
                                       bg=self.BG, fg=self.TEXT_PRIMARY,
                                       selectbackground=self.ACCENT_LIGHT,
                                       selectforeground=self.ACCENT,
                                       relief=tk.FLAT, bd=0, highlightthickness=1,
                                       highlightcolor=self.ACCENT,
                                       highlightbackground=self.BORDER)
        scrollbar.config(command=self.sel_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sel_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.sel_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        self.sel_info_var = tk.StringVar(value="Ctrl + 点击多选")
        self._make_label(section, "", tertiary=True).pack(fill=tk.X, pady=(4, 0))
        self.sel_info_label = tk.Label(section, textvariable=self.sel_info_var,
                                        font=("Microsoft YaHei UI", 8), fg=self.TEXT_TERTIARY,
                                        bg=self.BG_WHITE)
        self.sel_info_label.pack(fill=tk.X, pady=(2, 0))

    def _build_pixel_input_panel(self, parent):
        section = self._make_section(parent, "像素选区")

        row1 = self._make_row(section)
        self._make_label(row1, "宽").pack(side=tk.LEFT)
        self._make_input(row1, self.input_w, 6).pack(side=tk.LEFT, padx=4)
        self._make_label(row1, "高").pack(side=tk.LEFT, padx=(8, 0))
        self._make_input(row1, self.input_h, 6).pack(side=tk.LEFT, padx=4)

        self._make_btn(section, "在视图中心创建", self._create_pixel_selection_center).pack(fill=tk.X, pady=(6, 0))

    def _build_bg_remove_panel(self, parent):
        self.bg_frame_parent = parent
        section = self._make_section(parent, "去底图")
        self.bg_section = section
        self._build_bg_remove_contents()

    def _build_bg_remove_contents(self):
        for widget in self.bg_section.winfo_children():
            if widget == self.bg_section._header:
                continue
            widget.destroy()

        section = self.bg_section
        model_available = HAS_REMBG or _check_model_dir()

        if model_available:
            self._make_label(section, "AI 抠图模型", secondary=True).pack(anchor=tk.W)
            self.ai_model_combo = ttk.Combobox(section, textvariable=self.ai_model_var,
                                                values=[m[1] for m in AI_MODELS],
                                                state="readonly", width=28, style="TCombobox")
            self.ai_model_combo.pack(fill=tk.X, pady=(4, 0))
            self.ai_model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

            self.model_desc_var = tk.StringVar(value=AI_MODELS[0][2])
            self.model_desc_label = tk.Label(section, textvariable=self.model_desc_var,
                                              font=("Microsoft YaHei UI", 8), fg=self.TEXT_TERTIARY,
                                              bg=self.BG_WHITE, wraplength=240, justify=tk.LEFT)
            self.model_desc_label.pack(anchor=tk.W, pady=(2, 0))

            self.model_cache_var = tk.StringVar(value="")
            self.model_cache_label = tk.Label(section, textvariable=self.model_cache_var,
                                               font=("Microsoft YaHei UI", 8), fg=self.ACCENT,
                                               bg=self.BG_WHITE)
            self.model_cache_label.pack(anchor=tk.W, pady=(0, 4))

            row_ai = self._make_row(section)
            self.ai_remove_btn = self._make_btn(row_ai, "AI 智能去底", self._remove_bg_ai, accent=True)
            self.ai_remove_btn.pack(side=tk.LEFT, padx=(0, 4))
            self._make_btn(row_ai, "还原原图", self._restore_original).pack(side=tk.LEFT)

            self.ai_progress = ttk.Progressbar(section, mode='determinate', length=200,
                                                maximum=100, style="Horizontal.TProgressbar")
            self.ai_progress.pack(fill=tk.X, pady=(6, 0))
            self.ai_progress_label = tk.Label(section, text="", font=("Microsoft YaHei UI", 8),
                                               fg=self.ACCENT, bg=self.BG_WHITE)
            self.ai_progress_label.pack(anchor=tk.W)

            section.add_separator()
            self._on_model_change()
        else:
            self._make_label(section, "AI 智能去底（可选功能）", secondary=True).pack(anchor=tk.W, pady=(0, 2))
            tk.Label(section, text="未检测到 model 文件夹。\n请将 model 文件夹放在程序同级目录下，\n即可使用 AI 智能抠图功能。",
                     font=("Microsoft YaHei UI", 9), fg=self.TEXT_SECONDARY,
                     bg=self.BG_WHITE, wraplength=240, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 4))

            tk.Label(section, text=f"期望路径: PictureCut\\model",
                     font=("Microsoft YaHei UI", 8), fg=self.ACCENT,
                     bg=self.BG_WHITE, wraplength=240).pack(anchor=tk.W, pady=(0, 4))

            missing_text = "\n".join(f"  • {ml}" for _, ml in AI_MODELS)
            tk.Label(section, text=f"需要的模型:\n{missing_text}",
                     font=("Microsoft YaHei UI", 8), fg=self.TEXT_SECONDARY,
                     bg=self.BG_WHITE, wraplength=240, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 4))

            if _REMBG_IMPORT_ERROR:
                tk.Label(section, text=f"错误信息: {_REMBG_IMPORT_ERROR}",
                         font=("Microsoft YaHei UI", 8), fg="#FF3B30",
                         bg=self.BG_WHITE, wraplength=240).pack(anchor=tk.W)

            section.add_separator()

        self._make_label(section, "阈值去底（简单快速）", secondary=True).pack(anchor=tk.W)
        row1 = self._make_row(section)
        self._make_label(row1, "阈值").pack(side=tk.LEFT)
        self._make_spinbox(row1, self.bg_threshold, 100, 255, 5, 5).pack(side=tk.LEFT, padx=4)
        self._make_label(row1, "RGB > 阈值 → 透明", tertiary=True).pack(side=tk.LEFT, padx=4)

        row2 = self._make_row(section)
        self._make_btn(row2, "阈值去底", self._remove_bg_threshold, accent=True).pack(side=tk.LEFT, padx=(0, 4))
        self._make_btn(row2, "还原原图", self._restore_original).pack(side=tk.LEFT)

        self.bg_status_var = tk.StringVar(value="")
        tk.Label(section, textvariable=self.bg_status_var, font=("Microsoft YaHei UI", 9),
                 fg=self.ACCENT, bg=self.BG_WHITE).pack(anchor=tk.W, pady=(4, 0))

    def _build_ai_crop_panel(self, parent):
        section = self._make_section(parent, "AI 自动拆图")

        tk.Label(section, text="自动识别集合图中的独立元素，拆分为单独文件",
                 font=("Microsoft YaHei UI", 8), fg=self.TEXT_TERTIARY,
                 bg=self.BG_WHITE, wraplength=240, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 6))

        pad_frame = self._make_row(section)
        self._make_label(pad_frame, "留白").pack(side=tk.LEFT)
        self._make_spinbox(pad_frame, self.crop_padding_var, 0, 500, 2, 5).pack(side=tk.LEFT, padx=4)
        self._make_label(pad_frame, "像素", tertiary=True).pack(side=tk.LEFT, padx=4)

        area_frame = self._make_row(section)
        self._make_label(area_frame, "最小面积").pack(side=tk.LEFT)
        tip_label = tk.Label(area_frame, text=" ?", font=("Microsoft YaHei UI", 9, "bold"),
                             fg=self.ACCENT, bg=self.BG_WHITE, cursor="question_arrow")
        tip_label.pack(side=tk.LEFT, padx=(2, 0))
        tip_label.bind("<Enter>", lambda e: self._show_tooltip(e, "面积小于此值的连通区域将被忽略，\n不会生成选区。默认100像素，\n可根据图中元素大小调整。"))
        tip_label.bind("<Leave>", lambda e: self._hide_tooltip())
        self._make_spinbox(area_frame, self.crop_min_area_var, 1, 100000, 50, 6).pack(side=tk.LEFT, padx=4)
        self._make_label(area_frame, "像素", tertiary=True).pack(side=tk.LEFT, padx=4)

        section.add_separator()

        btn_row = self._make_row(section)
        self.ai_split_btn = self._make_btn(btn_row, "执行拆图", self._auto_split, accent=True)
        self.ai_split_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._make_btn(btn_row, "清除拆图选区", self._clear_split_selections).pack(side=tk.LEFT)

        tk.Label(section, textvariable=self.crop_status_var,
                 font=("Microsoft YaHei UI", 9), fg=self.ACCENT,
                 bg=self.BG_WHITE).pack(anchor=tk.W, pady=(4, 0))

    def _detect_elements(self, img, min_area=100):
        from collections import deque

        if img.mode != "RGBA":
            img = img.convert("RGBA")

        arr = np.array(img)
        alpha = arr[:, :, 3]
        mask = alpha > 0
        height, width = mask.shape
        labels = np.zeros((height, width), dtype=np.int32)
        current_label = 0
        elements = []

        for y in range(height):
            for x in range(width):
                if mask[y, x] and labels[y, x] == 0:
                    current_label += 1
                    queue = deque([(y, x)])
                    labels[y, x] = current_label
                    min_y, max_y = y, y
                    min_x, max_x = x, x
                    area = 0

                    while queue:
                        cy, cx = queue.popleft()
                        area += 1
                        if cy < min_y: min_y = cy
                        if cy > max_y: max_y = cy
                        if cx < min_x: min_x = cx
                        if cx > max_x: max_x = cx

                        for dy, dx in [(-1, -1), (-1, 0), (-1, 1),
                                       (0, -1),           (0, 1),
                                       (1, -1),  (1, 0),  (1, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = current_label
                                queue.append((ny, nx))

                    if area >= min_area:
                        elements.append((min_x, min_y, max_x + 1, max_y + 1, area))

        elements.sort(key=lambda e: (e[1], e[0]))
        return elements

    def _auto_split(self):
        if self.source_image is None:
            messagebox.showinfo("提示", "请先上传图片")
            return
        self._blur_entries()

        try:
            padding = int(self.crop_padding_var.get())
        except (ValueError, tk.TclError):
            padding = 10
        padding = max(0, padding)

        try:
            min_area = int(self.crop_min_area_var.get())
        except (ValueError, tk.TclError):
            min_area = 100
        min_area = max(1, min_area)

        self.crop_status_var.set("正在识别独立元素...")
        self.status_var.set("AI拆图: 正在识别独立元素...")
        self.root.update_idletasks()

        img = self.source_image
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        elements = self._detect_elements(img, min_area)

        if not elements:
            messagebox.showinfo("提示", "未检测到独立元素。\n\n请确认图片为透明底PNG，且包含多个独立的不透明元素。")
            self.crop_status_var.set("未检测到独立元素")
            self.status_var.set("AI拆图: 未检测到独立元素")
            return

        self.selections.clear()
        self.selected_indices = []
        Selection._counter = 0

        iw, ih = img.width, img.height
        for ex1, ey1, ex2, ey2, area in elements:
            sx1 = max(0, ex1 - padding)
            sy1 = max(0, ey1 - padding)
            sx2 = min(iw, ex2 + padding)
            sy2 = min(ih, ey2 + padding)
            sel = Selection(sx1, sy1, sx2, sy2)
            self.selections.append(sel)

        self._push_history()
        self._update_listbox()
        self._redraw_canvas()

        self.crop_status_var.set(f"检测到 {len(elements)} 个独立元素")
        self.status_var.set(f"AI拆图完成: 检测到 {len(elements)} 个独立元素")

    def _clear_split_selections(self):
        if not self.selections:
            return
        if messagebox.askyesno("确认", "确定清除所有拆图选区？"):
            self.selections.clear()
            self.selected_indices = []
            Selection._counter = 0
            self._push_history()
            self._update_listbox()
            self._redraw_canvas()
            self.crop_status_var.set("")
            self.status_var.set("已清除拆图选区")

    def _build_align_panel(self, parent):
        section = self._make_section(parent, "对齐与排列")

        self._make_label(section, "对齐（以首个选中为基准）", secondary=True).pack(anchor=tk.W)
        row1 = self._make_row(section)
        for text, direction in [("左", "left"), ("右", "right"), ("顶", "top"), ("底", "bottom")]:
            self._make_btn(row1, text, lambda d=direction: self._align_selections(d), small=True).pack(side=tk.LEFT, padx=(0, 4))

        section.add_separator()

        self._make_label(section, "等间距排列", secondary=True).pack(anchor=tk.W)
        row2 = self._make_row(section)
        self._make_btn(row2, "水平等距", lambda: self._distribute_selections('horizontal'), small=True).pack(side=tk.LEFT, padx=(0, 4))
        self._make_btn(row2, "垂直等距", lambda: self._distribute_selections('vertical'), small=True).pack(side=tk.LEFT)

        section.add_separator()

        self._make_btn(section, "统一尺寸（与首个选中相同）", self._same_size_selections).pack(fill=tk.X)

    def _build_export_panel(self, parent):
        section = self._make_section(parent, "导出设置")

        row1 = self._make_row(section)
        self._make_label(row1, "前缀").pack(side=tk.LEFT)
        self._make_input(row1, self.naming_prefix, 8).pack(side=tk.LEFT, padx=4)
        self._make_label(row1, "位数").pack(side=tk.LEFT, padx=(8, 0))
        self._make_spinbox(row1, self.naming_digits, 1, 6, 1, 3).pack(side=tk.LEFT, padx=4)

        self._make_label(section, "示例: icon_001.png, icon_002.png ...", tertiary=True).pack(anchor=tk.W, pady=(2, 0))

    def _on_model_change(self, event=None):
        sel_text = self.ai_model_var.get()
        for mid, mlabel, mdesc in AI_MODELS:
            if mid in sel_text or mlabel == sel_text:
                self.ai_model_var.set(mlabel)
                self.model_desc_var.set(mdesc)
                cached = _check_model_cached(mid)
                if cached:
                    self.model_cache_var.set("● 模型已就绪，可离线使用")
                else:
                    self.model_cache_var.set("○ 模型未下载，首次使用需联网下载")
                break

        missing = _get_missing_models()
        if missing:
            names = "\n".join(f"  • {ml}" for _, ml in missing)
            self.model_cache_var.set(f"缺少 {len(missing)} 个模型:\n{names}")

    def _get_model_id(self):
        sel_text = self.ai_model_var.get()
        for mid, mlabel, mdesc in AI_MODELS:
            if mid in sel_text or mlabel == sel_text:
                return mid
        return "birefnet-general-lite"

    def _start_progress(self):
        self._progress_value = 0
        self._progress_start = 0
        self.ai_progress['value'] = 0
        self.ai_progress_label.config(text="0%")
        self._tick_progress()

    def _tick_progress(self):
        if not self._ai_processing:
            return
        if self._progress_value < 90:
            remaining = 90 - self._progress_value
            increment = max(0.3, remaining * 0.04)
            self._progress_value += increment
            if self._progress_value > 90:
                self._progress_value = 90
            display_pct = int(self._progress_value)
            self.ai_progress['value'] = self._progress_value
            self.ai_progress_label.config(text=f"{display_pct}%")
            self._progress_timer = self.root.after(80, self._tick_progress)

    def _stop_progress(self, success=True):
        self._ai_processing = False
        if self._progress_timer:
            self.root.after_cancel(self._progress_timer)
            self._progress_timer = None
        if success:
            self.ai_progress['value'] = 100
            self.ai_progress_label.config(text="100% 完成")
        else:
            self.ai_progress_label.config(text="处理失败")

    def _remove_bg_threshold(self):
        if self.source_image is None:
            messagebox.showinfo("提示", "请先上传图片")
            return
        self._blur_entries()
        try:
            threshold = int(self.bg_threshold.get())
        except (ValueError, tk.TclError):
            threshold = 240
        threshold = max(0, min(255, threshold))

        self.status_var.set("正在阈值去底...")
        self.bg_status_var.set("处理中...")
        self.root.update_idletasks()

        img = self.source_image
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        arr = np.array(img)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        white_mask = (r >= threshold) & (g >= threshold) & (b >= threshold)
        arr[white_mask, 3] = 0

        self.source_image = Image.fromarray(arr, "RGBA")
        self._redraw_canvas()
        self.status_var.set(f"已阈值去底 (阈值={threshold})")
        self.bg_status_var.set("阈值去底完成")

    def _remove_bg_ai(self):
        if not HAS_REMBG:
            if not _try_import_rembg():
                messagebox.showinfo(
                    "AI去底不可用",
                    f"未检测到 AI 去底模块。\n\n"
                    f"请将 model 文件夹放在程序同级目录下:\n"
                    f"{_MODEL_DIR}\n\n"
                    f"错误: {_REMBG_IMPORT_ERROR}"
                )
                return
        if self.source_image is None:
            messagebox.showinfo("提示", "请先上传图片")
            return
        if self._ai_processing:
            return

        self._blur_entries()
        model_id = self._get_model_id()
        cached = _check_model_cached(model_id)

        if not cached:
            model_label = ""
            for mid, mlabel, mdesc in AI_MODELS:
                if mid == model_id:
                    model_label = mlabel
                    break
            proceed = messagebox.askyesno(
                "模型未下载",
                f"模型 [{model_label}] 尚未下载到本地。\n\n"
                f"请将模型文件放入:\n{_U2NET_HOME}\n\n"
                f"是否继续？（将自动联网下载）"
            )
            if not proceed:
                return

        self._ai_processing = True
        self.ai_remove_btn.configure_state("disabled")
        self.bg_status_var.set(f"AI去底处理中 [{model_id}]...")
        self.status_var.set(f"AI去底处理中 [{model_id}]，请稍候...")
        self._start_progress()
        self.root.update_idletasks()

        source_img = self.source_image
        is_frozen = getattr(sys, 'frozen', False)

        def do_remove():
            tmp_dir = tempfile.mkdtemp(prefix="pc_rembg_")
            input_path = os.path.join(tmp_dir, "input.png")
            output_path = os.path.join(tmp_dir, "output.png")

            try:
                source_img.save(input_path)

                if is_frozen:
                    import shutil as _shutil
                    py_exe = _shutil.which("python") or _shutil.which("python3")
                    if not py_exe:
                        for ver in range(14, 8, -1):
                            for p in [
                                f"C:\\Python3{ver}\\python.exe",
                                os.path.join(os.environ.get("LOCALAPPDATA", ""),
                                             "Programs", "Python", f"Python3{ver}", "python.exe"),
                            ]:
                                if os.path.isfile(p):
                                    py_exe = p
                                    break
                            if py_exe:
                                break
                    if not py_exe:
                        raise RuntimeError(
                            "未找到系统 Python，AI 去底功能需要系统安装 Python。\n"
                            "请安装 Python 3.10+ 并确保可在命令行中使用 python 命令。"
                        )

                    script_content = (
                        "import sys, os, importlib.util\n"
                        f"sys.path.insert(0, {_MODEL_DIR!r})\n"
                        f"os.environ['U2NET_HOME'] = {_U2NET_HOME!r}\n"
                        f"ort_capi = os.path.join({_MODEL_DIR!r}, 'onnxruntime', 'capi')\n"
                        "if os.path.isdir(ort_capi):\n"
                        "    if hasattr(os, 'add_dll_directory'):\n"
                        "        try: os.add_dll_directory(ort_capi)\n"
                        "        except OSError: pass\n"
                        "    os.environ['PATH'] = ort_capi + os.pathsep + os.environ.get('PATH', '')\n"
                        f"rembg_init = os.path.join({_MODEL_DIR!r}, 'rembg', '__init__.py')\n"
                        "spec = importlib.util.spec_from_file_location('rembg', rembg_init, "
                        f"submodule_search_locations=[os.path.join({_MODEL_DIR!r}, 'rembg')])\n"
                        "mod = importlib.util.module_from_spec(spec)\n"
                        "spec.loader.exec_module(mod)\n"
                        "from PIL import Image\n"
                        f"img = Image.open({input_path!r})\n"
                        f"session = mod.new_session({model_id!r})\n"
                        "result = mod.remove(img, session=session)\n"
                        f"result.save({output_path!r})\n"
                        "print('OK')\n"
                    )
                    script_path = os.path.join(tmp_dir, "_run_rembg.py")
                    with open(script_path, "w", encoding="utf-8") as f:
                        f.write(script_content)

                    env = os.environ.copy()
                    env["PYTHONUNBUFFERED"] = "1"
                    proc = subprocess.run(
                        [py_exe, script_path],
                        capture_output=True, text=True,
                        timeout=300, env=env,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    if proc.returncode != 0:
                        err_out = (proc.stderr or "").strip() or (proc.stdout or "").strip() or "未知错误"
                        raise RuntimeError(f"子进程执行失败: {err_out[:500]}")
                else:
                    global _rembg_remove_fn, _rembg_new_session_fn
                    session = _rembg_new_session_fn(model_id)
                    result = _rembg_remove_fn(source_img, session=session)
                    result.save(output_path)

                if not os.path.isfile(output_path):
                    raise RuntimeError("输出文件未生成")

                result_img = Image.open(output_path).convert("RGBA")
                self.root.after(0, lambda ri=result_img: self._on_ai_remove_done(ri, model_id))
            except Exception as e:
                err_str = str(e)
                self.root.after(0, lambda es=err_str: self._on_ai_remove_error(es, model_id))
            finally:
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

        threading.Thread(target=do_remove, daemon=True).start()

    def _on_ai_remove_done(self, result, model_name):
        self._stop_progress(success=True)
        self.source_image = result
        self._redraw_canvas()
        self.ai_remove_btn.configure_state("normal")
        self.status_var.set(f"AI去底完成 [{model_name}]")
        self.bg_status_var.set(f"AI去底完成 [{model_name}]")
        self._on_model_change()

    def _on_ai_remove_error(self, err_msg, model_name):
        self._stop_progress(success=False)
        self.ai_remove_btn.configure_state("normal")
        self.bg_status_var.set(f"AI去底失败 [{model_name}]")
        if self.original_image is not None:
            self.source_image = self.original_image.copy()
            self._redraw_canvas()
        messagebox.showerror(
            "AI去底失败",
            f"模型: {model_name}\n\n"
            f"错误信息:\n{err_msg}\n\n"
            f"可能原因:\n"
            f"• model 文件夹中缺少 onnxruntime 或其 DLL\n"
            f"• 模型未下载且无法联网\n"
            f"• 内存不足 — 尝试使用更轻量的模型 (如 BiRefNet-Lite)\n\n"
            f"已自动还原为原始图片。"
        )

    def _restore_original(self):
        if self.original_image is None:
            messagebox.showinfo("提示", "请先上传图片")
            return
        self._blur_entries()
        self.source_image = self.original_image.copy()
        self.crop_status_var.set("")
        self._redraw_canvas()
        self.status_var.set("已还原为原始图片")

    def _fit_to_window(self):
        if self.source_image is None:
            return
        self.root.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            self.root.after(100, self._fit_to_window)
            return
        iw = self.source_image.width
        ih = self.source_image.height
        margin = 40
        sx = (cw - margin) / iw
        sy = (ch - margin) / ih
        self.zoom = min(sx, sy, 1.0)
        self.pan_x = (cw - iw * self.zoom) / 2
        self.pan_y = (ch - ih * self.zoom) / 2
        self._update_zoom_display()
        self._redraw_canvas()

    def _zoom_original(self):
        if self.source_image is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        iw = self.source_image.width
        ih = self.source_image.height
        self.zoom = 1.0
        self.pan_x = (cw - iw) / 2
        self.pan_y = (ch - ih) / 2
        self._update_zoom_display()
        self._redraw_canvas()

    def _zoom_by(self, factor, cx=None, cy=None):
        if self.source_image is None:
            return
        old_zoom = self.zoom
        self.zoom = max(0.05, min(self.zoom * factor, 20.0))
        if cx is None:
            cx = self.canvas.winfo_width() / 2
            cy = self.canvas.winfo_height() / 2
        self.pan_x = cx - (cx - self.pan_x) * (self.zoom / old_zoom)
        self.pan_y = cy - (cy - self.pan_y) * (self.zoom / old_zoom)
        self._update_zoom_display()
        self._redraw_canvas()

    def _on_zoom_entry(self, event=None):
        try:
            val = self.zoom_var.get().replace("%", "").strip()
            self.zoom = max(0.05, min(float(val) / 100.0, 20.0))
            self._update_zoom_display()
            self._redraw_canvas()
        except ValueError:
            pass

    def _update_zoom_display(self):
        self.zoom_var.set(f"{self.zoom * 100:.0f}%")

    def _on_mouse_wheel(self, event):
        if event.delta > 0:
            self._zoom_by(1.1, event.x, event.y)
        else:
            self._zoom_by(0.9, event.x, event.y)

    def _on_pan_start(self, event):
        self._pan_start_x = event.x
        self._pan_start_y = event.y
        self._pan_start_px = self.pan_x
        self._pan_start_py = self.pan_y
        self.canvas.config(cursor="fleur")

    def _on_pan_move(self, event):
        self.pan_x = self._pan_start_px + event.x - self._pan_start_x
        self.pan_y = self._pan_start_py + event.y - self._pan_start_y
        self._redraw_canvas()

    def _on_pan_end(self, event):
        self.canvas.config(cursor="crosshair")

    def _on_canvas_resize(self, event=None):
        if self.source_image is not None:
            self._redraw_canvas()

    def _redraw_canvas(self):
        self.canvas.delete("all")
        if self.source_image is None:
            cx = self.canvas.winfo_width() / 2
            cy = self.canvas.winfo_height() / 2
            self.canvas.create_text(cx, cy - 20, text="拖拽图片到此处",
                                     fill="#86868B", font=("Microsoft YaHei UI", 18))
            self.canvas.create_text(cx, cy + 15, text="或点击「上传图片」",
                                     fill="#AEAEB2", font=("Microsoft YaHei UI", 13))
            self.canvas.create_text(cx, cy + 45, text="支持 PNG / JPG / BMP / TIFF / WEBP",
                                     fill="#636366", font=("Microsoft YaHei UI", 10))
            return
        self._draw_image()
        if self.grid_visible.get():
            self._draw_grid()
        self._draw_selections()

    def _draw_image(self):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 1 or ch < 1:
            return
        vis_x1, vis_y1 = self._canvas_to_img(0, 0)
        vis_x2, vis_y2 = self._canvas_to_img(cw, ch)
        iw = self.source_image.width
        ih = self.source_image.height
        crop_x1 = max(0, int(vis_x1) - 1)
        crop_y1 = max(0, int(vis_y1) - 1)
        crop_x2 = min(iw, int(vis_x2) + 2)
        crop_y2 = min(ih, int(vis_y2) + 2)
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return
        cropped = self.source_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        display_w = max(1, int((crop_x2 - crop_x1) * self.zoom))
        display_h = max(1, int((crop_y2 - crop_y1) * self.zoom))
        max_dim = 4096
        if display_w > max_dim or display_h > max_dim:
            ratio = min(max_dim / display_w, max_dim / display_h)
            display_w = int(display_w * ratio)
            display_h = int(display_h * ratio)
        if display_w < 1 or display_h < 1:
            return
        resized = cropped.resize((display_w, display_h), Image.BILINEAR)
        self.tk_image = ImageTk.PhotoImage(resized)
        img_cx, img_cy = self._img_to_canvas(crop_x1, crop_y1)
        self.canvas.create_image(img_cx, img_cy, anchor=tk.NW, image=self.tk_image)

    def _draw_grid(self):
        gs = self.grid_size.get()
        if gs < 2:
            return
        iw = self.source_image.width
        ih = self.source_image.height
        x1c, y1c = self._img_to_canvas(0, 0)
        x2c, y2c = self._img_to_canvas(iw, ih)
        if gs * self.zoom < 4:
            return
        x = 0
        while x <= iw:
            cx, _ = self._img_to_canvas(x, 0)
            if x1c <= cx <= x2c:
                self.canvas.create_line(cx, y1c, cx, y2c, fill="#3A3A3C", dash=(2, 4))
            x += gs
        y = 0
        while y <= ih:
            _, cy = self._img_to_canvas(0, y)
            if y1c <= cy <= y2c:
                self.canvas.create_line(x1c, cy, x2c, cy, fill="#3A3A3C", dash=(2, 4))
            y += gs

    def _draw_selections(self):
        selected_set = set(self.selected_indices)
        for i, sel in enumerate(self.selections):
            self._draw_one_selection(sel, i in selected_set)

    def _draw_one_selection(self, sel, is_selected):
        cx1, cy1 = self._img_to_canvas(sel.x1, sel.y1)
        cx2, cy2 = self._img_to_canvas(sel.x2, sel.y2)
        outline_color = self.ACCENT if is_selected else self.SEL_COLOR_ALT
        fill_alpha = "#0A1A2A" if is_selected else "#1A1A0A"
        self.canvas.create_rectangle(cx1, cy1, cx2, cy2, fill=fill_alpha, stipple="gray25",
                                      outline=outline_color, width=2)
        label_y = cy1 - 14 if cy1 > 20 else cy1 + 2
        self.canvas.create_text(cx1 + 2, label_y, text=sel.name, anchor=tk.NW,
                                 fill=outline_color, font=("Consolas", 9, "bold"))
        if is_selected:
            for name, (hx, hy) in sel.get_handles().items():
                chx, chy = self._img_to_canvas(hx, hy)
                hs = self.HANDLE_SIZE
                self.canvas.create_rectangle(chx - hs, chy - hs, chx + hs, chy + hs,
                                              fill="white", outline=outline_color, width=1)

    def _find_selection_at(self, img_x, img_y):
        for i in range(len(self.selections) - 1, -1, -1):
            if self.selections[i].contains(img_x, img_y):
                return i
        return -1

    def _on_canvas_press(self, event):
        self._blur_entries()
        if self.source_image is None:
            return
        img_x, img_y = self._canvas_to_img(event.x, event.y)
        ctrl = bool(event.state & 0x4)

        selected_set = set(self.selected_indices)
        if selected_set:
            for idx in self.selected_indices:
                sel = self.selections[idx]
                handle = sel.hit_handle(img_x, img_y, margin=6 / self.zoom)
                if handle:
                    self.drag_state = {'type': 'resize', 'handle': handle, 'sel_idx': idx,
                                       'orig': (sel.x1, sel.y1, sel.x2, sel.y2), 'start': (img_x, img_y)}
                    return

        idx = self._find_selection_at(img_x, img_y)
        if idx >= 0:
            if ctrl:
                if idx in selected_set:
                    self.selected_indices.remove(idx)
                else:
                    self.selected_indices.append(idx)
            else:
                if idx not in selected_set:
                    self.selected_indices = [idx]
            orig_positions = {si: (self.selections[si].x1, self.selections[si].y1,
                                    self.selections[si].x2, self.selections[si].y2)
                              for si in self.selected_indices}
            self.drag_state = {'type': 'move', 'orig_positions': orig_positions, 'start': (img_x, img_y)}
            self._update_listbox()
            self._redraw_canvas()
        else:
            if not ctrl:
                self.selected_indices = []
            self.drag_state = None
            self.drawing = True
            self.draw_start = (img_x, img_y)
            self._update_listbox()
            self._redraw_canvas()

    def _on_canvas_drag(self, event):
        if self.source_image is None:
            return
        img_x, img_y = self._canvas_to_img(event.x, event.y)

        if self.drawing and self.draw_start is not None:
            self.canvas.delete("temp_rect")
            sx, sy = self.draw_start
            if self.snap_to_grid.get():
                img_x, img_y = self._snap_value(img_x), self._snap_value(img_y)
            cx1, cy1 = self._img_to_canvas(sx, sy)
            cx2, cy2 = self._img_to_canvas(img_x, img_y)
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=self.ACCENT, width=2, dash=(4, 4), tags="temp_rect")
            return

        if self.drag_state is None:
            return
        ds = self.drag_state

        if ds['type'] == 'move':
            dx = img_x - ds['start'][0]
            dy = img_y - ds['start'][1]
            for si, (ox1, oy1, ox2, oy2) in ds['orig_positions'].items():
                if si < len(self.selections):
                    sel = self.selections[si]
                    nx1, ny1, nx2, ny2 = ox1 + dx, oy1 + dy, ox2 + dx, oy2 + dy
                    if self.snap_to_grid.get():
                        gs = self.grid_size.get()
                        if gs >= 2:
                            sx1 = self._snap_value(nx1)
                            nx1, nx2 = sx1, ox2 + dx + (sx1 - nx1)
                            sy1 = self._snap_value(ny1)
                            ny1, ny2 = sy1, oy2 + dy + (sy1 - ny1)
                    sel.x1, sel.y1, sel.x2, sel.y2 = nx1, ny1, nx2, ny2
            self._redraw_canvas()

        elif ds['type'] == 'resize':
            idx = ds['sel_idx']
            sel = self.selections[idx]
            ox1, oy1, ox2, oy2 = ds['orig']
            h = ds['handle']
            dx, dy = img_x - ds['start'][0], img_y - ds['start'][1]
            nx1, ny1, nx2, ny2 = ox1, oy1, ox2, oy2
            if 'w' in h: nx1 = ox1 + dx
            if 'e' in h: nx2 = ox2 + dx
            if 'n' in h: ny1 = oy1 + dy
            if 's' in h: ny2 = oy2 + dy
            if h in ('n', 's'): nx1, nx2 = ox1, ox2
            if h in ('w', 'e'): ny1, ny2 = oy1, oy2
            if self.snap_to_grid.get():
                nx1, ny1, nx2, ny2 = self._snap_value(nx1), self._snap_value(ny1), self._snap_value(nx2), self._snap_value(ny2)
            sel.x1, sel.y1, sel.x2, sel.y2 = min(nx1, nx2), min(ny1, ny2), max(nx1, nx2), max(ny1, ny2)
            self._redraw_canvas()

    def _on_canvas_release(self, event):
        if self.source_image is None:
            return
        img_x, img_y = self._canvas_to_img(event.x, event.y)

        if self.drawing and self.draw_start is not None:
            self.drawing = False
            self.canvas.delete("temp_rect")
            sx, sy = self.draw_start
            if self.snap_to_grid.get():
                img_x, img_y = self._snap_value(img_x), self._snap_value(img_y)
                sx, sy = self._snap_value(sx), self._snap_value(sy)
            x1, y1, x2, y2 = min(sx, img_x), min(sy, img_y), max(sx, img_x), max(sy, img_y)
            iw, ih = self.source_image.width, self.source_image.height
            x1, y1 = max(0, min(x1, iw)), max(0, min(y1, ih))
            x2, y2 = max(0, min(x2, iw)), max(0, min(y2, ih))
            if (x2 - x1) >= self.MIN_SELECTION and (y2 - y1) >= self.MIN_SELECTION:
                sel = Selection(x1, y1, x2, y2)
                self.selections.append(sel)
                self.selected_indices = [len(self.selections) - 1]
                self._push_history()
                self._update_listbox()
                self._redraw_canvas()
                self.status_var.set(f"已添加选区: {sel.name}  ({sel.width:.0f}x{sel.height:.0f})")
            else:
                self.status_var.set("选区太小，已忽略")
            self.draw_start = None
            return

        if self.drag_state is not None:
            self._push_history()
            self._update_listbox()
        self.drag_state = None

    def _push_history(self):
        if self._history_lock:
            return
        state = [sel.snapshot() for sel in self.selections]
        sel_indices = list(self.selected_indices)
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        self.history.append((state, sel_indices))
        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[-self.MAX_HISTORY:]
        self.history_index = len(self.history) - 1

    def _restore_state(self, state, sel_indices):
        self.selections = [Selection.from_snapshot(s) for s in state]
        self.selected_indices = [idx for idx in sel_indices if 0 <= idx < len(self.selections)]
        self._update_listbox()
        self._redraw_canvas()

    def _on_undo(self, event=None):
        if self._is_entry_focused(event):
            return
        if self.history_index > 0:
            self.history_index -= 1
            self._history_lock = True
            self._restore_state(*self.history[self.history_index])
            self._history_lock = False
            self.status_var.set("撤销")

    def _on_redo(self, event=None):
        if self._is_entry_focused(event):
            return
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self._history_lock = True
            self._restore_state(*self.history[self.history_index])
            self._history_lock = False
            self.status_var.set("重做")

    def _on_copy(self, event=None):
        if self._is_entry_focused(event):
            return
        if not self.selected_indices or self.source_image is None:
            return
        self.clipboard = [self.selections[idx].snapshot() for idx in self.selected_indices if 0 <= idx < len(self.selections)]
        if self.clipboard:
            self.status_var.set(f"已复制 {len(self.clipboard)} 个选区")

    def _on_paste(self, event=None):
        if self._is_entry_focused(event):
            return
        if not self.clipboard or self.source_image is None:
            return
        offset = 20
        new_indices = []
        iw, ih = self.source_image.width, self.source_image.height
        for x1, y1, x2, y2, name in self.clipboard:
            nx1, ny1, nx2, ny2 = x1 + offset, y1 + offset, x2 + offset, y2 + offset
            if nx2 > iw:
                s = nx2 - iw + 5; nx1 -= s; nx2 -= s
            if ny2 > ih:
                s = ny2 - ih + 5; ny1 -= s; ny2 -= s
            sel = Selection(nx1, ny1, nx2, ny2)
            new_indices.append(len(self.selections))
            self.selections.append(sel)
        self.selected_indices = new_indices
        self._push_history()
        self._update_listbox()
        self._redraw_canvas()
        self.status_var.set(f"已粘贴 {len(new_indices)} 个选区")

    def _on_arrow_key(self, event=None):
        if self._is_entry_focused(event):
            return
        if not self.selected_indices or self.source_image is None:
            return
        step = 10 if (event.state & 0x1) else 1
        dx, dy = 0, 0
        if event.keysym == 'Left': dx = -step
        elif event.keysym == 'Right': dx = step
        elif event.keysym == 'Up': dy = -step
        elif event.keysym == 'Down': dy = step
        if dx == 0 and dy == 0:
            return
        for idx in self.selected_indices:
            if 0 <= idx < len(self.selections):
                sel = self.selections[idx]
                sel.x1 += dx; sel.y1 += dy; sel.x2 += dx; sel.y2 += dy
        self._push_history()
        self._update_listbox()
        self._redraw_canvas()

    def _create_pixel_selection(self, x1, y1, w, h):
        if self.source_image is None:
            messagebox.showinfo("提示", "请先上传图片")
            return
        iw, ih = self.source_image.width, self.source_image.height
        x2, y2 = min(x1 + w, iw), min(y1 + h, ih)
        x1, y1 = max(0, x1), max(0, y1)
        if (x2 - x1) < self.MIN_SELECTION or (y2 - y1) < self.MIN_SELECTION:
            messagebox.showinfo("提示", "选区尺寸太小")
            return
        sel = Selection(x1, y1, x2, y2)
        self.selections.append(sel)
        self.selected_indices = [len(self.selections) - 1]
        self._push_history()
        self._update_listbox()
        self._redraw_canvas()
        self.status_var.set(f"已创建像素选区: {sel.name}  ({sel.width:.0f}x{sel.height:.0f})")

    def _create_pixel_selection_center(self):
        self._blur_entries()
        try:
            w, h = int(self.input_w.get()), int(self.input_h.get())
        except (ValueError, tk.TclError):
            messagebox.showinfo("提示", "请输入有效的宽高数值")
            return
        if w < 1 or h < 1:
            messagebox.showinfo("提示", "宽高必须大于0")
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        img_cx, img_cy = self._canvas_to_img(cw / 2, ch / 2)
        self._create_pixel_selection(img_cx - w / 2, img_cy - h / 2, w, h)

    def _get_selected_selections(self):
        result, seen = [], set()
        for idx in self.selected_indices:
            if 0 <= idx < len(self.selections) and idx not in seen:
                result.append((idx, self.selections[idx]))
                seen.add(idx)
        return result

    def _align_selections(self, direction):
        selected = self._get_selected_selections()
        if len(selected) < 2:
            messagebox.showinfo("提示", "至少需要选中2个选区才能对齐 (Ctrl+点击多选)")
            return
        ref_idx, ref = selected[0]
        for idx, sel in selected[1:]:
            w, h = sel.width, sel.height
            if direction == 'left': sel.x1, sel.x2 = ref.x1, ref.x1 + w
            elif direction == 'right': sel.x2, sel.x1 = ref.x2, ref.x2 - w
            elif direction == 'top': sel.y1, sel.y2 = ref.y1, ref.y1 + h
            elif direction == 'bottom': sel.y2, sel.y1 = ref.y2, ref.y2 - h
        self._push_history()
        self._update_listbox()
        self._redraw_canvas()
        self.status_var.set(f"已执行对齐: {direction}")

    def _distribute_selections(self, direction):
        selected = self._get_selected_selections()
        if len(selected) < 3:
            messagebox.showinfo("提示", "至少需要选中3个选区才能等间距排列 (Ctrl+点击多选)")
            return
        if direction == 'horizontal':
            sorted_sels = sorted(selected, key=lambda x: x[1].cx)
            first, last = sorted_sels[0][1], sorted_sels[-1][1]
            total_span = last.x2 - first.x1
            total_sel_w = sum(s.width for _, s in sorted_sels)
            if total_span <= total_sel_w:
                messagebox.showinfo("提示", "选区间距不足，无法等间距排列")
                return
            gap = (total_span - total_sel_w) / (len(sorted_sels) - 1)
            cur_x = first.x1
            for _, sel in sorted_sels:
                sel.x1, sel.x2 = cur_x, cur_x + sel.width
                cur_x = sel.x2 + gap
        elif direction == 'vertical':
            sorted_sels = sorted(selected, key=lambda x: x[1].cy)
            first, last = sorted_sels[0][1], sorted_sels[-1][1]
            total_span = last.y2 - first.y1
            total_sel_h = sum(s.height for _, s in sorted_sels)
            if total_span <= total_sel_h:
                messagebox.showinfo("提示", "选区间距不足，无法等间距排列")
                return
            gap = (total_span - total_sel_h) / (len(sorted_sels) - 1)
            cur_y = first.y1
            for _, sel in sorted_sels:
                sel.y1, sel.y2 = cur_y, cur_y + sel.height
                cur_y = sel.y2 + gap
        self._push_history()
        self._update_listbox()
        self._redraw_canvas()
        self.status_var.set(f"已执行等间距排列: {direction}")

    def _same_size_selections(self):
        selected = self._get_selected_selections()
        if len(selected) < 2:
            messagebox.showinfo("提示", "至少需要选中2个选区 (Ctrl+点击多选)")
            return
        ref_w, ref_h = selected[0][1].width, selected[0][1].height
        for idx, sel in selected[1:]:
            sel.x2, sel.y2 = sel.x1 + ref_w, sel.y1 + ref_h
        self._push_history()
        self._update_listbox()
        self._redraw_canvas()
        self.status_var.set(f"已统一尺寸为: {ref_w:.0f}x{ref_h:.0f}")

    def _update_listbox(self):
        self.sel_listbox.delete(0, tk.END)
        selected_set = set(self.selected_indices)
        for i, sel in enumerate(self.selections):
            prefix = "▶ " if i in selected_set else "  "
            self.sel_listbox.insert(tk.END, f"{prefix}{sel.name}  ({sel.width:.0f}x{sel.height:.0f})")
        for idx in self.selected_indices:
            if 0 <= idx < self.sel_listbox.size():
                self.sel_listbox.selection_set(idx)
                self.sel_listbox.see(idx)
        count = len(self.selected_indices)
        self.sel_info_var.set(f"已选中 {count} 个选区" if count else "Ctrl + 点击多选")

    def _on_listbox_select(self, event):
        sel_indices = self.sel_listbox.curselection()
        if sel_indices:
            self.selected_indices = list(sel_indices)
            self._redraw_canvas()

    def _delete_selected(self):
        if not self.selected_indices or self.source_image is None:
            return
        count = len(self.selected_indices)
        for idx in sorted(set(self.selected_indices), reverse=True):
            if 0 <= idx < len(self.selections):
                del self.selections[idx]
        self.selected_indices = []
        self._push_history()
        self._update_listbox()
        self._redraw_canvas()
        self.status_var.set(f"已删除 {count} 个选区")

    def _clear_selections(self):
        if not self.selections:
            return
        if messagebox.askyesno("确认", "确定清除所有选区？"):
            self.selections.clear()
            self.selected_indices = []
            Selection._counter = 0
            self._push_history()
            self._update_listbox()
            self._redraw_canvas()
            self.status_var.set("已清除所有选区")

    def _deselect(self):
        self.selected_indices = []
        self._update_listbox()
        self._redraw_canvas()

    def _export_all(self):
        if self.source_image is None:
            messagebox.showinfo("提示", "请先上传图片")
            return
        self._blur_entries()
        save_dir = filedialog.askdirectory(title="选择导出文件夹")
        if not save_dir:
            return
        prefix = self.naming_prefix.get().strip() or "icon"
        try:
            digits = int(self.naming_digits.get())
        except (ValueError, tk.TclError):
            digits = 3
        try:
            exported = 0
            if self.selections:
                for i, sel in enumerate(self.selections):
                    x1 = max(0, int(round(sel.x1)))
                    y1 = max(0, int(round(sel.y1)))
                    x2 = min(self.source_image.width, int(round(sel.x2)))
                    y2 = min(self.source_image.height, int(round(sel.y2)))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    cropped = self.source_image.crop((x1, y1, x2, y2))
                    if cropped.mode != "RGBA":
                        cropped = cropped.convert("RGBA")
                    filename = f"{prefix}_{exported + 1:0{digits}d}.png"
                    filepath = os.path.join(save_dir, filename)
                    cropped.save(filepath, format="PNG")
                    exported += 1
            else:
                img = self.source_image
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                filename = f"{prefix}_001.png"
                filepath = os.path.join(save_dir, filename)
                img.save(filepath, format="PNG")
                exported = 1
            self.status_var.set(f"已导出 {exported} 个文件到 {save_dir}")
            messagebox.showinfo("导出成功", f"已导出 {exported} 个PNG文件到:\n{save_dir}")
        except Exception as e:
            messagebox.showerror("导出失败", f"导出时出错:\n{e}")

    def _bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_by(1.1, e.x, e.y))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_by(0.9, e.x, e.y))
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_end)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self.root.bind("<Control-z>", self._on_undo)
        self.root.bind("<Control-y>", self._on_redo)
        self.root.bind("<Control-c>", self._on_copy)
        self.root.bind("<Control-v>", self._on_paste)
        self.root.bind("<Delete>", lambda e: self._delete_selected())
        self.root.bind("<Escape>", lambda e: self._deselect())
        self.root.bind("<Left>", self._on_arrow_key)
        self.root.bind("<Right>", self._on_arrow_key)
        self.root.bind("<Up>", self._on_arrow_key)
        self.root.bind("<Down>", self._on_arrow_key)

        if HAS_WINDND:
            windnd.hook_dropfiles(self.canvas, func=self._on_drop_files)

    def _blur_entries(self):
        self.root.focus_set()

    def _is_entry_focused(self, event=None):
        focus = self.root.focus_get()
        if focus and isinstance(focus, (tk.Entry, ttk.Entry, tk.Text, ttk.Spinbox)):
            return True
        return False

    def _on_drop_files(self, file_paths):
        for fp in file_paths:
            try:
                path = fp.decode('gbk')
            except Exception:
                path = fp.decode('utf-8', errors='ignore')
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')):
                self._load_image_file(path)
                return
        messagebox.showinfo("提示", "请拖入图片文件 (PNG/JPG/BMP/TIFF/WEBP)")

    def _img_to_canvas(self, ix, iy):
        return ix * self.zoom + self.pan_x, iy * self.zoom + self.pan_y

    def _canvas_to_img(self, cx, cy):
        return (cx - self.pan_x) / self.zoom, (cy - self.pan_y) / self.zoom

    def _snap_value(self, val):
        if not self.snap_to_grid.get():
            return val
        gs = self.grid_size.get()
        if gs < 2:
            return val
        return round(val / gs) * gs

    def _upload_image(self):
        filepath = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"), ("所有文件", "*.*")]
        )
        if not filepath:
            return
        self._load_image_file(filepath)

    def _load_image_file(self, filepath):
        try:
            img = Image.open(filepath)
            if img.mode not in ("RGBA", "RGB"):
                img = img.convert("RGBA")
            self.source_image = img
            self.original_image = img.copy()
            self.crop_status_var.set("")
            self.selections.clear()
            self.selected_indices = []
            Selection._counter = 0
            self.history.clear()
            self.history_index = -1
            self._push_history()
            self._update_listbox()
            self.root.update_idletasks()
            self._fit_to_window()
            self.status_var.set(f"已加载: {os.path.basename(filepath)}  ({img.width}x{img.height})")
        except Exception as e:
            messagebox.showerror("错误", f"无法打开图片:\n{e}")

    def _animate_entrance(self):
        self.root.attributes("-alpha", 0.0)
        self._fade_in(0)

    def _fade_in(self, step):
        alpha = min(1.0, step * 0.08)
        try:
            self.root.attributes("-alpha", alpha)
        except Exception:
            return
        if alpha < 1.0:
            self.root.after(20, lambda: self._fade_in(step + 1))


def main():
    import traceback
    log_path = os.path.join(os.path.expanduser("~"), "picture_cut_debug.log")

    try:
        root = tk.Tk()
    except Exception as e:
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"tk.Tk() FAILED:\n{traceback.format_exc()}\n")
        except Exception:
            pass
        return

    try:
        root.tk.call("tk", "scaling", 1.25)
    except Exception:
        pass

    try:
        app = PictureCutApp(root)
    except Exception as e:
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"PictureCutApp init FAILED:\n{traceback.format_exc()}\n")
        except Exception:
            pass
        try:
            messagebox.showerror("启动失败", f"程序初始化失败:\n\n{e}")
        except Exception:
            pass
        return

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"mainloop FAILED:\n{traceback.format_exc()}\n")
        except Exception:
            pass
        try:
            messagebox.showerror("程序异常", f"发生未预期的错误:\n\n{traceback.format_exc()}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
