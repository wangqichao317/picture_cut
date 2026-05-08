import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import os
import zipfile
import io
import threading
import numpy as np
import glob

try:
    import windnd
    HAS_WINDND = True
except ImportError:
    HAS_WINDND = False

try:
    from rembg import remove, new_session
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False

_rembg_sessions = {}

AI_MODELS = [
    ("u2netp", "u2netp - 轻量快速 (~~4MB)",
     "通用场景 | 速度快，适合快速预览 | 精度中等，边缘略粗"),
    ("u2net", "u2net - 标准精度 (~~176MB)",
     "通用场景 | 精度高，边缘较细腻 | 速度中等，模型较大"),
    ("birefnet-general", "BiRefNet-General - 高精度通用",
     "复杂背景/精细边缘 | 当前最高精度 | 速度较慢，适合最终出图"),
    ("birefnet-general-lite", "BiRefNet-Lite - 轻量通用",
     "通用场景 | 精度与速度平衡 | 比BiRefNet-General快"),
    ("birefnet-portrait", "BiRefNet-Portrait - 人像专用",
     "人物/角色/半身像 | 人像边缘精细 | 不适合物品/场景"),
    ("isnet-general-use", "ISNet-General - 通用",
     "通用场景 | 精度较好 | 速度中等"),
    ("isnet-anime", "ISNet-Anime - 动漫专用",
     "二次元/动漫/插画 | 动漫线条保留好 | 不适合照片"),
    ("bria-rmbg", "BRIA-RMBG - 商业去背",
     "商业素材/电商图 | 边缘干净 | 适合白底商品图"),
    ("ben_custom", "BEN2 - 边缘精细",
     "发光/渐变/半透明边缘 | 边缘过渡最自然 | 速度较慢"),
    ("silueta", "Silueta - 轻量",
     "简单快速 | 适合背景简单的图 | 精度一般"),
]


def _get_rembg_session(model_name):
    if model_name not in _rembg_sessions:
        _rembg_sessions[model_name] = new_session(model_name)
    return _rembg_sessions[model_name]


def _check_model_cached(model_name):
    home = os.path.expanduser("~")
    possible_dirs = [
        os.path.join(home, ".u2net"),
        os.path.join(home, ".rembg"),
    ]
    try:
        from rembg.sessions import sessions_names
        import rembg
        rembg_dir = getattr(rembg, 'HOME', None)
        if rembg_dir:
            possible_dirs.insert(0, rembg_dir)
    except Exception:
        pass

    for d in possible_dirs:
        if os.path.isdir(d):
            pattern = os.path.join(d, f"*{model_name}*")
            if glob.glob(pattern):
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


class PictureCutApp:
    HANDLE_SIZE = 5
    MIN_SELECTION = 5
    MAX_HISTORY = 50

    def __init__(self, root):
        self.root = root
        self.root.title("图集拆分工具 - Picture Cut")
        self.root.geometry("1280x800")
        self.root.minsize(1024, 680)

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
        self.ai_model_var = tk.StringVar(value="u2netp")

        self._ai_processing = False
        self._progress_timer = None
        self._progress_value = 0
        self.status_var = tk.StringVar(value="就绪")

        self._build_ui()
        self._bind_events()
        self._push_history()

    def _build_ui(self):
        self._build_toolbar()
        self._build_main_area()
        self._build_status_bar()

    def _build_toolbar(self):
        toolbar = ttk.Frame(self.root, padding=(6, 3))
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(toolbar, text="图集拆分工具", font=("Microsoft YaHei UI", 12, "bold")).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        ttk.Button(toolbar, text="上传图片", command=self._upload_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="适应窗口", command=self._fit_to_window).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="1:1", command=self._zoom_original, width=4).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)
        ttk.Label(toolbar, text="缩放:").pack(side=tk.LEFT)
        self.zoom_var = tk.StringVar(value="100%")
        zoom_entry = ttk.Entry(toolbar, textvariable=self.zoom_var, width=6)
        zoom_entry.pack(side=tk.LEFT, padx=2)
        zoom_entry.bind("<Return>", self._on_zoom_entry)
        ttk.Button(toolbar, text="+", width=3, command=lambda: self._zoom_by(1.25)).pack(side=tk.LEFT, padx=1)
        ttk.Button(toolbar, text="-", width=3, command=lambda: self._zoom_by(0.8)).pack(side=tk.LEFT, padx=1)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)
        ttk.Checkbutton(toolbar, text="网格", variable=self.grid_visible, command=self._redraw_canvas).pack(side=tk.LEFT, padx=2)
        ttk.Label(toolbar, text="大小:").pack(side=tk.LEFT)
        gs = ttk.Spinbox(toolbar, from_=8, to=256, increment=8, textvariable=self.grid_size, width=4, command=self._redraw_canvas)
        gs.pack(side=tk.LEFT, padx=2)
        gs.bind("<Return>", lambda e: self._redraw_canvas())
        ttk.Checkbutton(toolbar, text="吸附", variable=self.snap_to_grid).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)
        ttk.Button(toolbar, text="导出ZIP", command=self._export_all).pack(side=tk.LEFT, padx=2)

    def _build_status_bar(self):
        status_frame = ttk.Frame(self.root, padding=(6, 2))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status_frame, textvariable=self.status_var, font=("Microsoft YaHei UI", 8),
                  foreground="#888").pack(side=tk.LEFT)

    def _build_main_area(self):
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=2)

        canvas_frame = ttk.LabelFrame(main, text="画布 (左键框选/移动 | 右键平移 | 滚轮缩放 | Ctrl+点击多选)", padding=3)
        main.add(canvas_frame, weight=3)

        self.canvas = tk.Canvas(canvas_frame, bg="#2b2b2b", cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        right_outer = ttk.Frame(main, width=270)
        main.add(right_outer, weight=0)

        r_canvas = tk.Canvas(right_outer, highlightthickness=0, width=260)
        r_scroll = ttk.Scrollbar(right_outer, orient=tk.VERTICAL, command=r_canvas.yview)
        self.right_inner = ttk.Frame(r_canvas)

        self.right_inner.bind("<Configure>", lambda e: r_canvas.configure(scrollregion=r_canvas.bbox("all")))
        r_canvas.create_window((0, 0), window=self.right_inner, anchor=tk.NW)
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
        self._build_align_panel(self.right_inner)
        self._build_export_panel(self.right_inner)

    def _build_selection_panel(self, parent):
        sel_frame = ttk.LabelFrame(parent, text="选区列表", padding=4)
        sel_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 3))

        btn_row = ttk.Frame(sel_frame)
        btn_row.pack(fill=tk.X, pady=(0, 3))
        ttk.Button(btn_row, text="删除", command=self._delete_selected, width=6).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_row, text="清除全部", command=self._clear_selections, width=8).pack(side=tk.LEFT, padx=1)

        list_frame = ttk.Frame(sel_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.sel_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, font=("Consolas", 10),
                                       yscrollcommand=scrollbar.set, activestyle="none", height=5)
        scrollbar.config(command=self.sel_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sel_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.sel_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        info_frame = ttk.Frame(sel_frame)
        info_frame.pack(fill=tk.X, pady=(3, 0))
        self.sel_info_var = tk.StringVar(value="未选中选区 | Ctrl+点击多选")
        ttk.Label(info_frame, textvariable=self.sel_info_var, font=("Microsoft YaHei UI", 8), foreground="gray").pack(fill=tk.X)

    def _build_pixel_input_panel(self, parent):
        px_frame = ttk.LabelFrame(parent, text="像素选区", padding=4)
        px_frame.pack(fill=tk.X, pady=(0, 3))

        row1 = ttk.Frame(px_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="宽:").pack(side=tk.LEFT)
        self.input_w_entry = ttk.Entry(row1, textvariable=self.input_w, width=6)
        self.input_w_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(row1, text="高:").pack(side=tk.LEFT, padx=(6, 0))
        self.input_h_entry = ttk.Entry(row1, textvariable=self.input_h, width=6)
        self.input_h_entry.pack(side=tk.LEFT, padx=2)

        ttk.Button(px_frame, text="在视图中心创建", command=self._create_pixel_selection_center).pack(fill=tk.X, pady=2)

    def _build_bg_remove_panel(self, parent):
        bg_frame = ttk.LabelFrame(parent, text="去底图 (背景→透明)", padding=4)
        bg_frame.pack(fill=tk.X, pady=(0, 3))

        if HAS_REMBG:
            ttk.Label(bg_frame, text="AI抠图模型:", font=("Microsoft YaHei UI", 8), foreground="gray").pack(anchor=tk.W)
            self.ai_model_combo = ttk.Combobox(bg_frame, textvariable=self.ai_model_var,
                                                values=[m[1] for m in AI_MODELS],
                                                state="readonly", width=28)
            self.ai_model_combo.pack(fill=tk.X, pady=2)
            self.ai_model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

            self.model_desc_var = tk.StringVar(value=AI_MODELS[0][2])
            self.model_desc_label = ttk.Label(bg_frame, textvariable=self.model_desc_var,
                                               font=("Microsoft YaHei UI", 7), foreground="#aaa", wraplength=240)
            self.model_desc_label.pack(anchor=tk.W, pady=(0, 2))

            self.model_cache_var = tk.StringVar(value="")
            ttk.Label(bg_frame, textvariable=self.model_cache_var,
                      font=("Microsoft YaHei UI", 7), foreground="#66aaff").pack(anchor=tk.W)

            row_ai = ttk.Frame(bg_frame)
            row_ai.pack(fill=tk.X, pady=3)
            self.ai_remove_btn = ttk.Button(row_ai, text="AI智能去底", command=self._remove_bg_ai)
            self.ai_remove_btn.pack(side=tk.LEFT, padx=1)
            ttk.Button(row_ai, text="还原原图", command=self._restore_original).pack(side=tk.LEFT, padx=1)

            self.ai_progress = ttk.Progressbar(bg_frame, mode='determinate', length=200, maximum=100)
            self.ai_progress.pack(fill=tk.X, pady=(2, 0))
            self.ai_progress_label = ttk.Label(bg_frame, text="", font=("Microsoft YaHei UI", 7), foreground="#00aaff")
            self.ai_progress_label.pack(anchor=tk.W)

            ttk.Separator(bg_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

            self._on_model_change()
        else:
            ttk.Label(bg_frame, text="AI抠图不可用 (需安装 rembg)", font=("Microsoft YaHei UI", 8), foreground="#cc6666").pack(anchor=tk.W)
            ttk.Separator(bg_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        ttk.Label(bg_frame, text="阈值去底 (简单快速):", font=("Microsoft YaHei UI", 8), foreground="gray").pack(anchor=tk.W)
        row1 = ttk.Frame(bg_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="阈值:").pack(side=tk.LEFT)
        self.threshold_spin = ttk.Spinbox(row1, from_=100, to=255, increment=5, textvariable=self.bg_threshold, width=5)
        self.threshold_spin.pack(side=tk.LEFT, padx=2)
        ttk.Label(row1, text="(RGB>阈值→透明)", font=("Microsoft YaHei UI", 7), foreground="gray").pack(side=tk.LEFT, padx=2)

        row2 = ttk.Frame(bg_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Button(row2, text="阈值去底", command=self._remove_bg_threshold).pack(side=tk.LEFT, padx=1)
        if not HAS_REMBG:
            ttk.Button(row2, text="还原原图", command=self._restore_original).pack(side=tk.LEFT, padx=1)

        self.bg_status_var = tk.StringVar(value="")
        ttk.Label(bg_frame, textvariable=self.bg_status_var, font=("Microsoft YaHei UI", 8), foreground="#00aaff").pack(anchor=tk.W, pady=(2, 0))

    def _build_align_panel(self, parent):
        align_frame = ttk.LabelFrame(parent, text="对齐与排列", padding=4)
        align_frame.pack(fill=tk.X, pady=(0, 3))

        ttk.Label(align_frame, text="对齐 (以首个选中为基准):", font=("Microsoft YaHei UI", 8), foreground="gray").pack(anchor=tk.W)
        row1 = ttk.Frame(align_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Button(row1, text="左对齐", width=6, command=lambda: self._align_selections('left')).pack(side=tk.LEFT, padx=1)
        ttk.Button(row1, text="右对齐", width=6, command=lambda: self._align_selections('right')).pack(side=tk.LEFT, padx=1)
        ttk.Button(row1, text="顶对齐", width=6, command=lambda: self._align_selections('top')).pack(side=tk.LEFT, padx=1)
        ttk.Button(row1, text="底对齐", width=6, command=lambda: self._align_selections('bottom')).pack(side=tk.LEFT, padx=1)

        ttk.Separator(align_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        ttk.Label(align_frame, text="等间距排列:", font=("Microsoft YaHei UI", 8), foreground="gray").pack(anchor=tk.W)
        row2 = ttk.Frame(align_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Button(row2, text="水平等距", width=10, command=lambda: self._distribute_selections('horizontal')).pack(side=tk.LEFT, padx=1)
        ttk.Button(row2, text="垂直等距", width=10, command=lambda: self._distribute_selections('vertical')).pack(side=tk.LEFT, padx=1)

        ttk.Separator(align_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        ttk.Button(align_frame, text="统一尺寸 (与首个选中相同)", command=self._same_size_selections).pack(fill=tk.X, pady=2)

    def _build_export_panel(self, parent):
        exp_frame = ttk.LabelFrame(parent, text="导出设置", padding=4)
        exp_frame.pack(fill=tk.X, pady=(0, 3))

        row1 = ttk.Frame(exp_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="前缀:").pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.naming_prefix, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(row1, text="位数:").pack(side=tk.LEFT, padx=(6, 0))
        ttk.Spinbox(row1, from_=1, to=6, textvariable=self.naming_digits, width=3).pack(side=tk.LEFT, padx=2)

        ttk.Label(exp_frame, text="示例: icon_001.png, icon_002.png ...", font=("Microsoft YaHei UI", 7), foreground="gray").pack(anchor=tk.W)

    def _on_model_change(self, event=None):
        sel_text = self.ai_model_var.get()
        for mid, mlabel, mdesc in AI_MODELS:
            if mid in sel_text or mlabel == sel_text:
                self.ai_model_var.set(mlabel)
                self.model_desc_var.set(mdesc)
                cached = _check_model_cached(mid)
                if cached:
                    self.model_cache_var.set("● 模型已下载，可离线使用")
                else:
                    self.model_cache_var.set("○ 模型未下载，首次使用需联网下载")
                break

    def _get_model_id(self):
        sel_text = self.ai_model_var.get()
        for mid, mlabel, mdesc in AI_MODELS:
            if mid in sel_text or mlabel == sel_text:
                return mid
        return "u2netp"

    def _start_progress(self):
        self._progress_value = 0
        self.ai_progress['value'] = 0
        self.ai_progress_label.config(text="0%")
        self._tick_progress()

    def _tick_progress(self):
        if not self._ai_processing:
            return
        if self._progress_value < 90:
            self._progress_value += 1
            self.ai_progress['value'] = self._progress_value
            self.ai_progress_label.config(text=f"{self._progress_value}%")
            delay = 200 if self._progress_value < 30 else 400
            if self._progress_value > 60:
                delay = 800
            self._progress_timer = self.root.after(delay, self._tick_progress)

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
            messagebox.showinfo("提示", "AI去底需要安装 rembg 库\n\npip install rembg onnxruntime")
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
                f"首次使用需要联网下载模型文件，下载后可永久离线使用。\n\n"
                f"是否继续？（将自动下载模型）"
            )
            if not proceed:
                return

        self._ai_processing = True
        self.ai_remove_btn.config(state="disabled")
        self.bg_status_var.set(f"AI去底处理中 [{model_id}]...")
        self.status_var.set(f"AI去底处理中 [{model_id}]，请稍候...")
        self._start_progress()
        self.root.update_idletasks()

        source_ref = self.source_image

        def do_remove():
            try:
                session = _get_rembg_session(model_id)
                result = remove(source_ref, session=session)
                self.root.after(0, lambda: self._on_ai_remove_done(result, model_id))
            except Exception as e:
                err_str = str(e)
                self.root.after(0, lambda es=err_str: self._on_ai_remove_error(es, model_id))

        threading.Thread(target=do_remove, daemon=True).start()

    def _on_ai_remove_done(self, result, model_name):
        self._stop_progress(success=True)
        self.source_image = result
        self._redraw_canvas()
        self.ai_remove_btn.config(state="normal")
        self.status_var.set(f"AI去底完成 [{model_name}]")
        self.bg_status_var.set(f"AI去底完成 [{model_name}]")
        self._on_model_change()

    def _on_ai_remove_error(self, err_msg, model_name):
        self._stop_progress(success=False)
        self.ai_remove_btn.config(state="normal")
        self.bg_status_var.set(f"AI去底失败 [{model_name}]")
        if self.original_image is not None:
            self.source_image = self.original_image.copy()
            self._redraw_canvas()
        messagebox.showerror(
            "AI去底失败",
            f"模型: {model_name}\n\n"
            f"错误信息:\n{err_msg}\n\n"
            f"可能原因:\n"
            f"• 模型未下载且无法联网 — 请连接网络后重试\n"
            f"• 内存不足 — 尝试使用更小的模型 (如 u2netp)\n"
            f"• 图片格式不支持 — 尝试其他格式\n\n"
            f"已自动还原为原始图片。"
        )

    def _restore_original(self):
        if self.original_image is None:
            messagebox.showinfo("提示", "请先上传图片")
            return
        self._blur_entries()
        self.source_image = self.original_image.copy()
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
            self.canvas.create_text(cx, cy - 15, text="拖拽图片到此处 或 点击「上传图片」",
                                     fill="#888", font=("Microsoft YaHei UI", 16))
            self.canvas.create_text(cx, cy + 20, text="支持 PNG / JPG / BMP / TIFF / WEBP",
                                     fill="#666", font=("Microsoft YaHei UI", 10))
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
                self.canvas.create_line(cx, y1c, cx, y2c, fill="#555", dash=(2, 4))
            x += gs
        y = 0
        while y <= ih:
            _, cy = self._img_to_canvas(0, y)
            if y1c <= cy <= y2c:
                self.canvas.create_line(x1c, cy, x2c, cy, fill="#555", dash=(2, 4))
            y += gs

    def _draw_selections(self):
        selected_set = set(self.selected_indices)
        for i, sel in enumerate(self.selections):
            self._draw_one_selection(sel, i in selected_set)

    def _draw_one_selection(self, sel, is_selected):
        cx1, cy1 = self._img_to_canvas(sel.x1, sel.y1)
        cx2, cy2 = self._img_to_canvas(sel.x2, sel.y2)
        outline_color = "#00ddff" if is_selected else "#ffcc00"
        fill_alpha = "#0a1a2a" if is_selected else "#1a1a0a"
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
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#00ff00", width=2, dash=(4, 4), tags="temp_rect")
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
        self.sel_info_var.set(f"已选中 {count} 个选区 | Ctrl+点击多选" if count else "未选中选区 | Ctrl+点击多选")

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
        if not self.selections:
            messagebox.showinfo("提示", "没有选区可导出，请先框选区域")
            return
        if self.source_image is None:
            messagebox.showinfo("提示", "请先上传图片")
            return
        self._blur_entries()
        save_path = filedialog.asksaveasfilename(
            title="导出ZIP文件", defaultextension=".zip",
            filetypes=[("ZIP压缩包", "*.zip")], initialfile="exported_sprites.zip")
        if not save_path:
            return
        prefix = self.naming_prefix.get().strip() or "icon"
        try:
            digits = int(self.naming_digits.get())
        except (ValueError, tk.TclError):
            digits = 3
        try:
            exported = 0
            with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zf:
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
                    buf = io.BytesIO()
                    cropped.save(buf, format="PNG")
                    zf.writestr(filename, buf.getvalue())
                    exported += 1
            self.status_var.set(f"已导出 {exported} 个选区到 {os.path.basename(save_path)}")
            messagebox.showinfo("导出成功", f"已导出 {exported} 个PNG文件到:\n{save_path}")
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


def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.25)
    except Exception:
        pass
    app = PictureCutApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        try:
            import traceback
            messagebox.showerror("程序异常", f"发生未预期的错误:\n\n{traceback.format_exc()}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
