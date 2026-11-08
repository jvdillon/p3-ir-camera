"""Tk workstation: UI owns presentation; Acquisition exclusively owns USB."""

from dataclasses import asdict
from tkinter import ttk

import queue
import time
import tkinter as tk

import numpy as np

from .acquisition import Acquisition
from .analysis import AnalysisState
from .analysis_ui import AnalysisWorkspace
from .canvas import ThermalCanvas
from .comparison_ui import ComparisonPanel
from .export import (
    load_raw_image,
    load_snapshot,
    save_snapshot,
    snapshot_display_settings,
)
from .help_ui import SHORTCUTS, HelpPanel
from .i18n import Translator, filedialog, messagebox
from .image_ui import image_dialog
from .palette_ui import PalettePanel
from .palettes import PaletteLibrary, TemperaturePalette
from .preferences import Preferences
from .processing import (
    PALETTES,
    DisplaySettings,
    Processor,
    focus_peaking,
    legend_colors,
    orient,
    pixel_label,
    sensor_coordinates,
    temperature,
)
from .scrolling import ScrollArea, install_scroll_routing
from .sidebar import Sidebar
from .widgets import button, caption, checkbox


class ThermalApp:
    """Professional desktop controls with raw-accurate pixel inspection."""

    RECONNECT_DELAY = 2.0

    def __init__(self, root, model="p3", demo=False):
        self.root, self.model, self.demo = root, model, demo
        self.worker = None
        self.frame = None
        self.analysis = AnalysisState()
        self.analysis_shape = None
        self.measured = None
        self.workspace = None
        self.recorder = None
        self.offline = False
        self.snapshot_metadata = {}
        self.library = PaletteLibrary(PALETTES)
        self._combos = {}
        self._render_key = None
        self.processor = Processor()
        self.settings = DisplaySettings()
        self.custom_auto = tk.BooleanVar(value=False)
        self.rotation = 0
        self.preferences = Preferences(model=model)
        self.mirror = self.preferences.mirror
        self.translator = Translator(root, self.preferences.language)
        self.paused = False
        self.closing = False
        self._waiting = False
        self._retry_at = 0.0
        self._connection_error = ""
        self.root.title("P3 Thermal Studio" + (" · DEMO" if demo else ""))
        self.root.minsize(520, 360)
        self._style()
        self._build()
        install_scroll_routing(root)
        self.workspace = AnalysisWorkspace(self)
        self.comparison = ComparisonPanel(self.sidebar, self)
        self.sidebar.add(self.comparison, text="Compare")
        self.help_panel = HelpPanel(self.viewer_stack, self.translator, self.close_help)
        help_menu = ttk.Frame(self.sidebar.viewport, padding=12)
        caption(
            help_menu,
            "Help fills the main workspace. Select another category to return.",
            wraplength=400,
            fill="x",
        )
        self.sidebar.add(help_menu, text="Help")
        settings = ttk.Frame(self.sidebar.viewport, padding=12)
        self.sidebar.add(settings, text="Settings")
        caption(settings, "Language")
        self.language = tk.StringVar(
            value="Polski" if self.preferences.language == "pl" else "English"
        )
        self.language_box = ttk.Combobox(
            settings,
            values=("English", "Polski"),
            textvariable=self.language,
            state="readonly",
        )
        self.language_box.pack(fill="x", pady=8)
        self.language_box.bind("<<ComboboxSelected>>", self.change_language)
        caption(
            settings,
            "Language changes immediately and is remembered.",
            wraplength=400,
            fill="x",
        )
        self.sidebar.on_select = self.show_section
        self.show_section(self.sidebar.selection.get())
        self.translator.refresh()
        self._translation_at = 0.0
        self.root.after_idle(self.initial_layout)
        root.protocol("WM_DELETE_WINDOW", self.close)
        for binding, _, _, action in SHORTCUTS:
            root.bind(binding, lambda event, name=action: self.shortcut(event, name))
        self.connect()
        self._poll_id = self.root.after(40, self.poll)

    def initial_layout(self):
        """Reserve usable space for the default controls and pixel readouts at startup."""
        if self.closing:
            return
        self.root.update_idletasks()
        menu_width = (
            max(
                self.sidebar.pages["View"].winfo_reqwidth(),
                self.sidebar.navigation.winfo_reqwidth(),
            )
            + 12
        )
        info_height = self.info_area.content.winfo_reqheight() + 8
        image_width = max(512, self.info_area.content.winfo_reqwidth())
        content_height = max(
            384 + info_height,
            self.sidebar.pages["View"].winfo_reqheight()
            + self.sidebar.navigation_area.winfo_reqheight(),
        )
        header_height = (
            sum(
                child.winfo_reqheight()
                for child in self.root.winfo_children()
                if child is not self.body
            )
            + 24
        )
        width = min(menu_width + image_width + 12, self.root.winfo_screenwidth() - 80)
        height = min(
            content_height + header_height, self.root.winfo_screenheight() - 100
        )
        self.root.geometry(f"{max(520, width)}x{max(360, height)}")

        # Window-manager geometry arrives asynchronously; place dividers after mapping.
        def place_panes():
            if not self.closing:
                self.body.sashpos(0, max(200, self.body.winfo_width() - menu_width))
                self.image_info_split.sashpos(
                    0, max(80, self.image_info_split.winfo_height() - info_height)
                )

        self.root.after(100, place_panes)

    def _style(self):
        self.root.configure(background="#18212e")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            ".", background="#18212e", foreground="#e7edf5", font=("TkDefaultFont", 10)
        )
        style.configure("TButton", padding=(10, 7), background="#2b3b50")
        style.configure(
            "Section.TButton",
            padding=(8, 3),
            borderwidth=1,
            relief="solid",
            bordercolor="#48596d",
            lightcolor="#48596d",
            darkcolor="#48596d",
            background="#18212e",
            font=("TkDefaultFont", 9),
        )
        style.configure(
            "Selected.Section.TButton", background="#236b89", foreground="#ffffff"
        )
        style.map("Section.TButton", background=[("active", "#30475e")])
        style.configure("TPanedwindow", sashwidth=7)
        style.map("TButton", background=[("active", "#385673")])
        style.configure(
            "TCombobox", fieldbackground="#f0f4f9", foreground="#142030", padding=5
        )
        for widget in ("TEntry", "TSpinbox", "TCombobox"):
            style.configure(
                widget,
                fieldbackground="#f0f4f9",
                foreground="#142030",
                insertcolor="#142030",
            )
            style.map(
                widget,
                foreground=[("disabled", "#526173"), ("readonly", "#142030")],
                fieldbackground=[("readonly", "#f0f4f9"), ("disabled", "#d6dee8")],
            )
        style.configure(
            "Treeview",
            background="#202e40",
            fieldbackground="#202e40",
            foreground="#f3f6fa",
            rowheight=29,
        )
        style.map(
            "Treeview",
            background=[("selected", "#275f91")],
            foreground=[("selected", "#ffffff")],
        )
        style.configure(
            "Treeview.Heading", background="#31465e", foreground="#ffffff", padding=6
        )
        for widget in ("TCheckbutton", "TRadiobutton"):
            style.map(
                widget,
                foreground=[("disabled", "#9caec2"), ("active", "#ffffff")],
                background=[("active", "#26384e")],
            )
        style.configure("TNotebook.Tab", background="#2b3b50", padding=(10, 8))
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#385673")],
            foreground=[("selected", "#ffffff")],
        )
        style.configure("Title.TLabel", font=("TkDefaultFont", 20, "bold"))
        style.configure(
            "Readout.TLabel", font=("TkFixedFont", 12), foreground="#76dfd0"
        )
        style.configure("TLabelframe", padding=12)
        style.configure("TLabelframe.Label", foreground="#91a8c1")

    def _build(self):
        header = ttk.Frame(self.root, padding=(12, 6))
        header.pack(fill="x")
        self.utility_bar = ttk.Frame(header)
        self.utility_bar.pack(side="right")
        ttk.Label(header, text="THERMAL STUDIO", style="Title.TLabel").pack(side="left")
        caption(header, f"  {self.model.upper()} / radiometric workspace", side="left")
        action_area = ScrollArea(self.root, horizontal_only=True)
        action_area.pack(fill="x", padx=10)
        toolbar = action_area.content
        self.connect_button = ttk.Button(
            toolbar, text="Reconnect", command=self.connect
        )
        self.connect_button.pack(side="left", padx=3)
        self.pause_button = ttk.Button(toolbar, text="Freeze", command=self.pause)
        self.pause_button.pack(side="left", padx=3)
        button(toolbar, "Open RAW…", self.open_snapshot, side="left", padx=3)
        button(toolbar, "Save data…", self.export, side="left", padx=3)
        button(toolbar, "Save image…", self.screenshot, side="left", padx=3)
        self.shutter_button = ttk.Button(
            toolbar, text="Shutter / NUC", command=lambda: self.command("shutter", None)
        )
        self.shutter_button.pack(side="left", padx=3)
        body = self.body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True)
        self.viewer_stack = ttk.Frame(body)
        body.add(self.viewer_stack, weight=3)
        left = self.live_panel = ttk.Frame(self.viewer_stack)
        left.pack(fill="both", expand=True)
        self.image_info_split = ttk.Panedwindow(left, orient="vertical")
        self.image_info_split.pack(fill="both", expand=True)
        image_host = ttk.Frame(self.image_info_split, height=480)
        self.image_info_split.add(image_host, weight=4)
        self.canvas = ThermalCanvas(
            image_host,
            self.pixel,
            lambda scale: self.zoom_text.set(
                f"Zoom  {scale * 100:.0f}%  ·  wheel to zoom / drag to pan"
            ),
        )
        self.canvas.on_gesture = self.gesture
        self.canvas.pack(fill="both", expand=True)
        self.info_area = ScrollArea(self.image_info_split, height=150)
        self.image_info_split.add(self.info_area, weight=1)
        info = self.info_area.content
        navigation = ttk.Frame(info, padding=(0, 5))
        navigation.pack(fill="x")
        for label, action in [
            ("−", lambda: self.canvas.zoom(0.8)),
            ("+", lambda: self.canvas.zoom(1.25)),
            ("←", lambda: self.canvas.move_image(-60, 0)),
            ("→", lambda: self.canvas.move_image(60, 0)),
            ("↑", lambda: self.canvas.move_image(0, -60)),
            ("↓", lambda: self.canvas.move_image(0, 60)),
            ("Fit", self.canvas.fit),
            ("Pixels", self.canvas.inspect),
        ]:
            ttk.Button(navigation, text=label, width=5, command=action).pack(
                side="left", padx=2
            )
        self.show_legend = tk.BooleanVar(value=True)
        checkbox(navigation, "Legend", self.show_legend, self.render, side="right")
        self.zoom_text = tk.StringVar(value="Zoom")
        self.pixel_text = tk.StringVar(value="Pixel: move the pointer over the image")
        ttk.Label(info, textvariable=self.zoom_text, padding=(8, 7)).pack(fill="x")
        ttk.Label(
            info, textvariable=self.pixel_text, style="Readout.TLabel", padding=(8, 7)
        ).pack(fill="x")
        self.legend = tk.StringVar()
        ttk.Label(info, textvariable=self.legend, padding=8).pack(fill="x")
        side = ttk.Frame(body, width=540)
        body.add(side, weight=2)
        tabs = self.sidebar = Sidebar(side, side, self.utility_bar)
        tabs.pack(fill="both", expand=True)
        display = ttk.Frame(tabs.viewport, padding=12)
        tabs.add(display, text="View")
        self.mode = self.combo(
            display,
            "Source",
            ["Temperature", "Filtered temperature", "Raw counts", "Factory brightness"],
            self.settings.mode,
        )
        self.palette = self.combo(
            display,
            "Palette",
            list(PALETTES) + sorted(self.library.palettes),
            self.settings.palette,
            lambda: self.palette_changed(self.palette.get()),
        )
        self.fixed_scale = tk.BooleanVar(value=self.settings.range_mode == "Fixed")
        checkbox(
            display,
            "Fixed scale",
            self.fixed_scale,
            self.update_settings,
            anchor="w",
            pady=6,
        )
        self.range_mode = tk.StringVar(
            value="Fixed" if self.fixed_scale.get() else "Auto percentile"
        )
        self.low = tk.StringVar(value="15")
        self.high = tk.StringVar(value="40")
        for label, variable in [("Minimum °C", self.low), ("Maximum °C", self.high)]:
            caption(display, label)
            ttk.Entry(display, textvariable=variable, width=22).pack(
                fill="x", pady=(2, 6)
            )
        button(display, "Apply range", self.update_settings)
        self.palette_note = tk.StringVar(
            value="Factory palette: range follows Auto / Fixed."
        )
        ttk.Label(display, textvariable=self.palette_note, wraplength=290).pack(
            anchor="w", pady=8
        )
        filters = ttk.Frame(tabs.viewport, padding=12)
        tabs.add(filters, text="Filters")
        self.peaking = tk.BooleanVar(value=False)
        self.peaking_threshold = tk.DoubleVar(value=0.35)
        checkbox(
            filters,
            "Focus peaking · green edge overlay",
            self.peaking,
            self.render,
            anchor="w",
            pady=6,
        )
        caption(filters, "Peaking threshold (lower highlights more edges)")
        ttk.Scale(
            filters,
            from_=0.05,
            to=0.95,
            variable=self.peaking_threshold,
            command=lambda _: self.render(),
        ).pack(fill="x")
        self.detail = tk.BooleanVar()
        self.clahe = tk.BooleanVar()
        self.clahe_switch = ttk.Checkbutton(
            filters,
            text="CLAHE · local contrast",
            variable=self.clahe,
            command=self.update_settings,
        )
        self.clahe_switch.pack(anchor="w", pady=8)
        self.dde_switch = ttk.Checkbutton(
            filters,
            text="DDE · edge sharpening",
            variable=self.detail,
            command=self.update_settings,
        )
        self.dde_switch.pack(anchor="w", pady=8)
        self.dde_strength = tk.DoubleVar(value=1.5)
        self.dde_caption = tk.StringVar(value="DDE strength: 1.50")
        ttk.Label(filters, textvariable=self.dde_caption).pack(anchor="w")
        self.dde_slider = ttk.Scale(
            filters,
            from_=0.0,
            to=4.0,
            variable=self.dde_strength,
            command=lambda value: self.update_settings(),
        )
        self.dde_slider.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(
            filters, text="X³ · unavailable in current USB driver", state="disabled"
        ).pack(anchor="w", pady=8)
        caption(
            filters,
            "X³ needs a verified camera command or the manufacturer's algorithm. It is not ordinary zoom.",
            wraplength=290,
            anchor="w",
            pady=8,
        )
        caption(filters, "Temporal filter: current-frame weight")
        self.alpha = tk.DoubleVar(value=0.35)
        ttk.Scale(
            filters,
            from_=0.05,
            to=1,
            variable=self.alpha,
            command=lambda value: self.update_settings(),
        ).pack(fill="x")
        caption(
            filters,
            "Temporal filtering applies only to Filtered temperature. Pixel measurements stay independent of display filters.\n\nCLAHE and DDE work with every palette; enhanced colors no longer indicate exact temperature thresholds.",
            wraplength=290,
            anchor="w",
            pady=15,
        )
        view = ttk.Frame(tabs.viewport, padding=12)
        tabs.add(view, text="Sensor")
        for label, fn in [
            ("Fit image", self.canvas.fit),
            ("Inspect pixels · Ctrl+X", self.canvas.inspect),
            ("Rotate 90°", self.rotate),
            ("Mirror", self.flip),
        ]:
            button(view, label, fn, fill="x", pady=2)
        self.remember_mirror = tk.BooleanVar(value=self.preferences.remember)
        checkbox(
            view,
            "Remember mirror for this camera",
            self.remember_mirror,
            self.save_preferences,
            anchor="w",
            pady=6,
        )
        self.spots = tk.BooleanVar(value=True)
        checkbox(
            view,
            "Minimum / maximum markers",
            self.spots,
            self.render,
            anchor="w",
            pady=5,
        )
        self.gain = self.combo(
            view,
            "Sensor gain",
            ["HIGH", "LOW"],
            "HIGH",
            lambda: self.command("gain", self.gain.get()),
        )
        ttk.Label(
            view,
            text="LOW: experimental radiometry. Tested firmware\nreturned an offset that NUC did not resolve.",
            foreground="#f4ba76",
            wraplength=310,
        ).pack(anchor="w", pady=8)
        if self.demo:
            self.shutter_button.state(["disabled"])
        self.statistics = tk.StringVar(value="No frame")
        ttk.Label(
            view, textvariable=self.statistics, justify="left", style="Readout.TLabel"
        ).pack(anchor="w", pady=8)
        self.palette_panel = PalettePanel(tabs, self.library, self.palette_changed)
        tabs.add(self.palette_panel, text="Palettes")
        self.status = tk.StringVar(value="Starting…")
        ttk.Label(self.root, textvariable=self.status, padding=(16, 10)).pack(
            fill="x", side="bottom", before=body
        )

    def combo(self, parent, label, values, value, callback=None):
        caption(parent, label)
        variable = tk.StringVar(value=value)
        widget = ttk.Combobox(
            parent, textvariable=variable, values=values, state="readonly", width=25
        )
        widget.pack(fill="x", pady=(3, 9))
        self._combos[label] = widget

        def selected(_event):
            self.translator.sync_source(widget)
            (callback or self.update_settings)()

        widget.bind("<<ComboboxSelected>>", selected)
        return variable

    def update_settings(self):
        try:
            low, high = float(self.low.get()), float(self.high.get())
            if not np.isfinite([low, high]).all() or high <= low:
                raise ValueError()
        except ValueError:
            self.status.set(
                "Invalid range: enter finite values with maximum > minimum."
            )
            return
        custom = self.library.palettes.get(self.palette.get())
        filters_active = self.clahe.get() or self.detail.get()
        if (custom or self.palette.get() == "White hot / red peak") and filters_active:
            note = "Enhancement is visual; colors no longer represent exact thresholds."
        else:
            note = "Auto scales to the frame; Fixed uses the min/max fields below."
        self.palette_note.set(note)
        self.clahe_switch.state(["!disabled"])
        self.dde_switch.state(["!disabled"])
        self.dde_slider.state(["!disabled"])
        self.dde_caption.set(f"DDE strength: {self.dde_strength.get():.2f}")
        self.settings = DisplaySettings(
            self.mode.get(),
            self.palette.get(),
            "Fixed" if self.fixed_scale.get() else "Auto percentile",
            low,
            high,
            self.alpha.get(),
            self.detail.get(),
            self.clahe.get(),
            self.dde_strength.get(),
            self.custom_auto.get(),
            self.settings.enhance_custom_palette,
            self.settings.enhancements_enabled,
        )
        self.processor.reset()
        self._render_key = None
        self.render()

    def connect(self):
        """Start one connection attempt; retries never overlap USB cleanup."""
        if self.closing:
            return
        if self.offline:
            self.return_live()
            return
        if self.worker is not None and self.worker.is_alive():
            self.status.set("Camera worker is already active.")
            return
        self._waiting = False
        self.paused = False
        self.pause_button.configure(text="Freeze")
        self.processor.reset()
        self._render_key = None
        self.worker = Acquisition(self.model, self.demo)
        self.worker.recorder = self.recorder
        self.worker.start()

    def ensure_comparison_stream(self):
        """Reconnect comparison acquisition while preserving an imported working frame."""
        if self.closing or (self.worker and self.worker.is_alive()):
            return
        if time.monotonic() < self._retry_at:
            return
        self._retry_at = time.monotonic() + self.RECONNECT_DELAY
        self.worker = Acquisition(self.model, self.demo)
        self.worker.recorder = self.recorder
        self.worker.start()

    def command(self, name, value):
        if self.offline:
            self.status.set("Imported snapshot · return to live to control the camera.")
        elif self.demo:
            self.status.set("Sensor controls are unavailable in demo mode.")
        elif self.worker and self.worker.is_alive():
            self.worker.commands.put((name, value))
            self.status.set(f"Queued: {name} {value or ''}")
        else:
            self.status.set("Reconnect the camera before issuing a sensor command.")

    def poll(self):
        if self.closing:
            return
        if self.worker:
            while True:
                try:
                    event = self.worker.events.get_nowait()
                    if not self.offline:
                        self.status.set(event)
                    if event.startswith("Disconnected / error:"):
                        self._connection_error = event
                    if event.startswith("Applied:") and not self.offline:
                        self.processor.reset()
                        self._render_key = None
                except queue.Empty:
                    break
            try:
                frame = self.worker.frames.get_nowait()
                self.comparison.receive_live(frame)
                if not self.paused and not self.offline:
                    self.frame = frame
                    self.render()
            except queue.Empty:
                pass
            alive = self.worker.is_alive()
            self.connect_button.state(
                ["disabled"] if alive and not self.offline else ["!disabled"]
            )
            if not alive:
                if self.comparison.latest_live is not None:
                    self.comparison.clear_live()
                if self.offline and self.comparison.needs_live and not self.demo:
                    self.ensure_comparison_stream()
            if not alive and not self.demo and not self.offline:
                if not self._waiting:
                    self._waiting = True
                    self._retry_at = time.monotonic() + self.RECONNECT_DELAY
                    self.frame = None
                    self.measured = None
                    self.canvas.clear("No camera · waiting for reconnection…")
                    self.pixel_text.set("Pixel: no camera")
                    self.statistics.set("No camera connected")
                    self.legend.set("")
                    self.status.set(
                        "No camera · retrying automatically. " + self._connection_error
                    )
                elif time.monotonic() >= self._retry_at:
                    self.connect()
        if self.workspace is not None:
            self.workspace.update_values()
        if time.monotonic() >= self._translation_at:
            self.translator.refresh()
            self._translation_at = time.monotonic() + 0.2
        self._poll_id = self.root.after(40, self.poll)

    def render(self):
        if self.frame is None:
            return
        custom = self.library.palettes.get(self.settings.palette)
        if (
            self.analysis_shape is not None
            and self.analysis_shape != self.frame.raw.shape
        ):
            self.analysis.layers.clear()
            self.analysis.regions.clear()
            self.analysis.touch()
            if self.workspace:
                self.workspace.undo.clear()
                self.workspace.refresh_state()
        self.analysis_shape = self.frame.raw.shape
        key = (
            id(self.frame),
            repr(self.settings),
            repr(custom),
            self.analysis.revision,
            self.correction_active,
        )
        if key != self._render_key:
            self.measured = (
                self.analysis.celsius(self.frame.raw)
                if self.correction_active
                else temperature(self.frame.raw)
            )
            self._rendered = self.processor.render(
                self.frame.raw,
                self.frame.brightness,
                self.settings,
                custom,
                self.measured,
            )
            self._render_key = key
        assert self.measured is not None
        rgb, limits = self._rendered
        rgb = rgb.copy()
        invalid = ~np.isfinite(self.measured)
        rgb[invalid] = (80, 80, 80)
        if self.analysis.isotherm:
            selected = (self.measured >= self.analysis.iso_min) & (
                self.measured <= self.analysis.iso_max
            )
            color = self.analysis.iso_color
            rgb[selected] = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
        self.export_rgb = orient(rgb, self.rotation, self.mirror).copy()
        if (
            self.workspace
            and self.canvas.tool in ("Brush", "Eraser")
            and self.workspace.show_mask.get()
        ):
            index = self.workspace.selected_layer()
            if index is not None and index < len(self.analysis.layers):
                mask = self.analysis.layers[index].mask
                rgb[mask] = (0.5 * rgb[mask] + 0.5 * np.array([0, 255, 255])).astype(
                    np.uint8
                )
        if self.peaking.get():
            rgb = focus_peaking(rgb, self.frame.raw, self.peaking_threshold.get())
        raw = orient(self.frame.raw, self.rotation, self.mirror)
        coords = sensor_coordinates(self.frame.raw.shape, self.rotation, self.mirror)
        self.canvas.regions = self.analysis.regions
        oriented_measured = orient(self.measured, self.rotation, self.mirror)
        self.canvas.native_measurements = self.measured
        self.canvas.measurements = oriented_measured
        self.canvas.corrected = self.correction_active
        self.canvas.markers = self.spots.get()
        self.canvas.show_legend = self.show_legend.get()
        unit = "counts" if self.settings.mode == "Raw counts" else "°C"
        bands = self.processor.temperature_bands
        self.canvas.legend_data = (
            (legend_colors(self.settings, custom), *limits, unit)
            if bands is None and limits
            else (legend_colors(self.settings, custom), 0.0, 255.0, "°C min/max", bands)
        )
        self.canvas.set_frame(
            orient(rgb, self.rotation, self.mirror),
            raw,
            coords,
        )
        data = self.measured[np.isfinite(self.measured)]
        minimum, maximum, mean = (
            (data.min(), data.max(), data.mean()) if data.size else (float("nan"),) * 3
        )
        label = (
            "CORRECTED · ESTIMATE" if self.correction_active else "SENSOR TEMPERATURE"
        )
        self.statistics.set(
            f"{label}\nMin   {minimum:.6f} °C\nMax   {maximum:.6f} °C\nMean  {mean:.6f} °C\n\n{self.frame.raw.shape[1]} × {self.frame.raw.shape[0]} pixels\nNative step: 0.015625 K"
        )
        unit = (
            "counts"
            if self.settings.mode == "Raw counts"
            else "brightness"
            if self.settings.mode == "Factory brightness"
            else "°C"
        )
        self.legend.set(
            f"Palette range: {limits[0]:.6f} → {limits[1]:.6f} {unit}"
            if limits
            else "Legend: observed °C min/max in each color band; local contrast can make ranges overlap."
        )

    def pixel(self, sample):
        if sample is None:
            self.pixel_text.set("Pixel: move the pointer over the image")
        else:
            x, y, raw = sample
            text = f"Pixel ({x}, {y})   {pixel_label(raw)}   RAW {raw}"
            if self.correction_active and self.canvas.native_measurements is not None:
                text += (
                    f"\nCorrected estimate ≈ "
                    f"{self.canvas.native_measurements[y, x]:.3f} °C"
                )
            self.pixel_text.set(text)

    def pause(self):
        if self.offline:
            self.return_live()
            return
        self.paused = not self.paused
        self.pause_button.configure(text="Resume" if self.paused else "Freeze")
        self.render()
        self.status.set(
            "Frozen frame · acquisition continues"
            if self.paused
            else "Live view resumed"
        )

    def rotate(self):
        self.rotation = (self.rotation + 1) % 4
        self.canvas.auto_fit = True
        self.render()

    def flip(self):
        self.mirror = not self.mirror
        if not self.offline:
            self.save_preferences()
        self.render()

    def save_preferences(self):
        try:
            self.preferences.save(self.mirror, self.remember_mirror.get())
        except (OSError, ValueError) as exc:
            self.status.set(f"Cannot save preferences: {exc}")

    @property
    def correction_active(self):
        """Live correction is an explicit opt-in; frozen analysis remains available."""
        return self.analysis.radiometry.enabled and (
            self.paused
            or self.offline
            or bool(self.workspace and self.workspace.live_correction.get())
        )

    def export(self):
        if self.frame is None:
            self.status.set("No frame to save.")
            return
        frame = self.frame
        metadata = self.capture_metadata()
        corrected = self.analysis.celsius(frame.raw) if self.correction_active else None
        path = filedialog.asksaveasfilename(
            defaultextension=".npz", filetypes=[("Thermal snapshot", "*.npz")]
        )
        if path:
            try:
                save_snapshot(path, frame, metadata, corrected_celsius=corrected)
                self.status.set(f"Saved thermal data: {path}")
            except OSError as exc:
                messagebox.showerror("Save failed", str(exc))

    def capture_metadata(self):
        metadata = {
            **self.snapshot_metadata,
            "snapshot_version": 2,
            "model": self.snapshot_metadata.get("model", self.model)
            if self.offline
            else self.model,
            "demo": self.snapshot_metadata.get("demo", False)
            if self.offline
            else self.demo,
            "display": asdict(self.settings),
            "rotation_quarters_ccw": self.rotation,
            "mirror": self.mirror,
        }
        custom = self.library.palettes.get(self.settings.palette)
        metadata["palette_spec"] = custom.to_dict() if custom else None
        metadata["analysis"] = self.analysis.to_dict()
        return metadata

    def show_section(self, name):
        """Give comparison the image area while retaining sidebar navigation."""
        self.help_panel.pack_forget()
        if name == "Help":
            self.live_panel.pack_forget()
            self.comparison.images.pack_forget()
            self.help_panel.pack(fill="both", expand=True)
        elif name == "Compare":
            self.live_panel.pack_forget()
            self.comparison.images.pack(fill="both", expand=True)
        else:
            self.comparison.images.pack_forget()
            self.live_panel.pack(fill="both", expand=True)
            if name == "View" and self.workspace:
                self.workspace.set_tool("Pan")

    def open_workspace(self):
        if self.workspace is None:
            self.workspace = AnalysisWorkspace(self)
            self.sidebar.select("Measurements")
        else:
            self.sidebar.select(
                "View"
                if self.sidebar.selection.get()
                in ("Measurements", "RAW editing", "Video")
                else "Measurements"
            )
            self.workspace.refresh_state()

    def gesture(self, phase, tool, start, end):
        if self.workspace:
            self.workspace.gesture(phase, tool, start, end)

    def screenshot(self):
        if self.frame is None or self.canvas.rgb is None:
            self.status.set("No image to save.")
            return
        image_dialog(
            self.root,
            self.frame,
            self.export_rgb.copy(),
            lambda path: self.status.set(f"Saved image: {path}"),
            self.canvas.legend_data if self.show_legend.get() else None,
            host=self.sidebar,
        )

    def palette_changed(self, name):
        self._combos["Palette"].configure(
            values=list(PALETTES) + sorted(self.library.palettes)
        )
        if name:
            self.palette.set(name)
        elif (
            self.palette.get() not in PALETTES
            and self.palette.get() not in self.library.palettes
        ):
            self.palette.set("Inferno")
        self.update_settings()

    def open_snapshot(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Radiometric data", "*.npz *.npy *.png"),
                ("Full thermal snapshot", "*.npz"),
                ("Standalone 16-bit RAW", "*.png *.npy"),
            ]
        )
        if path:
            try:
                self.import_snapshot(path)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                messagebox.showerror("Cannot open snapshot", str(exc))

    def import_snapshot(self, path):
        """Enter offline analysis; background camera activity cannot replace this frame."""
        from pathlib import Path

        frame, metadata = (
            load_raw_image(path)
            if Path(path).suffix.lower() in (".png", ".npy")
            else load_snapshot(path)
        )
        self.apply_import(frame, metadata)
        self.status.set(f"Imported frozen frame · {path}")

    def apply_import(self, frame, metadata):
        """Restore a snapshot or the first frame of a radiometric sequence."""
        restored_analysis = AnalysisState.from_dict(
            metadata.get("analysis"), frame.raw.shape
        )
        restored = snapshot_display_settings(metadata)
        rotation = metadata.get("rotation_quarters_ccw", 0)
        mirror = metadata.get("mirror", False)
        if (
            type(rotation) is not int
            or rotation not in range(4)
            or type(mirror) is not bool
        ):
            raise ValueError("Invalid snapshot orientation")
        palette_data = metadata.get("palette_spec")
        selected = (
            restored.palette
            if restored.palette in PALETTES or restored.palette in self.library.palettes
            else "Inferno"
        )
        if palette_data is not None:
            palette = TemperaturePalette.from_dict(palette_data)
            # Never replace an existing custom preset merely by opening a snapshot.
            original = palette.name
            index = 1
            while palette.name in PALETTES or (
                palette.name in self.library.palettes
                and self.library.palettes[palette.name] != palette
            ):
                index += 1
                palette = TemperaturePalette(
                    f"{original[:65]} ({index})", palette.stops, palette.interpolation
                )
            self.library.save(palette)
            self.palette_panel.refresh()
            selected = palette.name
        if self.workspace:
            self.workspace.stop_playback()
        self.analysis = restored_analysis
        self.analysis_shape = frame.raw.shape
        self.offline = True
        self.paused = True
        self.snapshot_metadata = metadata
        self.frame = frame
        self.rotation = rotation
        self.mirror = mirror
        sources = ["Temperature", "Filtered temperature", "Raw counts"]
        if metadata.get("brightness_origin") != "derived_from_raw":
            sources.append("Factory brightness")
        self._combos["Source"].configure(values=sources)
        self.mode.set(restored.mode if restored.mode in sources else "Temperature")
        self.fixed_scale.set(restored.range_mode == "Fixed")
        self.range_mode.set("Fixed" if self.fixed_scale.get() else "Auto percentile")
        self.low.set(str(restored.minimum))
        self.high.set(str(restored.maximum))
        self.alpha.set(restored.alpha)
        self.clahe.set(restored.clahe)
        self.detail.set(restored.detail)
        self.dde_strength.set(restored.dde_strength)
        self.custom_auto.set(restored.custom_auto_scale)
        self.settings = restored
        self.canvas.auto_fit = True
        self.pause_button.configure(text="Return to live")
        self._render_key = None
        self.processor.reset()
        self.palette_changed(selected)
        if self.workspace:
            self.workspace.undo.clear()
            self.workspace.refresh_state()
        self.status.set("Imported radiometric frame")

    def return_live(self):
        if self.workspace:
            self.workspace.stop_playback()
        self.offline = False
        self.paused = False
        self.mirror = self.preferences.mirror
        self.snapshot_metadata = {}
        self._combos["Source"].configure(
            values=[
                "Temperature",
                "Filtered temperature",
                "Raw counts",
                "Factory brightness",
            ]
        )
        self._waiting = False
        self.pause_button.configure(text="Freeze")
        self.processor.reset()
        self._render_key = None
        self.frame = None
        self.measured = None
        self.canvas.clear("Waiting for live camera…")
        self.pixel_text.set("Pixel: waiting for live camera")
        self.statistics.set("Waiting for live camera")
        self.legend.set("")
        self.status.set("Returning to live camera…")
        if self.worker is None or not self.worker.is_alive():
            self.connect()

    def help(self):
        self.help_panel.refresh()
        self.sidebar.select("Help")

    def close_help(self):
        self.sidebar.select("View")

    def change_language(self, event=None):
        selected = self.language_box.get() if event is not None else self.language.get()
        language = "pl" if selected == "Polski" else "en"
        try:
            self.preferences.save(
                self.preferences.mirror, self.remember_mirror.get(), language
            )
        except (OSError, ValueError) as exc:
            self.status.set(f"Cannot save preferences: {exc}")
            return
        self.translator.language = language
        if self.workspace:
            self.workspace.refresh_state()
        self.translator.refresh()
        self.help_panel.refresh()
        self.canvas.redraw()
        self.comparison.render()

    def shortcut(self, event, action):
        """Preserve text editing and direct image shortcuts to the active comparison slot."""
        if (
            isinstance(
                event.widget, (tk.Text, tk.Entry, ttk.Entry, ttk.Combobox, ttk.Spinbox)
            )
            and action != "help"
        ):
            return None
        canvas = (
            (
                self.comparison.split_canvas
                if self.comparison.split_view.get()
                else self.comparison.canvases[self.comparison.active_index]
            )
            if self.sidebar.selection.get() == "Compare"
            else self.canvas
        )
        if action == "fit":
            canvas.gesture_start = canvas.gesture_last = None
            canvas.fit()
        elif action == "inspect":
            canvas.inspect()
        elif action == "zoom_in":
            canvas.zoom(1.25)
        elif action == "zoom_out":
            canvas.zoom(0.8)
        elif action == "save":
            self.export()
        elif action == "help":
            self.help()
        return "break"

    def close(self):
        """Keep event processing alive until the USB owner completes cleanup."""
        self.closing = True
        if self.workspace:
            self.workspace.stop_playback()
        self.root.after_cancel(self._poll_id)
        self.status.set("Closing · stopping stream and releasing USB…")
        if self.worker:
            self.worker.stop_event.set()
        self._await_close()

    def _await_close(self):
        if self.worker and self.worker.is_alive():
            self.root.after(50, self._await_close)
        elif self.recorder and self.recorder.is_alive():
            self.recorder.stop()
            self.root.after(50, self._await_close)
        else:
            # Dispose pending Tk/Matplotlib redraws before their widget commands vanish.
            for timer in self.root.tk.call("after", "info"):
                self.root.tk.call("after", "cancel", timer)
            self.root.destroy()
