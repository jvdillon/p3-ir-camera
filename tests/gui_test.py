"""Opt-in real Tk smoke test: P3_GUI_TEST=1 python -m pytest tests/gui_test.py."""

import contextlib
import os
import time

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("P3_GUI_TEST") != "1", reason="requires desktop session"
)


@pytest.mark.filterwarnings("ignore:Due to '_pack_'.*:DeprecationWarning")
def test_desktop_zoom_sources_and_shutdown():
    import tkinter as tk

    import numpy as np

    from p3_thermal.app import ThermalApp

    root = tk.Tk()
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    app = ThermalApp(root, demo=os.environ.get("P3_GUI_CAMERA") != "1")
    try:
        deadline = time.monotonic() + 12
        while app.frame is None and time.monotonic() < deadline:
            root.update()
            time.sleep(0.02)
        assert app.frame is not None
        app.pause()
        for mode in (
            "Temperature",
            "Filtered temperature",
            "Raw counts",
            "Factory brightness",
        ):
            app.mode.set(mode)
            app.update_settings()
            root.update()
        app.mode.set("Factory brightness")
        for palette in ("White hot", "White hot / red peak", "Inferno"):
            app.palette.set(palette)
            app.palette_changed(palette)
            app.update_settings()
            assert app.settings.mode == "Factory brightness"
        app.clahe.set(True)
        app.update_settings()
        assert app.canvas.legend_data[3] == "°C min/max"
        assert len(app.canvas.legend_data[4]) == 5
        app.detail.set(True)
        app.dde_strength.set(3.0)
        app.update_settings()
        assert app.settings.dde_strength == 3.0
        app.canvas.inspect()
        root.update()
        labels = [
            app.canvas.itemcget(item, "text")
            for item in app.canvas.find_all()
            if app.canvas.type(item) == "text"
        ]
        assert any("°C" in label and "." in label for label in labels)
        app.rotate()
        app.flip()
        app.canvas.pick(
            app.canvas.offset[0] + 0.5 * app.canvas.pixel_scale,
            app.canvas.offset[1] + 0.5 * app.canvas.pixel_scale,
        )
        expected_measurements = np.fliplr(np.rot90(app.measured, 1))
        np.testing.assert_array_equal(app.canvas.measurements, expected_measurements)
        np.testing.assert_array_equal(app.canvas.native_measurements, app.measured)
        app.analysis.radiometry.enabled = True
        app.render()
        app.pixel((0, 0, int(app.canvas.raw[0, 0])))
        assert f"{app.measured[0, 0]:.3f}" in app.pixel_text.get()
        assert "RAW" in app.pixel_text.get()
        assert not errors
    finally:
        app.close()
        deadline = time.monotonic() + 4
        while app.worker.is_alive() and time.monotonic() < deadline:
            root.update()
            time.sleep(0.02)
        assert not app.worker.is_alive()


def test_disconnect_waits_reconnects_and_close_cancels_retries(monkeypatch):
    import queue
    import threading
    import tkinter as tk

    import numpy as np

    from p3_thermal import app as module
    from p3_thermal.acquisition import Frame

    workers = []

    class FakeAcquisition:
        def __init__(self, *args):
            self.frames = queue.Queue()
            self.events = queue.Queue()
            self.stop_event = threading.Event()
            self.alive = False
            workers.append(self)

        def start(self):
            # First attempt fails (app starts without a camera).
            self.alive = len(workers) > 1
            if self.alive:
                self.frames.put(
                    Frame(
                        np.full((8, 8), 19000, np.uint16),
                        np.zeros((8, 8), np.uint8),
                        time.monotonic(),
                    )
                )
            else:
                self.events.put("Disconnected / error: camera not found")

        def is_alive(self):
            return self.alive and not self.stop_event.is_set()

    monkeypatch.setattr(module, "Acquisition", FakeAcquisition)
    monkeypatch.setattr(module.ThermalApp, "RECONNECT_DELAY", 60.0)
    root = tk.Tk()
    app = module.ThermalApp(root)

    def until(predicate):
        deadline = time.monotonic() + 2
        while not predicate() and time.monotonic() < deadline:
            root.update()
            time.sleep(0.01)
        assert predicate()

    try:
        until(lambda: app._waiting)
        assert root.winfo_exists() and not app.closing
        assert "No camera" in app.status.get()
        app._retry_at = 0
        until(lambda: app.frame is not None)
        assert len(workers) == 2
        app.paused = True
        workers[-1].alive = False
        until(lambda: app._waiting)
        assert app.frame is None and app.canvas.raw is None
        assert not app.closing
        app._retry_at = 0
        until(lambda: app.frame is not None)
        assert len(workers) == 3 and not app.paused
        workers[-1].alive = False
        until(lambda: app._waiting)
        count = len(workers)
        app.close()
        time.sleep(0.2)
        assert len(workers) == count
    finally:
        if not app.closing:
            app.close()


@pytest.fixture(autouse=True)
def isolated_palette_config(monkeypatch, tmp_path):
    """GUI tests never read or overwrite the user's saved palettes."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    yield
    import tkinter as tk

    deadline = time.monotonic() + 5
    while (
        getattr(tk, "_default_root", None) is not None and time.monotonic() < deadline
    ):
        with contextlib.suppress(tk.TclError):
            tk._default_root.update()
        time.sleep(0.01)
    remaining = getattr(tk, "_default_root", None)
    if remaining is not None:
        captured = [
            cell.cell_contents
            for cell in (
                getattr(remaining.report_callback_exception, "__closure__", None) or ()
            )
        ]
        remaining.destroy()
        pytest.fail(f"Window cleanup did not finish: {captured}")


def test_imported_frame_survives_live_frames_and_disconnect(monkeypatch, tmp_path):
    from dataclasses import asdict

    import tkinter as tk

    import numpy as np

    from p3_thermal.acquisition import Frame
    from p3_thermal.app import ThermalApp
    from p3_thermal.export import save_snapshot
    from p3_thermal.palettes import TemperaturePalette
    from p3_thermal.processing import DisplaySettings

    root = tk.Tk()
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    app = ThermalApp(root, demo=True)
    raw = np.arange(192, dtype=np.uint16).reshape(12, 16) + 19000
    frame = Frame(raw, np.zeros_like(raw, dtype=np.uint8), 10)
    palette = TemperaturePalette.from_dict(
        {
            "version": 1,
            "name": "Imported board",
            "stops": [[20, "#000000"], [30, "#FFFFFF"]],
        }
    )
    path = tmp_path / "capture.npz"
    save_snapshot(
        path,
        frame,
        {
            "snapshot_version": 2,
            "palette_spec": palette.to_dict(),
            "display": asdict(DisplaySettings(palette=palette.name)),
            "rotation_quarters_ccw": 1,
            "mirror": True,
        },
    )
    try:
        root.update()
        app.import_snapshot(path)
        app.canvas.inspect()
        root.update()
        imported_rgb = app.canvas.rgb.copy()
        app.palette.set("White hot / red peak")
        app.update_settings()
        assert not np.array_equal(imported_rgb, app.canvas.rgb)
        np.testing.assert_array_equal(app.frame.raw, raw)
        app.canvas.move_image(60, -60)
        app.worker.stop_event.set()
        app.worker.join(2)
        app.demo = False  # exercise the same disconnected-camera path as live mode
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            root.update()
            time.sleep(0.01)
        assert app.offline and app.paused and not app.closing
        np.testing.assert_array_equal(app.frame.raw, raw)
        assert app.canvas.legend_data[-1] == "°C"
        app.library.delete(palette.name)
        app.palette_changed(None)
        assert "Inferno" in app._combos["Palette"]["values"]
        assert not errors
    finally:
        app.close()


def test_analysis_workspace_roi_layers_recording_and_playback(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import tkinter as tk

    import numpy as np

    from p3_thermal.app import ThermalApp
    from p3_thermal.export import load_snapshot

    root = tk.Tk()
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    app = ThermalApp(root, demo=True)

    def pump(seconds=0.1):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            root.update()
            time.sleep(0.01)

    try:
        pump(0.2)
        assert app.frame is not None
        app.open_workspace()
        assert app.sidebar.selection.get() == "Measurements"
        app.open_workspace()
        assert app.sidebar.selection.get() == "View"
        app.open_workspace()
        panel = app.workspace
        app.pause()
        panel.tool_box.set("Rectangle")
        panel.tool_box.event_generate("<<ComboboxSelected>>")
        root.update()
        assert panel.tool.get() == app.canvas.tool == "Rectangle"
        app.rotate()
        app.flip()
        panel.set_tool("Line")
        a, b = app.canvas.screen_point((20, 20)), app.canvas.screen_point((30, 30))
        app.canvas.start_pan(SimpleNamespace(x=a[0], y=a[1]))
        app.canvas.pan(SimpleNamespace(x=b[0], y=b[1]))
        assert app.canvas.find_withtag("roi_preview")
        app.canvas.finish_gesture(SimpleNamespace(x=b[0], y=b[1]))
        assert not app.canvas.find_withtag("roi_preview")
        assert app.analysis.regions[0].start == (20, 20)
        assert app.analysis.regions[0].end == (30, 30)
        for tool in ("Circle", "Rectangle"):
            panel.set_tool(tool)
            app.canvas.start_pan(SimpleNamespace(x=a[0], y=a[1]))
            app.canvas.pan(SimpleNamespace(x=b[0], y=b[1]))
            assert app.canvas.find_withtag("roi_preview")
            app.canvas.finish_gesture(SimpleNamespace(x=b[0], y=b[1]))
            assert not app.canvas.find_withtag("roi_preview")
        panel.rois.selection_set("0")
        panel.line_profile()
        pump(0.1)
        panel.add_layer()
        panel.set_tool("Brush")
        panel.gesture("begin", "Brush", (25, 25), (25, 25))
        panel.gesture("move", "Brush", (25, 25), (28, 25))
        assert app.analysis.layers[0].mask[25, 26]
        panel.undo_stroke()
        assert not app.analysis.layers[0].mask.any()
        panel.gesture("begin", "Brush", (25, 25), (25, 25))
        panel.enabled.set(True)
        panel.params["emissivity"].set("0.8")
        panel.apply_radiometry()
        assert app.analysis.radiometry.enabled
        assert not np.array_equal(app.measured, app.frame.raw / 64 - 273.15)
        panel.iso_enabled.set(True)
        panel.iso_min.set("-100")
        panel.iso_max.set("1000")
        panel.apply_isotherm()
        assert app.analysis.isotherm
        project = tmp_path / "project.npz"
        monkeypatch.setattr(
            "p3_thermal.app.filedialog.asksaveasfilename", lambda **kw: str(project)
        )
        app.export()
        frame, metadata = load_snapshot(project)
        assert "analysis" in metadata
        np.testing.assert_array_equal(frame.raw, app.frame.raw)
        app.comparison.load(0, project)
        app.comparison.load(1, project)
        assert "mean +0.000" in app.comparison.summary.get()
        app.sidebar.select("Compare")
        pump(0.1)
        original_shape = app.comparison.canvases[0].raw.shape
        app.comparison.rotate()
        assert app.comparison.canvases[0].raw.shape == original_shape[::-1]
        assert app.comparison.canvases[1].raw.shape == original_shape[::-1]
        app.comparison.flip()
        assert app.comparison.mirror
        app.comparison.split_view.set(True)
        app.comparison.toggle_split()
        app.comparison.split_position.set(30)
        app.comparison.update_split()
        pump(0.05)
        assert app.comparison.split_controls.winfo_ismapped()
        assert app.comparison.split_canvas.find_withtag("split_boundary")
        old_position = app.comparison.split_position.get()
        boundary_x = app.comparison.split_boundary_x()
        app.comparison.start_split_drag(SimpleNamespace(x=boundary_x, y=20))
        app.comparison.drag_split(SimpleNamespace(x=boundary_x + 20, y=20))
        app.comparison.finish_split_drag(SimpleNamespace(x=boundary_x + 20, y=20))
        assert app.comparison.split_position.get() != old_position
        app.comparison.split_view.set(False)
        app.comparison.toggle_split()
        assert not any(
            isinstance(child, tk.Toplevel) for child in root.winfo_children()
        )
        app.peaking.set(True)
        app.render()
        app.pause()
        assert not app.correction_active
        panel.live_correction.set(True)
        panel.changed()
        assert app.correction_active
        app.pause()
        app.screenshot()
        assert app.sidebar.selection.get() == "Save image"
        app.palette_panel.edit()
        assert app.sidebar.selection.get() == "Palette editor"
        assert not any(
            isinstance(child, tk.Toplevel) for child in root.winfo_children()
        )
        sequence = tmp_path / "recording.p3v"
        monkeypatch.setattr(
            "p3_thermal.analysis_ui.filedialog.asksaveasfilename",
            lambda **kw: str(sequence),
        )
        panel.start_recording()
        pump(0.3)
        panel.stop_recording()
        app.recorder.join(3)
        assert app.recorder.written >= 2 and app.recorder.error is None
        monkeypatch.setattr(
            "p3_thermal.analysis_ui.filedialog.askopenfilename",
            lambda **kw: str(sequence),
        )
        pump(0.15)
        assert app.sidebar.selection.get() == "Video"
        assert app.offline and panel.sequence is not None
        panel.step(1)
        assert panel.position.get() == 1
        app.palette.set("White hot")
        app.update_settings()
        panel.play()
        pump(0.4)
        panel.stop_playback()
        assert not errors
    finally:
        app.close()
        end = time.monotonic() + 3
        while app.worker.is_alive() and time.monotonic() < end:
            root.update()
            time.sleep(0.01)
        with contextlib.suppress(tk.TclError):
            root.update()


def test_live_comparison_language_navigation_and_full_help(tmp_path):
    from types import SimpleNamespace

    import tkinter as tk

    import numpy as np

    from p3_thermal.app import ThermalApp
    from p3_thermal.export import save_snapshot
    from p3_thermal.help_ui import SHORTCUTS
    from p3_thermal.preferences import Preferences

    root = tk.Tk()
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    app = ThermalApp(root, demo=True)

    def pump(seconds=0.2):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            root.update()
            time.sleep(0.01)

    try:
        pump()
        app.pause()
        app.sidebar.buttons["Compare"].invoke()
        comparison = app.comparison
        comparison.freeze(0)
        reference = comparison.items[0][0].raw.copy()
        comparison.live(1)
        pump()
        first_time = comparison.items[1][0].timestamp
        pump()
        assert comparison.items[1][0].timestamp > first_time
        np.testing.assert_array_equal(comparison.items[0][0].raw, reference)
        assert app.paused
        comparison.freeze(1)
        frozen_time = comparison.items[1][0].timestamp
        pump()
        assert comparison.items[1][0].timestamp == frozen_time
        path = tmp_path / "reference.npz"
        save_snapshot(path, app.frame, app.capture_metadata())
        app.import_snapshot(path)
        working = app.frame
        comparison.load(0, path)
        comparison.live(1)
        pump()
        assert app.offline and app.frame is working
        comparison.clear_live()
        assert comparison.items[0] is not None and comparison.items[1] is None
        pump()
        assert comparison.items[1] is not None
        app.sidebar.buttons["Settings"].invoke()
        app.language.set("Polski")
        app.change_language()
        pump()
        assert app.sidebar.buttons["View"].cget("text") == "Widok"
        assert app.sidebar.selection.get() == "Settings"
        source = app._combos["Source"]
        source.set("Wartości RAW")
        source.event_generate("<<ComboboxSelected>>")
        assert app.mode.get() == app.settings.mode == "Raw counts"
        app.translator.refresh()
        assert source.get() == "Wartości RAW"
        assert Preferences(model="p3").language == "pl"
        app.help()
        pump()
        content = app.help_panel.text.get("1.0", "end")
        assert "INSTRUKCJA OBSŁUGI" in content
        assert all(key in content for _, key, _, _ in SHORTCUTS)
        assert app.help_panel.winfo_width() > 500
        assert app.help_panel.winfo_height() > 400
        assert not any(
            isinstance(child, tk.Toplevel) for child in root.winfo_children()
        )
        app.shortcut(SimpleNamespace(widget=app.help_panel.text), "zoom_in")
        app.language.set("English")
        app.change_language()
        assert app.sidebar.buttons["View"].cget("text") == "View"
        assert source.get() == "Raw counts"
        assert app.settings.mode == "Raw counts"
        assert not errors
    finally:
        app.close()
        deadline = time.monotonic() + 3
        while app.worker.is_alive() and time.monotonic() < deadline:
            root.update()
            time.sleep(0.01)
        with contextlib.suppress(tk.TclError):
            root.update()


def test_compact_navigation_and_scrollable_resizable_panels():
    import tkinter as tk

    from p3_thermal.app import ThermalApp

    root = tk.Tk()
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    app = ThermalApp(root, demo=True)

    def pump():
        deadline = time.monotonic() + 0.15
        while time.monotonic() < deadline:
            root.update()
            time.sleep(0.01)

    try:
        pump()
        sections = [
            button
            for name, button in app.sidebar.buttons.items()
            if name not in ("Help", "Settings")
        ]
        assert len({button.winfo_y() for button in sections}) == 2
        assert (
            max(button.winfo_height() for button in sections)
            < app.pause_button.winfo_height()
        )
        assert app.sidebar.navigation_area.master is app.sidebar.master
        assert app.sidebar.navigation_area.winfo_rootx() == app.sidebar.winfo_rootx()
        assert app.sidebar.navigation_area.winfo_rooty() < app.sidebar.winfo_rooty()
        assert app.sidebar.buttons["Help"].master is app.utility_bar
        assert app.sidebar.buttons["Settings"].master is app.utility_bar
        assert "Analysis / RAW / Video" not in [
            button.cget("text") for button in sections
        ]
        root.geometry("520x360")
        pump()
        assert root.winfo_width() == 520 and root.winfo_height() == 360
        app.body.sashpos(0, 290)
        app.image_info_split.sashpos(0, 85)
        app.sidebar.select("RAW editing")
        pump()
        assert app.sidebar.navigation_area.horizontal.winfo_ismapped()
        assert app.sidebar.vertical.winfo_ismapped()
        bar = app.sidebar.vertical
        assert (
            root.winfo_containing(
                bar.winfo_rootx() + bar.winfo_width() // 2, bar.winfo_rooty() + 20
            )
            is bar
        ), "Page content covers the menu scrollbar"
        assert app.sidebar.horizontal.winfo_ismapped()
        assert app.info_area.vertical.winfo_ismapped()
        assert app.info_area.horizontal.winfo_ismapped()
        app.sidebar.viewport.yview_moveto(1)
        app.info_area.viewport.yview_moveto(1)
        pump()
        assert app.sidebar.viewport.yview()[1] == 1
        assert app.info_area.viewport.yview()[1] == 1
        old_height = app.info_area.winfo_height()
        app.image_info_split.sashpos(0, 40)
        pump()
        assert app.info_area.winfo_height() > old_height
        app.sidebar.select("View")
        pump()
        assert app.sidebar.viewport.yview()[0] == 0
        assert app.sidebar.content is app.sidebar.pages["View"]
        app.language.set("Polski")
        app.change_language()
        pump()
        assert len({button.winfo_y() for button in sections}) == 2
        assert app.sidebar.buttons["Settings"].winfo_ismapped()
        assert not errors
    finally:
        app.close()


def test_palette_editor_save_reopen_and_scroll_recovery():
    from tkinter import ttk

    import tkinter as tk

    from p3_thermal.app import ThermalApp

    root = tk.Tk()
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    app = ThermalApp(root, demo=True)

    def pump():
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            root.update()
            time.sleep(0.01)

    def children(widget):
        for child in widget.winfo_children():
            yield child
            yield from children(child)

    def click(widget, text):
        next(
            child
            for child in children(widget)
            if isinstance(child, ttk.Button) and child.cget("text") == text
        ).invoke()
        pump()

    try:
        pump()
        assert root.state() == "normal"
        assert not root.attributes("-fullscreen")
        app.sidebar.select("Palettes")
        click(app.palette_panel, "New…")
        page = app.sidebar.pages["Palette editor"]
        assert page.winfo_height() >= page.winfo_reqheight() > 300
        from p3_thermal.color_ui import ColorWheel

        click(page, "Color…")
        wheel = next(child for child in children(page) if isinstance(child, ColorWheel))
        assert wheel.winfo_manager() == "pack"
        wheel.color.set("#FF0000")
        assert wheel.saturation == wheel.value.get() == 1
        wheel.canvas.event_generate("<Button-1>", x=120, y=120)
        assert wheel.saturation < 0.01
        wheel.value.set(0)
        wheel.publish()
        assert wheel.color.get() == "#000000"
        wheel.color.set("#00FF00")
        assert abs(wheel.hue - 1 / 3) < 1e-6
        click(page, "Add")
        click(page, "Color…")
        assert not wheel.winfo_manager()
        entry = next(child for child in children(page) if isinstance(child, ttk.Entry))
        entry.delete(0, "end")
        entry.insert(0, "Audit palette")
        click(page, "Save and use")
        assert app.palette.get() == "Audit palette"
        assert app.library.palettes["Audit palette"].stops[0][0] == 15
        assert (30.0, "#00FF00") in app.library.palettes["Audit palette"].stops
        click(app.palette_panel, "Edit…")
        assert app.sidebar.pages["Palette editor"] is not page
        assert app.sidebar.content.winfo_height() > 300
        root.geometry("520x360")
        pump()
        assert app.sidebar.vertical.winfo_ismapped()
        app.sidebar.viewport.yview_moveto(1)
        app.sidebar.select("Settings")
        root.geometry("1400x950")
        pump()
        app.body.sashpos(0, 800)
        pump()
        assert not app.sidebar.vertical.winfo_ismapped()
        assert not app.sidebar.horizontal.winfo_ismapped()
        app.sidebar.viewport.event_generate("<Button-5>")
        pump()
        assert app.sidebar.viewport.yview() == (0.0, 1.0)
        app.language.set("Polski")
        app.change_language()
        app.palette_panel.edit_selected()
        pump()
        assert app.sidebar.content.winfo_height() > 300
        assert "Audit palette" in app.palette_panel.listing.cget("values")
        assert not errors
    finally:
        app.close()


def test_comparison_native_readouts_and_button_zoom():
    from tkinter import ttk

    import tkinter as tk

    import numpy as np

    from p3_thermal.acquisition import Frame
    from p3_thermal.analysis import AnalysisState, Radiometry
    from p3_thermal.app import ThermalApp

    root = tk.Tk()
    app = ThermalApp(root, demo=True)
    try:
        root.update()
        comparison = app.comparison
        app.sidebar.select("Compare")
        raw = np.arange(15, dtype=np.uint16).reshape(3, 5) + 19000
        comparison.items = [
            (
                Frame(raw + offset, np.zeros(raw.shape, np.uint8), 0),
                AnalysisState(radiometry=Radiometry(enabled=True, emissivity=0.8)),
                name,
            )
            for offset, name in ((0, "A"), (1000, "B"))
        ]
        comparison.correct.set(True)
        comparison.render()
        root.update()
        controls = next(
            child
            for child in comparison.image_boxes[0].winfo_children()
            if isinstance(child, ttk.Frame)
        )
        zoom = next(
            child for child in controls.winfo_children() if child.cget("text") == "+"
        )
        old_scale = comparison.canvases[0].pixel_scale
        zoom.invoke()
        assert comparison.canvases[0].pixel_scale > old_scale
        assert comparison.canvases[0].pixel_scale == comparison.canvases[1].pixel_scale
        comparison.split_view.set(True)
        comparison.toggle_split()
        for rotation in range(4):
            for mirror in (False, True):
                comparison.rotation, comparison.mirror = rotation, mirror
                comparison.render()
                canvas = comparison.split_canvas
                for column, slot in ((0, 0), (canvas.raw.shape[1] - 1, 1)):
                    sy, sx = canvas.coordinates[0, column]
                    frame, state, _ = comparison.items[slot]
                    assert canvas.raw[0, column] == frame.raw[sy, sx]
                    canvas.pick(
                        canvas.offset[0] + (column + 0.5) * canvas.pixel_scale,
                        canvas.offset[1] + 0.5 * canvas.pixel_scale,
                    )
                    assert f"({sx}, {sy})" in comparison.labels[slot].get()
                    assert (
                        f"{state.celsius(frame.raw)[sy, sx]:.6f}"
                        in comparison.labels[slot].get()
                    )
        comparison.items[1] = None
        comparison.render()
        assert comparison.split_canvas.raw is None
    finally:
        app.close()
