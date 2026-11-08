# P3 Thermal Studio

**A desktop workspace for USB thermal imaging and radiometric analysis.**

View live thermal images, inspect native sensor pixels, measure regions of interest, compare frames and record radiometric sequences. P3 Thermal Studio combines a Python USB driver with a modular Tk/ttk application, with English and Polish interfaces.

![P3 Thermal Studio main workspace with a thermal image, temperature legend and display controls](screenshots/thermal-studio-workspace.png)

*Current application running in demo mode with synthetic thermal data. All screenshots below also use demo data; no camera is required to explore the interface.*

[Quick start](#quick-start) · [Screenshots](#screenshots) · [Features](#features-in-detail) · [Limitations](#camera-behavior-and-limitations) · [Documentation](#documentation-and-development)

## At a glance

| Capability | What you can do |
| --- | --- |
| Live viewing | Switch thermal sources and palettes, adjust contrast, zoom, rotate and mirror. |
| Native measurements | Inspect temperature and RAW counts; add spots, rectangles, circles and line profiles. |
| Radiometric files | Save RAW snapshots, reopen them for analysis and record/play back `.p3v` sequences. |
| Comparison | Compare saved frames or a frozen reference against the live stream with a shared scale and B − A statistics. |
| Image export | Export color PNG/JPEG images or native 16-bit RAW PNG data. |
| Offline exploration | Use synthetic demo frames or open saved data without camera measurements. |

### Supported cameras

| Model | Native resolution | USB VID:PID | Verification status |
| --- | --- | --- | --- |
| P3 | 256 × 192 | `3474:45A2` | Previously tested on Linux; see gain limitations below. |
| P1 | 160 × 120 | `3474:45C2` | Driver configuration available; hardware verification still needed. |

## Screenshots

### Region measurements

Draw regions directly on the thermal image and inspect their minimum, maximum and mean temperatures. Line selections support profile plots, and region statistics can be exported to CSV.

![Measurement workspace showing a rectangular hot-spot region, a line selection and temperature statistics](screenshots/thermal-studio-measurements.png)

### Frame comparison

Compare a frozen reference with the running demo stream. The comparison workspace provides separate A/B views, common display limits and temperature-difference statistics.

![Comparison workspace showing frozen reference A alongside live demo frame B](screenshots/thermal-studio-compare.png)

## Quick start

Requires **Python 3.10+**, **Tk 8.6+** and a graphical desktop session. Run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m p3-viewer --demo
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell. Demo mode generates synthetic thermal frames and requires no camera.

### Connect a camera

```bash
pip install -e .
p3-viewer                  # P3 camera
p3-viewer --model p1       # P1 camera
python p3_viewer.py        # Alternative entry point
```

Project installation supplies Python dependencies, including NumPy, OpenCV, PyUSB and Matplotlib. Tk is a system component: install `python3-tk` on Debian/Ubuntu or `tk` on Arch, and use a Python interpreter built with Tk support. A graphical desktop session is required. OpenCV's Qt viewer is not used.

### USB access on Linux

Put these rules in `/etc/udev/rules.d/99-p3-ir.rules`:

```udev
SUBSYSTEM=="usb", ATTR{idVendor}=="3474", ATTR{idProduct}=="45c2", TAG+="uaccess"
SUBSYSTEM=="usb", ATTR{idVendor}=="3474", ATTR{idProduct}=="45a2", TAG+="uaccess"
```

Reload using `sudo udevadm control --reload-rules`, then reconnect the camera. These rules grant access to the active local desktop session.

### Other platforms

Windows requires a libusb/WinUSB backend; previous setups used Zadig with VID `3474`, PID `45C2` or `45A2`. P1 and Windows/macOS hardware operation need separate verification.

## Workspace

The thermal image stays on the left. Use the compact, two-row category strip **directly above the right settings panel**: View, Filters, Sensor, Palettes, Measurements, RAW editing, Video or Compare. Help and Settings are separate compact buttons in the header. Drag the vertical divider to resize the sidebar, or the horizontal divider below the image to resize its navigation/readout panel. Both control panels scroll vertically and horizontally when needed; the category strip scrolls horizontally in narrow windows. Startup uses a normal window sized from the requested controls and available screen space; it does not maximize or enter fullscreen. The window can be reduced to 520×360. Analysis, palette editing, image export options and plots use this same window. File and color pickers remain native dialogs.

Use the wheel or on-screen +/− buttons to zoom, and Pan or the arrow buttons to move the image. Fit restores the full frame; Pixels enables detailed inspection. Under the image, the left readout shows native sensor coordinates, temperature and RAW counts. Rotation, mirroring and zoom do not alter measurements.

| Source | Meaning |
| --- | --- |
| Temperature | Native sensor values converted to °C; optional radiometric correction applies when enabled. |
| Filtered temperature | Temporal EMA smoothing of the displayed temperature image; cursor measurements remain current. |
| Raw counts | Original 16-bit thermal codes, displayed with a RAW-unit scale. |
| Factory brightness | Separate 8-bit image processed by the camera; intensity is not a linear temperature scale. |

```text
Temperature °C = RAW / 64 − 273.15
Encoding step = 0.015625 K
RAW 19000 → 23.725000 °C
```

The six decimal places preserve all values in this encoding; they do not claim six-decimal sensor accuracy. Measurements never come from RGB or interpolated export images. Optional corrections are labeled as estimates and retain the original reading.

## Features in detail

- Factory palettes: Inferno, Magma, Viridis, Turbo, Rainbow, White hot and Black hot.
- Auto percentile adapts the display range using percentiles 1–99; Fixed uses a specified temperature range after Apply range.
- CLAHE improves local contrast. DDE independently sharpens edges with strength 0–4; zero has no effect, and smooth areas may change little.
- The image legend runs from maximum at the top to minimum at the bottom. With CLAHE, DDE or Factory brightness, five color bands report observed temperature min/max, because local processing has no unique inverse temperature scale.
- Custom palettes define absolute temperature/color stops with linear gradients or discrete bands. JSON presets persist between sessions; factory palettes cannot be overwritten or removed. Auto adapts their range to the frame; Fixed uses the entered limits. CLAHE and DDE work with custom palettes too, affecting display colors without changing measurements.
- Measurements provides spots, rectangles, circles and lines, with live drag previews, minimum/maximum/mean, line profiles, CSV export and isotherms.
- RAW editing provides experimental emissivity and environmental correction, including painted material layers with an eraser and undo.
- Video records native radiometric frames to `.p3v`. After Stop recording drains the writer queue successfully, the completed sequence opens automatically for analysis. Empty or failed recordings do not replace the current image.

Freeze stops display updates without stopping acquisition. Open RAW loads NPZ, native 16-bit PNG or uint16 NPY; imported frames remain editable and can be recolored and exported. Return to live resumes camera viewing. Save data preserves RAW and analysis settings in NPZ. Save image offers smooth JPEG, native RAW PNG or color PNG, with an optional legend for color images.

### Comparison and advanced viewing

- **Compare:** two saved images or a frozen reference against the live camera, with a shared scale, pixel inspection and B − A statistics. Each slot offers Open image, Freeze and Live. Freeze copies that slot or the working frame if empty; live comparison continues when the main view is frozen/offline.
- **White hot / red peak:** grayscale with the upper end of the display scale in red. Disable CLAHE/DDE when interpreting red as temperature; enhanced edges can also become red.
- **Custom palette scaling:** the View **Fixed scale** checkbox selects manual limits; otherwise the range uses frame percentiles. Older NPZ files with `custom_auto_scale` retain their explicit min/max behavior.
- **Focus peaking:** adjustable green thermal-edge overlay in Filters, excluded from measurements and exports.
- **Remember mirror:** Sensor saves live-view mirroring per camera model and restores it automatically; imported orientation does not replace this preference.
- **Live correction:** RAW editing has an explicit experimental switch for applying enabled emissivity correction to the running stream. It starts disabled; material masks do not track motion.

See [operation details](docs/OPERATIONS.md) for exact scale behavior, persistence and comparison limitations.

Help opens a large scrollable panel in the main workspace and documents every registered shortcut, mouse navigation and all tool categories. In Compare, image shortcuts target the last clicked A/B image; text fields keep their standard editing shortcuts.

## Camera behavior and limitations

Unplugging the camera **does not close the application**. The live view displays a disconnected status and retries every two seconds after cleanup. Reconnecting resumes viewing; imported offline frames remain visible during USB failures. Reconnect expedites a pending attempt. Closing waits for acquisition and recording cleanup.

Use **HIGH** for measurements. On the previously tested P3 firmware `00.00.02.18`, LOW produced roughly −34°C for a scene reading about 20–25°C in HIGH; shutter calibration did not remove the difference. LOW remains experimental, without an invented offset. AUTO gain is unsupported.

**X³ remains disabled:** no verified camera command or implementation is available in this driver. JPEG enlargement does not implement the camera's resolution enhancement. Lock-in remains a separate, untested historical experiment.

Radiometric correction is an experimental broad-band model, not calibrated P3 spectral inversion. It cannot guarantee accurate correction of solar reflections. Recording FPS samples arriving frames; it does not change sensor timing or synthesize measurements.

## Shortcuts

| Action | Shortcut |
| --- | --- |
| Pixel inspection at 12800% | Ctrl+X |
| Save thermal data | Ctrl+S |
| Cancel drawing and fit image | Escape or double click |
| Full workspace help | F1 |
| Zoom | + / − or mouse wheel |

Pixel grid lines appear at 2800%, full temperature labels at 9600%, and zoom is limited to 25600%. The cursor readout remains available at every scale.

## Documentation and development

- [Measurements, RAW editing and video](docs/ANALYSIS.md)
- [Optimization audit and verification scope](docs/OPTIMIZATION.md)
- [Architecture and extension contracts](docs/ARCHITECTURE.md)
- [Formats, troubleshooting and verification](docs/OPERATIONS.md)
- [USB protocol](P3_PROTOCOL.md)
- [Historical lock-in experiment](LOCK-IN.md)
- [Archived demo notes and attribution](docs/LEGACY_DEMO.md)

```bash
pip install -e '.[dev]'
python -m pytest -q
# Optional real-window tests in a graphical desktop session:
P3_GUI_TEST=1 python -m pytest tests/gui_test.py -q
```

Tests are not part of application startup. `p3_viewer.py` is the entry point; application modules live in `p3_thermal`.

## License and acknowledgements

Distributed under the [Apache License 2.0](LICENSE). This project is independent of the camera manufacturer.

Original project by **Joshua V. Dillon**; current package authors include **Piotr Przybył** and **Joshua V. Dillon**. Historical contributor acknowledgements are preserved in the [archived demo documentation](docs/LEGACY_DEMO.md).
