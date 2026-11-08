# Operation, formats and verification

## Typical measurement

1. Start `p3-viewer` (P3), `p3-viewer --model p1`, or `p3-viewer --demo`.
2. In View, choose Temperature and a palette. Use Fixed and Apply range for consistent temperature limits between scenes.
3. Hover over a pixel to read native coordinates, six-decimal °C and RAW below the image. Wheel/+/- zoom, Pan/arrows move, and Fit/Escape restore the full frame.
4. Freeze holds the displayed frame while acquisition continues. Save data preserves radiometry; Save image opens export settings in the sidebar.
5. Select Measurements, RAW editing, Video or Compare using the category buttons. The compact category strip above the right settings panel selects the sidebar content; selecting View restores Pan. Help and Settings are separate header buttons.
6. Close using the window's X and allow acquisition/recording cleanup to finish.

Temporal filtering changes presentation, not the cursor's native sample. Corrected live measurements require both correction parameters to be enabled and the explicit live-correction switch. Frozen/offline correction remains independently available.

## NPZ format

```python
import json
import numpy as np

with np.load("capture.npz", allow_pickle=False) as capture:
    raw = capture["raw"]                 # H×W uint16, native sensor orientation
    brightness = capture["brightness"]   # H×W uint8
    metadata = json.loads(str(capture["metadata"]))
    celsius = raw.astype(np.float64) / 64 - 273.15
    print(f"{celsius[10, 20]:.6f} °C")    # Native x=20, y=10
```

Metadata includes camera model, demo flag, display settings, rotation/mirror, UTC export time, host monotonic frame time, RAW unit and conversion. UTC is save time, not exposure time. Version 2 separates CLAHE/DDE settings and includes custom palette definitions. Legacy combined `detail` flags migrate to both filters. Analysis metadata preserves regions, masks, isotherms and correction parameters; `corrected_celsius` is an optional derived export.

Open RAW accepts NPZ, uint16 NPY and native 16-bit single-channel PNG. Imports validate types, size, units and settings without pickle; unpacked NPZ content is limited to 32 MB. Standalone PNG/NPY has no factory brightness or measurement metadata, so brightness is explicitly reconstructed and Factory brightness is unavailable.

Imports restore display settings and orientation and stay frozen despite incoming USB frames or disconnection. Custom palette name conflicts create a new name without overwriting existing presets. Return to live resumes normal acquisition. Imported orientation never overwrites your saved camera mirror preference.

## Image export

| Format | Content |
| --- | --- |
| JPEG | Full current color frame; bicubic enlargement 1–8×, default 3×; quality 1–100, default 95. |
| Native RAW PNG | Original uint16 sensor matrix in native orientation, without filters, scaling or legend. |
| Color PNG | Lossless current RGB image in sensor resolution with the selected orientation. |
| NPZ via Save data | Both native camera planes and metadata, including analysis and palette settings. |

Image options capture the frame when opened, so live updates cannot change the selected export. Include legend appends a scale to color images only and increases output dimensions. Exports include the full frame, not the viewport crop, pixel grid or ROI markers. Focus peaking and the emissivity painting mask are viewing aids and are excluded. RAW PNG may look dark in an ordinary viewer; JPEG smoothing does not add sensor measurements.

## User palettes and automatic scale

Palettes → New opens the embedded editor. Provide a name and 2–64 strictly increasing finite temperature/color stops. Color toggles an embedded HSV wheel: drag to select hue/saturation, adjust Brightness, or enter #RRGGBB directly. Selecting an existing stop synchronizes the wheel; Add/Update commits the chosen color to that stop. Remove deletes the selected stop. Save and use persists the preset. A different name creates a copy. Factory palette names are protected in both UI and data validation.

`linear` interpolates RGB between stops. `steps` uses a stop's color until the next threshold, which belongs to the new band. Values outside the range use endpoint colors.

```json
{
  "version": 1,
  "name": "PCB 15–50 C",
  "interpolation": "linear",
  "stops": [[15, "#0000FF"], [25, "#00FF00"], [50, "#FF0000"]]
}
```

Palette stops define relative spacing within the selected display range. With **Fixed scale**, the entered endpoints are used; otherwise Auto uses percentiles 1–99. To use the preset's original °C scale, select Temperature, enter its first and last stop as Fixed limits, and disable CLAHE/DDE. Display colors use a 256-entry ramp; native temperature readings retain full precision. Older snapshots containing `custom_auto_scale=true` use frame extrema instead of percentiles, without modifying their preset. CLAHE/DDE apply to custom palettes and replace the linear legend with observed temperature bands.

The library lives in `$XDG_CONFIG_HOME/p3-thermal-studio/palettes.json`, otherwise `%APPDATA%/p3-thermal-studio/palettes.json`, otherwise `~/.config/p3-thermal-studio/palettes.json`. Saves are atomic. A corrupt library is reported and protected from automatic overwrite.

## White hot / red peak and focus peaking

**White hot / red peak** uses grayscale and a red upper tail covering approximately the top 5% of the display range. Auto uses percentiles; Fixed uses your limits. With CLAHE/DDE enabled, red can mark enhanced edges rather than the hottest native samples. Pixel readouts and observed temperature bands remain authoritative. Ordinary White hot remains available separately.

**Focus peaking**, in Filters, marks strong native thermal gradients in green. Lowering the threshold highlights more edges. Gaussian smoothing followed by Sobel gradients reduces isolated noise sensitivity. Peaking is a focusing aid, not autofocus or a calibrated sharpness measure. It changes neither RAW nor temperature readings and is excluded from export. Green marks overlay the palette, so use the ordinary legend for the underlying image.

## Comparing two images

Select Compare and load A and B from NPZ, RAW PNG or uint16 NPY. Alternatively, Freeze A/B copies the displayed slot (the working frame when empty), and Live A/B connects that slot to camera acquisition. Freeze A + Live B gives a fixed reference against the current scene. A file import or another Freeze replaces only that slot. Live frames arrive even while the main working view is frozen or imported; its data is not replaced. Unplugging clears live slots while preserving saved/frozen references, and comparison acquisition retries automatically. Both images occupy the main image area and share the same temperature minimum/maximum and palette. User palettes are stretched to this shared range. Each image has independent zoom/pan, Fit and pixel inspection. Comparison data is separate from the live or imported working frame.

Apply each file's saved correction optionally uses that file's radiometric settings. Equal-sized frames show mean/min/max **B − A** for matching native sensor coordinates. Different-sized frames remain visually comparable but have no pixelwise difference statistic. No registration or motion compensation is performed: physically align the scene before interpreting pixelwise differences. RGB JPEGs do not contain radiometric data and are not supported here.

## Mirror and live correction

Sensor → Remember mirror stores live-view mirror changes automatically in `settings-p3.json` or `settings-p1.json` alongside the palette library. It is enabled by default and applies on startup and Return to live. Disable it to clear the stored mirror. Offline project orientation stays with that project.

RAW editing → Apply correction to live stream is an explicit experimental opt-in, off at startup. Also enable radiometric correction and Apply parameters. It is not recommended for unvalidated measurements; painted material masks stay at fixed sensor positions and do not follow moving objects. Disabling the live switch leaves frozen/offline correction available. See [the analysis guide](ANALYSIS.md) for model limitations.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Camera not found | Check model, cable, USB IDs and permissions. Reconnect expedites automatic retry. |
| Access denied / busy | Check udev/WinUSB and close other software using the camera. |
| No display / Tk | Use a desktop session and a Python installation with system Tk support. |
| Disconnected | The window stays open and retries. Offline imported frames remain visible. |
| Invalid range | Enter finite values with maximum greater than minimum and apply. |
| Unexpected colors | Check source, local enhancements, custom auto-scale and isotherms. Use Temperature with a fixed linear scale for direct comparisons. |
| DDE seems inactive | Try a frozen frame with edges and increase strength. Flat areas need not change. |
| LOW gives implausible temperatures | Return to HIGH; LOW radiometry has not been verified. |
| Recording does not auto-open | Wait for draining; inspect writer error and frame count. Failed/empty recordings do not replace the image. |

## Verification and limits

```bash
python -m pytest -q
P3_GUI_TEST=1 python -m pytest tests/gui_test.py -q
# Optional physical-camera variation:
P3_GUI_TEST=1 P3_GUI_CAMERA=1 python -m pytest tests/gui_test.py -q
```

Headless tests cover protocol behavior, all 65536 encoded temperature values, orientation, picking, filters, palette ranges, exports, ROI geometry, radiometry, masks and sequence sampling. GUI tests cover drawing previews, embedded tools, plots, projects, automatic sequence loading, comparison, live-correction opt-in and simulated disconnect/reconnect. Tk tests isolate the configuration directory from user presets.

Earlier hardware checks used **P3 firmware 00.00.02.18**: native 256×192 frames, about 25 FPS after startup, shutter commands, reopening after close, source switching and temperature inspection were observed. Startup took approximately five seconds. LOW showed around −34°C for a scene around 20–25°C in HIGH; NUC did not remove the difference. No arbitrary offset was fitted.

Physical unplug detection was exercised previously; automatic reconnect in the same window is covered with a simulated device and still needs a separate physical reconnect check. P1 and other operating systems, long recordings, and absolute corrected-temperature accuracy require independent hardware verification. These UI changes do not establish new calibration evidence. Lock-in and X³ remain unavailable in the application.

## Language, navigation and full-size help

The two-row category strip directly above the right settings panel uses compact flat buttons with an active-section highlight. It scrolls horizontally when the window is narrow. Help and Settings live separately in the header; the redundant Analysis / RAW / Video button has been removed. The sidebar and the bottom image-control/readout panel both have automatic vertical and horizontal scrollbars. Drag the main vertical divider to change sidebar width; drag the horizontal divider between the image and its readouts to change their heights. Mouse wheel scrolls a control panel; Shift+wheel scrolls horizontally. Wheel over a thermal image still zooms. The application supports windows down to 520×360.

Settings → Language offers English (default) and Polski; changes apply immediately and persist with model preferences. Internal source/ROI identifiers and portable NPZ/JSON data remain unchanged. User-entered material names and external driver diagnostics are not rewritten.

Help (toolbar or F1) fills the main workspace with a scrollable guide. It documents Ctrl+X inspection, Ctrl+S project export, +/− zoom, Escape to cancel drawing and fit, F1 help, mouse wheel, drag, double click and standard keyboard focus navigation. In Compare, inspection/zoom/fit act on the last clicked image, defaulting to A. Ctrl+S saves the working project. Editable controls retain their normal cut/copy/text-entry behavior.

The settings sidebar keeps its vertical scrollbar visible even when the selected page fits, so its location stays predictable. Other overflow bars remain automatic. Section buttons use a subtle one-pixel outline.
