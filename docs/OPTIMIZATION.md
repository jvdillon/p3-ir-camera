# Optimization and correctness audit

This audit targets the current `p3_thermal` application and its `p3_viewer.py` entry point. Legacy demos and the unverified X³/lock-in experiments are not promoted to supported features.

## Changes

- **Palette editor:** canvas windows now follow a page's natural requested height. Selecting an empty editor before adding its controls no longer locks it to one pixel. Creating, saving and reopening presets is exercised in a real Tk window, including Polish presentation.
- **Layout:** startup measures controls and selects a normal window bounded by screen space. Divider placement waits for window-manager geometry. Both scroll axes disappear when unnecessary and wheel input leaves fitting content in place. Small windows retain access to the bottom of editors.
- **Coordinates:** native measurements remain native after rotation/mirroring. Shared cached orientation maps and inverse canvas maps remove repeated coordinate-array construction and full-array searches for each ROI marker.
- **Comparison:** the split boundary selects matching RGB, RAW and corrected temperature data from A/B. Readouts remain visible in split mode. On-screen zoom buttons synchronize linked views; missing sources clear stale split data. The common palette ramp is reused rather than rebuilt with temperature-dependent stops every frame.
- **Rendering:** one normalization/filter/color pipeline covers all sources and palettes. Each processor reuses its CLAHE operator; a bounded cache holds immutable 256-color ramps. Factory brightness gets observed Celsius bands even without software enhancement. These optimizations affect presentation, not native precision.
- **Analysis and video:** unchanged ROI tables skip recomputation and existing rows are updated in place. Rectangle sampling allocates only its bounds; circle grids use broadcasting. Sequences cache their timeline and stream analysis through one read-only database connection. Frame stepping clamps at the last frame; successful snapshot import stops old playback.
- **File handling:** JSON, NPZ and image exports share atomic replacement, preserving an existing file after failed encoding or writing. NPZ validates archive entries, expanded size and advertised array dimensions before NumPy allocation, with pickle disabled. Sequence readers bound frame counts, metadata, dimensions, payload sizes and decompression; trailing compressed garbage is rejected. Imported correction/filter switches are validated.
- **Readability:** small shared widget builders replace repeated control construction. Comments and documentation remain; obsolete descriptions of filters bypassing custom palettes were corrected. Simplification removes duplicated processing and UI logic, while additional validation and regression coverage necessarily add code.

## Processing measurements

A local same-process comparison against the pre-audit processor used deterministic 192×256 uint16 RAW/uint8 brightness arrays, a three-stop custom palette, 80 frames per run, and the fastest of three runs. The final run measured:

| Render case | Before, ms/frame | After, ms/frame |
| --- | ---: | ---: |
| Temperature | 2.177 | 1.930 |
| Custom palette | 3.840 | 1.595 |
| CLAHE + DDE | 4.195 | 3.825 |

These timings cover processing only, excluding USB, Tk painting and recording. Absolute times vary with machine load and power state; they are not camera-FPS claims.

## Verification

- `python -m pytest -q`: **110 passed**, 8 desktop tests skipped by default.
- `P3_GUI_TEST=1 python -m pytest tests/gui_test.py -q`: **8 passed** in a graphical session with demo frames and simulated disconnections.
- Ruff passed for the application, entry point and tests; scoped BasedPyright passed for `p3_thermal` and `p3_viewer.py`.

Coverage includes native precision, processing/filter combinations, palette validation/persistence, lossless RAW and snapshot round trips, color exports, ROI shapes/profiles, isotherms, emissivity layers/correction, recording/playback, frozen/live comparison, orientation, language, help, resize/scroll behavior, disconnect/reconnect and cleanup. Regression tests also exercise interrupted saves and oversized advertised NPZ shapes before allocation.

No P3 camera was enumerated by `lsusb` during this audit. Physical image quality, camera gain/shutter commands and real cable reconnection therefore still need a hardware acceptance run. X³ remains unavailable without a verified protocol command; radiometric correction remains an experimental model. Automated coverage does not establish calibration accuracy or exhaustive correctness for every device/OS combination.
