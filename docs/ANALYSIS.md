# Measurements, RAW editing and radiometric video

Choose **Measurements**, **RAW editing** or **Video** using the compact category strip above the right settings panel. The image, controls, palette editor, export options and profiles remain in one window. Resize the sidebar using the vertical divider; use its scrollbars to reach all controls in narrow or short windows. The horizontal divider below the image adjusts the height of the separately scrollable navigation and pixel-readout panel.

## Temperature, Raw counts and Factory brightness

Raw counts are the original uint16 thermal values. Temperature converts them using `°C = RAW / 64 − 273.15`: RAW 19000 becomes 23.725000°C. Automatic contrast can make these views look alike because conversion is linear. Their scale units differ, and Fixed temperature limits do not apply to Raw counts. Correction changes the derived temperature plane, never the RAW codes.

Factory brightness is a separate, camera-processed 8-bit channel. Its intensity is not a linear Celsius scale. CLAHE/DDE and Factory brightness therefore show observed temperature ranges for color bands, rather than claiming an exact global inverse scale. Exact pixel readings remain below the image.

Imported NPZ projects and video frames remain recolorable, zoomable and editable. A standalone RAW PNG/NPY lacks the factory brightness channel, so Factory brightness is disabled for those imports.

## ROI tools

| Tool | Gesture |
| --- | --- |
| Spot | Click a sensor pixel. |
| Rectangle | Drag between opposite corners. |
| Circle | Drag from center to a point on the circumference. |
| Line | Drag from the first endpoint to the second. |
| Pan | Drag the image without creating measurements. |

A dashed preview follows the pointer while drawing lines, rectangles and circles; release commits the ROI. Coordinates remain attached to native sensor pixels through rotation, mirroring, pan and zoom. Rectangles include their endpoints; circles include pixel centers inside the radius. Lines sample nearest native pixels in endpoint order without temperature interpolation. Up to 100 ROIs are supported.

The table reports minimum, maximum, mean and valid sample count. Regions show minimum/maximum markers; spots show their temperature. Delete selected removes a region; Export ROI CSV saves the statistics. When correction is enabled, statistics use corrected estimates. Invalid results are excluded, and entirely invalid regions show no fabricated value.

### Profiles and isotherms

Select a Line in the table and click Line profile. The embedded Profile section plots temperature against distance in **sensor pixels**, not millimeters. It updates during live acquisition, playback and correction editing. Matplotlib's toolbar provides plot navigation and image export.

Enable Isotherm, enter inclusive lower/upper temperature limits, choose a color and apply. For a threshold of 50°C, use 50 as the minimum and a sufficiently high maximum. Isotherms use the active temperature plane; the legend describes the underlying palette, not the single-color overlay.

## RAW editing

Enable experimental radiometric correction, enter parameters and click Apply radiometric parameters. Correction works on imported/frozen frames. For live acquisition, also enable **Apply correction to live stream (experimental)**; this separate opt-in starts disabled. Painted layers remain fixed to sensor coordinates, so live correction is not recommended without validating the scene and model. Disabling it restores original readings. The cursor retains original temperature/RAW and labels the modeled result **Corrected estimate**.

| Parameter | Meaning |
| --- | --- |
| Global emissivity ε | 0.01–1, used where no enabled material mask overrides it. |
| Reflected / background °C | Apparent temperature of reflected radiation; not necessarily air temperature. |
| Air temperature °C | Atmosphere between the object and camera. |
| Object distance m | Path length for the optional atmospheric transmission model. |
| Relative humidity % | Humidity used in that atmospheric model. |

Distance/humidity modeling has its own enable switch. With ε=1 and atmospheric correction disabled, the model leaves native temperatures unchanged.

### Model and limitations

This is an **experimental broad-band T⁴ estimate**, not inversion of a calibrated P3 spectral response. Manufacturer calibration constants and corrections already applied by firmware are unknown. Entering emissivity does not guarantee absolute accuracy or remove solar reflections. Missing/saturated sensor information cannot be recovered.

The model balances temperatures in kelvin:

```text
T_apparent⁴ = τ [ε T_object⁴ + (1−ε) T_reflected⁴] + (1−τ) T_air⁴
```

It solves for object temperature per pixel. Transmission τ is 1 when atmospheric correction is disabled or distance is zero. Invalid transmission is rejected. Negative inverted radiance produces NaN and a gray pixel, not an invented 0 K measurement.

Parameter meanings are described in [FLIR measurement documentation](https://docs.flir.com/T810605/en-US/latest/s10.html). Generic atmospheric constants come from [Thermimage raw2temp](https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R), not P3 calibration. The implementation models one object-to-camera path without an IR window, using the square root of the full distance; see the [ThermImageJ update](https://github.com/gtatters/ThermImageJ/releases) for this distinction.

Material names alone do not determine emissivity: oxidation, coatings, texture, moisture and viewing angle matter. The application stores your selected values rather than assigning automatic material constants.

### Painting emissivity layers

1. Open a RAW image or freeze a live frame.
2. Enter a material name and emissivity, click Add layer and select it.
3. Choose Brush and a radius in sensor pixels, then paint the image. Selecting a brush automatically freezes a live view.
4. Eraser removes the selected layer's assignment, revealing a lower layer or the global value. Undo stroke retains up to 20 strokes.
5. Update ε edits the value; Toggle enables/disables a layer; Delete removes it. Move up/down changes priority: **later, lower layers override earlier ones**.
6. Enable correction and apply parameters to include masks in modeled results.

The turquoise active-mask overlay is optional and excluded from image export. Up to 32 layers are supported. Changing sensor dimensions clears incompatible ROIs/masks but preserves global parameters. Layers are fixed to image coordinates; they do not track moving materials in video.

Save RAW + correction project writes NPZ with original raw/brightness, parameters, packed masks, ROI, isotherms and palette. Active correction also exports float64 `corrected_celsius`. Import recomputes from the original data rather than treating that derived plane as RAW.

Standalone uint16 PNG/NPY lacks measurement metadata and factory brightness; enter environmental parameters manually. Ordinary JPEG and 8-bit PNG cannot be imported as radiometric RAW.

## Recording and playback

1. Return to live and select Video. Choose 1, 2, 5, 10, 15 or 25 FPS.
2. Record selects a **new** `.p3v` file; existing recordings are not overwritten.
3. Monitor committed frame count, actual average FPS and dropped frames. Stop recording signals the writer to drain its queue.
4. A successful nonempty recording **opens automatically when the writer finishes**, and Video becomes the selected sidebar section. Failed or empty recordings do not replace the current image. Opening is suppressed during application shutdown.
5. The timeline selects a frame; frame buttons step; Play / pause follows recorded timestamp intervals. Open sequence also loads existing recordings.
6. Recolor frames, add ROIs/correction or export selected frames as NPZ/JPEG/PNG. Return to live stops playback.
7. Temperature over time computes the selected ROI's mean, or the whole frame's mean if no ROI is selected, using current correction parameters. Computation runs outside the GUI thread and appears in the embedded Profile section.

### FPS and storage

FPS controls **sampling of arriving frames**, not unknown hardware timing registers. The previously tested P3 stream produced about 25 FPS. Sampling at 15 FPS selects among these arrivals and may produce unequal intervals; actual timestamps are preserved. Experimental rates allow 0.1–240 FPS, but requesting 100 FPS from a 25 FPS source still saves at most 25 real measurements per second.

Recording precedes the GUI latest-frame queue. Its own queue holds 128 frames; slow storage causes counted drops rather than blocking USB indefinitely. Each saved RAW/brightness plane is losslessly compressed, but temporal recording is not guaranteed lossless under overload. USB gaps remain visible in timestamps; an active recording can resume receiving frames after reconnection. Closing the application waits for queue draining.

Analysis metadata is captured at recording start. Later UI parameter changes are not stored as timed events; original per-frame data permits analysis with new parameters afterward.

### `.p3v` format

SQLite tables are `metadata(key,value)` and `frames(id,timestamp,height,width,raw,brightness)`. Each frame stores independently zlib-compressed little-endian uint16 RAW and uint8 brightness. Host monotonic timestamps are not sensor exposure timestamps. Each completed frame is committed transactionally; earlier committed frames can survive an interrupted recording. Empty or invalid sequences are rejected.

```python
from p3_thermal.recording import Sequence
from p3_thermal.analysis import AnalysisState

video = Sequence("measurement.p3v")
frame = video.frame(0)
analysis = AnalysisState.from_dict(video.metadata.get("analysis"), frame.raw.shape)
celsius = analysis.celsius(frame.raw)
print(video.times[0], celsius.mean())
```

Tests verify geometry, native line samples, model balance, invalid results, layer priority, painting/erasing, serialization, RAW import and FPS sampling. GUI tests exercise transformed ROI drawing, profiles, correction, projects, recording and automatic playback loading. This verifies implementation, not absolute radiometric calibration or sustained hardware/storage performance.
