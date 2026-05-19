#!/usr/bin/env python3
"""Thermal Camera UVC Virtual Camera Driver.

SPDX-License-Identifier: Apache-2.0

Streams colormapped thermal frames to a v4l2loopback virtual device
so Zoom, VLC, OBS, or any webcam app can see the thermal camera.

Power-saving design:
  - When no app is reading the virtual camera, the physical camera stays
    idle (like any other UVC camera).  A low-rate standby frame keeps the
    v4l2loopback format valid so apps can discover the device.
  - When an app opens the virtual camera, the physical camera wakes up
    and streams real thermal data.
  - When the last reader closes, the camera goes back to idle.
"""

from __future__ import annotations

import glob
import os
import signal
import struct
import sys
import time

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[misc, assignment]

import cv2
import numpy as np

from .p3_camera import P3Camera, get_model_config

from .viewer import (
    ColormapID, apply_colormap, agc_temporal, dde, tnr,
)

# ── v4l2 constants ───────────────────────────────────────────────────────────

V4L2_BUF_TYPE_VIDEO_OUTPUT = 2
V4L2_PIX_FMT_YUYV = 0x56595559  # 'YUYV' — universally supported by webcam apps
V4L2_FIELD_NONE = 1
V4L2_COLORSPACE_SMPTE170M = 6


def _detect_v4l2_fmt_layout():
    """Auto-detect struct v4l2_format size and VIDIOC_S_FMT code."""
    import subprocess as _sp
    src = (
        '#include <stdio.h>\n#include <stddef.h>\n#include <linux/videodev2.h>\n'
        'int main(){printf("%zu %zu %lu\\n",sizeof(struct v4l2_format),'
        'offsetof(struct v4l2_format,fmt),(unsigned long)VIDIOC_S_FMT);}'
    )
    try:
        _sp.run(["gcc", "-x", "c", "-o", "/tmp/_v4l2sz", "-"], input=src.encode(),
                capture_output=True, timeout=5, check=True)
        out = _sp.check_output("/tmp/_v4l2sz", timeout=3).decode().split()
        return int(out[0]), int(out[1]), int(out[2])
    except Exception:
        return (208, 8, 0xc0d05605)


V4L2_FMT_SIZE, _FMT_UNION_OFF, VIDIOC_S_FMT = _detect_v4l2_fmt_layout()


def _bgr_to_yuyv(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to YUYV (YUY2) packed format."""
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0]
    u = yuv[:, :, 1]
    v = yuv[:, :, 2]
    # Subsample U and V horizontally (average pairs of pixels)
    u_sub = ((u[:, 0::2].astype(np.uint16) + u[:, 1::2].astype(np.uint16)) >> 1).astype(np.uint8)
    v_sub = ((v[:, 0::2].astype(np.uint16) + v[:, 1::2].astype(np.uint16)) >> 1).astype(np.uint8)
    h, w = bgr.shape[:2]
    yuyv = np.empty((h, w, 2), dtype=np.uint8)
    yuyv[:, 0::2, 0] = y[:, 0::2]  # Y0
    yuyv[:, 0::2, 1] = u_sub        # U
    yuyv[:, 1::2, 0] = y[:, 1::2]  # Y1
    yuyv[:, 1::2, 1] = v_sub        # V
    return yuyv


class V4L2Writer:
    """Write YUYV frames to a v4l2loopback device via ioctl + write()."""

    def __init__(self, device: str, width: int, height: int):
        self.device = device
        self.width = width
        self.height = height
        self.fd: int = -1
        self._open()

    def _open(self):
        if fcntl is None:
            raise RuntimeError("v4l2loopback writer requires Linux (fcntl.ioctl)")
        self.fd = os.open(self.device, os.O_RDWR)
        off = _FMT_UNION_OFF
        buf = bytearray(V4L2_FMT_SIZE)
        struct.pack_into("<I", buf, 0, V4L2_BUF_TYPE_VIDEO_OUTPUT)
        struct.pack_into("<I", buf, off + 0, self.width)
        struct.pack_into("<I", buf, off + 4, self.height)
        struct.pack_into("<I", buf, off + 8, V4L2_PIX_FMT_YUYV)
        struct.pack_into("<I", buf, off + 12, V4L2_FIELD_NONE)
        struct.pack_into("<I", buf, off + 24, V4L2_COLORSPACE_SMPTE170M)
        try:
            fcntl.ioctl(self.fd, VIDIOC_S_FMT, buf)
        except OSError:
            os.close(self.fd)
            self.fd = -1
            raise

    def write(self, bgr: np.ndarray) -> None:
        """Accept BGR frame, convert to YUYV, and write."""
        yuyv = _bgr_to_yuyv(bgr)
        os.write(self.fd, yuyv.tobytes())

    def close(self):
        if self.fd >= 0:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = -1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── device discovery ─────────────────────────────────────────────────────────

def find_v4l2_loopback_device() -> str | None:
    for dev in sorted(glob.glob("/sys/devices/virtual/video4linux/video*")):
        name_file = os.path.join(dev, "name")
        try:
            with open(name_file) as f:
                name = f.read().strip().lower()
            if "loopback" in name or "dummy" in name or "thermal" in name:
                return f"/dev/{os.path.basename(dev)}"
        except OSError:
            continue
    return None


def _has_readers(dev_path: str, own_fd: int) -> bool:
    """Check if any OTHER process has the v4l2loopback device open."""
    try:
        dev_stat = os.stat(dev_path)
        dev_rdev = dev_stat.st_rdev
    except OSError:
        return False
    my_pid = os.getpid()
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return False
    for pid in pids:
        if int(pid) == my_pid:
            continue
        fd_dir = f"/proc/{pid}/fd"
        try:
            for fd_name in os.listdir(fd_dir):
                try:
                    target = os.stat(f"{fd_dir}/{fd_name}")
                    if target.st_rdev == dev_rdev:
                        return True
                except OSError:
                    continue
        except OSError:
            continue
    return False


def _usb_camera_present(vid: str = "3474", pid: str = "45a2") -> bool:
    """Check if the physical USB camera is still connected."""
    try:
        for d in os.listdir("/sys/bus/usb/devices"):
            base = f"/sys/bus/usb/devices/{d}"
            try:
                with open(f"{base}/idVendor") as f:
                    if f.read().strip() != vid:
                        continue
                with open(f"{base}/idProduct") as f:
                    if f.read().strip() in (pid, "45c2"):
                        return True
            except OSError:
                continue
    except OSError:
        pass
    return False


def _make_standby_frame(w: int, h: int) -> np.ndarray:
    """Generate a dark standby frame (BGR) with a subtle indicator."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (20, 20, 25)
    cx, cy = w // 2, h // 2
    r = min(w, h) // 8
    cv2.circle(frame, (cx, cy), r, (40, 60, 40), 2, cv2.LINE_AA)
    cv2.line(frame, (cx - r // 2, cy), (cx + r // 2, cy), (40, 70, 40), 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - r // 2), (cx, cy + r // 2), (40, 70, 40), 1, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Standby", (cx - 45, cy + r + 24), font, 0.5, (50, 70, 50), 1, cv2.LINE_AA)
    return frame


# ── main loop ────────────────────────────────────────────────────────────────

def run_uvc(
    model: str = "p3",
    device: str | None = None,
    colormap: str = "rainbow",
    out_w: int = 640,
    out_h: int = 480,
) -> None:
    if sys.platform == "win32" or fcntl is None:
        raise RuntimeError("UVC virtual webcam requires Linux (v4l2loopback + fcntl).")
    cmap_map = {n.lower(): v for n, v in ColormapID.__members__.items()}
    cmap_id = cmap_map.get(colormap.lower(), ColormapID.RAINBOW)

    running = True

    def _stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    standby_bgr = _make_standby_frame(out_w, out_h)

    while running:
        camera = None
        writer = None
        streaming = False
        try:
            dev_path = device or find_v4l2_loopback_device()
            if dev_path is None or not os.path.exists(dev_path):
                time.sleep(5)
                continue

            writer = V4L2Writer(dev_path, out_w, out_h)
            print(f"Ready on {dev_path} ({out_w}x{out_h}) — standby")

            prev = None
            frame_count = 0
            t0 = time.time()
            last_reader_check = 0.0
            readers_active = False

            while running:
                now = time.monotonic()

                # Periodically check for readers (every ~1s)
                if now - last_reader_check > 1.0:
                    last_reader_check = now
                    readers_active = _has_readers(dev_path, writer.fd)

                if not readers_active:
                    # === STANDBY: camera off, send idle frame at ~0.5 fps ===
                    if streaming:
                        try:
                            camera.stop_streaming()
                        except Exception:
                            pass
                        camera = None
                        streaming = False
                        prev = None
                        frame_count = 0
                        print("\n  No readers — camera idle (standby)")
                    if not _usb_camera_present():
                        print("\n  Camera disconnected (standby).")
                        break
                    try:
                        writer.write(standby_bgr)
                    except OSError:
                        break
                    time.sleep(2.0)
                    continue

                # === ACTIVE: readers present, stream real thermal data ===
                if not streaming:
                    config = get_model_config(model)
                    camera = P3Camera(config=config)
                    camera.connect()
                    name, ver = camera.init()
                    camera.start_streaming()
                    streaming = True
                    t0 = time.time()
                    frame_count = 0
                    print(f"  Reader detected — streaming [{ColormapID(cmap_id).name}]"
                          f" ({name} fw:{ver})")

                try:
                    ir, thermal = camera.read_frame_both()
                except Exception:
                    if not running:
                        break
                    raise
                if thermal is None:
                    continue

                thermal = tnr(thermal, prev, 0.5)
                prev = thermal.copy()

                img = ir.copy() if ir is not None else agc_temporal(thermal, 1.0)
                h, w = img.shape[:2]
                img = np.asarray(
                    cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC),
                    dtype=np.uint8,
                )
                img = np.asarray(clahe.apply(img), dtype=np.uint8)
                img = dde(img, 0.3)
                bgr = apply_colormap(img, cmap_id)
                bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

                try:
                    writer.write(bgr)
                except OSError:
                    break

                frame_count += 1
                if frame_count % 50 == 0:
                    elapsed = time.time() - t0
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"\r  {frame_count} frames, {fps:.1f} fps → {dev_path}",
                          end="", flush=True)

        except Exception as e:
            if not running:
                break
            if not _usb_camera_present():
                print(f"\nCamera disconnected.")
                break
            print(f"\nError ({e}), retrying in 3s...")
            time.sleep(3)
            continue

        finally:
            if streaming and camera:
                try:
                    camera.stop_streaming()
                except Exception:
                    pass
            if writer:
                writer.close()

    print("\nStopped.")


def main():
    if sys.platform == "win32":
        print(
            "thermal-camera-viewer-uvc: virtual webcam is Linux-only (v4l2loopback).\n"
            "On Windows, use the desktop viewer:  python -m thermal_camera_viewer",
            file=sys.stderr,
        )
        raise SystemExit(2)
    import argparse
    parser = argparse.ArgumentParser(
        description="Stream thermal camera as a virtual webcam (UVC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  thermal-camera-viewer-uvc\n"
            "  thermal-camera-viewer-uvc --device /dev/video10\n"
            "  thermal-camera-viewer-uvc --colormap ironbow\n"
        ),
    )
    parser.add_argument("--model", choices=["p1", "p3"], default="p3")
    parser.add_argument("--device", type=str, default=None,
                        help="v4l2loopback device (e.g. /dev/video10)")
    parser.add_argument("--colormap", type=str, default="rainbow",
                        choices=[c.name.lower() for c in ColormapID],
                        help="Color palette (default: rainbow)")
    parser.add_argument("--width", type=int, default=640,
                        help="Output width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Output height (default: 480)")
    args = parser.parse_args()

    run_uvc(
        model=args.model,
        device=args.device,
        colormap=args.colormap,
        out_w=args.width,
        out_h=args.height,
    )


if __name__ == "__main__":
    main()
