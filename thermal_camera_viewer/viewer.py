#!/usr/bin/env python3
"""Thermal Camera Viewer — Qt-based thermal camera viewer.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from enum import IntEnum
from typing import Optional, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from PyQt5.QtCore import (
    QPoint, QPointF, QRect, QRectF, QSize, Qt, QThread, QTimer, pyqtSignal,
)
from PyQt5.QtGui import (
    QBrush, QColor, QCursor, QFont, QFontMetrics, QIcon, QImage,
    QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QPolygon, QPolygonF,
)
from PyQt5.QtWidgets import (
    QAction, QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QSizePolicy, QStatusBar, QToolBar, QWidget,
)

from .p3_camera import (
    GainMode, Model, P3Camera, get_model_config, raw_to_celsius_corrected,
)

# ── colormaps ────────────────────────────────────────────────────────────────

class ColormapID(IntEnum):
    WHITE_HOT = 0
    BLACK_HOT = 1
    RAINBOW = 2
    IRONBOW = 3
    MILITARY = 4
    SEPIA = 5

COLORMAPS: dict[ColormapID, NDArray[np.uint8]] = {}

def _cv_lut(cm: int) -> NDArray[np.uint8]:
    g = np.arange(256, dtype=np.uint8).reshape(1, 256)
    return cast(NDArray[np.uint8], cv2.applyColorMap(g, cm).reshape(256, 3))

def _init_colormaps() -> None:
    r = np.arange(256, dtype=np.uint8)
    def _gray(b, g, rv):
        lut = np.zeros((256, 3), dtype=np.uint8)
        lut[:, 0] = (r * b).astype(np.uint8)
        lut[:, 1] = (r * g).astype(np.uint8)
        lut[:, 2] = (r * rv).astype(np.uint8)
        return lut
    COLORMAPS[ColormapID.WHITE_HOT] = _gray(1, 1, 1)
    COLORMAPS[ColormapID.BLACK_HOT] = _gray(1, 1, 1)[::-1].copy()
    COLORMAPS[ColormapID.RAINBOW] = _cv_lut(cv2.COLORMAP_JET)
    COLORMAPS[ColormapID.IRONBOW] = _cv_lut(cv2.COLORMAP_INFERNO)
    COLORMAPS[ColormapID.MILITARY] = _gray(0.2, 1, 0.3)
    COLORMAPS[ColormapID.SEPIA] = _gray(0.4, 0.7, 1)

_init_colormaps()

def apply_colormap(u8: NDArray[np.uint8], cid: int) -> NDArray[np.uint8]:
    return COLORMAPS[ColormapID(cid)][u8]

# ── ISP helpers ──────────────────────────────────────────────────────────────

_ema_lo: float | None = None
_ema_hi: float | None = None

def agc_temporal(img: NDArray[np.uint16], pct: float = 1.0, alpha: float = 0.1) -> NDArray[np.uint8]:
    global _ema_lo, _ema_hi
    lo = float(np.percentile(img, pct))
    hi = float(np.percentile(img, 100.0 - pct))
    if _ema_lo is None:
        _ema_lo, _ema_hi = lo, hi
    else:
        _ema_lo = alpha * lo + (1 - alpha) * _ema_lo
        _ema_hi = alpha * hi + (1 - alpha) * _ema_hi
    if _ema_hi <= _ema_lo:
        return np.zeros(img.shape, dtype=np.uint8)
    n = (img.astype(np.float32) - _ema_lo) / (_ema_hi - _ema_lo)
    return (np.clip(n, 0, 1) * 255).astype(np.uint8)

def dde(u8: NDArray[np.uint8], s: float = 0.3) -> NDArray[np.uint8]:
    if s <= 0: return u8
    bl = cv2.GaussianBlur(u8, (3, 3), 0).astype(np.float32)
    return np.clip(u8.astype(np.float32) + s * (u8.astype(np.float32) - bl), 0, 255).astype(np.uint8)

def tnr(cur: NDArray[np.uint16], prev: NDArray[np.uint16] | None, a: float = 0.5) -> NDArray[np.uint16]:
    if prev is None: return cur
    return (a * cur.astype(np.float32) + (1 - a) * prev.astype(np.float32)).astype(np.uint16)

# ── temperature helpers ──────────────────────────────────────────────────────

def c2f(c: float) -> float: return c * 1.8 + 32.0

def fmt_t(c: float, f: bool) -> str:
    return f"{c2f(c):.1f} °F" if f else f"{c:.1f} °C"

# ── icon factory ─────────────────────────────────────────────────────────────

_ICON_SZ = 28

def _mk(draw_fn) -> QIcon:
    pm = QPixmap(_ICON_SZ, _ICON_SZ)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    draw_fn(p)
    p.end()
    return QIcon(pm)

def _pen(p: QPainter, c: QColor = QColor(210, 210, 210), w: float = 1.8):
    pen = QPen(c, w)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)

def icon_screenshot() -> QIcon:
    def d(p: QPainter):
        _pen(p)
        p.drawRoundedRect(3, 8, 22, 15, 2, 2)
        p.drawEllipse(10, 11, 8, 8)
        p.drawRect(11, 5, 6, 4)
    return _mk(d)

def icon_record() -> QIcon:
    def d(p: QPainter):
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(220, 50, 50))
        p.drawEllipse(5, 5, 18, 18)
    return _mk(d)

def icon_stop_rec() -> QIcon:
    def d(p: QPainter):
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(220, 50, 50))
        p.drawRect(6, 6, 16, 16)
    return _mk(d)

def icon_rot_cw() -> QIcon:
    def d(p: QPainter):
        _pen(p)
        p.drawArc(5, 5, 18, 18, 60 * 16, -300 * 16)
        p.drawLine(20, 6, 20, 12)
        p.drawLine(20, 6, 15, 8)
    return _mk(d)

def icon_rot_ccw() -> QIcon:
    def d(p: QPainter):
        _pen(p)
        p.drawArc(5, 5, 18, 18, 120 * 16, 300 * 16)
        p.drawLine(8, 6, 8, 12)
        p.drawLine(8, 6, 13, 8)
    return _mk(d)

def icon_flip_h() -> QIcon:
    def d(p: QPainter):
        _pen(p)
        p.drawLine(4, 14, 10, 8)
        p.drawLine(4, 14, 10, 20)
        p.drawLine(24, 14, 18, 8)
        p.drawLine(24, 14, 18, 20)
        _pen(p, QColor(100, 100, 100))
        p.drawLine(14, 5, 14, 23)
    return _mk(d)

def icon_flip_v() -> QIcon:
    def d(p: QPainter):
        _pen(p)
        p.drawLine(14, 4, 8, 10)
        p.drawLine(14, 4, 20, 10)
        p.drawLine(14, 24, 8, 18)
        p.drawLine(14, 24, 20, 18)
        _pen(p, QColor(100, 100, 100))
        p.drawLine(5, 14, 23, 14)
    return _mk(d)

def icon_zoom_in() -> QIcon:
    def d(p: QPainter):
        _pen(p)
        p.drawEllipse(4, 4, 16, 16)
        p.drawLine(12, 7, 12, 17)
        p.drawLine(7, 12, 17, 12)
        _pen(p, w=2.2)
        p.drawLine(18, 18, 24, 24)
    return _mk(d)

def icon_zoom_out() -> QIcon:
    def d(p: QPainter):
        _pen(p)
        p.drawEllipse(4, 4, 16, 16)
        p.drawLine(7, 12, 17, 12)
        _pen(p, w=2.2)
        p.drawLine(18, 18, 24, 24)
    return _mk(d)

def icon_palette() -> QIcon:
    def d(p: QPainter):
        p.setPen(Qt.NoPen)
        for i, c in enumerate([QColor(30, 0, 180), QColor(180, 0, 140), QColor(220, 100, 0), QColor(255, 230, 50)]):
            p.setBrush(c)
            x = 4 + (i % 2) * 11
            y = 4 + (i // 2) * 11
            p.drawRect(x, y, 10, 10)
    return _mk(d)

def icon_thermo() -> QIcon:
    def d(p: QPainter):
        _pen(p, QColor(210, 210, 210))
        p.drawRoundedRect(11, 3, 6, 16, 3, 3)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(220, 60, 60))
        p.drawEllipse(10, 17, 8, 8)
        p.drawRect(12, 10, 4, 9)
    return _mk(d)

def icon_hotspot() -> QIcon:
    def d(p: QPainter):
        _pen(p, QColor(220, 60, 60))
        p.drawEllipse(7, 7, 14, 14)
        p.drawEllipse(11, 11, 6, 6)
        _pen(p, QColor(210, 210, 210))
        p.drawLine(14, 2, 14, 8)
        p.drawLine(14, 20, 14, 26)
        p.drawLine(2, 14, 8, 14)
        p.drawLine(20, 14, 26, 14)
    return _mk(d)

def icon_reticule() -> QIcon:
    def d(p: QPainter):
        _pen(p, QColor(0, 220, 220))
        p.drawLine(14, 4, 14, 11)
        p.drawLine(14, 17, 14, 24)
        p.drawLine(4, 14, 11, 14)
        p.drawLine(17, 14, 24, 14)
    return _mk(d)

def icon_colorbar() -> QIcon:
    def d(p: QPainter):
        for i in range(20):
            frac = i / 19.0
            r = int(frac * 255)
            b = int((1 - frac) * 255)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(b, int(frac * 100), r))
            p.drawRect(9, 4 + i, 10, 1)
        _pen(p)
        p.setBrush(Qt.NoBrush)
        p.drawRect(9, 4, 10, 20)
    return _mk(d)

def icon_shutter() -> QIcon:
    def d(p: QPainter):
        _pen(p)
        p.drawEllipse(4, 4, 20, 20)
        for angle in range(0, 360, 60):
            p.save()
            p.translate(14, 14)
            p.rotate(angle)
            p.drawLine(0, -4, 0, -10)
            p.restore()
    return _mk(d)

def icon_gain() -> QIcon:
    def d(p: QPainter):
        _pen(p, QColor(80, 200, 120), 3)
        for i, h in enumerate([6, 11, 17]):
            x = 6 + i * 7
            p.drawLine(x, 24 - h, x, 24)
    return _mk(d)

def icon_emissivity() -> QIcon:
    def d(p: QPainter):
        _pen(p, QColor(255, 180, 50))
        for i in range(3):
            y = 7 + i * 5
            path = QPainterPath()
            path.moveTo(6, y)
            path.cubicTo(10, y - 3, 18, y + 3, 22, y)
            p.drawPath(path)
    return _mk(d)

def icon_enhanced() -> QIcon:
    def d(p: QPainter):
        _pen(p, QColor(255, 220, 80), 1.5)
        pts = []
        for i in range(5):
            import math
            a = math.radians(-90 + i * 72)
            pts.append(QPointF(14 + 10 * math.cos(a), 14 + 10 * math.sin(a)))
            a2 = math.radians(-90 + i * 72 + 36)
            pts.append(QPointF(14 + 4 * math.cos(a2), 14 + 4 * math.sin(a2)))
        p.drawPolygon(QPolygonF(pts))
    return _mk(d)

def icon_help() -> QIcon:
    def d(p: QPainter):
        _pen(p)
        p.drawEllipse(4, 4, 20, 20)
        f = QFont("sans-serif", 13, QFont.Bold)
        p.setFont(f)
        p.drawText(QRect(4, 3, 20, 20), Qt.AlignCenter, "?")
    return _mk(d)

# ── camera thread ────────────────────────────────────────────────────────────

class CameraThread(QThread):
    frame_ready = pyqtSignal(object, object)
    status_msg = pyqtSignal(str)
    error_msg = pyqtSignal(str)
    cam_info = pyqtSignal(str, str)

    def __init__(self, model: str = "p3"):
        super().__init__()
        self._model = model
        self._running = False
        self.camera: P3Camera | None = None

    def run(self):
        try:
            config = get_model_config(self._model)
            self.camera = P3Camera(config=config)
            self.camera.connect()
            name, ver = self.camera.init()
            self.cam_info.emit(name, ver)
            self.camera.start_streaming()
            self.status_msg.emit("Streaming")
        except Exception as e:
            self.error_msg.emit(str(e))
            return

        self._running = True
        prev: NDArray[np.uint16] | None = None
        while self._running:
            try:
                ir, thermal = self.camera.read_frame_both()
                if thermal is None:
                    continue
                thermal = tnr(thermal, prev, 0.5)
                prev = thermal.copy()
                self.frame_ready.emit(ir, thermal)
            except Exception:
                if not self._running:
                    break

        try:
            self.camera.stop_streaming()
        except Exception:
            pass

    def stop(self):
        self._running = False
        self.wait(3000)

# ── thermal image widget ─────────────────────────────────────────────────────

class ThermalWidget(QWidget):
    mouse_info = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)

        # state
        self.colormap_idx: int = ColormapID.RAINBOW
        self.rotation: int = 0
        self.flip_h: bool = False
        self.flip_v: bool = False
        self.enhanced: bool = True
        self.use_f: bool = False
        self.show_reticule: bool = True
        self.show_colorbar: bool = True
        self.show_hotspots: bool = True
        self.dde_strength: float = 0.3

        # frame data
        self._thermal: NDArray[np.uint16] | None = None
        self._temps: NDArray[np.float32] | None = None
        self._pixmap: QPixmap | None = None
        self._proc_w: int = 0
        self._proc_h: int = 0
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # display mapping
        self._zoom: float = 1.0
        self._scale: float = 1.0
        self._off_x: float = 0
        self._off_y: float = 0

        # mouse
        self._mouse_pos: QPoint | None = None
        self._roi_start: QPoint | None = None
        self._roi_end: QPoint | None = None
        self._roi_fixed: QRect | None = None

    # ── frame processing ──

    def update_frame(self, ir_brightness: NDArray[np.uint8] | None, thermal: NDArray[np.uint16]) -> None:
        self._thermal = thermal
        self._temps = raw_to_celsius_corrected(thermal, self._env_params())

        if ir_brightness is not None:
            img = ir_brightness.copy()
        else:
            img = agc_temporal(thermal, 1.0)

        if self.enhanced:
            h, w = img.shape[:2]
            img = np.asarray(cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC), dtype=np.uint8)
            img = np.asarray(self._clahe.apply(img), dtype=np.uint8)
            img = dde(img, self.dde_strength)

        bgr = apply_colormap(img, self.colormap_idx)
        if self.flip_h:
            bgr = cv2.flip(bgr, 1)
        if self.flip_v:
            bgr = cv2.flip(bgr, 0)
        if self.rotation == 90:
            bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            bgr = cv2.rotate(bgr, cv2.ROTATE_180)
        elif self.rotation == 270:
            bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

        self._proc_h, self._proc_w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, self._proc_w, self._proc_h, self._proc_w * 3, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def _env_params(self):
        mw = self._main_window()
        if mw and mw._cam_thread and mw._cam_thread.camera:
            return mw._cam_thread.camera.env_params
        from .p3_camera import EnvParams
        return EnvParams()

    def _main_window(self) -> Optional["MainWindow"]:
        w = self.window()
        return w if isinstance(w, MainWindow) else None

    def get_bgr_frame(self) -> NDArray[np.uint8] | None:
        if self._pixmap is None:
            return None
        qimg = self._pixmap.toImage().convertToFormat(QImage.Format_RGB888)
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        arr = np.array(ptr).reshape(qimg.height(), qimg.width(), 3)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # ── coordinate transforms ──

    def _compute_mapping(self):
        if self._proc_w == 0 or self._proc_h == 0:
            return
        ww, wh = self.width(), self.height()
        base = min(ww / self._proc_w, wh / self._proc_h)
        self._scale = base * self._zoom
        disp_w = self._proc_w * self._scale
        disp_h = self._proc_h * self._scale
        self._off_x = (ww - disp_w) / 2
        self._off_y = (wh - disp_h) / 2

    def _w2img(self, pos: QPoint) -> Optional[tuple[float, float]]:
        if self._scale <= 0:
            return None
        ix = (pos.x() - self._off_x) / self._scale
        iy = (pos.y() - self._off_y) / self._scale
        if 0 <= ix < self._proc_w and 0 <= iy < self._proc_h:
            return (ix, iy)
        return None

    def _img2thermal(self, ix: float, iy: float) -> Optional[tuple[int, int]]:
        if self._thermal is None:
            return None
        th, tw = self._thermal.shape

        if self.rotation == 90:
            px, py = iy, self._proc_w - 1 - ix
        elif self.rotation == 180:
            px, py = self._proc_w - 1 - ix, self._proc_h - 1 - iy
        elif self.rotation == 270:
            px, py = self._proc_h - 1 - iy, ix
        else:
            px, py = ix, iy

        ew = (self._proc_h if self.rotation in (90, 270) else self._proc_w)
        eh = (self._proc_w if self.rotation in (90, 270) else self._proc_h)
        if self.flip_h:
            px = ew - 1 - px
        if self.flip_v:
            py = eh - 1 - py

        col = int(px / ew * tw)
        row = int(py / eh * th)
        if 0 <= row < th and 0 <= col < tw:
            return (row, col)
        return None

    def _w2thermal(self, pos: QPoint) -> Optional[tuple[int, int]]:
        img = self._w2img(pos)
        if img is None:
            return None
        return self._img2thermal(*img)

    def _thermal2w(self, row: int, col: int) -> QPoint:
        if self._thermal is None:
            return QPoint(0, 0)
        th, tw = self._thermal.shape
        ew = (self._proc_h if self.rotation in (90, 270) else self._proc_w)
        eh = (self._proc_w if self.rotation in (90, 270) else self._proc_h)
        px = (col + 0.5) / tw * ew
        py = (row + 0.5) / th * eh
        if self.flip_h:
            px = ew - 1 - px
        if self.flip_v:
            py = eh - 1 - py
        if self.rotation == 90:
            ix, iy = eh - 1 - py, px
        elif self.rotation == 180:
            ix, iy = ew - 1 - px, eh - 1 - py
        elif self.rotation == 270:
            ix, iy = py, ew - 1 - px
        else:
            ix, iy = px, py
        wx = int(ix * self._scale + self._off_x)
        wy = int(iy * self._scale + self._off_y)
        return QPoint(wx, wy)

    # ── mouse events ──

    def mouseMoveEvent(self, e):
        self._mouse_pos = e.pos()
        if self._roi_start is not None and e.buttons() & Qt.LeftButton:
            self._roi_end = e.pos()
        tc = self._w2thermal(e.pos())
        if tc and self._temps is not None:
            t = float(self._temps[tc[0], tc[1]])
            self.mouse_info.emit(f"Cursor: {fmt_t(t, self.use_f)}  ({tc[1]}, {tc[0]})")
        else:
            self.mouse_info.emit("")
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._roi_start = e.pos()
            self._roi_end = e.pos()
            self._roi_fixed = None
        elif e.button() == Qt.RightButton:
            self._roi_fixed = None
            self._roi_start = None
            self._roi_end = None
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._roi_start is not None:
            self._roi_end = e.pos()
            r = QRect(self._roi_start, self._roi_end).normalized()
            if r.width() > 8 and r.height() > 8:
                self._roi_fixed = r
            else:
                self._roi_fixed = None
            self._roi_start = None
            self._roi_end = None
            self.update()

    def enterEvent(self, e):
        self.setCursor(QCursor(Qt.BlankCursor))

    def leaveEvent(self, e):
        self.setCursor(QCursor(Qt.ArrowCursor))
        self._mouse_pos = None
        self.mouse_info.emit("")
        self.update()

    # ── painting ──

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        p.fillRect(self.rect(), QColor(20, 20, 25))
        self._compute_mapping()

        if self._pixmap:
            target = QRectF(self._off_x, self._off_y,
                            self._proc_w * self._scale, self._proc_h * self._scale)
            p.drawPixmap(target.toRect(), self._pixmap)

        self._paint_crosshair(p)
        self._paint_reticule(p)
        self._paint_hotspots(p)
        self._paint_roi(p)
        self._paint_colorbar(p)
        self._paint_status(p)
        p.end()

    def _adaptive_font(self, base: int = 11) -> QFont:
        sz = max(9, int(base * min(self.width(), self.height()) / 600))
        return QFont("sans-serif", sz)

    def _draw_label(self, p: QPainter, x: int, y: int, text: str,
                    fg: QColor, font: QFont | None = None, bg_alpha: int = 110,
                    shadow: bool = True):
        if font is None:
            font = self._adaptive_font()
        p.setFont(font)
        fm = QFontMetrics(font)
        r = fm.boundingRect(text)
        if bg_alpha > 0:
            pad = 4
            bg_rect = QRect(x - pad, y - r.height() - pad,
                            r.width() + pad * 3, r.height() + pad * 2)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, bg_alpha))
            p.drawRoundedRect(bg_rect, 3, 3)
        if shadow:
            p.setPen(QColor(0, 0, 0, 180))
            p.drawText(x + 1, y + 1, text)
        p.setPen(fg)
        p.drawText(x, y, text)
        return r

    def _paint_crosshair(self, p: QPainter):
        if self._mouse_pos is None:
            return
        img = self._w2img(self._mouse_pos)
        if img is None:
            return
        mx, my = self._mouse_pos.x(), self._mouse_pos.y()
        r = max(6, int(10 * self._scale))
        gap = max(2, int(3 * self._scale))
        lw = max(1, int(1.2 * self._scale))
        p.setPen(QPen(QColor(0, 0, 0, 120), lw + 2, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(mx - r, my, mx - gap, my)
        p.drawLine(mx + gap, my, mx + r, my)
        p.drawLine(mx, my - r, mx, my - gap)
        p.drawLine(mx, my + gap, mx, my + r)
        p.setPen(QPen(QColor(0, 230, 0, 220), lw, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(mx - r, my, mx - gap, my)
        p.drawLine(mx + gap, my, mx + r, my)
        p.drawLine(mx, my - r, mx, my - gap)
        p.drawLine(mx, my + gap, mx, my + r)

    def _paint_reticule(self, p: QPainter):
        if not self.show_reticule or self._temps is None:
            return
        cx, cy = self.width() // 2, self.height() // 2
        tc = self._w2thermal(QPoint(cx, cy))
        if tc is None:
            return
        t = float(self._temps[tc[0], tc[1]])
        r = max(10, int(16 * self._scale))
        gap = max(3, int(4 * self._scale))
        pen = QPen(QColor(0, 230, 230, 200), max(1, int(1.5 * self._scale)))
        p.setPen(pen)
        p.drawLine(cx - r, cy, cx - gap, cy)
        p.drawLine(cx + gap, cy, cx + r, cy)
        p.drawLine(cx, cy - r, cx, cy - gap)
        p.drawLine(cx, cy + gap, cx, cy + r)
        self._draw_label(p, cx + r + 4, cy - 2, fmt_t(t, self.use_f), QColor(0, 230, 230))

    def _paint_hotspots(self, p: QPainter):
        if not self.show_hotspots or self._temps is None:
            return
        if self._roi_fixed is not None:
            return
        min_pos = np.unravel_index(np.argmin(self._temps), self._temps.shape)
        max_pos = np.unravel_index(np.argmax(self._temps), self._temps.shape)
        self._paint_marker(p, max_pos, float(np.max(self._temps)), QColor(255, 50, 50), "MAX")
        self._paint_marker(p, min_pos, float(np.min(self._temps)), QColor(50, 120, 255), "MIN")

    def _paint_marker(self, p: QPainter, tpos: tuple, temp: float, color: QColor, label: str):
        pt = self._thermal2w(int(tpos[0]), int(tpos[1]))
        r = max(2, int(3 * self._scale))
        pen = QPen(color, max(1, int(1.2 * self._scale)))
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRect(pt.x() - r, pt.y() - r, r * 2, r * 2)
        text = f"{label} {fmt_t(temp, self.use_f)}"
        font = self._adaptive_font(8)
        ty = pt.y() - r - 4 if pt.y() - r - 16 > 10 else pt.y() + r + 14
        self._draw_label(p, pt.x() - r, ty, text, QColor(255, 255, 255), font=font, bg_alpha=0)

    def _paint_roi(self, p: QPainter):
        # dragging preview
        if self._roi_start is not None and self._roi_end is not None:
            r = QRect(self._roi_start, self._roi_end).normalized()
            pen = QPen(QColor(255, 255, 0, 200), max(1, int(2 * self._scale)))
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.setBrush(QColor(255, 255, 0, 25))
            p.drawRect(r)
            return

        if self._roi_fixed is None or self._thermal is None or self._temps is None:
            return

        r = self._roi_fixed
        pen = QPen(QColor(255, 255, 0, 220), max(1, int(2 * self._scale)))
        p.setPen(pen)
        p.setBrush(QColor(255, 255, 0, 20))
        p.drawRect(r)

        # map ROI corners to thermal
        tl = self._w2thermal(r.topLeft())
        br = self._w2thermal(r.bottomRight())
        if tl is None or br is None:
            return

        r1, c1 = tl
        r2, c2 = br
        rmin, rmax = min(r1, r2), max(r1, r2)
        cmin, cmax = min(c1, c2), max(c1, c2)
        th, tw = self._thermal.shape
        rmin, rmax = max(0, rmin), min(th - 1, rmax)
        cmin, cmax = max(0, cmin), min(tw - 1, cmax)
        if rmax <= rmin or cmax <= cmin:
            return

        roi = self._temps[rmin:rmax + 1, cmin:cmax + 1]
        if roi.size == 0:
            return

        roi_min = float(np.min(roi))
        roi_max = float(np.max(roi))
        roi_avg = float(np.mean(roi))
        min_loc = np.unravel_index(np.argmin(roi), roi.shape)
        max_loc = np.unravel_index(np.argmax(roi), roi.shape)

        self._paint_marker(p, (rmin + max_loc[0], cmin + max_loc[1]), roi_max, QColor(255, 50, 50), "MAX")
        self._paint_marker(p, (rmin + min_loc[0], cmin + min_loc[1]), roi_min, QColor(50, 120, 255), "MIN")

        avg_text = f"Avg {fmt_t(roi_avg, self.use_f)}"
        font = self._adaptive_font(8)
        ty = r.bottom() + 18
        self._draw_label(p, r.left(), ty, avg_text, QColor(255, 255, 0),
                         font=font, bg_alpha=0)

    def _paint_colorbar(self, p: QPainter):
        if not self.show_colorbar or self._temps is None:
            return
        ww, wh = self.width(), self.height()
        bar_w = max(12, int(16 * self._scale))
        bar_h = int(wh * 0.45)
        x0 = ww - bar_w - max(50, int(55 * self._scale))
        y0 = (wh - bar_h) // 2
        if x0 < 0:
            return

        lut = COLORMAPS[ColormapID(self.colormap_idx)]
        for i in range(bar_h):
            idx = int((1 - i / bar_h) * 255)
            bgr = lut[idx]
            p.setPen(QColor(int(bgr[2]), int(bgr[1]), int(bgr[0])))
            p.drawLine(x0, y0 + i, x0 + bar_w, y0 + i)

        p.setPen(QColor(200, 200, 200))
        p.setBrush(Qt.NoBrush)
        p.drawRect(x0, y0, bar_w, bar_h)

        t_min = float(np.min(self._temps))
        t_max = float(np.max(self._temps))
        font = self._adaptive_font(9)
        p.setFont(font)
        for i in range(5):
            frac = i / 4.0
            ty = y0 + int((1 - frac) * bar_h)
            temp = t_min + frac * (t_max - t_min)
            p.setPen(QColor(200, 200, 200, 180))
            p.drawLine(x0, ty, x0 + bar_w, ty)
            p.drawText(x0 + bar_w + 4, ty + 4, fmt_t(temp, self.use_f))

    def _paint_status(self, p: QPainter):
        if self._temps is None:
            return
        t_min = float(np.min(self._temps))
        t_max = float(np.max(self._temps))
        font = self._adaptive_font(10)
        p.setFont(font)
        text = f"Range: {fmt_t(t_min, self.use_f)} ~ {fmt_t(t_max, self.use_f)}  |  {ColormapID(self.colormap_idx).name}"
        p.setPen(Qt.NoPen)
        fm = QFontMetrics(font)
        tw = fm.horizontalAdvance(text)
        p.setBrush(QColor(0, 0, 0, 100))
        p.drawRoundedRect(4, 4, tw + 16, fm.height() + 8, 4, 4)
        p.setPen(QColor(220, 220, 220))
        p.drawText(12, fm.ascent() + 8, text)

# ── main window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, model: str = "p3"):
        super().__init__()
        self.setWindowTitle("Thermal Camera Viewer")
        self.resize(1024, 680)

        self._model = model
        self._recording = False
        self._ffmpeg_proc: subprocess.Popen | None = None
        self._video_file = ""
        self._fps = 0.0
        self._fps_count = 0
        self._fps_time = time.time()

        # central widget
        self._thermal_w = ThermalWidget(self)
        self.setCentralWidget(self._thermal_w)

        # toolbar
        self._setup_toolbar()

        # status bar
        self._status_cursor = QLabel("")
        self._status_fps = QLabel("")
        self._status_info = QLabel("")
        sb = self.statusBar()
        sb.addWidget(self._status_cursor, 2)
        sb.addWidget(self._status_info, 2)
        sb.addPermanentWidget(self._status_fps, 0)
        self._thermal_w.mouse_info.connect(self._status_cursor.setText)

        # fps timer
        self._fps_timer = QTimer()
        self._fps_timer.timeout.connect(self._update_fps_label)
        self._fps_timer.start(1000)

        # camera thread
        self._cam_thread = CameraThread(model)
        self._cam_thread.frame_ready.connect(self._on_frame)
        self._cam_thread.cam_info.connect(self._on_cam_info)
        self._cam_thread.error_msg.connect(self._on_cam_error)
        self._cam_thread.status_msg.connect(lambda s: self._status_info.setText(s))
        self._cam_thread.start()

    def _setup_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setIconSize(QSize(28, 28))
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonIconOnly)

        def act(icon: QIcon, tip: str, slot, checkable=False) -> QAction:
            a = QAction(icon, tip, self)
            a.setToolTip(tip)
            a.setStatusTip(tip)
            if checkable:
                a.setCheckable(True)
                a.toggled.connect(slot)
            else:
                a.triggered.connect(slot)
            tb.addAction(a)
            return a

        act(icon_screenshot(), "Screenshot  [Space]", self._on_screenshot)
        self._act_rec = act(icon_record(), "Start / Stop recording  [F5]", self._on_record)
        tb.addSeparator()
        act(icon_rot_cw(), "Rotate 90° clockwise  [R]", self._on_rot_cw)
        act(icon_rot_ccw(), "Rotate 90° counter-clockwise  [Shift+R]", self._on_rot_ccw)
        self._act_flip_h = act(icon_flip_h(), "Flip horizontal  [M]",
            lambda c: setattr(self._thermal_w, "flip_h", c), checkable=True)
        self._act_flip_v = act(icon_flip_v(), "Flip vertical  [V]",
            lambda c: setattr(self._thermal_w, "flip_v", c), checkable=True)
        tb.addSeparator()
        act(icon_zoom_in(), "Zoom in  [+]", self._on_zoom_in)
        act(icon_zoom_out(), "Zoom out  [-]", self._on_zoom_out)
        tb.addSeparator()
        act(icon_palette(), "Cycle color palette  [C]", self._on_palette)
        act(icon_thermo(), "Toggle °C / °F  [F]", self._on_unit)
        self._act_hotspot = act(icon_hotspot(), "Show hotspot markers  [N]",
            lambda c: setattr(self._thermal_w, "show_hotspots", c), checkable=True)
        self._act_hotspot.setChecked(True)
        self._act_reticule = act(icon_reticule(), "Center reticule  [T]",
            lambda c: setattr(self._thermal_w, "show_reticule", c), checkable=True)
        self._act_reticule.setChecked(True)
        self._act_colorbar = act(icon_colorbar(), "Color bar  [B]",
            lambda c: setattr(self._thermal_w, "show_colorbar", c), checkable=True)
        self._act_colorbar.setChecked(True)
        tb.addSeparator()
        act(icon_shutter(), "Trigger shutter / NUC  [S]", self._on_shutter)
        act(icon_gain(), "Toggle gain high/low  [G]", self._on_gain)
        act(icon_emissivity(), "Cycle emissivity  [E]", self._on_emissivity)
        self._act_enhanced = act(icon_enhanced(), "Enhanced mode (CLAHE+DDE)  [P]",
            lambda c: setattr(self._thermal_w, "enhanced", c), checkable=True)
        self._act_enhanced.setChecked(True)
        tb.addSeparator()
        act(icon_help(), "Help  [H]", self._on_help)

    # ── actions ──


    @staticmethod
    def _xdg_dir(xdg_var: str, fallback: str) -> str:
        if sys.platform == "win32":
            base = os.environ.get("USERPROFILE", os.path.expanduser("~"))
            if "PICTURE" in xdg_var.upper():
                d = os.path.join(base, "Pictures")
            else:
                d = os.path.join(base, "Videos")
        else:
            d = os.environ.get(xdg_var, "")
            if not d:
                d = os.path.join(os.path.expanduser("~"), fallback)
        os.makedirs(d, exist_ok=True)
        return d

    def _on_screenshot(self):
        bgr = self._thermal_w.get_bgr_frame()
        if bgr is None:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        d = self._xdg_dir("XDG_PICTURES_DIR", "Pictures")
        fn = os.path.join(d, f"thermal_camera_{ts}.png")
        cv2.imwrite(fn, bgr)
        self.statusBar().showMessage(f"Saved: {fn}", 3000)

    def _on_record(self):
        if not self._recording:
            ts = time.strftime("%Y%m%d_%H%M%S")
            d = self._xdg_dir("XDG_VIDEOS_DIR", "Videos")
            self._video_file = os.path.join(d, f"thermal_camera_{ts}.mp4")
            pw, ph = self._thermal_w._proc_w, self._thermal_w._proc_h
            if pw > 0 and ph > 0:
                self._ffmpeg_proc = subprocess.Popen(
                    [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-f", "rawvideo", "-vcodec", "rawvideo",
                        "-pix_fmt", "bgr24",
                        "-s", f"{pw}x{ph}", "-r", "25",
                        "-i", "pipe:0",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                        "-pix_fmt", "yuv420p",
                        self._video_file,
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                self._recording = True
                self._act_rec.setIcon(icon_stop_rec())
                self._act_rec.setToolTip("Stop recording  [F5]")
                self.statusBar().showMessage(f"Recording: {self._video_file}", 3000)
        else:
            self._recording = False
            if self._ffmpeg_proc and self._ffmpeg_proc.stdin:
                try:
                    self._ffmpeg_proc.stdin.close()
                except OSError:
                    pass
                self._ffmpeg_proc.wait(timeout=10)
                self._ffmpeg_proc = None
            self._act_rec.setIcon(icon_record())
            self._act_rec.setToolTip("Start recording  [F5]")
            self.statusBar().showMessage(f"Saved: {self._video_file}", 3000)

    def _on_rot_cw(self):
        self._thermal_w.rotation = (self._thermal_w.rotation + 90) % 360

    def _on_rot_ccw(self):
        self._thermal_w.rotation = (self._thermal_w.rotation - 90) % 360

    def _on_zoom_in(self):
        w = self._thermal_w
        w._zoom = min(8.0, w._zoom * 1.25)
        self.statusBar().showMessage(f"Zoom: {w._zoom:.1f}x", 1500)

    def _on_zoom_out(self):
        w = self._thermal_w
        z = w._zoom / 1.25
        if z < 1.05:
            z = 1.0
        w._zoom = z
        self.statusBar().showMessage(f"Zoom: {w._zoom:.1f}x", 1500)

    def _on_palette(self):
        w = self._thermal_w
        w.colormap_idx = (w.colormap_idx + 1) % len(ColormapID)
        self.statusBar().showMessage(f"Palette: {ColormapID(w.colormap_idx).name}", 2000)

    def _on_unit(self):
        self._thermal_w.use_f = not self._thermal_w.use_f
        u = "Fahrenheit" if self._thermal_w.use_f else "Celsius"
        self.statusBar().showMessage(f"Unit: {u}", 2000)

    def _on_shutter(self):
        cam = self._cam_thread.camera
        if cam:
            try:
                cam.trigger_shutter()
                self.statusBar().showMessage("Shutter triggered", 2000)
            except Exception as e:
                self.statusBar().showMessage(f"Shutter error: {e}", 3000)

    def _on_gain(self):
        cam = self._cam_thread.camera
        if cam:
            nm = GainMode.LOW if cam.gain_mode == GainMode.HIGH else GainMode.HIGH
            cam.set_gain_mode(nm)
            self.statusBar().showMessage(f"Gain: {nm.name}", 2000)

    def _on_emissivity(self):
        cam = self._cam_thread.camera
        if cam:
            vals = [0.95, 0.90, 0.85, 0.80, 0.70, 0.50, 0.30, 0.10]
            cur = cam.env_params.emissivity
            idx = 0
            for i, v in enumerate(vals):
                if abs(cur - v) < 0.01:
                    idx = (i + 1) % len(vals)
                    break
            cam.env_params.emissivity = vals[idx]
            self.statusBar().showMessage(f"Emissivity: {vals[idx]:.2f}", 2000)

    def _on_help(self):
        QMessageBox.information(self, "Thermal Camera Viewer — Help",
            "<h3>Thermal Camera Viewer</h3>"
            "<p><b>Mouse:</b> Hover to see temperature. Left-drag to draw ROI box. Right-click to clear ROI.</p>"
            "<p><b>ROI box</b> shows MAX (red), MIN (blue) markers at exact positions and average temperature.</p>"
            "<hr>"
            "<b>Keyboard shortcuts:</b><br>"
            "<table>"
            "<tr><td>Space</td><td>Screenshot</td><td>F5</td><td>Record</td></tr>"
            "<tr><td>R / Shift+R</td><td>Rotate CW/CCW</td><td>M</td><td>Flip H</td></tr>"
            "<tr><td>V</td><td>Flip V</td><td>+/-</td><td>Zoom</td></tr>"
            "<tr><td>C</td><td>Palette</td><td>F</td><td>°C/°F</td></tr>"
            "<tr><td>N</td><td>Hotspots</td><td>T</td><td>Reticule</td></tr>"
            "<tr><td>B</td><td>Colorbar</td><td>P</td><td>Enhanced</td></tr>"
            "<tr><td>S</td><td>Shutter</td><td>G</td><td>Gain</td></tr>"
            "<tr><td>E</td><td>Emissivity</td><td>Q</td><td>Quit</td></tr>"
            "</table>"
        )

    # ── frame handling ──

    def _on_frame(self, ir, thermal):
        self._thermal_w.update_frame(ir, thermal)
        self._fps_count += 1
        if self._recording and self._ffmpeg_proc and self._ffmpeg_proc.stdin:
            bgr = self._thermal_w.get_bgr_frame()
            if bgr is not None:
                try:
                    self._ffmpeg_proc.stdin.write(bgr.tobytes())
                except (BrokenPipeError, OSError):
                    self._recording = False
                    self._ffmpeg_proc = None
                    self._act_rec.setIcon(icon_record())
                    self.statusBar().showMessage("Recording interrupted", 3000)

    def _on_cam_info(self, name, ver):
        self._status_info.setText(f"{name}  fw:{ver}")

    def _on_cam_error(self, msg):
        QMessageBox.critical(self, "Camera Error", f"Failed to connect:\n{msg}\n\nMake sure no other viewer is running.")

    def _update_fps_label(self):
        now = time.time()
        dt = now - self._fps_time
        if dt > 0:
            self._fps = self._fps_count / dt
        self._fps_count = 0
        self._fps_time = now
        self._status_fps.setText(f"{self._fps:.0f} fps")

    # ── keyboard shortcuts ──

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_Q:
            self.close()
        elif k == Qt.Key_Space:
            self._on_screenshot()
        elif k == Qt.Key_F5:
            self._on_record()
        elif k == Qt.Key_R:
            if e.modifiers() & Qt.ShiftModifier:
                self._on_rot_ccw()
            else:
                self._on_rot_cw()
        elif k == Qt.Key_M:
            self._act_flip_h.setChecked(not self._act_flip_h.isChecked())
        elif k == Qt.Key_V:
            self._act_flip_v.setChecked(not self._act_flip_v.isChecked())
        elif k == Qt.Key_Plus or k == Qt.Key_Equal:
            self._on_zoom_in()
        elif k == Qt.Key_Minus:
            self._on_zoom_out()
        elif k == Qt.Key_C:
            self._on_palette()
        elif k == Qt.Key_F:
            self._on_unit()
        elif k == Qt.Key_N:
            self._act_hotspot.setChecked(not self._act_hotspot.isChecked())
        elif k == Qt.Key_T:
            self._act_reticule.setChecked(not self._act_reticule.isChecked())
        elif k == Qt.Key_B:
            self._act_colorbar.setChecked(not self._act_colorbar.isChecked())
        elif k == Qt.Key_P:
            self._act_enhanced.setChecked(not self._act_enhanced.isChecked())
        elif k == Qt.Key_S:
            self._on_shutter()
        elif k == Qt.Key_G:
            self._on_gain()
        elif k == Qt.Key_E:
            self._on_emissivity()
        elif k == Qt.Key_H:
            self._on_help()
        elif k == Qt.Key_0:
            self._thermal_w._zoom = 1.0
            self.statusBar().showMessage("Zoom: 1.0x (reset)", 1500)
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        if self._recording and self._ffmpeg_proc:
            try:
                if self._ffmpeg_proc.stdin:
                    self._ffmpeg_proc.stdin.close()
                self._ffmpeg_proc.wait(timeout=10)
            except Exception:
                self._ffmpeg_proc.kill()
            self._ffmpeg_proc = None
        self._cam_thread.stop()
        e.accept()


def _stop_uvc_driver() -> None:
    """Stop only the UVC driver Python process so the viewer can claim the camera.

    Does NOT touch the watcher — the watcher keeps v4l2loopback alive and
    will automatically restart the UVC driver once the viewer exits.
    """
    if sys.platform == "win32":
        return
    import subprocess
    pid = os.getpid()
    for pat in (
        "from thermal_camera_viewer.uvc_driver",
        "thermal_camera_viewer.uvc_driver",
    ):
        try:
            out = subprocess.check_output(["pgrep", "-f", pat], text=True)
            for line in out.strip().split("\n"):
                p = int(line.strip())
                if p != pid:
                    os.kill(p, signal.SIGTERM)
        except Exception:
            pass


def main() -> None:
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(description="Thermal Camera Viewer")
    parser.add_argument("--model", choices=["p1", "p3"], default="p3")
    args = parser.parse_args()

    # Kill only the UVC driver so the viewer can claim the camera.
    # The watcher stays alive and will re-spawn the driver after we exit.
    _stop_uvc_driver()

    # Also kill stale viewer instances (Linux/macOS: pgrep; not used on Windows)
    if sys.platform != "win32":
        pid = os.getpid()
        try:
            out = subprocess.check_output(
                ["pgrep", "-f", "thermal_camera_viewer.viewer"], text=True
            )
            for line in out.strip().split("\n"):
                p = int(line.strip())
                if p != pid:
                    os.kill(p, signal.SIGTERM)
        except Exception:
            pass
        time.sleep(0.4)

    app = QApplication(sys.argv)
    app.setApplicationName("Thermal Camera Viewer")

    win = MainWindow(model=args.model)
    win.show()
    ret = app.exec_()
    sys.exit(ret)


if __name__ == "__main__":
    main()
