#!/bin/bash
set -e

APP_NAME="Thermal Camera Viewer"
BUNDLE_ID="com.thermalcameraviewer.app"
VERSION="3.3.0"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIST_DIR="$SCRIPT_DIR/dist"
APP_DIR="$DIST_DIR/${APP_NAME}.app"

echo "=== Building ${APP_NAME} v${VERSION} for macOS ==="
cd "$SCRIPT_DIR"

# Python used inside the .app launcher (Finder gives a minimal PATH, so "python3"
# often resolves to /usr/bin without Homebrew site-packages). Callers may set:
#   THERMAL_CAMERA_VIEWER_PYTHON=/path/to/python3
BUILD_PY="${THERMAL_CAMERA_VIEWER_PYTHON:-$(command -v python3 2>/dev/null || true)}"
if [ -z "$BUILD_PY" ] || [ ! -x "$BUILD_PY" ]; then
    BUILD_PY="python3"
fi

# Interpreter used only to run OpenCV for generating icon.icns. Defaults to
# BUILD_PY; override with THERMAL_CAMERA_VIEWER_ICON_PYTHON. Homebrew PEP 668
# installs often have no global cv2 — we create a throwaway venv when needed.
ICON_PY="${THERMAL_CAMERA_VIEWER_ICON_PYTHON:-$BUILD_PY}"
TCV_AUTO_ICON_VENV=""
cleanup_tcv_iconvenv() {
    [ -n "${TCV_AUTO_ICON_VENV:-}" ] && rm -rf "$TCV_AUTO_ICON_VENV"
}
if [ -n "$ICON_PY" ] && ! "$ICON_PY" -c "import cv2, numpy" 2>/dev/null; then
    if [ -x "$BUILD_PY" ] && "$BUILD_PY" -c "import venv" 2>/dev/null; then
        echo "Note: host Python has no OpenCV; using a temp venv to render the app icon..."
        TCV_AUTO_ICON_VENV="$(mktemp -d "${TMPDIR:-/tmp}/tcv-iconbuild.XXXXXX")"
        trap cleanup_tcv_iconvenv EXIT INT TERM
        "$BUILD_PY" -m venv "$TCV_AUTO_ICON_VENV"
        "$TCV_AUTO_ICON_VENV/bin/pip" install -q numpy opencv-python-headless
        ICON_PY="$TCV_AUTO_ICON_VENV/bin/python3"
    fi
fi
if [ -z "$ICON_PY" ] || ! "$ICON_PY" -c "import cv2, numpy" 2>/dev/null; then
    echo "Warning: could not find Python with OpenCV; app will use the default macOS icon."
    ICON_PY=""
fi

# ── check we're on macOS or allow cross-build ──
if [ "$(uname)" != "Darwin" ]; then
    echo "Note: Building macOS .app bundle on $(uname). The bundle is portable"
    echo "      but must be run on macOS with dependencies installed."
fi

# ── clean previous build ──
rm -rf "$APP_DIR"

# ── create .app bundle structure ──
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources/thermal_camera_viewer"

# ── generate .icns icon ──
ICON_BUILT=false
if [ -n "$ICON_PY" ]; then
    echo "Generating app icon..."
    "$ICON_PY" -c "
import numpy as np, cv2
S = 512
img = np.zeros((S, S, 4), dtype=np.uint8)
r = S // 5
cv2.rectangle(img, (r//2, r//2), (S - r//2, S - r//2), (46, 30, 26, 255), -1)
mx, my = int(S*0.18), int(S*0.28)
mw, mh = int(S*0.64), int(S*0.38)
for y in range(my, my + mh):
    for x in range(mx, mx + mw):
        frac = (x - mx) / mw; fy = (y - my) / mh
        img[y, x] = [int(min(255,(1-frac)*300)), int(min(255,frac*fy*200)), int(min(255,frac*400)), 230]
cx, cy = S // 2, int(S * 0.47); cr = int(S * 0.06)
cv2.line(img, (cx-cr, cy), (cx+cr, cy), (0, 255, 0, 200), max(1, S//80))
cv2.line(img, (cx, cy-cr), (cx, cy+cr), (0, 255, 0, 200), max(1, S//80))
cv2.circle(img, (int(S*.65), int(S*.35)), max(2, S//20), (0, 200, 255, 200), -1)
cv2.rectangle(img, (mx-2, my-2), (mx+mw+2, my+mh+2), (120,120,120,200), max(1,S//60))
tx, ty2 = int(S*.25), int(S*.74); tw2, th2 = int(S*.50), int(S*.10)
cv2.rectangle(img, (tx, ty2), (tx+tw2, ty2+th2), (20,20,20,220), -1)
fs = max(0.3, S / 600.0)
cv2.putText(img, '36.5 C', (tx+int(S*.05), ty2+th2-max(2,S//40)),
            cv2.FONT_HERSHEY_SIMPLEX, fs, (68,100,255,255), max(1,S//120))
cv2.imwrite('/tmp/_tcv_icon_512.png', img)
" 2>/dev/null

    if [ -f /tmp/_tcv_icon_512.png ]; then
        if [ "$(uname)" = "Darwin" ] && command -v sips &>/dev/null; then
            # macOS: use sips + iconutil to create .icns
            ICONSET="/tmp/_tcv_icon.iconset"
            mkdir -p "$ICONSET"
            for SIZE in 16 32 64 128 256 512; do
                sips -z $SIZE $SIZE /tmp/_tcv_icon_512.png --out "$ICONSET/icon_${SIZE}x${SIZE}.png" >/dev/null 2>&1
                D=$((SIZE * 2))
                if [ $D -le 512 ]; then
                    sips -z $D $D /tmp/_tcv_icon_512.png --out "$ICONSET/icon_${SIZE}x${SIZE}@2x.png" >/dev/null 2>&1
                fi
            done
            cp /tmp/_tcv_icon_512.png "$ICONSET/icon_512x512.png"
            cp /tmp/_tcv_icon_512.png "$ICONSET/icon_256x256@2x.png"
            iconutil -c icns "$ICONSET" -o "$APP_DIR/Contents/Resources/icon.icns" 2>/dev/null && ICON_BUILT=true
            rm -rf "$ICONSET"
        else
            # Cross-build: just include the PNG (macOS will use it as fallback)
            cp /tmp/_tcv_icon_512.png "$APP_DIR/Contents/Resources/icon.png"
            echo "  (PNG icon — convert to .icns on macOS with iconutil for best results)"
        fi
        rm -f /tmp/_tcv_icon_512.png
        echo "  Icon generated."
    fi
fi

trap - EXIT INT TERM 2>/dev/null || true
cleanup_tcv_iconvenv

# ── Info.plist ──
# CFBundleIconFile must be the basename without extension (see Apple TN).
cat > "$APP_DIR/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundleExecutable</key>
    <string>thermal-camera-viewer</string>
    <key>CFBundleIconFile</key>
    <string>icon</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHumanReadableCopyright</key>
    <string>Apache License 2.0</string>
</dict>
</plist>
PLIST

# ── launcher script ──
# NOTE: Double-clicking the .app yields a tiny PATH (no Homebrew). Prepend
# standard brew locations and embed the interpreter path used at build/install time.
cat > "$APP_DIR/Contents/MacOS/thermal-camera-viewer" << LAUNCHER
#!/bin/bash
DIR="\$(cd "\$(dirname "\$0")/../Resources" && pwd)"
export PYTHONPATH="\$DIR\${PYTHONPATH:+:\$PYTHONPATH}"
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/local/sbin:/usr/bin:/bin:/usr/sbin:/sbin:\$PATH"

# Bundled venv (created by install.sh) avoids Homebrew PEP 668 pip restrictions.
if [ -x "\$DIR/venv/bin/python3" ]; then
  PY="\$DIR/venv/bin/python3"
else
  PY='${BUILD_PY}'
  if [ ! -x "\$PY" ]; then
    PY="\$(command -v python3 2>/dev/null || true)"
  fi
fi
if [ -z "\$PY" ] || [ ! -x "\$PY" ]; then
  osascript -e 'display dialog "Thermal Camera Viewer could not find python3. Install Homebrew Python (brew install python@3) and run install.sh from the project folder." with title "Thermal Camera Viewer" buttons {"OK"} default button "OK" with icon stop' 2>/dev/null || true
  echo "thermal-camera-viewer: python3 not found" >&2
  exit 1
fi
if ! "\$PY" -c "import PyQt5" 2>/dev/null; then
  osascript -e 'display dialog "PyQt5 is missing. Homebrew Python blocks global pip (PEP 668). Run ./install.sh from the project folder — it creates an embedded venv inside the app with all dependencies." with title "Thermal Camera Viewer" buttons {"OK"} default button "OK" with icon stop' 2>/dev/null || true
  echo "thermal-camera-viewer: PyQt5 import failed for \$PY" >&2
  exit 1
fi
_QT_PLUGINS="\$("\$PY" -c "import os, PyQt5; print(os.path.join(os.path.dirname(PyQt5.__file__), 'Qt', 'plugins'))" 2>/dev/null || true)"
if [ -n "\$_QT_PLUGINS" ] && [ -d "\$_QT_PLUGINS" ]; then
  export QT_QPA_PLATFORM_PLUGIN_PATH="\$_QT_PLUGINS"
fi
exec "\$PY" -m thermal_camera_viewer "\$@"
LAUNCHER
chmod 755 "$APP_DIR/Contents/MacOS/thermal-camera-viewer"

# ── copy Python package ──
for f in __init__.py __main__.py viewer.py uvc_driver.py p3_camera.py; do
    cp "thermal_camera_viewer/$f" "$APP_DIR/Contents/Resources/thermal_camera_viewer/"
done

echo ""
echo "=== Build complete ==="
echo "App: $APP_DIR"
echo ""
echo "To install:"
echo "  cp -r \"$APP_DIR\" /Applications/"
echo "  ln -sf \"/Applications/${APP_NAME}.app/Contents/MacOS/thermal-camera-viewer\" /usr/local/bin/thermal-camera-viewer"
echo ""
echo "To run directly:"
echo "  open \"$APP_DIR\""
echo "  # or: $APP_DIR/Contents/MacOS/thermal-camera-viewer"
