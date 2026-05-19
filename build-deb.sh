#!/bin/bash
set -e

APP_NAME="thermal-camera-viewer"
VERSION="3.3.0"
ARCH="amd64"
PKG_DIR="${APP_NAME}_${VERSION}_${ARCH}"
INSTALL_PREFIX="/opt/${APP_NAME}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Building ${APP_NAME} ${VERSION} .deb package ==="
cd "$SCRIPT_DIR"

# ── generate PNG icons ──
echo "Generating PNG icons..."
for SIZE in 48 128 256; do
    python3 -c "
import numpy as np, cv2, math
S = ${SIZE}
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
cv2.imwrite('data/${APP_NAME}-${SIZE}.png', img)
print('  ${SIZE}x${SIZE} OK')
" || echo "Warning: could not generate ${SIZE}px icon"
done
cp -f "data/${APP_NAME}-256.png" "data/${APP_NAME}.png" 2>/dev/null || true

# ── clean previous build ──
rm -rf "$PKG_DIR" "${PKG_DIR}.deb"

# ── directory structure ──
mkdir -p "${PKG_DIR}/DEBIAN"
mkdir -p "${PKG_DIR}${INSTALL_PREFIX}/thermal_camera_viewer"
mkdir -p "${PKG_DIR}/usr/bin"
mkdir -p "${PKG_DIR}/usr/share/applications"
mkdir -p "${PKG_DIR}/usr/share/icons/hicolor/48x48/apps"
mkdir -p "${PKG_DIR}/usr/share/icons/hicolor/128x128/apps"
mkdir -p "${PKG_DIR}/usr/share/icons/hicolor/256x256/apps"
mkdir -p "${PKG_DIR}/usr/share/pixmaps"
mkdir -p "${PKG_DIR}/etc/udev/rules.d"
mkdir -p "${PKG_DIR}/etc/modprobe.d"
mkdir -p "${PKG_DIR}/etc/sudoers.d"
mkdir -p "${PKG_DIR}/usr/lib/python3/dist-packages"

# ── Python package ──
for f in __init__.py __main__.py viewer.py uvc_driver.py p3_camera.py; do
    cp "thermal_camera_viewer/$f" "${PKG_DIR}${INSTALL_PREFIX}/thermal_camera_viewer/"
done

# ── launcher scripts ──
cat > "${PKG_DIR}/usr/bin/${APP_NAME}" << 'LAUNCHER'
#!/bin/bash
exec python3 -m thermal_camera_viewer "$@"
LAUNCHER
chmod 755 "${PKG_DIR}/usr/bin/${APP_NAME}"

cat > "${PKG_DIR}/usr/bin/${APP_NAME}-uvc" << 'LAUNCHER'
#!/bin/bash
exec python3 -c "from thermal_camera_viewer.uvc_driver import main; main()" "$@"
LAUNCHER
chmod 755 "${PKG_DIR}/usr/bin/${APP_NAME}-uvc"

cat > "${PKG_DIR}/usr/bin/${APP_NAME}-uvc-watch" << 'WATCHER'
#!/bin/bash
# Lifecycle manager: started by udev on camera plug-in.
# Runs UVC driver in BACKGROUND, polls camera presence every 2s.
# On camera removal: kills driver, unloads v4l2loopback, exits.
USB_VID="3474"
USB_PID="45a2"
UVC_PID=""

camera_present() {
    grep -rqs "${USB_VID}" /sys/bus/usb/devices/*/idVendor 2>/dev/null &&
    grep -rqs "${USB_PID}" /sys/bus/usb/devices/*/idProduct 2>/dev/null
}

kill_uvc() {
    [ -n "$UVC_PID" ] && kill "$UVC_PID" 2>/dev/null
    wait "$UVC_PID" 2>/dev/null
    UVC_PID=""
    pkill -f "from thermal_camera_viewer.uvc_driver" 2>/dev/null
    pkill -f "thermal_camera_viewer\.uvc_driver" 2>/dev/null
}

cleanup() {
    kill_uvc
    sleep 0.5
    sudo /opt/thermal-camera-viewer/hotplug-remove.sh 2>/dev/null
}
trap cleanup EXIT

start_uvc() {
    thermal-camera-viewer-uvc "$@" &
    UVC_PID=$!
}

# Wait for /dev/video10 to appear, then notify PipeWire
for i in 1 2 3 4 5; do
    [ -e /dev/video10 ] && break
    sleep 1
done
if [ -e /dev/video10 ]; then
    # Restart PipeWire media-session so it creates a Video/Source node
    systemctl --user restart pipewire-media-session 2>/dev/null || true
fi

start_uvc "$@"

while camera_present; do
    if [ -n "$UVC_PID" ] && ! kill -0 "$UVC_PID" 2>/dev/null; then
        sleep 2
        camera_present && start_uvc "$@"
    fi
    sleep 2
done
# Camera gone → EXIT trap fires → cleanup
WATCHER
chmod 755 "${PKG_DIR}/usr/bin/${APP_NAME}-uvc-watch"

# ── .pth for Python ──
echo "${INSTALL_PREFIX}" > "${PKG_DIR}/usr/lib/python3/dist-packages/${APP_NAME}.pth"

# ── desktop entry ──
cp "data/${APP_NAME}.desktop" "${PKG_DIR}/usr/share/applications/"

# ── icons ──
[ -f "data/${APP_NAME}-48.png" ]  && cp "data/${APP_NAME}-48.png"  "${PKG_DIR}/usr/share/icons/hicolor/48x48/apps/${APP_NAME}.png"
[ -f "data/${APP_NAME}-128.png" ] && cp "data/${APP_NAME}-128.png" "${PKG_DIR}/usr/share/icons/hicolor/128x128/apps/${APP_NAME}.png"
[ -f "data/${APP_NAME}-256.png" ] && cp "data/${APP_NAME}-256.png" "${PKG_DIR}/usr/share/icons/hicolor/256x256/apps/${APP_NAME}.png"
[ -f "data/${APP_NAME}.png" ]     && cp "data/${APP_NAME}.png"     "${PKG_DIR}/usr/share/pixmaps/${APP_NAME}.png"

# ── udev rule: camera plug → load v4l2loopback + start watcher ──
# Removal is handled by the watcher itself (exits + cleanup on camera gone).
cat > "${PKG_DIR}/etc/udev/rules.d/99-thermal-camera-viewer.rules" << 'UDEV'
# USB thermal camera — permissions + hotplug
ACTION=="add", SUBSYSTEM=="usb", ATTR{idVendor}=="3474", ATTR{idProduct}=="45a2", MODE="0666", RUN+="/opt/thermal-camera-viewer/hotplug-add.sh"
ACTION=="add", SUBSYSTEM=="usb", ATTR{idVendor}=="3474", ATTR{idProduct}=="45c2", MODE="0666", RUN+="/opt/thermal-camera-viewer/hotplug-add.sh"
# v4l2loopback virtual camera — permissions when device appears
KERNEL=="video10", SUBSYSTEM=="video4linux", MODE="0666"
UDEV

# ── hotplug-add: runs as root from udev on camera plug-in ──
cat > "${PKG_DIR}${INSTALL_PREFIX}/hotplug-add.sh" << 'HOTPLUG_ADD'
#!/bin/bash
# Load v4l2loopback if not already loaded
if ! lsmod | grep -q v4l2loopback; then
    modprobe v4l2loopback devices=1 video_nr=10 card_label=ThermalCamera exclusive_caps=1
    sleep 0.5
    chmod 0666 /dev/video10 2>/dev/null
fi
# Start UVC watcher for each logged-in user (as user, not root)
for uid in $(loginctl list-users --no-legend 2>/dev/null | awk '{print $1}'); do
    user=$(id -nu "$uid" 2>/dev/null) || continue
    if ! pgrep -u "$uid" -f "thermal-camera-viewer-uvc-watch" >/dev/null 2>&1; then
        su - "$user" -c 'setsid thermal-camera-viewer-uvc-watch </dev/null >/dev/null 2>&1 &' 2>/dev/null
    fi
done
HOTPLUG_ADD
chmod 755 "${PKG_DIR}${INSTALL_PREFIX}/hotplug-add.sh"

# ── hotplug-remove: called by watcher via sudo on camera unplug ──
cat > "${PKG_DIR}${INSTALL_PREFIX}/hotplug-remove.sh" << 'HOTPLUG_REMOVE'
#!/bin/bash
# Kill any remaining UVC processes
pkill -f "from thermal_camera_viewer.uvc_driver" 2>/dev/null
pkill -f "thermal_camera_viewer\.uvc_driver" 2>/dev/null
sleep 1
# Force-close any leftover fds on video10, then unload
fuser -k /dev/video10 2>/dev/null
sleep 0.5
modprobe -r v4l2loopback 2>/dev/null
HOTPLUG_REMOVE
chmod 755 "${PKG_DIR}${INSTALL_PREFIX}/hotplug-remove.sh"

# ── sudoers: allow any user to run the remove helper without password ──
cat > "${PKG_DIR}/etc/sudoers.d/${APP_NAME}" << 'SUDOERS'
ALL ALL=(root) NOPASSWD: /opt/thermal-camera-viewer/hotplug-remove.sh
SUDOERS
chmod 440 "${PKG_DIR}/etc/sudoers.d/${APP_NAME}"

# ── modprobe.d (default params when modprobe is called) ──
cat > "${PKG_DIR}/etc/modprobe.d/${APP_NAME}.conf" << 'MODPROBE'
options v4l2loopback devices=1 video_nr=10 card_label=ThermalCamera exclusive_caps=1
MODPROBE

# ── installed size ──
INSTALLED_SIZE=$(du -sk "${PKG_DIR}" | cut -f1)

# ── control ──
cat > "${PKG_DIR}/DEBIAN/control" << CONTROL
Package: ${APP_NAME}
Version: ${VERSION}
Section: science
Priority: optional
Architecture: ${ARCH}
Installed-Size: ${INSTALLED_SIZE}
Depends: python3 (>= 3.10), python3-pyqt5, python3-numpy, python3-opencv, python3-usb, libusb-1.0-0, ffmpeg, v4l2loopback-dkms
Maintainer: Thermal Camera Viewer <user@localhost>
Description: USB thermal camera viewer & UVC virtual webcam driver
 Qt-based thermal camera viewer with ROI analysis, color palettes,
 hotspot tracking, and virtual UVC webcam output for Linux.
 .
 The UVC driver automatically starts when the camera is plugged in
 and enters standby when no app is reading from the virtual camera,
 saving power.  No sudo is needed at runtime.
 .
 Run 'thermal-camera-viewer' for the full-featured viewer.
CONTROL

# ── postinst: one-time setup (runs as root during dpkg install) ──
cat > "${PKG_DIR}/DEBIAN/postinst" << 'POSTINST'
#!/bin/bash
set -e
udevadm control --reload-rules 2>/dev/null || true
udevadm trigger 2>/dev/null || true
update-desktop-database /usr/share/applications 2>/dev/null || true
gtk-update-icon-cache -f /usr/share/icons/hicolor 2>/dev/null || true

# Only load v4l2loopback if the camera is currently plugged in
if lsusb -d 3474:45a2 >/dev/null 2>&1 || lsusb -d 3474:45c2 >/dev/null 2>&1; then
    modprobe -r v4l2loopback 2>/dev/null || true
    modprobe v4l2loopback devices=1 video_nr=10 card_label=ThermalCamera exclusive_caps=1 2>/dev/null || true
    sleep 0.5
    chmod 0666 /dev/video10 2>/dev/null || true
fi
POSTINST
chmod 755 "${PKG_DIR}/DEBIAN/postinst"

# ── prerm ──
cat > "${PKG_DIR}/DEBIAN/prerm" << 'PRERM'
#!/bin/bash
set -e
# Kill watcher, UVC driver, and viewer (avoid matching dpkg by using specific patterns)
pkill -f "thermal-camera-viewer-uvc-watch" 2>/dev/null || true
pkill -f "from thermal_camera_viewer.uvc_driver" 2>/dev/null || true
pkill -f "thermal_camera_viewer\.viewer" 2>/dev/null || true
pkill -f "python3 -m thermal_camera_viewer" 2>/dev/null || true
sleep 1
fuser -k /dev/video10 2>/dev/null || true
sleep 0.5
modprobe -r v4l2loopback 2>/dev/null || true
PRERM
chmod 755 "${PKG_DIR}/DEBIAN/prerm"

# ── postrm ──
cat > "${PKG_DIR}/DEBIAN/postrm" << 'POSTRM'
#!/bin/bash
set -e
update-desktop-database /usr/share/applications 2>/dev/null || true
gtk-update-icon-cache -f /usr/share/icons/hicolor 2>/dev/null || true
modprobe -r v4l2loopback 2>/dev/null || true
POSTRM
chmod 755 "${PKG_DIR}/DEBIAN/postrm"

# ── build ──
dpkg-deb --build --root-owner-group "$PKG_DIR"

echo ""
echo "=== Build complete ==="
echo "Package: ${PKG_DIR}.deb"
echo ""
echo "Install:"
echo "  sudo dpkg -i ${PKG_DIR}.deb"
echo "  sudo apt-get install -f"
echo ""
echo "Commands:"
echo "  thermal-camera-viewer       — full-featured viewer"
echo "  thermal-camera-viewer-uvc   — start UVC driver manually"
echo ""
echo "Uninstall:"
echo "  sudo dpkg -r ${APP_NAME}"
