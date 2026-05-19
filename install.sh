#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="thermal-camera-viewer"
OS="$(uname)"

# ─────────────────────────────────────────────────────────────
# macOS
# ─────────────────────────────────────────────────────────────
install_macos() {
    echo "=== Installing Thermal Camera Viewer for macOS ==="
    echo ""

    # Install Homebrew dependencies
    if ! command -v brew &>/dev/null; then
        echo "ERROR: Homebrew is required. Install from https://brew.sh"
        exit 1
    fi

    echo "[deps] Installing system dependencies via Homebrew..."
    brew install python@3 libusb ffmpeg 2>/dev/null || true

    # Build the .app bundle first (no global pip: Homebrew Python is PEP 668
    # "externally managed" and rejects pip install without --break-system-packages).
    echo "[build] Building .app bundle..."
    # Fallback interpreter for launcher when no bundled venv exists (e.g. build-macos.sh only).
    if [ -x /opt/homebrew/bin/python3 ]; then
        THERMAL_CAMERA_VIEWER_PYTHON="/opt/homebrew/bin/python3"
    elif [ -x /usr/local/bin/python3 ]; then
        THERMAL_CAMERA_VIEWER_PYTHON="/usr/local/bin/python3"
    else
        THERMAL_CAMERA_VIEWER_PYTHON="$(command -v python3)"
    fi
    export THERMAL_CAMERA_VIEWER_PYTHON
    bash "$SCRIPT_DIR/build-macos.sh"

    APP_BUNDLE="$SCRIPT_DIR/dist/Thermal Camera Viewer.app"
    if [ ! -d "$APP_BUNDLE" ]; then
        echo "ERROR: Build failed."
        exit 1
    fi

    echo "[deps] Creating embedded Python venv inside the app (PEP 668–safe)..."
    VENV="$APP_BUNDLE/Contents/Resources/venv"
    HOST_PY="$THERMAL_CAMERA_VIEWER_PYTHON"
    if [ -z "$HOST_PY" ] || [ ! -x "$HOST_PY" ]; then
        echo "ERROR: No usable python3 for venv (brew install python@3)."
        exit 1
    fi
    rm -rf "$VENV"
    "$HOST_PY" -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip setuptools wheel
    "$VENV/bin/pip" install pyqt5 numpy opencv-python-headless pyusb

    # Install to /Applications
    echo "[install] Installing to /Applications..."
    if [ -d "/Applications/Thermal Camera Viewer.app" ]; then
        rm -rf "/Applications/Thermal Camera Viewer.app"
    fi
    cp -R "$APP_BUNDLE" "/Applications/"

    # CLI symlink: /usr/local/bin is often root-owned on Apple Silicon; prefer
    # Homebrew's bin (user-writable) then ~/.local/bin. App itself is unaffected.
    LAUNCHER="/Applications/Thermal Camera Viewer.app/Contents/MacOS/thermal-camera-viewer"
    CLI_NAME="thermal-camera-viewer"
    CLI_LINK=""
    CLI_CANDIDATES=()
    if command -v brew &>/dev/null; then
        CLI_CANDIDATES+=("$(brew --prefix)/bin")
    fi
    CLI_CANDIDATES+=("/opt/homebrew/bin" "/usr/local/bin" "${HOME}/.local/bin")
    for d in "${CLI_CANDIDATES[@]}"; do
        [ -z "$d" ] && continue
        if mkdir -p "$d" 2>/dev/null && ln -sf "$LAUNCHER" "$d/$CLI_NAME" 2>/dev/null; then
            CLI_LINK="$d/$CLI_NAME"
            break
        fi
    done
    if [ -n "$CLI_LINK" ]; then
        echo "[install] CLI: $CLI_LINK (ensure this directory is on your PATH)"
    else
        echo "[install] Could not create CLI symlink in any standard location."
        echo "          Open the app from Applications, or run:"
        echo "          \"$LAUNCHER\""
    fi

    echo ""
    echo "=== Done ==="
    echo ""
    echo "  Open 'Thermal Camera Viewer' from Applications or Launchpad"
    if [ -n "$CLI_LINK" ]; then
        echo "  Or run: $CLI_NAME   ($CLI_LINK)"
    fi
    echo ""
    echo "Uninstall:"
    echo "  rm -rf '/Applications/Thermal Camera Viewer.app'"
    if [ -n "$CLI_LINK" ]; then
        echo "  rm -f \"$CLI_LINK\""
    fi
}

# ─────────────────────────────────────────────────────────────
# Linux
# ─────────────────────────────────────────────────────────────
install_linux() {
    echo "=== Installing Thermal Camera Viewer for Linux ==="
    echo ""

    # Build .deb if not present
    DEB=$(ls -t "$SCRIPT_DIR"/${APP_NAME}_*_amd64.deb 2>/dev/null | head -1)
    if [ -z "$DEB" ]; then
        echo "[build] No .deb found, building..."
        bash "$SCRIPT_DIR/build-deb.sh"
        DEB=$(ls -t "$SCRIPT_DIR"/${APP_NAME}_*_amd64.deb 2>/dev/null | head -1)
    fi

    if [ -z "$DEB" ]; then
        echo "ERROR: Build failed — no .deb produced."
        exit 1
    fi

    VERSION=$(dpkg-deb -f "$DEB" Version 2>/dev/null || echo "unknown")
    echo "Package: $DEB (v${VERSION})"
    echo ""

    # Fix broken dpkg state if needed
    if dpkg -s "$APP_NAME" &>/dev/null; then
        STATUS=$(dpkg -s "$APP_NAME" 2>/dev/null | grep '^Status:' || true)
        if echo "$STATUS" | grep -qE 'reinstreq|half-inst|unpacked'; then
            echo "[fix] Neutralizing broken prerm from previous install..."
            sudo bash -c "echo -e '#!/bin/bash\nexit 0' > /var/lib/dpkg/info/${APP_NAME}.prerm"
            sudo dpkg --remove --force-remove-reinstreq "$APP_NAME" 2>/dev/null || true
        else
            echo "[info] Removing previous version..."
            sudo dpkg -r "$APP_NAME" 2>/dev/null || true
        fi
        sleep 1
    fi

    sudo modprobe -r v4l2loopback 2>/dev/null || true

    echo "[install] Installing..."
    sudo dpkg -i "$DEB"
    sudo apt-get install -f -y

    echo ""
    echo "=== Done (v${VERSION}) ==="
    echo ""
    echo "  thermal-camera-viewer       — viewer app"
    echo "  thermal-camera-viewer-uvc   — UVC driver (manual start)"
    echo ""
    echo "Plug in your camera and /dev/video10 will appear automatically."
    echo "Uninstall:  sudo dpkg -r ${APP_NAME}"
}

# ─────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────
case "$OS" in
    Darwin)
        install_macos
        ;;
    Linux)
        install_linux
        ;;
    *)
        echo "ERROR: Unsupported OS: $OS"
        echo "Supported: Linux (Ubuntu/Debian), macOS"
        exit 1
        ;;
esac
