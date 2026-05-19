#!/bin/bash
# Remove all build artifacts so the project is ready for git upload.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

rm -rf thermal-camera-viewer_*_amd64/
rm -f  thermal-camera-viewer_*_amd64.deb
rm -rf dist/
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
rm -f  thermal_camera_viewer/*.pyc
rm -rf .venv venv .eggs .pytest_cache .mypy_cache
shopt -s nullglob 2>/dev/null || true
rm -rf *.egg-info
shopt -u nullglob 2>/dev/null || true

echo "Clean. Ready for upload."
