#Requires -Version 5.1
<#
.SYNOPSIS
  Install Python dependencies for Thermal Camera Viewer on Windows (venv).

.DESCRIPTION
  Creates .venv in the repo root, installs PyQt5 / OpenCV / PyUSB and
  libusb-package (bundled libusb-1.0 DLLs for PyUSB on Windows).

  USB: you must still assign WinUSB to the camera with Zadig (VID 3474).
  See README.md -> Installation -> Windows.
#>
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) {
    Write-Error "Python 3.10+ not found in PATH. Install from https://www.python.org/downloads/ or: winget install Python.Python.3.12"
    exit 1
}

$VenvDir = Join-Path $Root ".venv"
if (-not (Test-Path $VenvDir)) {
    & python -m venv $VenvDir
}

$Py = Join-Path $VenvDir "Scripts\python.exe"
$Pip = Join-Path $VenvDir "Scripts\pip.exe"
& $Py -m pip install -U pip setuptools wheel
& $Pip install libusb-package numpy opencv-python-headless pyusb PyQt5

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
Write-Host "Run viewer:"
Write-Host "  $Py -m thermal_camera_viewer"
Write-Host ""
Write-Host "USB (required once): Zadig -> Options -> List All Devices ->"
Write-Host "  select camera (VID 3474, PID 45A2 or 45C2) -> WinUSB driver."
Write-Host "  See: https://github.com/jvdillon/p3-ir-camera#usb-driver-windows"
Write-Host ""
Write-Host "Optional: add FFmpeg to PATH for MP4 recording (F5)."
