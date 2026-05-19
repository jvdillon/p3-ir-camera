# P3 USB Protocol

Protocol documentation for P3-series USB thermal cameras.

> **Note**: Protocol details determined through USB traffic analysis.
> Additional research by [@aeternium](https://github.com/jvdillon/p3-ir-camera/issues/2).

## Supported Devices

| Model | VID | PID | Resolution | Frame Size |
|-------|-----|-----|------------|------------|
| P3 | 0x3474 | 0x45A2 | 256×192 | 197,632 bytes |
| P1 | 0x3474 | 0x45C2 | 160×120 | 77,440 bytes |

Frame rate: ~25 fps

## USB Setup

```
Interface 0: Control commands
Interface 1, Alt 0: Inactive (streaming stopped)
Interface 1, Alt 1: Active (streaming enabled)
```

Detach kernel drivers, claim both interfaces before use.

## Control Transfers

| bRequest | bmRequestType | wIndex | Purpose |
|----------|---------------|--------|---------|
| 0x20 | 0x41 | 0 | Send 18-byte command |
| 0x21 | 0xC1 | 0 | Read response data |
| 0x22 | 0xC1 | 0 | Read status (1 byte) |
| 0xEE | 0x40 | 1 | Start streaming |

### USB Request Type Constants

```
bmRequestType = Direction | Type | Recipient
Direction: USB_DIR_OUT=0x00, USB_DIR_IN=0x80
Type:      USB_TYPE_STANDARD=0x00, USB_TYPE_CLASS=0x20, USB_TYPE_VENDOR=0x40
Recipient: USB_RECIP_DEVICE=0x00, USB_RECIP_INTERFACE=0x01

Common values:
  0x41 = OUT | VENDOR | INTERFACE (write vendor command)
  0xC1 = IN  | VENDOR | INTERFACE (read vendor response)
  0x40 = OUT | VENDOR | DEVICE (device-level write)
```

### Status Register

The Windows tool follows every control transfer with a status register read:

```python
status = dev.ctrl_transfer(0xC1, 0x22, 0, 0, 1)
```

Status values:
- `0x02` (0b10): After write command
- `0x03` (0b11): After read command

This appears to be an ACK mechanism. The device works without checking status,
but following this pattern improves reliability.

### Extended Status Register (Debug Logs)

Reading more than 1 byte from the status register returns debug/log messages
from the camera's internal logging system:

```python
# Read extended status (128 bytes)
data = dev.ctrl_transfer(0xC1, 0x22, 0, 0, 128)
# Debug messages typically appear starting around byte 64
debug_msg = data[64:]
```

Example debug messages:
- `[162] I/default.conf: default configuration already load`
- `[34064] I/std cmd: in : 1 1 81` (after register read command)
- `[91403] I/shutter: === Shutter close ===`
- `[91598] I/Preview: preview total tick: 348`

## Command Format (18 bytes)

```
Offset  Size  Description
0       2     Command type (LE)
2       2     Parameter (usually 0x0081)
4       2     Register ID (LE)
6       6     Reserved (zeros)
12      2     Response length (LE)
14      2     Reserved (zeros)
16      2     CRC16 checksum (LE)
```

Command types:
- 0x0101: Read register
- 0x1021: Status check
- 0x012f: Stream control
- 0x0136: Shutter/NUC

### Register Map

| Register | Name | Read Size | Description |
|----------|------|-----------|-------------|
| 0x01 | model | 30 | Model name (e.g., "P3") |
| 0x02 | fw_version | 12 | Firmware version (e.g., "00.00.02.17") |
| 0x06 | part_number | 64 | Part number (e.g., "P30-1Axxxxxxxx") |
| 0x07 | serial | 64 | Serial number |
| 0x0a | hw_version | 64 | Hardware revision (e.g., "P3-00.04") |
| 0x0f | model_long | 64 | Model name (64-byte version) |

## Pre-computed Commands (with CRC)

```python
COMMANDS = {
    # Register reads
    'read_name':        bytes.fromhex('0101810001000000000000001e0000004f90'),  # reg 0x01
    'read_version':     bytes.fromhex('0101810002000000000000000c0000001f63'),  # reg 0x02
    'read_part_number': bytes.fromhex('01018100060000000000000040000000654f'),  # reg 0x06
    'read_serial':      bytes.fromhex('01018100070000000000000040000000104c'),  # reg 0x07
    'read_hw_version':  bytes.fromhex('010181000a00000000000000400000001959'),  # reg 0x0a
    'read_model_long':  bytes.fromhex('010181000f0000000000000040000000b857'),  # reg 0x0f
    # Stream control
    'start_stream':     bytes.fromhex('012f81000000000000000000010000004930'),
    'gain_low':         bytes.fromhex('012f41000000000000000000000000003c3a'),
    'gain_high':        bytes.fromhex('012f41000100000000000000000000004939'),
    # Shutter
    'shutter':          bytes.fromhex('01364300000000000000000000000000cd0b'),
}
```

> **Note**: The camera does not verify CRCs. Invalid CRCs still work.

## Initialization Sequence

```python
# 1. Detach kernel drivers and claim interfaces
dev.set_configuration()
usb.util.claim_interface(dev, 0)
usb.util.claim_interface(dev, 1)

# 2. Read device info (with status checks)
dev.ctrl_transfer(0x41, 0x20, 0, 0, COMMANDS['read_name'])
dev.ctrl_transfer(0xC1, 0x22, 0, 0, 1)  # status
name = dev.ctrl_transfer(0xC1, 0x21, 0, 0, 30)
dev.ctrl_transfer(0xC1, 0x22, 0, 0, 1)  # status

# 3. Initial start streaming command
dev.ctrl_transfer(0x41, 0x20, 0, 0, COMMANDS['start_stream'])
dev.ctrl_transfer(0xC1, 0x22, 0, 0, 1)  # reads 0x02
resp = dev.ctrl_transfer(0xC1, 0x21, 0, 0, 1)  # reads 0x01 (or 0x35 if restarting)
dev.ctrl_transfer(0xC1, 0x22, 0, 0, 1)  # reads 0x03

# 4. Wait before configuring interface
time.sleep(1)

# 5. Enable streaming interface
dev.set_interface_altsetting(interface=1, alternate_setting=1)
dev.ctrl_transfer(0x40, 0xEE, 0, 1, None)

# 6. Wait for camera ready (~2 seconds)
time.sleep(2)

# 7. Issue async bulk read (Windows tool does this)
try:
    dev.read(0x81, frame_size, 100)
except:
    pass  # Expected to timeout

# 8. Final start stream
dev.ctrl_transfer(0x41, 0x20, 0, 0, COMMANDS['start_stream'])
dev.ctrl_transfer(0xC1, 0x22, 0, 0, 1)
resp = dev.ctrl_transfer(0xC1, 0x21, 0, 0, 1)  # reads 0x01 or 0x35
dev.ctrl_transfer(0xC1, 0x22, 0, 0, 1)
```

### start_stream Response Values

The `start_stream` command returns a 1-byte response:
- `0x01`: Normal start (camera was idle)
- `0x35` ('5'): Restart (camera was already streaming)

The 0x35 response indicates the stream is being restarted rather than started fresh.

## Stop Streaming

```python
dev.set_interface_altsetting(interface=1, alternate_setting=0)
```

## Frame Transfer

Frames are transmitted as **2 separate USB bulk transfers**:

| Transfer | Size (P3) | Contents |
|----------|-----------|----------|
| 1 | 197,644 | Start marker (12) + pixel data (197,632) |
| 2 | 12 | End marker |

```python
MARKER_SIZE = 12
frame_size = 2 * (2 * sensor_h + 2) * sensor_w  # P3: 197,632

# Read frame
buf = array.array("B", [0]) * frame_size
n = dev.read(0x81, buf, 10000)
if n != frame_size:
    raise FrameIncompleteError()

remaining = dev.read(0x81, MARKER_SIZE, 1000)
end_marker = dev.read(0x81, MARKER_SIZE, 1000)

# Complete frame data
frame_data = bytes(buf) + bytes(remaining)
```

### Frame Markers (12 bytes)

```
Offset  Size  Description
0       1     Length (always 0x0c = 12)
1       1     Sync byte
2       4     Counter 1 (LE) - same in start/end markers
6       4     Counter 2 (LE)
10      2     Counter 3 (LE) - wraps at 2048, increments ~40/frame
```

Sync byte values:
- `0x8c` / `0x8d`: Start marker (alternates even/odd frames)
- `0x8e` / `0x8f`: End marker (alternates even/odd frames)

### Marker Validation

For frame integrity verification:

1. **cnt1 matching**: The `cnt1` value in the start marker must match the `cnt1`
   value in the end marker. A mismatch indicates frame corruption or sync loss.

2. **cnt3 tracking**: The `cnt3` counter increments by approximately 40 per frame
   and wraps at 2048. Track this value to detect dropped frames:
   - Expected next `cnt3` = `(previous_cnt3 + 40) % 2048`
   - Large gaps indicate dropped frames
   - The previous frame's end marker `cnt3` equals the current frame's start marker `cnt3`

Example marker sequence:
```
Start [cnt1=0,       cnt3=0]    End [cnt1=0,       cnt3=40]
Start [cnt1=601097,  cnt3=40]   End [cnt1=601097,  cnt3=80]
Start [cnt1=1201118, cnt3=80]   End [cnt1=1201118, cnt3=120]
```

## Frame Structure

Total pixel data: `2 × (2 × sensor_h + 2) × sensor_w` bytes

For P3 (256×192): 2 × 386 × 256 = 197,632 bytes

```
Rows        Size (P3)      Description
0-191       98,304 bytes   IR brightness data (hardware AGC'd, 8-bit in low byte)
192-193     1,024 bytes    Metadata rows
194-385     98,304 bytes   Temperature data (16-bit raw values)
```

The 256×386 pixel buffer is structured as:
- **IR data**: Rows 0 to (sensor_h - 1)
- **Metadata**: 2 rows
- **Thermal data**: Rows (sensor_h + 2) to (2 × sensor_h + 1)

```python
# Parse frame (after removing 12-byte start marker)
pixels = np.frombuffer(frame_data[12:], dtype='<u2')
full = pixels.reshape((386, 256))

ir_brightness = full[:192, :]
metadata = full[192:194, :]
thermal = full[194:386, :]
```

## Temperature Conversion

Raw 16-bit values are in 1/64 Kelvin units:

```python
SCALE = 64  # 1/64 Kelvin units

def raw_to_celsius(raw):
    return (raw / SCALE) - 273.15

def celsius_to_raw(celsius):
    return int((celsius + 273.15) * SCALE)
```

## Gain Modes

- **High gain**: -20°C to 150°C (higher sensitivity)
- **Low gain**: 0°C to 550°C (extended range)

```python
# Set low gain (extended range)
dev.ctrl_transfer(0x41, 0x20, 0, 0, COMMANDS['gain_low'])

# Set high gain (higher sensitivity)
dev.ctrl_transfer(0x41, 0x20, 0, 0, COMMANDS['gain_high'])
```

## Shutter/NUC Calibration

Trigger shutter calibration (audible click):

```python
dev.ctrl_transfer(0x41, 0x20, 0, 0, COMMANDS['shutter'])
```

The camera automatically triggers shutter approximately every 90 seconds,
even when streaming is stopped. The camera appears to always be in acquire
mode unless unplugged or reset.

After manual shutter activation, the first frame is transmitted in a nonstandard fashion.
There are 3 USB bulk transfers:

| Transfer | Size (P3) | Contents |
|----------|-----------|----------|
| 1 | 204,800 | Start marker (12) + partial frame (9,204) + pixel data (195,584) |
| 2 | 2,060 | Second start (?) marker (12) +  remaining pixel data (2,048) |
| 3 | 12 | End marker (12) |

The partial frame data is the top 36 lines of the IR image minus the last 6 pixels and repeated
in the full frame.
The second start marker has the same sync byte as the first one, but shares counter 3
with the end marker.

## Display Pipeline

```python
import numpy as np
import cv2

# 1. Read and parse frame
frame_data = read_frame()  # includes start marker + all pixel data
pixels = np.frombuffer(frame_data[12:], dtype='<u2')
full = pixels.reshape((386, 256))

# 2. Extract thermal data
thermal = full[194:386, :].copy()

# 3. Normalize to 8-bit (percentile-based AGC)
low = np.percentile(thermal, 1)
high = np.percentile(thermal, 99)
img_8 = np.clip((thermal - low) / (high - low) * 255, 0, 255).astype(np.uint8)

# 4. Apply colormap
img_color = cv2.applyColorMap(img_8, cv2.COLORMAP_INFERNO)
```

## CRC16 Reference

CRC16-CCITT with polynomial 0x1021, initial value 0x0000:

```python
def crc16_ccitt(data):
    crc = 0x0000
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc
```

## Troubleshooting

**Camera in degraded state (purple/garbage image):**
USB reset and reinitialize, or replug the device.

**Frame sync lost / incomplete frames:**
Discard incomplete frames and continue reading. The 3-part transfer pattern
helps maintain sync. Use `array.array` to detect short reads.

**Temperature drift:**
Trigger shutter calibration, allow camera to warm up (~5 minutes).

**Stream won't start:**
Ensure the 2-second delay after `0xEE` control transfer before final
start_stream command.
