"""USB smoke test: open camera, start stream, read one frame, print markers."""

from __future__ import annotations

import contextlib

from p3_camera import MARKER_SIZE, P3Camera


def main() -> int:
    """Run a one-frame streaming smoke test."""
    camera = P3Camera()
    try:
        camera.connect()
        camera.start_streaming()

        # Use the driver's reassembly logic (same path as viewer), then report
        # markers and payload size from one complete frame.
        frame = camera.read_frame(include_end_marker=True)
        expected = camera.config.frame_read_size
        if len(frame) != expected:
            raise RuntimeError(
                f"Unexpected frame length: got={len(frame)} expected={expected}",
            )

        start_marker = frame[:MARKER_SIZE]
        end = frame[-MARKER_SIZE:]
        payload = frame[MARKER_SIZE:-MARKER_SIZE]

        print(f"Reading frame: {MARKER_SIZE + len(payload)} + {MARKER_SIZE} bytes")
        print("Start marker:", start_marker.hex())
        print("End marker  :", end.hex())
        print("Pixel payload bytes:", len(payload))
        return 0
    finally:
        with contextlib.suppress(Exception):
            camera.stop_streaming()
        camera.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
