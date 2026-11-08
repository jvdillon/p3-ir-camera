#!/usr/bin/env python3
"""Compatibility entry point for Thermal Studio; implementation is p3_thermal."""

import argparse
import logging


def main():
    """Launch the desktop application, optionally with synthetic camera frames."""
    parser = argparse.ArgumentParser(description="P1/P3 Thermal Studio")
    parser.add_argument("--model", choices=("p1", "p3"), default="p3")
    parser.add_argument(
        "--demo", action="store_true", help="Use synthetic frames without USB"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)
    try:
        import tkinter as tk

        from p3_thermal.app import ThermalApp
    except ImportError as exc:
        parser.exit(
            1, f"Desktop initialization failed: {exc}\nInstall Python Tk support.\n"
        )
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        parser.exit(
            1,
            f"Desktop initialization failed: {exc}\nInstall Python Tk support and run in a desktop session.\n",
        )
    app = ThermalApp(root, args.model, args.demo)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.close()
        root.mainloop()


if __name__ == "__main__":
    main()
