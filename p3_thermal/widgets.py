"""Small widget builders keep control declarations readable without a UI framework."""

from collections.abc import Callable
from tkinter import ttk


def button(parent, text, command, **layout):
    """Create and pack a standard action; an unspecified layout fills the row."""
    widget = ttk.Button(parent, text=text, command=command)
    widget.pack(**(layout or {"fill": "x"}))
    return widget


def caption(parent, text, *, wraplength=0, **layout):
    """Pack explanatory text with optional wrapping and explicit placement."""
    widget = ttk.Label(parent, text=text, wraplength=wraplength)
    widget.pack(**(layout or {"anchor": "w"}))
    return widget


def checkbox(
    parent, text, variable, command: Callable[[], object] | str = "", **layout
):
    """Keep model variables explicit while sharing standard checkbox layout."""
    widget = ttk.Checkbutton(parent, text=text, variable=variable, command=command)
    widget.pack(**(layout or {"anchor": "w"}))
    return widget
