"""Reusable two-axis scroll areas for controls, not thermal image navigation."""

from tkinter import ttk

import tkinter as tk


class ScrollArea(ttk.Frame):
    """Retain requested content size so narrow/short panes never clip controls."""

    def __init__(self, parent, *, horizontal_only=False, height=1):
        super().__init__(parent)
        self.horizontal_only = horizontal_only
        self.overflow_x = self.overflow_y = False
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.viewport = tk.Canvas(
            self, highlightthickness=0, background="#18212e", width=1, height=height
        )
        self.viewport.grid(row=0, column=0, sticky="nsew")
        self.horizontal = ttk.Scrollbar(
            self, orient="horizontal", command=self.viewport.xview
        )
        self.horizontal.grid(row=1, column=0, sticky="ew")
        self.vertical = ttk.Scrollbar(
            self, orient="vertical", command=self.viewport.yview
        )
        if not horizontal_only:
            self.vertical.grid(row=0, column=1, sticky="ns")
        self.viewport.configure(
            xscrollcommand=self.horizontal.set, yscrollcommand=self.vertical.set
        )
        self.content = ttk.Frame(self.viewport)
        self.item = self.viewport.create_window(0, 0, window=self.content, anchor="nw")
        self.content.bind("<Configure>", self.layout, add="+")
        self.viewport.bind("<Configure>", self.layout)

    def layout(self, event=None):
        if not self.content.winfo_exists():
            return
        requested_width = self.content.winfo_reqwidth()
        requested_height = self.content.winfo_reqheight()
        width, height = self.winfo_width(), self.winfo_height()
        bar_width = self.vertical.winfo_reqwidth()
        bar_height = self.horizontal.winfo_reqheight()
        x_needed = y_needed = False
        # Solve from an empty viewport, so visible bars cannot keep each other alive.
        for _ in range(3):
            x_needed = requested_width > max(1, width - (bar_width if y_needed else 0))
            y_needed = not self.horizontal_only and requested_height > max(
                1, height - (bar_height if x_needed else 0)
            )
        self.overflow_x, self.overflow_y = x_needed, y_needed
        self.horizontal.grid() if x_needed else self.horizontal.grid_remove()
        self.vertical.grid() if y_needed else self.vertical.grid_remove()
        view_width = max(1, width - (bar_width if y_needed else 0))
        view_height = max(1, height - (bar_height if x_needed else 0))
        content_width = max(view_width, requested_width)
        # Let Tk track requested height: editors populate after they are selected.
        self.viewport.itemconfigure(self.item, width=content_width, height=0)
        if self.horizontal_only:
            self.viewport.configure(height=requested_height)
            view_height = requested_height
        # A region smaller than the viewport permits unwanted canvas offsets.
        self.viewport.configure(
            scrollregion=(0, 0, content_width, max(view_height, requested_height))
        )
        if not x_needed:
            self.viewport.xview_moveto(0)
        if not y_needed:
            self.viewport.yview_moveto(0)

    def show(self, widget):
        """Attach a child of the viewport so Tk clips it away from the scrollbars."""
        self.content = widget
        self.viewport.itemconfigure(self.item, window=widget)
        self.viewport.xview_moveto(0)
        self.viewport.yview_moveto(0)
        self.layout()


def install_scroll_routing(root):
    """Route wheel gestures to the nearest control area; image canvases keep zoom."""

    def scroll(event):
        widget = event.widget
        if isinstance(widget, (ttk.Combobox, ttk.Spinbox, ttk.Scale, tk.Text)):
            return None
        area = widget
        while area is not None and not isinstance(area, ScrollArea):
            area = getattr(area, "master", None)
        if area is None:
            return None
        delta = (
            (-1 if event.num == 4 else 1)
            if event.num in (4, 5)
            else (-1 if event.delta > 0 else 1)
        )
        if area.horizontal_only or event.state & 1:
            if area.overflow_x:
                area.viewport.xview_scroll(delta * 3, "units")
        else:
            if area.overflow_y:
                area.viewport.yview_scroll(delta * 3, "units")
        return "break"

    for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        root.bind(sequence, scroll, add="+")
