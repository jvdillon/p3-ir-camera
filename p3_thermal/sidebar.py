"""Compact top navigation and a scrollable settings pane with stable page IDs."""

from collections.abc import Callable
from tkinter import ttk

import tkinter as tk

from .scrolling import ScrollArea


class Sidebar(ScrollArea):
    """Separate two-row section navigation from the resizable page viewport."""

    def __init__(self, parent, navigation_parent=None, utility_parent=None):
        super().__init__(parent)
        self.pages = {}
        self.buttons = {}
        self.on_select: Callable[[str], None] | None = None
        self.selection = tk.StringVar()
        self.navigation_area = ScrollArea(
            navigation_parent or parent, horizontal_only=True
        )
        self.navigation_area.pack(fill="x")
        self.navigation = self.navigation_area.content
        self.utilities = utility_parent or self.navigation

    def add(self, page, text):
        if page.master is not self.viewport:
            raise ValueError(
                "Sidebar pages must be children of the viewport for clipping"
            )
        self.pages[text] = page
        page.bind("<Configure>", self.layout, add="+")
        if text not in self.buttons:
            host = self.utilities if text in ("Help", "Settings") else self.navigation
            self.buttons[text] = ttk.Button(
                host,
                text=text,
                style="Section.TButton",
                command=lambda: self.select(text),
            )
            if text in ("Help", "Settings"):
                self.buttons[text].pack(side="left", padx=1, pady=2)
            self._layout_sections()
        if len(self.pages) == 1:
            self.select(text)

    def _layout_sections(self):
        """Keep the eight primary sections stable; editors extend the same two rows."""
        sections = [
            button
            for name, button in self.buttons.items()
            if name not in ("Help", "Settings")
        ]
        for index, button in enumerate(sections):
            row, column = (
                divmod(index, 4)
                if index < 8
                else ((index - 8) % 2, 4 + (index - 8) // 2)
            )
            button.grid(row=row, column=column, sticky="ew", padx=1, pady=2)

    def select(self, name):
        self.selection.set(name)
        for key, button in self.buttons.items():
            button.configure(
                style="Selected.Section.TButton" if key == name else "Section.TButton"
            )
        self.show(self.pages[name])
        if self.on_select is not None:
            self.on_select(name)

    def remove(self, title):
        page = self.pages.pop(title, None)
        if page is not None:
            page.destroy()
        button = self.buttons.pop(title, None)
        if button is not None:
            button.destroy()
        self._layout_sections()

    def editor(self, title):
        old = self.pages.pop(title, None)
        if old is not None:
            old.destroy()
        page = ttk.Frame(self.viewport, padding=8)
        self.add(page, title)
        self.select(title)
        return page
