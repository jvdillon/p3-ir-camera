"""Presentation-only localization; model values and portable files stay in English.

Widget adapters keep translated combobox choices separate from their source variables.
Labels mirror source variables without rewriting measurements, filenames or user names.
"""

from tkinter import (
    filedialog as _filedialog,
    messagebox as _messagebox,
    ttk,
)

import re
import tkinter as tk

from .translations import DYNAMIC_TEMPLATES, POLISH


PATTERNS = [
    (
        re.compile(
            "^" + re.escape(source).replace(re.escape("{}"), "(.*?)") + "$", re.DOTALL
        ),
        target,
    )
    for source, target in DYNAMIC_TEMPLATES.items()
]


DYNAMIC_PREFIXES = (
    "No camera · retrying automatically. ",
    "Experimental correction · atmospheric transmission ",
    "Corrected estimate ≈ ",
    "Saved thermal data: ",
    "Saved image: ",
    "Palette range: ",
    "Disconnected / error:",
    "Cannot save preferences: ",
    "DDE strength: ",
    "Common range: ",
    "Frames ",
    "Frame ",
    "Pixel (",
    "Zoom  ",
    "Applied:",
    "Queued:",
)


class Translator:
    def __init__(self, root, language="en"):
        self.root, self.language = root, language
        self.records = {}
        root.translator = self

    def text(self, value):
        value = str(value)
        if self.language != "pl":
            return value
        if value in POLISH:
            return POLISH[value]
        for pattern, target in PATTERNS:
            match = pattern.fullmatch(value)
            if match:
                return target.format(*match.groups())
        if "\n" in value:
            return "\n".join(self.text(line) for line in value.split("\n"))
        # Only known dynamic prefixes are translated; user paths/names are not searched.
        for source in DYNAMIC_PREFIXES:
            if value.startswith(source):
                return POLISH.get(source, source) + value[len(source) :]
        if value.startswith(("A: ", "B: ")):
            prefix, content = value[:3], value[3:]
            return prefix + POLISH.get(content, content)
        return value

    def refresh(self):
        """Discover new embedded editors and update presentation without rebuilding state."""
        alive = set()

        def visit(widget):
            key = str(widget)
            alive.add(key)
            record = self.records.setdefault(key, {})
            try:
                if (
                    isinstance(widget, ttk.Combobox)
                    and str(widget.cget("state")) == "readonly"
                ):
                    self._combo(widget, record)
                elif (
                    "textvariable" in tuple(widget.keys())
                    and widget.cget("textvariable")
                    and isinstance(widget, (ttk.Label, tk.Label))
                ):
                    if "source_var" not in record:
                        record["source_var"] = str(widget.cget("textvariable"))
                        record["display_var"] = tk.StringVar(master=self.root)
                        widget.configure(textvariable=record["display_var"])
                    value = self.text(self.root.getvar(record["source_var"]))
                    if record["display_var"].get() != value:
                        record["display_var"].set(value)
                elif "text" in tuple(widget.keys()):
                    current = str(widget.cget("text"))
                    if current != record.get("translated"):
                        record["original"] = current
                    translated = self.text(record.get("original", current))
                    if current != translated:
                        widget.configure({"text": translated})
                    record["translated"] = translated
                if isinstance(widget, ttk.Treeview):
                    headings = record.setdefault("headings", {})
                    for column in widget.cget("columns"):
                        headings.setdefault(column, widget.heading(column, "text"))
                        translated = self.text(headings[column])
                        if widget.heading(column, "text") != translated:
                            widget.heading(column, text=translated)
                for child in widget.winfo_children():
                    visit(child)
            except tk.TclError:
                pass

        visit(self.root)
        self.records = {
            key: value for key, value in self.records.items() if key in alive
        }

    def _combo(self, widget, record):
        if "source_var" not in record:
            record["source_var"] = str(widget.cget("textvariable"))
            record["display_var"] = tk.StringVar(master=self.root)
            record["values"] = tuple(widget.cget("values"))

            def selected(*_):
                if record.get("updating"):
                    return
                display = record["display_var"].get()
                mapping = dict(
                    zip(
                        record.get("translated_values", ()),
                        record["values"],
                        strict=False,
                    )
                )
                self.root.setvar(record["source_var"], mapping.get(display, display))

            record["display_var"].trace_add("write", selected)
            widget.configure(textvariable=record["display_var"])
        current = tuple(widget.cget("values"))
        if current != record.get("translated_values"):
            record["values"] = current
        values = tuple(self.text(value) for value in record["values"])
        record["updating"] = True
        try:
            record["translated_values"] = values
            if current != values:
                widget.configure(values=values)
            display = self.text(self.root.getvar(record["source_var"]))
            if record["display_var"].get() != display:
                record["display_var"].set(display)
        finally:
            record["updating"] = False

    def source_value(self, widget):
        """Return a Combobox's stable source value for the current label."""
        record = self.records.get(str(widget), {})
        display = widget.get()
        values = record.get("values", ())
        translated = record.get("translated_values", ())
        return dict(zip(translated, values, strict=False)).get(display, display)

    def sync_source(self, widget):
        """Copy a Combobox's current display value back to its source variable."""
        record = self.records.get(str(widget), {})
        source_var = record.get("source_var")
        if source_var:
            self.root.setvar(source_var, self.source_value(widget))


def translate(value):
    root = getattr(tk, "_default_root", None)
    translator = getattr(root, "translator", None)
    return translator.text(value) if translator else value


class Dialogs:
    """Translate native dialog text at the boundary, leaving returned paths unchanged."""

    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        function = getattr(self.module, name)

        def call(*args, **kwargs):
            args = tuple(
                translate(value) if isinstance(value, str) else value for value in args
            )
            for key in ("title", "message", "detail"):
                if key in kwargs:
                    kwargs[key] = translate(kwargs[key])
            if "filetypes" in kwargs:
                kwargs["filetypes"] = [
                    (translate(label), pattern)
                    for label, pattern in kwargs["filetypes"]
                ]
            return function(*args, **kwargs)

        return call


filedialog = Dialogs(_filedialog)
messagebox = Dialogs(_messagebox)
