"""Editor for named absolute-temperature palettes; persistence lives in palettes.py."""

from tkinter import ttk

import tkinter as tk

from .color_ui import ColorWheel
from .i18n import filedialog, messagebox, translate
from .palettes import TemperaturePalette, read_palette, write_json
from .widgets import button, caption


class PalettePanel(ttk.Frame):
    """Create, edit, import, export and remove user presets without touching factory LUTs."""

    def __init__(self, parent, library, on_change):
        super().__init__(parent.viewport, padding=12)
        self.host = parent
        self.library, self.on_change = library, on_change
        self.selected = tk.StringVar()
        caption(self, "User temperature palettes")
        self.listing = ttk.Combobox(
            self, textvariable=self.selected, state="readonly", width=23
        )
        self.listing.pack(fill="x", pady=8)
        for label, command in [
            ("New…", lambda: self.edit()),
            ("Edit…", self.edit_selected),
            ("Use palette", self.use),
            ("Import JSON…", self.import_file),
            ("Export JSON…", self.export_file),
            ("Delete user palette", self.delete),
        ]:
            button(self, label, command, fill="x", pady=3)
        caption(
            self,
            "Define temperature/color points in °C.\nLinear gradients or discrete bands.\nFactory palettes are protected.\n\nView controls the display range. CLAHE and DDE also apply to custom colors.",
            wraplength=290,
            anchor="w",
            pady=12,
        )
        if self.library.load_error:
            ttk.Label(
                self,
                text=f"Cannot load saved palettes: {self.library.load_error}",
                foreground="#f4ba76",
                wraplength=290,
            ).pack(anchor="w")
        self.refresh()

    def refresh(self):
        names = sorted(self.library.palettes)
        self.listing.configure(values=names)
        if self.selected.get() not in names:
            self.selected.set(names[0] if names else "")

    def use(self):
        if self.selected.get() in self.library.palettes:
            self.on_change(self.selected.get())

    def edit_selected(self):
        palette = self.library.palettes.get(self.selected.get())
        if palette:
            self.edit(palette)

    def edit(self, palette=None):
        dialog = self.host.editor("Palette editor")
        box = ttk.Frame(dialog, padding=16)
        box.pack(fill="both", expand=True)
        name = tk.StringVar(value=palette.name if palette else translate("My palette"))
        caption(box, "Name (new name creates a copy)")
        ttk.Entry(box, textvariable=name, width=36).pack(fill="x", pady=6)
        mode = tk.StringVar(value=palette.interpolation if palette else "linear")
        ttk.Combobox(
            box, values=("linear", "steps"), textvariable=mode, state="readonly"
        ).pack(fill="x")
        tree = ttk.Treeview(
            box,
            columns=("temp", "color"),
            show="headings",
            height=8,
            selectmode="browse",
        )
        tree.heading("temp", text="Temperature °C")
        tree.heading("color", text="Color #RRGGBB")
        tree.pack(fill="both", expand=True, pady=10)
        for temp, color in (
            palette.stops
            if palette
            else ((15, "#000080"), (25, "#00FF80"), (40, "#FF2000"))
        ):
            tree.insert("", "end", values=(temp, color))
        row = ttk.Frame(box)
        row.pack(fill="x")
        temp = tk.StringVar(value="30")
        color = tk.StringVar(value="#FFFFFF")
        ttk.Entry(row, textvariable=temp, width=12).pack(side="left")
        ttk.Entry(row, textvariable=color, width=12).pack(side="left", padx=5)

        wheel = ColorWheel(box, color)

        def choose():
            if wheel.winfo_manager():
                wheel.pack_forget()
            else:
                wheel.pack(after=row, fill="x", pady=8)

        button(row, "Color…", choose, side="left")

        def change(replace=False):
            try:
                value = float(temp.get())
                # Reuse parser validation for finite temperature and hex color.
                TemperaturePalette.from_dict(
                    {
                        "version": 1,
                        "name": "check",
                        "stops": [[value, color.get()], [value + 1, color.get()]],
                    }
                )
                selection = tree.selection()
                if replace and selection:
                    tree.item(selection[0], values=(value, color.get()))
                else:
                    tree.insert("", "end", values=(value, color.get()))
            except ValueError as exc:
                messagebox.showerror("Invalid stop", str(exc), parent=dialog)

        buttons = ttk.Frame(box)
        buttons.pack(fill="x", pady=6)
        button(buttons, "Add", change, side="left")
        button(buttons, "Update", lambda: change(True), side="left")
        button(buttons, "Remove", lambda: tree.delete(*tree.selection()), side="left")

        def selected(event):
            if tree.selection():
                t, c = tree.item(tree.selection()[0], "values")
                temp.set(t)
                color.set(c)

        tree.bind("<<TreeviewSelect>>", selected)

        def save():
            try:
                stops = sorted(
                    [
                        [
                            float(tree.item(item, "values")[0]),
                            tree.item(item, "values")[1],
                        ]
                        for item in tree.get_children()
                    ]
                )
                result = TemperaturePalette.from_dict(
                    {
                        "version": 1,
                        "name": name.get(),
                        "interpolation": mode.get(),
                        "stops": stops,
                    }
                )
                if (
                    result.name in self.library.palettes
                    and (palette is None or result.name != palette.name)
                    and not messagebox.askyesno(
                        "Replace palette?",
                        f"Replace user palette '{result.name}'?",
                        parent=dialog,
                    )
                ):
                    return
                self.library.save(result)
                self.refresh()
                self.selected.set(result.name)
                self.on_change(result.name)
                self.host.select("Palettes")
            except (ValueError, OSError) as exc:
                messagebox.showerror("Cannot save palette", str(exc), parent=dialog)

        button(box, "Save and use", save, fill="x", pady=8)

    def import_file(self):
        path = filedialog.askopenfilename(filetypes=[("Temperature palette", "*.json")])
        if not path:
            return
        try:
            palette = read_palette(path)
            if palette.name in self.library.palettes and not messagebox.askyesno(
                "Replace palette?", f"Replace '{palette.name}'?"
            ):
                return
            self.library.save(palette)
            self.refresh()
            self.selected.set(palette.name)
            self.on_change(palette.name)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Cannot import palette", str(exc))

    def export_file(self):
        palette = self.library.palettes.get(self.selected.get())
        if not palette:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("Temperature palette", "*.json")]
        )
        if path:
            try:
                write_json(path, palette.to_dict())
            except OSError as exc:
                messagebox.showerror("Cannot export palette", str(exc))

    def delete(self):
        name = self.selected.get()
        if name not in self.library.palettes:
            return
        if not messagebox.askyesno("Delete palette?", f"Delete user palette '{name}'?"):
            return
        try:
            self.library.delete(name)
            self.refresh()
            self.on_change(None)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Cannot delete palette", str(exc))
