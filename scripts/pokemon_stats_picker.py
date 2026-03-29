#!/usr/bin/env python3
"""
Tkinter browser for pokemon_stats.json (or pokemon_stats_by_tag.json): pick a grouping
(type, egg group, habitat, or trait tag when present) and inspect base-stat percentiles
(p25 / p50 / p75), BST, and Gen 9 move counts / % of species.

If the JSON includes ``by_trait_tag`` (from ``build_pokemon_stats_by_tag.py`` or
``--merge-into``), a "Trait tag" option appears in Group by.

Usage:
  python scripts/pokemon_stats_picker.py [path/to/pokemon_stats.json]

Defaults to repo-root pokemon_stats.json when run from the YGOrange tree.
"""

from __future__ import annotations

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional


STAT_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")
GROUP_LABELS = {
    "by_single_type": "Single type",
    "by_single_egg_group": "Single egg group",
    "by_habitat": "Habitat",
    "by_trait_tag": "Trait tag",
}
# Order for discovery; by_trait_tag is appended only if present in the loaded file.
GROUP_KEYS_PROBE_ORDER = (
    "by_single_type",
    "by_single_egg_group",
    "by_habitat",
    "by_trait_tag",
)

DATASET_KEYS = ("all", "fully_evolved")
DATASET_LABELS = (
    "All species (stats + moves)",
    "Fully evolved (stats only)",
)


def default_json_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    p = root / "pokemon_stats.json"
    if p.is_file():
        return p
    return Path.cwd() / "pokemon_stats.json"


def pct_bar(pct: float, width: int = 22) -> str:
    n = max(0, min(width, int(round((pct / 100.0) * width))))
    return "█" * n + "·" * (width - n)


class PokemonStatsPicker(tk.Tk):
    def __init__(self, initial_path: Path) -> None:
        super().__init__()
        self.title("Pokémon stats — group picker")
        self.geometry("1100x720")
        self.minsize(880, 520)

        self._path: Optional[Path] = None
        self._data: dict[str, Any] = {}
        self._bucket_keys: list[str] = []
        self._group_key_list: list[str] = []
        self._group_label_list: list[str] = []

        self._build_menu()
        self._build_ui()

        if initial_path.is_file():
            self.load_file(initial_path)
        else:
            self._set_status(f"No file at {initial_path} — use File → Open")

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        file_m = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_m)
        file_m.add_command(label="Open JSON…", command=self._open_file_dialog)
        file_m.add_separator()
        file_m.add_command(label="Quit", command=self.destroy)

    def _build_ui(self) -> None:
        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(main, width=300)
        main.add(left, weight=0)

        right = ttk.Frame(main)
        main.add(right, weight=1)

        ttk.Label(left, text="Dataset").pack(anchor=tk.W, padx=4, pady=(0, 2))
        self.var_dataset_label = tk.StringVar(value=DATASET_LABELS[0])
        ds = ttk.Combobox(
            left,
            textvariable=self.var_dataset_label,
            state="readonly",
            values=DATASET_LABELS,
            width=32,
        )
        ds.pack(fill=tk.X, padx=4, pady=(0, 8))
        ds.bind(
            "<<ComboboxSelected>>",
            lambda e: self._on_dataset_switch(),
        )

        ttk.Label(left, text="Group by").pack(anchor=tk.W, padx=4, pady=(0, 2))
        self.var_group_label = tk.StringVar(value=GROUP_LABELS["by_single_type"])
        self.grp = ttk.Combobox(
            left,
            textvariable=self.var_group_label,
            state="readonly",
            values=tuple(GROUP_LABELS[k] for k in GROUP_KEYS_PROBE_ORDER[:3]),
            width=32,
        )
        self.grp.pack(fill=tk.X, padx=4, pady=(0, 8))
        self.grp.bind("<<ComboboxSelected>>", lambda e: self._on_dataset_or_group_change())

        ttk.Label(left, text="Filter").pack(anchor=tk.W, padx=4, pady=(0, 2))
        self.var_filter = tk.StringVar()
        self.var_filter.trace_add("write", lambda *a: self._apply_filter())
        ttk.Entry(left, textvariable=self.var_filter).pack(fill=tk.X, padx=4, pady=(0, 6))

        lf = ttk.LabelFrame(left, text="Groups")
        lf.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        scroll = ttk.Scrollbar(lf)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = tk.Listbox(
            lf,
            yscrollcommand=scroll.set,
            exportselection=False,
            activestyle="dotbox",
            font=("Segoe UI", 10),
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", lambda e: self._on_select_bucket())

        self.meta_text = tk.Text(left, height=6, wrap=tk.WORD, font=("Segoe UI", 8))
        self.meta_text.pack(fill=tk.X, padx=4, pady=(8, 0))
        self.meta_text.config(state=tk.DISABLED)

        self.status = ttk.Label(left, text="", foreground="#555")
        self.status.pack(anchor=tk.W, padx=4, pady=6)

        self.title_label = ttk.Label(right, text="Select a group", font=("Segoe UI", 14, "bold"))
        self.title_label.pack(anchor=tk.W, pady=(0, 4))
        self.count_label = ttk.Label(right, text="", font=("Segoe UI", 11))
        self.count_label.pack(anchor=tk.W, pady=(0, 12))

        sf = ttk.LabelFrame(right, text="Base stats (percentiles among species in this group)")
        sf.pack(fill=tk.X, pady=(0, 10))
        cols = ("stat", "p25", "p50", "p75")
        self.tree_stats = ttk.Treeview(sf, columns=cols, show="headings", height=8)
        for c, w in zip(cols, (90, 72, 72, 72)):
            self.tree_stats.heading(c, text=c.upper() if c != "stat" else "Stat")
            self.tree_stats.column(c, width=w, anchor=tk.CENTER if c != "stat" else tk.W)
        self.tree_stats.pack(fill=tk.X, padx=6, pady=6)

        mf = ttk.LabelFrame(right, text="Gen 9 learnset moves (count & % of species in group)")
        mf.pack(fill=tk.BOTH, expand=True)
        mcols = ("move", "count", "pct", "bar")
        self.tree_moves = ttk.Treeview(mf, columns=mcols, show="headings", height=18)
        self.tree_moves.heading("move", text="Move")
        self.tree_moves.heading("count", text="Species")
        self.tree_moves.heading("pct", text="%")
        self.tree_moves.heading("bar", text="Share")
        self.tree_moves.column("move", width=200, anchor=tk.W)
        self.tree_moves.column("count", width=72, anchor=tk.E)
        self.tree_moves.column("pct", width=56, anchor=tk.E)
        self.tree_moves.column("bar", width=260, anchor=tk.W)
        mscroll = ttk.Scrollbar(mf, orient=tk.VERTICAL, command=self.tree_moves.yview)
        self.tree_moves.configure(yscrollcommand=mscroll.set)
        self.tree_moves.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=6)
        mscroll.pack(side=tk.RIGHT, fill=tk.Y, pady=6, padx=(0, 6))

    def _set_status(self, msg: str) -> None:
        self.status.config(text=msg)

    def _open_file_dialog(self) -> None:
        p = filedialog.askopenfilename(
            title="Open pokemon_stats.json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if p:
            self.load_file(Path(p))

    def load_file(self, path: Path) -> None:
        try:
            raw = path.read_text(encoding="utf-8")
            self._data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as e:
            messagebox.showerror("Load error", str(e))
            return
        self._path = path
        self.title(f"Pokémon stats — {path.name}")
        self._refresh_group_combobox_options()
        self._fill_meta()
        self._on_dataset_or_group_change()
        self._set_status(str(path))

    def _fill_meta(self) -> None:
        self.meta_text.config(state=tk.NORMAL)
        self.meta_text.delete("1.0", tk.END)
        meta = self._data.get("meta")
        if isinstance(meta, dict):
            lines = [f"{k}: {v}" for k, v in sorted(meta.items())]
            self.meta_text.insert(tk.END, "\n".join(lines))
        else:
            self.meta_text.insert(tk.END, "(no meta)")
        self.meta_text.config(state=tk.DISABLED)

    def _dataset_key(self) -> str:
        try:
            i = DATASET_LABELS.index(self.var_dataset_label.get())
        except ValueError:
            i = 0
        return DATASET_KEYS[i]

    def _get_root_for_group(self, gkey: str) -> Optional[dict[str, Any]]:
        if self._dataset_key() == "fully_evolved":
            fe = self._data.get("fully_evolved")
            if not isinstance(fe, dict):
                return None
            root = fe.get(gkey)
        else:
            root = self._data.get(gkey)
        return root if isinstance(root, dict) else None

    def _refresh_group_combobox_options(self) -> None:
        opts: list[tuple[str, str]] = []
        for gkey in GROUP_KEYS_PROBE_ORDER:
            if self._get_root_for_group(gkey) is not None:
                label = GROUP_LABELS.get(gkey, gkey)
                opts.append((gkey, label))
        if not opts:
            opts = [("by_single_type", GROUP_LABELS["by_single_type"])]
        self._group_key_list = [a for a, _ in opts]
        self._group_label_list = [b for _, b in opts]
        self.grp.configure(values=self._group_label_list)
        cur = self.var_group_label.get()
        if cur not in self._group_label_list:
            self.var_group_label.set(self._group_label_list[0])

    def _on_dataset_switch(self) -> None:
        self._refresh_group_combobox_options()
        self._on_dataset_or_group_change()

    def _group_key(self) -> str:
        if not self._group_label_list:
            return "by_single_type"
        try:
            i = self._group_label_list.index(self.var_group_label.get())
        except ValueError:
            i = 0
        return self._group_key_list[i]

    def _bucket_root(self) -> Optional[dict[str, Any]]:
        return self._get_root_for_group(self._group_key())

    def _on_dataset_or_group_change(self) -> None:
        self.listbox.delete(0, tk.END)
        self._clear_detail()
        root = self._bucket_root()
        if root is None:
            self._set_status("Missing data for this dataset / group.")
            return
        self._bucket_keys = sorted(root.keys(), key=str.casefold)
        self._apply_filter()

    def _apply_filter(self) -> None:
        q = self.var_filter.get().strip().casefold()
        self.listbox.delete(0, tk.END)
        root = self._bucket_root()
        if root is None:
            return
        shown: list[str] = []
        for k in self._bucket_keys:
            if not q or q in str(k).casefold():
                self.listbox.insert(tk.END, k)
                shown.append(k)
        if shown and self.listbox.size():
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(0)
            self.listbox.activate(0)
            self._show_bucket(shown[0], root[shown[0]])
        else:
            self._clear_detail()

    def _on_select_bucket(self) -> None:
        sel = self.listbox.curselection()
        if not sel:
            return
        name = self.listbox.get(sel[0])
        root = self._bucket_root()
        if root is None or name not in root:
            return
        self._show_bucket(name, root[name])

    def _clear_detail(self) -> None:
        self.title_label.config(text="Select a group")
        self.count_label.config(text="")
        for t in (self.tree_stats, self.tree_moves):
            for iid in t.get_children():
                t.delete(iid)

    def _show_bucket(self, name: str, bucket: Any) -> None:
        if not isinstance(bucket, dict):
            return
        gkey = self._group_key()
        g_human = GROUP_LABELS.get(gkey, gkey)
        ds = (
            "Fully evolved"
            if self._dataset_key() == "fully_evolved"
            else "All species"
        )
        self.title_label.config(text=f"{name}  —  {g_human}  ({ds})")

        n = bucket.get("species_count")
        try:
            n_int = int(n) if n is not None else 0
        except (TypeError, ValueError):
            n_int = 0
        self.count_label.config(text=f"Species in group: {n_int}")

        for iid in self.tree_stats.get_children():
            self.tree_stats.delete(iid)
        stats = bucket.get("stats")
        if isinstance(stats, dict):
            for sk in STAT_ORDER:
                row = stats.get(sk)
                if isinstance(row, dict):
                    self.tree_stats.insert(
                        "",
                        tk.END,
                        values=(
                            sk.upper(),
                            row.get("p25", ""),
                            row.get("p50", ""),
                            row.get("p75", ""),
                        ),
                    )
        bst = bucket.get("bst")
        if isinstance(bst, dict):
            self.tree_stats.insert(
                "",
                tk.END,
                values=("BST", bst.get("p25", ""), bst.get("p50", ""), bst.get("p75", "")),
            )

        for iid in self.tree_moves.get_children():
            self.tree_moves.delete(iid)
        moves = bucket.get("moves")
        if not isinstance(moves, list) or not moves:
            self.tree_moves.insert(
                "",
                tk.END,
                values=(
                    "—",
                    "—",
                    "—",
                    "No move list for this dataset (fully evolved has stats only).",
                ),
            )
            return
        denom = float(n_int) if n_int > 0 else 1.0
        for entry in moves:
            if not isinstance(entry, dict):
                continue
            mv = str(entry.get("move", ""))
            try:
                c = int(entry.get("species_count", 0))
            except (TypeError, ValueError):
                c = 0
            pct = 100.0 * c / denom
            self.tree_moves.insert(
                "",
                tk.END,
                values=(mv, c, f"{pct:.1f}%", pct_bar(pct)),
            )


def main() -> None:
    argv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_json_path()
    app = PokemonStatsPicker(argv_path)
    app.mainloop()


if __name__ == "__main__":
    main()
