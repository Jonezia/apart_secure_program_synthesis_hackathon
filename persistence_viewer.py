#!/usr/bin/env python3
"""
persistence_viewer.py — inspect and prune the MutDafny cache.

A small rich TUI over PersistenceManager showing, per source file, how much is
cached in the mutant categories and in the analysis (diffs / analyzed / problems),
plus scan progress and on-disk size. Lets you purge a file's analysis cache, its
mutant cache, or everything — at per-file granularity.

Keys:
  ↑/↓ or k/j   select file        r   refresh        p   detail ⇄ problems
  a            purge ANALYSIS      m   purge MUTANTS   x   purge ALL (both)
  y/n          confirm/cancel a pending purge          q   quit
"""

import select
import sys
import termios
import time
import tty
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from persistence_manager import PersistenceManager

ROOT = Path(__file__).parent
MUTDAFNY = ROOT / "mutdafny"
PM = PersistenceManager(MUTDAFNY)
REFRESH_HZ = 12

THEME = Theme(
    {
        "app.title": "bold bright_cyan",
        "app.dim": "grey42",
        "app.key": "bold black on bright_cyan",
        "row.sel": "bold black on bright_cyan",
        "stat.alive": "bold bright_green",
        "stat.killed": "bright_red",
        "stat.equiv": "bright_magenta",
        "stat.timeout": "bright_yellow",
        "stat.invalid": "grey42",
        "stat.total": "bold bright_blue",
        "stat.analysis": "bright_cyan",
        "warn": "bold bright_red",
        "ok": "bold bright_green",
        "border": "bright_cyan",
        "problem.id": "bold bright_yellow",
        "problem.title": "bold white",
        "problem.desc": "grey70",
        "diff.add": "green",
        "diff.del": "red",
        "diff.hunk": "bright_cyan",
        "diff.ctx": "grey50",
    }
)


def colorize_diff(text: str, max_lines: int = 14) -> Text:
    """Render a unified diff with +/- coloring, trimmed to its first hunk(s)."""
    lines = text.splitlines()
    # Skip the ---/+++ file header; start at the first hunk.
    start = next((i for i, l in enumerate(lines) if l.startswith("@@")), 0)
    shown = lines[start:start + max_lines]
    out = Text()
    for l in shown:
        if l.startswith("@@"):
            style = "diff.hunk"
        elif l.startswith("+"):
            style = "diff.add"
        elif l.startswith("-"):
            style = "diff.del"
        else:
            style = "diff.ctx"
        out.append(l + "\n", style=style)
    if start + max_lines < len(lines):
        out.append(f"  … ({len(lines) - start - max_lines} more lines)\n", style="app.dim")
    if not shown:
        out.append("(diff unavailable)\n", style="app.dim")
    return out


def human_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "K", "M", "G"):
        if f < 1024 or unit == "G":
            return f"{f:.0f}{unit}" if unit == "B" else f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}G"


class Viewer:
    def __init__(self):
        self.summaries: list[dict] = []
        self.sel = 0
        self.message = ""
        self.pending = None    # ("analysis"|"mutants"|"all", stem)
        self.mode = "detail"   # "detail" | "problems"
        self.quit = False
        self.refresh()

    def refresh(self):
        self.summaries = PM.summarize_all()
        if self.sel >= len(self.summaries):
            self.sel = max(0, len(self.summaries) - 1)

    def cur(self):
        if 0 <= self.sel < len(self.summaries):
            return self.summaries[self.sel]
        return None

    # ----------------------------------------------------------------- input
    def handle(self, ch: str):
        if self.pending:
            stem = self.pending[1]
            if ch in ("y", "Y"):
                kind = self.pending[0]
                {"analysis": PM.purge_analysis, "mutants": PM.purge_mutants,
                 "fixes": PM.purge_fixes, "all": PM.purge_all}[kind](stem)
                self.message = f"purged {kind} cache for {stem}"
                self.pending = None
                self.refresh()
            else:
                self.message = "purge cancelled"
                self.pending = None
            return

        if ch in ("q", "Q"):
            self.quit = True
        elif ch in ("k",) or ch == "UP":
            self.sel = max(0, self.sel - 1)
        elif ch in ("j",) or ch == "DOWN":
            self.sel = min(len(self.summaries) - 1, self.sel + 1)
        elif ch in ("r", "R"):
            self.refresh(); self.message = "refreshed"
        elif ch in ("p", "P"):
            self.mode = "problems" if self.mode == "detail" else "detail"
        elif ch in ("a", "m", "f", "x"):
            s = self.cur()
            if s:
                kind = {"a": "analysis", "m": "mutants", "f": "fixes", "x": "all"}[ch]
                self.pending = (kind, s["stem"])

    # --------------------------------------------------------------- render
    def render(self) -> Group:
        title = Text("MutDafny Cache Viewer", style="app.title")
        if not self.summaries:
            body = Panel(Text("No cached files found under mutdafny/mutants/.",
                              style="app.dim"), border_style="border")
            return Group(title, body, self._footer())

        table = Table(expand=True, border_style="border", header_style="app.title")
        table.add_column("#", justify="right", width=3)
        table.add_column("File", overflow="fold", ratio=3)
        table.add_column("alive", justify="right", style="stat.alive")
        table.add_column("kill", justify="right", style="stat.killed")
        table.add_column("t/o", justify="right", style="stat.timeout")
        table.add_column("inv", justify="right", style="stat.invalid")
        table.add_column("equiv", justify="right", style="stat.equiv")
        table.add_column("total", justify="right", style="stat.total")
        table.add_column("scan", justify="right", style="app.dim")
        table.add_column("diffs", justify="right", style="stat.analysis")
        table.add_column("anlz", justify="right", style="stat.analysis")
        table.add_column("prob", justify="right", style="stat.analysis")
        table.add_column("fixes", justify="right", style="ok")
        table.add_column("size", justify="right", style="app.dim")

        for i, s in enumerate(self.summaries):
            c = s["counts"]
            scan = f"{s['consumed']}/{s['targets_total']}" if s["scanned"] else "-"
            fixes = f"{s.get('fixes_applied', 0)}/{s.get('fixes', 0)}"
            row_style = "row.sel" if i == self.sel else ""
            table.add_row(
                str(i + 1), s["stem"],
                str(c["alive"]), str(c["killed"]), str(c["timed-out"]),
                str(c["invalid"]), str(c["equivalent"]), str(c["total"]),
                scan, str(s["diffs"]), str(s["analyzed"]), str(s["problems"]),
                fixes, human_bytes(s["mutant_bytes"]),
                style=row_style,
            )

        lower = self._problems() if self.mode == "problems" else self._detail()
        parts = [title, table, lower]
        if self.pending:
            kind, stem = self.pending
            parts.append(Text(f"  Purge {kind.upper()} cache for '{stem}'? (y/n)",
                              style="warn"))
        elif self.message:
            parts.append(Text(f"  {self.message}", style="ok"))
        parts.append(self._footer())
        return Group(*parts)

    def _detail(self) -> Panel:
        s = self.cur()
        if not s:
            return Panel(Text(""), border_style="border")
        lines = Text()
        lines.append(f"{s['stem']}\n", style="app.title")
        lines.append(f"  source   {s['source'] or '(unknown)'}\n", style="app.dim")
        lines.append(f"  scanned  {s['scanned']}   targets {s['targets_total']}   "
                     f"consumed {s['consumed']}   remaining {s['remaining']}   "
                     f"phase {s['phase']}\n", style="app.dim")
        lines.append(f"  mutants  {s['counts']['total']} across categories   "
                     f"size {human_bytes(s['mutant_bytes'])}\n", style="app.dim")
        cs = PM.load_problems(s["stem"])
        n_ctx = cs.get("total_contextual", 0)
        ctx_note = f"   contextual {n_ctx}" if n_ctx else ""
        lines.append(f"  analysis diffs {s['diffs']}   analyzed {s['analyzed']}   "
                     f"problems {s['problems']}{ctx_note}\n", style="app.dim")
        uvm = cs.get("unverified_methods", {})
        if uvm:
            lines.append(
                f"  unverified   {len(uvm)} method(s) with no postcondition "
                f"({cs.get('total_unverified', 0)} mutants) — add ensures manually\n",
                style="stat.timeout",
            )
        lines.append(f"  fixes    {s.get('fixes_applied', 0)} applied / "
                     f"{s.get('fixes', 0)} total\n", style="app.dim")
        lines.append(f"  updated  {s['updated'] or '-'}", style="app.dim")
        return Panel(lines, title="selected (p: problems)", border_style="border",
                     title_align="left")

    def _representative_diff(self, stem: str, cluster: dict) -> Text:
        """The diff of the first mutant assigned to this problem, colorized."""
        mutants = cluster.get("mutants", [])
        if not mutants:
            return Text("  (no mutant recorded)\n", style="app.dim")
        name = mutants[0]
        dfile = PM.diffs_dir("alive", stem) / (name[: -len(".dfy")] + ".diff")
        if not dfile.exists():
            return Text(f"  (diff missing for {name})\n", style="app.dim")
        body = colorize_diff(dfile.read_text())
        head = Text(f"  e.g. {name}\n", style="app.dim")
        return Text.assemble(head, body)

    def _problems(self, max_clusters: int = 4) -> Panel:
        s = self.cur()
        if not s:
            return Panel(Text(""), border_style="border")
        stem = s["stem"]
        cs = PM.load_problems(stem)
        clusters = list(cs.get("clusters", {}).items())
        rows: list = [Text(f"{stem} — {len(clusters)} spec problem(s) in the analysis cache",
                           style="app.title")]
        if not clusters:
            rows.append(Text("  (no problems cached — run analysis on this file first)",
                             style="app.dim"))
            return Panel(Group(*rows), title="problems (p: detail)",
                         border_style="border", title_align="left")
        for cid, c in clusters[:max_clusters]:
            header = Text()
            header.append(f"\n  [{cid}] ", style="problem.id")
            header.append(c.get("title", ""), style="problem.title")
            header.append(f"   ({c.get('mutant_count', 0)} mutants)", style="app.dim")
            rows.append(header)
            desc = c.get("description", "")
            if desc:
                rows.append(Padding(Text(desc, style="problem.desc"), (0, 0, 0, 6)))
            oq = c.get("open_questions", "")
            if oq:
                rows.append(Padding(Text(f"unknown: {oq}", style="app.dim"), (0, 0, 0, 6)))
            rows.append(Padding(self._representative_diff(stem, c), (0, 0, 0, 4)))
        if len(clusters) > max_clusters:
            rows.append(Text(f"  … {len(clusters) - max_clusters} more problem(s)",
                             style="app.dim"))
        # Unverified methods section
        uvm = cs.get("unverified_methods", {})
        rows.append(Text(f"\nUNVERIFIED METHODS ({len(uvm)})", style="app.title"))
        if not uvm:
            rows.append(Text("  (none — all mutated methods have spec clauses)", style="app.dim"))
        else:
            rows.append(Text(
                "  No ensures/decreases — mutants cannot be killed by spec fixes:",
                style="stat.timeout",
            ))
            for name, info in uvm.items():
                rows.append(Text(
                    f"  · {name}  ({info.get('mutant_count', 0)} mutant(s))",
                    style="problem.title",
                ))
        return Panel(Group(*rows), title="problems (p: detail)",
                     border_style="border", title_align="left")

    def _footer(self) -> Text:
        f = Text()
        for key, label in [("↑/↓", "select"), ("p", "detail/problems"),
                           ("a", "purge analysis"), ("m", "purge mutants+analysis"),
                           ("f", "purge fixes"), ("x", "purge all"),
                           ("r", "refresh"), ("q", "quit")]:
            f.append(f" {key} ", style="app.key")
            f.append(f" {label}  ", style="app.dim")
        return f


def main():
    if not MUTDAFNY.exists():
        sys.exit(f"[error] mutdafny dir not found at {MUTDAFNY}")
    viewer = Viewer()
    console = Console(theme=THEME)

    if not sys.stdin.isatty():
        # Non-interactive: print a one-shot summary table and exit.
        console.print(viewer.render())
        return

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        with Live(viewer.render(), console=console, screen=True,
                  refresh_per_second=REFRESH_HZ, auto_refresh=False) as live:
            while not viewer.quit:
                live.update(viewer.render())
                live.refresh()
                r, _, _ = select.select([sys.stdin], [], [], 1 / REFRESH_HZ)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch == "\x1b":   # arrow-key escape sequence
                    seq = sys.stdin.read(2)
                    ch = {"[A": "UP", "[B": "DOWN"}.get(seq, "")
                if ch:
                    viewer.handle(ch)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == "__main__":
    main()
