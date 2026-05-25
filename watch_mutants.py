#!/usr/bin/env python3
"""
Watch mutants/alive for new .dfy files, fetch ground truths, and produce diffs.
Every BATCH_SIZE new mutants, the batch is sent to Gemini for behavioural
equivalence classification and online cluster tracking (see analyze_mutants.py).

Persistence:
  .diff_lock              — JSON list of already-diffed filenames
  analysis/cluster_state.json — cumulative cluster assignments and problem list

Console: a live-redrawn TUI panel; all diffs are written to alive/diffs/.
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).parent
MUTDAFNY = ROOT / "mutdafny"
ALIVE_DIR = MUTDAFNY / "mutants" / "alive"
GTS_DIR = MUTDAFNY / "mutants" / "gts"
DIFFS_DIR = ALIVE_DIR / "diffs"
ANALYSIS_DIR = ALIVE_DIR / "analysis"
CLUSTER_STATE_FILE = ANALYSIS_DIR / "cluster_state.json"
GT_SOURCE = MUTDAFNY / "original"
GT_SOURCE_FALLBACK = MUTDAFNY / "DafnyBench" / "DafnyBench" / "dataset" / "ground_truth"
LOCK_FILE = ALIVE_DIR / ".diff_lock"
# Mutants the formal filter proved behaviourally equivalent. We diff these for
# inspection but run no LLM analysis on them.
EQUIV_DIR = MUTDAFNY / "mutants" / "equivalent"
EQUIV_DIFFS_DIR = EQUIV_DIR / "diffs"
EQUIV_LOCK_FILE = EQUIV_DIR / ".diff_lock"
POLL_INTERVAL = 2
BATCH_SIZE = int(os.getenv("MUTANT_BATCH_SIZE", "10"))


# ---------------------------------------------------------------------------
# TUI panel
# ---------------------------------------------------------------------------

class TUIPanel:
    """
    Renders a fixed-height panel to stdout and redraws in-place using ANSI
    cursor movement.  All other output should go to stderr to avoid collision.
    """

    def __init__(self, model: str, batch_size: int) -> None:
        self._model = model
        self._batch_size = batch_size
        self._last_height = 0

    def _w(self) -> int:
        return min(shutil.get_terminal_size().columns - 1, 78)

    def _row(self, text: str = "") -> str:
        w = self._w()
        inner = w - 2
        return "│" + text[:inner].ljust(inner) + "│"

    def _div(self, label: str = "") -> str:
        w = self._w()
        inner = w - 2
        if label:
            rest = inner - len(label) - 3
            return "├─ " + label + " " + "─" * max(rest, 0) + "┤"
        return "├" + "─" * inner + "┤"

    def redraw(
        self,
        state: dict,
        pending: list[str],
        last_batch: dict | None,
    ) -> None:
        w = self._w()
        inner = w - 2

        if self._last_height:
            sys.stdout.write(f"\033[{self._last_height}A\033[J")

        rows: list[str] = []

        # ── top border / header ────────────────────────────────────────────
        hdr = f"─ MutDafny Live · {self._model} "
        rows.append("┌" + hdr + "─" * max(inner - len(hdr), 0) + "┐")

        # ── stats ──────────────────────────────────────────────────────────
        total = state.get("total_processed", 0)
        weak  = state.get("total_weak_spec", 0)
        equiv = state.get("total_equivalent", 0)
        pct   = f"{weak / total:.0%}" if total else "─"
        rows.append(self._row(f"  Processed: {total}   WEAK_SPEC: {weak} ({pct})   EQUIV: {equiv}"))

        # ── clusters ───────────────────────────────────────────────────────
        clusters = state.get("clusters", {})
        rows.append(self._div(f"SPEC PROBLEMS ({len(clusters)})"))
        if clusters:
            for cid, c in clusters.items():
                cnt   = c.get("mutant_count", 0)
                title = c.get("title", "")
                desc  = c.get("description", "")
                tag   = f"{cnt:4} mut"
                left  = f"  {cid}  {title}"
                gap   = inner - len(left) - len(tag)
                if gap < 1:
                    left = left[: inner - len(tag) - 1]
                    gap = 1
                rows.append(self._row(left + " " * gap + tag))
                if desc:
                    # Full description, word-wrapped to the panel width (no truncation).
                    indent = "       "
                    for wrapped in textwrap.wrap(desc, width=inner - len(indent)) or [""]:
                        rows.append(self._row(indent + wrapped))
        else:
            rows.append(self._row("  (no problems identified yet)"))

        # ── pending ────────────────────────────────────────────────────────
        rows.append(self._div(f"PENDING ({len(pending)} / {self._batch_size} to flush)"))
        show = pending[-8:] if len(pending) > 8 else pending
        if show:
            if len(pending) > 8:
                rows.append(self._row(f"  … {len(pending) - 8} earlier …"))
            for name in show:
                rows.append(self._row("  " + name[:inner - 2]))
        else:
            rows.append(self._row("  (none — watching for new mutants)"))

        # ── last batch summary ─────────────────────────────────────────────
        if last_batch is not None:
            rows.append(self._div("LAST BATCH"))
            rows.append(self._row(
                f"  +{last_batch['weak']} WEAK  "
                f"+{last_batch['equiv']} EQUIV  │  "
                f"{last_batch['new_problems']} new problem(s)  "
                f"{last_batch['merges']} merge(s)"
            ))

        # ── bottom border ──────────────────────────────────────────────────
        rows.append("└" + "─" * inner + "┘")

        sys.stdout.write("\n".join(rows) + "\n")
        sys.stdout.flush()
        self._last_height = len(rows)


# ---------------------------------------------------------------------------
# Diff utilities
# ---------------------------------------------------------------------------

def load_processed(lock_file: Path = LOCK_FILE) -> set:
    if lock_file.exists():
        try:
            return set(json.loads(lock_file.read_text()))
        except (json.JSONDecodeError, OSError):
            return set()
    return set()


def save_processed(processed: set, lock_file: Path = LOCK_FILE) -> None:
    lock_file.write_text(json.dumps(sorted(processed), indent=2))


def gt_name_for(mutant_name: str) -> str:
    stem = mutant_name[: -len(".dfy")] if mutant_name.endswith(".dfy") else mutant_name
    return stem.split("__")[0] + ".dfy"


def ensure_gt(gt_name: str) -> Path | None:
    gt_dest = GTS_DIR / gt_name
    if gt_dest.exists():
        return gt_dest

    gt_src = GT_SOURCE / gt_name
    if not gt_src.exists():
        gt_src = GT_SOURCE_FALLBACK / gt_name
        if not gt_src.exists():
            print(f"  [warn] GT not found for {gt_name}", file=sys.stderr)
            return None
        print(f"  [warn] using raw DafnyBench source for {gt_name}", file=sys.stderr)

    GTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(gt_src, gt_dest)
    return gt_dest


def compute_diff(mutant_path: Path, gt_path: Path) -> str:
    result = subprocess.run(
        ["diff", "-uw", str(gt_path), str(mutant_path)],
        capture_output=True,
        text=True,
    )
    return result.stdout


def process_mutant(mutant_path: Path, diffs_dir: Path = DIFFS_DIR) -> dict | None:
    """Diff the mutant against its GT. Returns {"name", "diff"} or None."""
    name = mutant_path.name
    gt_name = gt_name_for(name)
    gt_path = ensure_gt(gt_name)
    if gt_path is None:
        return None

    diff_text = compute_diff(mutant_path, gt_path)

    diffs_dir.mkdir(parents=True, exist_ok=True)
    diff_file = diffs_dir / (name[: -len(".dfy")] + ".diff")
    diff_file.write_text(diff_text)

    return {"name": name, "diff": diff_text}


def scan_equivalent(processed: set) -> None:
    """
    Diff any new mutants in mutants/equivalent against their ground truth and
    write them to mutants/equivalent/diffs. These were proved behaviourally
    equivalent by the formal filter, so no further analysis is performed.
    """
    if not EQUIV_DIR.exists():
        return
    try:
        current = {
            e.name
            for e in os.scandir(EQUIV_DIR)
            if e.is_file()
            and e.name.endswith(".dfy")
            and not e.name.endswith(".equiv.dfy")  # skip generated harnesses
        }
    except OSError:
        return

    new_files = current - processed
    if not new_files:
        return

    for name in sorted(new_files):
        try:
            process_mutant(EQUIV_DIR / name, EQUIV_DIFFS_DIR)
        except Exception as exc:
            print(f"  [error] equivalent {name}: {exc}", file=sys.stderr)
        finally:
            processed.add(name)

    save_processed(processed, EQUIV_LOCK_FILE)


# ---------------------------------------------------------------------------
# Batch flush
# ---------------------------------------------------------------------------

def flush_batch(
    batch: list[dict],
    analyser,
    cluster_state: dict,
) -> dict:
    """
    Send batch to LLM, mutate cluster_state in-place, return batch summary.
    """
    from analyze_mutants import save_cluster_state

    try:
        _results, summary = analyser.analyse_batch(
            batch, GTS_DIR, ANALYSIS_DIR, cluster_state
        )
    except Exception as exc:
        print(f"[llm]  batch analysis failed: {exc}", file=sys.stderr)
        summary = {"weak": 0, "equiv": 0, "new_clusters": 0, "merges": 0}

    save_cluster_state(cluster_state, CLUSTER_STATE_FILE)
    return summary


# ---------------------------------------------------------------------------
# Main watch loop
# ---------------------------------------------------------------------------

def watch() -> None:
    if not ALIVE_DIR.exists():
        sys.exit(f"[error] alive directory does not exist: {ALIVE_DIR}")

    GTS_DIR.mkdir(parents=True, exist_ok=True)
    DIFFS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    EQUIV_DIFFS_DIR.mkdir(parents=True, exist_ok=True)

    # Lazy-import analyser
    analyser = None
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    try:
        from analyze_mutants import MutantAnalyser
        analyser = MutantAnalyser()
    except Exception as exc:
        print(f"[warn] LLM analyser unavailable: {exc}", file=sys.stderr)
        print("[warn] Diffs will still be produced; set GEMINI_API_KEY to enable analysis.", file=sys.stderr)

    from analyze_mutants import load_cluster_state
    cluster_state = load_cluster_state(CLUSTER_STATE_FILE)

    processed = load_processed()
    equiv_processed = load_processed(EQUIV_LOCK_FILE)
    pending_batch: list[dict] = []
    last_batch: dict | None = None

    panel = TUIPanel(model=model_name, batch_size=BATCH_SIZE)
    panel.redraw(cluster_state, [], last_batch)

    while True:
        try:
            current = {
                e.name
                for e in os.scandir(ALIVE_DIR)
                if e.is_file() and e.name.endswith(".dfy")
            }
        except OSError as exc:
            print(f"[error] scan failed: {exc}", file=sys.stderr)
            time.sleep(POLL_INTERVAL)
            continue

        new_files = current - processed
        if new_files:
            for name in sorted(new_files):
                try:
                    entry = process_mutant(ALIVE_DIR / name)
                    if entry is not None and analyser is not None:
                        pending_batch.append(entry)
                except Exception as exc:
                    print(f"  [error] {name}: {exc}", file=sys.stderr)
                finally:
                    processed.add(name)

            save_processed(processed)
            panel.redraw(cluster_state, [m["name"] for m in pending_batch], last_batch)

            if analyser is not None and len(pending_batch) >= BATCH_SIZE:
                last_batch = flush_batch(pending_batch, analyser, cluster_state)
                pending_batch.clear()
                panel.redraw(cluster_state, [], last_batch)

        # Diff-only pass over proved-equivalent mutants (no analysis).
        scan_equivalent(equiv_processed)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    watch()
