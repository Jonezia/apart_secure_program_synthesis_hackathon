#!/usr/bin/env python3
"""
launcher.py — orchestrate MutDafny mutation generation + live mutant analysis
across a list of Dafny files, behind a single tabbed TUI.

For every input file (or every .dfy under an input directory) the launcher:
  1. Runs run.sh (single-mutation, deterministic) until either
        N_TRUE_POSITIVE alive-after-equiv-filter mutants, or
        M_TOTAL total mutants
     are reached, then escalates to randomised multi-mutation mode if the
     deterministic targets are exhausted before N is hit.
  2. Streams run.sh stdout into the "MutDafny" tab.
  3. Diffs every new alive mutant against its ground truth and (if a Gemini key
     is configured) clusters them into specification problems — rendered live in
     the "Watch" tab, ported from watch_mutants.py's panel.
  4. The "Summary" tab shows per-file progress and problems. All persistence
     (per-file mutant cache, diffs/analysis, manifests) is owned by
     PersistenceManager under mutdafny/mutants/<category>/<stem>/.

Warm starts (both sides resume from the cache, via PersistenceManager):
  * Mutation: the manager scans once and feeds run.sh only the REMAINING targets
    (full - consumed); run.sh logs each consumed target, so the next session
    continues exactly where it stopped.
  * Analysis: already-diffed/analyzed mutants are read off disk, so the watcher
    only diffs/clusters what the analysis cache hasn't covered yet.

Keys:  1/2/3 select tab · Tab cycles · q quits (finishes nothing in flight).
"""

import os
import select
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.segment import Segment
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

import debug_log
import watch_mutants as wm
from persistence_manager import PersistenceManager, stem_of
from refinement import RefinementEngine

# ---------------------------------------------------------------------------
# Tunable knobs (env-overridable)
# ---------------------------------------------------------------------------

N_TRUE_POSITIVE = int(os.getenv("LAUNCHER_N_TP", "256"))   # alive-after-equiv-filter cap
M_TOTAL = int(os.getenv("LAUNCHER_M_TOTAL", "1024"))       # total mutants cap (incl. killed)
MULTI_MUT_K = int(os.getenv("LAUNCHER_MULTI_K", "2"))      # mutations/program in multi phase
BATCH_SIZE = int(os.getenv("MUTANT_BATCH_SIZE", "5"))           # trigger threshold (on-the-fly)
MUTANT_FIRST_BATCH_SIZE = int(os.getenv("MUTANT_FIRST_BATCH_SIZE", "5"))  # max chunk (mutant-first)
POLL_INTERVAL = 0.5                                        # worker/threshold poll (s)
REFRESH_HZ = 8
MUTDAFNY_SCROLLBACK = int(os.getenv("MUTDAFNY_SCROLLBACK", "32"))  # lines shown in MutDafny tab
# Mirror run.sh's hard wall-clock backstop so the activity timer flips red when a
# dafny call is past the point it should have been killed (a genuine stall).
DAFNY_HARD_TIMEOUT = int(os.getenv("DAFNY_HARD_TIMEOUT", "20"))
# A cluster with this many WEAK_SPEC mutants triggers spec refinement (default mode pauses
# mutdafny; human mode just queues a proposal). Also: every remaining problem is refined when
# the mutation targets are exhausted.
N_PER_PROBLEM = int(os.getenv("LAUNCHER_N_PER_PROBLEM", "5"))

ROOT = Path(__file__).parent
MUTDAFNY = ROOT / "mutdafny"
RUN_SH = MUTDAFNY / "run.sh"
MUT_DIR = MUTDAFNY / "mutants"
GTS_DIR = MUT_DIR / "gts"
STATUS_FILE = MUT_DIR / ".mutdafny_status"   # run.sh writes its current activity here

# On-disk persistence (per-file mutant cache, manifests, warm-start) is owned by
# the PersistenceManager; the launcher just orchestrates and renders.
PM = PersistenceManager(MUTDAFNY)

# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------

THEME = Theme(
    {
        "app.title": "bold bright_cyan",
        "app.dim": "grey42",
        "app.key": "bold black on bright_cyan",
        "tab.active": "bold black on bright_cyan",
        "tab.idle": "bright_cyan",
        "stat.alive": "bold bright_green",
        "stat.total": "bold bright_blue",
        "stat.killed": "bright_red",
        "stat.equiv": "bright_magenta",
        "stat.timeout": "bright_yellow",
        "stat.invalid": "grey42",
        "problem.id": "bold bright_yellow",
        "problem.title": "bold white",
        "problem.desc": "grey70",
        "problem.unknown": "italic grey58",
        "fix.applied": "bold bright_green",
        "fix.proposed": "bold bright_yellow",
        "fix.reverted": "strike grey50",
        "fix.dismissed": "grey42",
        "row.sel": "bold black on bright_cyan",
        "status.pending": "grey42",
        "status.running": "bold bright_yellow",
        "status.done": "bold bright_green",
        "status.error": "bold bright_red",
        "border": "bright_cyan",
        "border.dim": "grey35",
        "diff.add": "green",
        "diff.del": "red",
        "diff.hunk": "bright_cyan",
        "diff.ctx": "grey50",
        # mutant-first mode variants (magenta replaces cyan)
        "border.magenta": "bright_magenta",
        "tab.active.magenta": "bold black on bright_magenta",
        "tab.idle.magenta": "bright_magenta",
        "app.title.magenta": "bold bright_magenta",
        # mutant-only mode variants (yellow replaces cyan)
        "border.yellow": "yellow",
        "tab.active.yellow": "bold black on yellow",
        "tab.idle.yellow": "yellow",
        "app.title.yellow": "bold yellow",
    }
)


def _border_key(app) -> str:
    if app.mutant_only: return "border.yellow"
    if app.analysis_mode == "mutant_first": return "border.magenta"
    return "border"

def _tab_active_key(app) -> str:
    if app.mutant_only: return "tab.active.yellow"
    if app.analysis_mode == "mutant_first": return "tab.active.magenta"
    return "tab.active"

def _tab_idle_key(app) -> str:
    if app.mutant_only: return "tab.idle.yellow"
    if app.analysis_mode == "mutant_first": return "tab.idle.magenta"
    return "tab.idle"

def _title_key(app) -> str:
    if app.mutant_only: return "app.title.yellow"
    if app.analysis_mode == "mutant_first": return "app.title.magenta"
    return "app.title"


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

@dataclass
class FileState:
    path: Path
    stem: str
    status: str = "pending"        # pending | running | done | error
    phase: str = "-"               # scan | single | multi | analysing | saved
    alive: int = 0
    killed: int = 0
    timed_out: int = 0
    invalid: int = 0
    equivalent: int = 0
    total: int = 0
    problems: dict = field(default_factory=dict)   # per-file cluster_state
    note: str = ""
    fixes_applied: int = 0

    @property
    def weak(self) -> int:
        return self.problems.get("total_weak_spec", 0)


@dataclass
class AppState:
    files: list = field(default_factory=list)
    current: int = -1
    active_tab: int = 0            # 0 MutDafny, 1 Watch, 2 Summary
    quit: bool = False
    finished: bool = False
    mutdafny_lines: deque = field(default_factory=lambda: deque(maxlen=400))
    # Watch-tab live state for the file currently being processed
    pending_names: list = field(default_factory=list)
    last_batch: dict = field(default_factory=dict)
    llm_enabled: bool = False
    # Live Gemini call indicator
    analysis_inflight: int = 0       # examples in the call currently hitting the API
    analysis_queued: int = 0         # examples in dispatched batches blocked on the lock
    analysis_started: float = 0.0    # monotonic start of the in-flight call (0 = idle)
    # Live mutdafny activity (from run.sh's status side-channel)
    mutdafny_status: str = ""        # what run.sh is currently doing
    mutdafny_status_ts: float = 0.0  # wall-clock mtime of the status file (for elapsed)
    run_active: bool = False         # a run.sh subprocess is currently running
    # Iterative refinement
    max_files: int = 0               # quit after this many files (0 = no limit)
    human_mode: bool = False         # propose-only; user applies fixes via the Proposed tab
    refine_status: str = ""          # what the refinement engine is currently doing
    proposed_sel: int = 0            # selection index in the Proposed tab
    refine_actions: list = field(default_factory=list)  # queued (kind, entry_id) from keys
    scroll: int = 0                  # pager scroll offset for the active tab
    viewport: int = 20               # current pager interior height (rows); updated each render
    # Queue 2: WEAK_SPEC mutants awaiting problem assignment (populated by equiv pass,
    # consumed by categ pass after all equiv threads finish)
    pending_categ: list = field(default_factory=list)
    # Count of mutants in dispatched equiv threads (not yet classified).
    # pending_names has been cleared but threads haven't finished.
    equiv_dispatched: int = 0
    # Watch tab navigation
    watch_view: str = "problems"     # "problems" | "diffs"
    watch_sel: int = 0               # selected problem index in problems view
    watch_diff_pid: str = ""         # problem id whose diffs are displayed
    watch_diff_sel: int = 0          # selected mutant index in diffs view
    # Debug tab navigation
    debug_sel: int = 0               # selected entry index in debug log
    debug_expanded: set = field(default_factory=set)  # set of entry ids that are expanded
    # Analysis mode
    analysis_mode: str = "on_the_fly"  # "on_the_fly" | "mutant_first"
    mode_switch_ts: float = 0.0        # monotonic time when switched from mutant_first→on_the_fly
    mutant_only: bool = False          # --mutant-only: harvest only, skip analysis entirely
    allow_overconstrain: bool = False  # --allow-overconstrain: skip GT verification check
    lock: threading.Lock = field(default_factory=threading.Lock)

    def cur(self):
        if 0 <= self.current < len(self.files):
            return self.files[self.current]
        return None


TAB_NAMES = ["MutDafny", "Watch", "Proposed", "Summary", "Debug"]


# ---------------------------------------------------------------------------
# Stray-file cleanup (per-file mutant counting now lives in PersistenceManager)
# ---------------------------------------------------------------------------

def clean_stray_files() -> None:
    """Remove run.sh leftovers in the mutdafny cwd after an interrupted run.
    targets.csv is intentionally excluded — its lifecycle is owned by the manager
    (finalize_run reads the leftover, then removes it)."""
    for pattern in ("*.dfy", "*.equiv.dfy", "*.equiv.skip", "elapsed-time.csv"):
        for f in MUTDAFNY.glob(pattern):
            try:
                f.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# run.sh subprocess management
# ---------------------------------------------------------------------------

def spawn_run(file_path: Path, multi_k: int | None, extra_env: dict | None = None) -> subprocess.Popen:
    cmd = ["bash", str(RUN_SH), str(file_path.resolve())]
    if multi_k is not None:
        cmd += ["--num_mutations", str(multi_k)]
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(
        cmd,
        cwd=str(MUTDAFNY),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        start_new_session=True,   # own process group → killpg stops dafny children too
    )


def kill_run(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        for _ in range(20):
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        if proc.poll() is None:
            os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


# ---------------------------------------------------------------------------
# Worker pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    def __init__(self, app: AppState):
        self.app = app
        self.analyser = None
        self.fixer = None
        self.engine = None
        self.processed_alive: set = set()   # alive mutant filenames already diffed (current file)
        # LLM analysis runs in background threads so its (blocking) network calls
        # never stall the worker loop / stdout draining / threshold checks.
        self._analysis_busy = threading.Lock()   # serialises cluster_state mutation
        self._flush_threads: list = []
        self._proposed_pids: set = set()    # human mode: problems already proposed (current file)
        self._propose_threads: list = []
        self._gt_text_cache: dict = {}      # GT filename → text (unverified-method detection)
        self._conditions_cache: dict = {}   # mutant name → covering conditions dict
        self._last_fs: "FileState | None" = None  # most-recent active file (for flush watcher)

    # -- LLM analyser (optional) --------------------------------------------
    def _init_analyser(self):
        try:
            from analyze_mutants import MutantAnalyser
            self.analyser = MutantAnalyser()
            with self.app.lock:
                self.app.llm_enabled = True
        except Exception as exc:
            self.analyser = None
            with self.app.lock:
                self.app.llm_enabled = False
            self._log(f"[watch] LLM analyser disabled: {exc}")
        try:
            from analyze_mutants import SpecFixer
            self.fixer = SpecFixer()
        except Exception as exc:
            self.fixer = None
            self._log(f"[refine] fix proposer disabled: {exc}")
        self.engine = RefinementEngine(
            PM, self.fixer, self.analyser, GTS_DIR, log=self._refine_log,
            allow_overconstrain=app.allow_overconstrain,
        )

    def _refine_log(self, msg: str):
        self._log(msg)
        debug_log.log("refine", msg)

    def _set_refine_status(self, msg: str):
        with self.app.lock:
            self.app.refine_status = msg

    def _log(self, line: str):
        with self.app.lock:
            self.app.mutdafny_lines.append(line.rstrip("\n"))

    # -- main loop ----------------------------------------------------------
    def run(self):
        if not self.app.mutant_only:
            self._init_analyser()

        watcher = threading.Thread(target=self._flush_watcher, daemon=True)
        watcher.start()

        for idx, fs in enumerate(self.app.files):
            if self.app.quit:
                break
            # Warm-start the analysis side from the cache: cluster_state, the set of
            # already-diffed mutants (don't re-diff), and any diffed-but-unanalyzed
            # mutants (re-queue for the LLM so analysis resumes where it left off).
            problems = PM.load_problems(fs.stem)
            diffed = PM.diffed_mutants(fs.stem)
            resume_pending = []
            if self.analyser is not None:
                # Exclude mutants already classified as targeting unverified methods
                already_excluded = {
                    m for info in problems.get("unverified_methods", {}).values()
                      for m in info.get("mutants", [])
                }
                resume_pending = sorted(diffed - PM.analyzed_mutants(fs.stem) - already_excluded)
            with self.app.lock:
                self.app.current = idx
                fs.status = "running"
                fs.problems = problems
                self.app.pending_names = list(resume_pending)
                self.app.pending_categ = []
                self.app.equiv_dispatched = 0
                self.app.last_batch = {}
            self.processed_alive = set(diffed)
            self._gt_text_cache = {}
            self._flush_threads = []
            self._proposed_pids = set()
            self._propose_threads = []
            try:
                self._process_file(fs)
                with self.app.lock:
                    fs.status = "done"
                    fs.phase = "saved"
            except Exception as exc:
                with self.app.lock:
                    fs.status = "error"
                    fs.note = str(exc)[:120]
                self._log(f"[launcher] error on {fs.stem}: {exc}")
            with self.app.lock:
                limit = self.app.max_files
            if limit and (idx + 1) >= limit:
                break

        with self.app.lock:
            self.app.finished = True

    def _process_file(self, fs: FileState):
        self._last_fs = fs
        self._log(f"[launcher] === {fs.path.name} ===")
        PM.ensure_scanned(fs.path)
        self._recount(fs)
        remaining = PM.remaining_targets(fs.stem)
        if not remaining:
            self._log(f"[launcher] all {len(PM.load_manifest(fs.stem).get('targets', []))} "
                      f"targets already consumed → resuming in multi-mutation phase")

        # Phase A: deterministic single-mutation over the REMAINING targets only.
        if remaining:
            with self.app.lock:
                fs.phase = "single"
            self._run_resumable(fs, multi_k=None)

        # Phase B: randomised multi-mutation, only if N not yet reached and budget remains.
        self._recount(fs)
        if not self.app.quit and fs.alive < N_TRUE_POSITIVE and fs.total < M_TOTAL:
            with self.app.lock:
                fs.phase = "multi"
            self._run_resumable(fs, multi_k=MULTI_MUT_K)

        # Drain-before-advance: flush the analysis backlog over every alive mutant (this runs
        # even when phases A/B did nothing because all mutations were already cached), wait for
        # background analysis, then settle refinement before the file is marked done.
        # In mutant-only mode, skip analysis entirely — just save the manifest.
        if self.app.mutant_only:
            self._save_results(fs)
            return
        with self.app.lock:
            fs.phase = "analysing"
        self._sweep_alive(fs, final=True)
        self._drain_flushes()        # wait for all equiv threads (queue 1)
        self._do_categ_pass(fs)      # assign problems once queue 1 is empty (queue 2)
        self._settle_refinement(fs)  # fix proposals once queue 2 is empty (queue 3)
        self._save_results(fs)

    def _run_resumable(self, fs: FileState, multi_k):
        """Run a generation phase until exhausted or budget is hit."""
        while not self.app.quit:
            if fs.alive >= N_TRUE_POSITIVE or fs.total >= M_TOTAL:
                return
            if multi_k is None and not PM.remaining_targets(fs.stem):
                return
            self._run_phase(fs, multi_k)
            return   # threshold | exhausted | quit

    def _settle_refinement(self, fs: FileState):
        """End-of-file: refine every remaining problem (default) or wait for the user to
        clear the proposal queue (human). All problems qualify regardless of mutant count."""
        if self.engine is None or self.fixer is None:
            return
        if self.app.human_mode:
            self._human_settle(fs)
            return
        with self.app.lock:
            fs.phase = "refining"
        while not self.app.quit:
            pids = self.engine.pending_problems(fs, 1)
            if not pids:
                break
            self._set_refine_status(f"refining {pids[0]} ({len(pids)} pending)")
            self.engine.refine_problem(fs, pids[0])
            self._recount(fs)
        self._set_refine_status("")

    def _poll_status(self):
        """Read run.sh's status side-channel so the TUI can show live activity."""
        try:
            st = STATUS_FILE.stat()
            text = STATUS_FILE.read_text().strip()
        except OSError:
            return
        with self.app.lock:
            self.app.mutdafny_status = text
            self.app.mutdafny_status_ts = st.st_mtime

    def _run_phase(self, fs: FileState, multi_k) -> str:
        """Run one run.sh invocation. Returns the reason it ended:
        'threshold' | 'exhausted' | 'quit'."""
        multi = multi_k is not None
        # Manager writes targets.csv (remapped remaining only) and returns run.sh's env.
        env = PM.prepare_run(fs.path, multi=multi)
        # Once spec fixes are applied, run.sh mutates the patched program (current.dfy).
        program = PM.active_source(fs.stem, fs.path)
        # Stale status from a previous run should not look "live".
        try:
            STATUS_FILE.unlink()
        except OSError:
            pass
        with self.app.lock:
            self.app.run_active = True
            self.app.mutdafny_status = ""
        proc = spawn_run(program, multi_k, extra_env=env)
        q: "deque[str]" = deque()
        reader_done = threading.Event()

        def _reader():
            try:
                for line in proc.stdout:          # type: ignore[union-attr]
                    q.append(line)
            finally:
                reader_done.set()

        rt = threading.Thread(target=_reader, daemon=True)
        rt.start()

        reason = "exhausted"
        while True:
            drained = False
            while q:
                self._log(q.popleft())
                drained = True
            self._recount(fs)
            self._poll_status()
            self._sweep_alive(fs, final=False)

            if fs.alive >= N_TRUE_POSITIVE or fs.total >= M_TOTAL:
                self._log(
                    f"[launcher] threshold reached (alive={fs.alive}/{N_TRUE_POSITIVE}, "
                    f"total={fs.total}/{M_TOTAL}) → stopping {fs.phase} phase"
                )
                kill_run(proc)
                reason = "threshold"
                break
            if self.app.quit:
                kill_run(proc)
                reason = "quit"
                break
            if reader_done.is_set() and proc.poll() is not None and not q and not drained:
                break
            time.sleep(POLL_INTERVAL)

        # drain trailing output + leftovers
        while q:
            self._log(q.popleft())
        # Fold this run's consumption back into the manifest BEFORE cleaning strays
        # (multi mode derives consumption from the leftover targets.csv).
        PM.finalize_run(fs.path, multi=multi)
        clean_stray_files()
        self._recount(fs)
        with self.app.lock:
            self.app.run_active = False
        return reason

    def _pause_for_refine(self, fs: FileState):
        """Default mode: the qualifying problem to refine next, or None. Snapshot under lock
        because background analysis threads mutate the clusters concurrently."""
        if self.engine is None or self.fixer is None:
            return None
        with self.app.lock:
            return self.engine.qualifying_problem(fs, 1)

    def _dispatch_human_proposals(self, fs: FileState):
        """Human mode: generate a proposal for every problem that doesn't yet have one."""
        if self.engine is None or self.fixer is None:
            return
        with self.app.lock:
            pids = self.engine.pending_problems(fs, 1)
        for pid in pids:
            if pid in self._proposed_pids:
                continue
            self._proposed_pids.add(pid)
            t = threading.Thread(target=self._propose_bg, args=(fs, pid), daemon=True)
            t.start()
            self._propose_threads.append(t)

    def _propose_bg(self, fs: FileState, pid: str):
        try:
            self.engine.propose_only(fs, pid)
        except Exception as exc:
            self._log(f"[refine] proposal failed for {pid}: {exc}")

    def _do_refinement_pass(self, fs: FileState):
        """Paused mid-generation (default mode): drain analysis so clusters are current, then
        refine every currently-qualifying problem before generation warm-resumes."""
        with self.app.lock:
            fs.phase = "refining"
        # Make sure cluster membership reflects all generated alive mutants.
        self._sweep_alive(fs, final=True)
        self._drain_flushes()
        while not self.app.quit:
            pid = self._pause_for_refine(fs)
            if pid is None:
                break
            self._set_refine_status(f"refining {pid}")
            self.engine.refine_problem(fs, pid)
            self._recount(fs)
        self._set_refine_status("")
        with self.app.lock:
            fs.phase = "single" if PM.remaining_targets(fs.stem) else "multi"

    def _human_settle(self, fs: FileState):
        """Human mode end-of-file gate: ensure a proposal exists for every problem, then wait
        for the user to apply or dismiss each one (processing queued key actions) before the
        file is allowed to complete."""
        with self.app.lock:
            fs.phase = "proposing"
        self._dispatch_human_proposals(fs)
        for t in list(self._propose_threads):
            t.join(timeout=60)
        while not self.app.quit:
            action = None
            with self.app.lock:
                if self.app.refine_actions:
                    action = self.app.refine_actions.pop(0)
            if action is not None:
                kind, eid = action
                try:
                    if kind == "apply":
                        self.engine.apply_proposal(fs, eid)
                    elif kind == "dismiss":
                        self.engine.dismiss_proposal(fs, eid)
                except Exception as exc:
                    self._log(f"[refine] {kind} {eid} failed: {exc}")
                self._recount(fs)
                continue
            stack = PM.load_fix_stack(fs.stem)
            if not any(e.get("status") == "unapplied" for e in stack.get("entries", [])):
                break
            time.sleep(POLL_INTERVAL)
        with self.app.lock:
            fs.phase = "analysing"

    def _recount(self, fs: FileState):
        c = PM.counts(fs.stem)
        applied = len(PM.applied_fixes(fs.stem))
        with self.app.lock:
            fs.alive = c["alive"]
            fs.killed = c["killed"]
            fs.timed_out = c["timed-out"]
            fs.invalid = c["invalid"]
            fs.equivalent = c["equivalent"]
            fs.total = c["total"]
            fs.fixes_applied = applied

    # -- watch / analysis ---------------------------------------------------
    def _should_flush(self) -> bool:
        """Return True when analysis batches should be dispatched right now."""
        with self.app.lock:
            mode = self.app.analysis_mode
            ts = self.app.mode_switch_ts
        if mode == "mutant_first":
            return False
        if ts > 0.0:                           # debounce: switched from mutant_first, waiting 5 s
            if time.monotonic() - ts < 5.0:
                return False
            with self.app.lock:               # debounce expired — clear the timestamp
                self.app.mode_switch_ts = 0.0
        return True

    def _sweep_alive(self, fs: FileState, final: bool):
        """Diff newly-generated alive mutants and queue them for background clustering.
        Skips analysis queueing for mutants targeting methods with no spec clauses or
        methods marked {:verify false}."""
        from analyze_mutants import (
            parse_mutant_filename, _find_method_at_offset, _method_has_spec,
            _method_has_verify_false, _collect_covering_conditions,
        )
        gen_alive = sorted(set(PM.list_mutants("alive", fs.stem)) - self.processed_alive)
        diffs_dir = PM.diffs_dir("alive", fs.stem)
        alive_dir = PM.cat_dir("alive", fs.stem)
        new_names = []
        uvm_dirty = False

        for name in gen_alive:
            try:
                entry = wm.process_mutant(alive_dir / name, diffs_dir)
                if entry is None:
                    continue
                if self.analyser is None:
                    continue
                # Unverified-method filter: skip analysis for methods with no spec clauses
                info = parse_mutant_filename(name)
                gt_name = info.get("gt_name", "")
                pos_str = info.get("pos", "")
                skip = False
                if gt_name and pos_str and pos_str != "-":
                    try:
                        char_offset = int(pos_str.split("-")[0])
                        if gt_name not in self._gt_text_cache:
                            gt_path = GTS_DIR / gt_name
                            self._gt_text_cache[gt_name] = (
                                gt_path.read_text() if gt_path.exists() else ""
                            )
                        gt_text = self._gt_text_cache[gt_name]
                        if gt_text:
                            method = _find_method_at_offset(gt_text, char_offset)
                            if method and _method_has_verify_false(gt_text, method):
                                # {:verify false}: Dafny skips verification → unkillable.
                                # Tracked in unverified_methods so warm restarts exclude them
                                # and fs.alive = total_processed + total_unverified holds.
                                uvm = fs.problems.setdefault("unverified_methods", {})
                                vf_key = f"[{method}] (verify_false)"
                                if vf_key not in uvm:
                                    uvm[vf_key] = {"name": method, "verify_false": True,
                                                   "mutant_count": 0, "mutants": []}
                                uvm[vf_key]["mutants"].append(name)
                                uvm[vf_key]["mutant_count"] += 1
                                fs.problems["total_unverified"] = (
                                    fs.problems.get("total_unverified", 0) + 1
                                )
                                uvm_dirty = True
                                skip = True
                            elif method and not _method_has_spec(gt_text, method):
                                uvm = fs.problems.setdefault("unverified_methods", {})
                                if method not in uvm:
                                    uvm[method] = {"name": method, "mutant_count": 0,
                                                   "mutants": []}
                                uvm[method]["mutants"].append(name)
                                uvm[method]["mutant_count"] += 1
                                fs.problems["total_unverified"] = (
                                    fs.problems.get("total_unverified", 0) + 1
                                )
                                uvm_dirty = True
                                skip = True
                            else:
                                cond = _collect_covering_conditions(gt_text, char_offset)
                                self._conditions_cache[name] = cond
                    except (ValueError, OSError):
                        pass
                if not skip:
                    new_names.append(name)
            except Exception as exc:
                self._log(f"[watch] diff failed {name}: {exc}")
            finally:
                self.processed_alive.add(name)

        if uvm_dirty:
            PM.save_problems(fs.stem, fs.problems)

        if new_names:
            with self.app.lock:
                self.app.pending_names.extend(new_names)

        if self.analyser is not None:
            self._maybe_flush(fs, force=final)

    def _maybe_flush(self, fs: FileState, force: bool):
        """Dispatch pending analysis batches, chunked for large backlogs (mutant-first mode).
        Never dispatches during the 5-second debounce after leaving mutant-first mode."""
        if self.analyser is None:
            return
        if not self._should_flush():
            return
        with self.app.lock:
            pending = list(self.app.pending_names)
            if not pending or (not force and len(pending) < BATCH_SIZE):
                return
            self.app.pending_names = []          # claim now so we don't re-send

        diffs_dir = PM.diffs_dir("alive", fs.stem)
        all_entries = []
        for name in pending:
            dfile = diffs_dir / (name[: -len(".dfy")] + ".diff")
            diff_text = dfile.read_text() if dfile.exists() else ""
            entry: dict = {"name": name, "diff": diff_text}
            cond = self._conditions_cache.get(name)
            if cond:
                entry["conditions"] = cond
            all_entries.append(entry)

        # Chunk large backlogs into MUTANT_FIRST_BATCH_SIZE-sized batches so each
        # Gemini request sees enough mutants for good clustering without hitting limits.
        chunk_size = (MUTANT_FIRST_BATCH_SIZE
                      if len(all_entries) > BATCH_SIZE else len(all_entries))
        with self.app.lock:
            self.app.equiv_dispatched += len(all_entries)
        for i in range(0, len(all_entries), chunk_size):
            chunk = all_entries[i:i + chunk_size]
            t = threading.Thread(target=self._do_equiv_flush, args=(fs, chunk), daemon=True)
            t.start()
            self._flush_threads.append(t)

    def _do_equiv_flush(self, fs: FileState, batch: list):
        """Queue-1 worker: classify a batch of mutants as EQUIV/WEAK_SPEC/CONTEXTUAL.
        WEAK_SPEC results are deposited into pending_categ (queue 2) for later processing
        once all equiv threads have drained."""
        analysis_dir = PM.analysis_dir(fs.stem)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        n = len(batch)
        with self.app.lock:
            self.app.analysis_queued += n
        with self._analysis_busy:
            with self.app.lock:
                self.app.analysis_queued -= n
                self.app.analysis_inflight = n
                self.app.analysis_started = time.monotonic()
            try:
                results, summary = self.analyser.classify_equiv(
                    batch, GTS_DIR, analysis_dir, fs.problems
                )
                PM.save_problems(fs.stem, fs.problems)
                weak_spec = [r for r in results if r.get("verdict") == "WEAK_SPEC"]
                with self.app.lock:
                    self.app.pending_categ.extend(weak_spec)
                    self.app.last_batch = summary
            except Exception as exc:
                self._log(f"[watch] equiv analysis failed: {exc}")
            finally:
                with self.app.lock:
                    self.app.analysis_inflight = 0
                    self.app.analysis_started = 0.0
                    self.app.equiv_dispatched = max(0, self.app.equiv_dispatched - n)

    def _do_categ_pass(self, fs: FileState):
        """Queue-2: assign all accumulated WEAK_SPEC mutants to problem clusters.
        Called synchronously in the worker thread after all equiv threads have drained,
        so queue 1 is guaranteed empty when this runs."""
        with self.app.lock:
            batch = list(self.app.pending_categ)
            self.app.pending_categ = []
        if not batch or self.analyser is None:
            return
        analysis_dir = PM.analysis_dir(fs.stem)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"[watch] categ pass: {len(batch)} WEAK_SPEC mutant(s)")
        with self.app.lock:
            self.app.analysis_inflight = len(batch)
            self.app.analysis_started = time.monotonic()
        try:
            _results, summary = self.analyser.assign_problems(
                batch, GTS_DIR, analysis_dir, fs.problems
            )
            PM.save_problems(fs.stem, fs.problems)
            with self.app.lock:
                lb = dict(self.app.last_batch)
                lb.update(summary)
                self.app.last_batch = lb
        except Exception as exc:
            self._log(f"[watch] categ pass failed: {exc}")
        finally:
            with self.app.lock:
                self.app.analysis_inflight = 0
                self.app.analysis_started = 0.0

    def _flush_watcher(self):
        """Background thread: dispatch pending_names once mode switches back to on_the_fly."""
        while not self.app.quit:
            time.sleep(2)
            with self.app.lock:
                has_pending = bool(self.app.pending_names)
                last_fs = self._last_fs
            if has_pending and last_fs and self._should_flush():
                self._maybe_flush(last_fs, force=True)

    def _drain_flushes(self):
        """Wait for all outstanding equiv analysis threads before advancing to categ.
        Checks app.quit every second so the user can still exit cleanly."""
        for t in self._flush_threads:
            while t.is_alive() and not self.app.quit:
                t.join(timeout=1.0)
        self._flush_threads = []

    # -- persistence --------------------------------------------------------
    def _save_results(self, fs: FileState):
        # Mutants are already in their per-file category dirs (run.sh placed them)
        # and diffs/analysis are written in place; just persist problems + manifest.
        PM.save_problems(fs.stem, fs.problems)
        m = PM.load_manifest(fs.stem, fs.path)
        m["thresholds"] = {"N_true_positive": N_TRUE_POSITIVE, "M_total": M_TOTAL}
        PM.save_manifest(m)
        self._log(
            f"[launcher] saved {fs.stem}: alive={fs.alive} total={fs.total} "
            f"weak={fs.weak} problems={len(fs.problems.get('clusters', {}))}"
        )


# ---------------------------------------------------------------------------
# Keyboard listener (raw tty)
# ---------------------------------------------------------------------------

def _read_escape_sequence() -> str:
    """Read the CSI/SS3 portion of an escape sequence (after the ESC byte).
    Returns a canonical name or empty string for bare ESC (timeout)."""
    buf = ""
    while True:
        r, _, _ = select.select([sys.stdin], [], [], 0.15)
        if not r:
            break
        c = sys.stdin.read(1)
        if not c:
            break
        buf += c
        if buf and buf[-1].isalpha():
            break
    return {
        "[A": "UP",    "[B": "DOWN",   "[C": "RIGHT",  "[D": "LEFT",
        "[1;2A": "SUP", "[1;2B": "SDOWN",
        "[1;2C": "SRIGHT", "[1;2D": "SLEFT",
        "[Z": "STAB",   # Shift+Tab (terminal-dependent)
        "m": "ALTM",    # Alt+M (ESC + m)
        "": "ESC",
    }.get(buf, "")


def keyboard_thread(app: AppState):
    if not sys.stdin.isatty():
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not app.quit:
            r, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not r:
                continue
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                ch = _read_escape_sequence()
            with app.lock:
                if ch in ("q", "Q"):
                    app.quit = True
                elif ch in ("1", "2", "3", "4", "5"):
                    new_tab = int(ch) - 1
                    if app.active_tab != new_tab:
                        app.watch_view = "problems"
                    app.active_tab = new_tab
                    app.scroll = 0
                elif ch == "\t":
                    app.watch_view = "problems"
                    app.active_tab = (app.active_tab + 1) % len(TAB_NAMES)
                    app.scroll = 0
                # UP/DOWN always scroll the viewport — independent of j/k selection
                elif ch == "UP":
                    app.scroll = max(0, app.scroll - 1)
                elif ch == "DOWN":
                    app.scroll += 1
                elif ch == "SUP":
                    app.scroll = max(0, app.scroll - app.viewport)
                elif ch == "SDOWN":
                    app.scroll += app.viewport
                elif ch in ("STAB", "ALTM"):     # Shift+Tab / Alt+M: toggle analysis mode
                    if app.analysis_mode == "mutant_first":
                        app.analysis_mode = "on_the_fly"
                        app.mode_switch_ts = time.monotonic()
                    else:
                        app.analysis_mode = "mutant_first"
                        app.mode_switch_ts = 0.0  # cancel any pending debounce
                    app.scroll = 0
                elif app.active_tab == 4:        # Debug tab: j/k navigate, Enter expand
                    es = debug_log.entries()
                    n = max(1, len(es))
                    if ch == "k":
                        app.debug_sel = max(0, app.debug_sel - 1)
                    elif ch == "j":
                        app.debug_sel = min(n - 1, app.debug_sel + 1)
                    elif ch in ("\r", "\n") and es:
                        idx = app.debug_sel % n
                        eid = es[idx]["id"]
                        if eid in app.debug_expanded:
                            app.debug_expanded.discard(eid)
                        else:
                            app.debug_expanded.add(eid)
                elif app.active_tab == 1:        # Watch tab
                    clusters = list(app.cur().problems.get("clusters", {}).items()) \
                        if app.cur() else []
                    if app.watch_view == "diffs":
                        # j/k select mutant within diffs view
                        if ch == "k":
                            app.watch_diff_sel = max(0, app.watch_diff_sel - 1)
                        elif ch == "j":
                            app.watch_diff_sel += 1  # clamped in renderer
                        elif ch in ("ESC", "\x7f"):
                            app.watch_view = "problems"
                            app.scroll = 0
                    else:
                        # j/k select problem; Enter to enter diffs view
                        if ch == "k":
                            app.watch_sel = max(0, app.watch_sel - 1)
                        elif ch == "j":
                            app.watch_sel = min(max(0, len(clusters) - 1), app.watch_sel + 1)
                        elif ch in ("\r", "\n") and clusters:
                            sel = min(app.watch_sel, len(clusters) - 1)
                            app.watch_diff_pid = clusters[sel][0]
                            app.watch_view = "diffs"
                            app.watch_diff_sel = 0
                            app.scroll = 0
                elif app.active_tab == 2:        # Proposed tab: j/k select, a/d act
                    proposals = _proposal_entries(app)
                    if ch == "k":
                        app.proposed_sel = max(0, app.proposed_sel - 1)
                    elif ch == "j":
                        app.proposed_sel = min(max(0, len(proposals) - 1), app.proposed_sel + 1)
                    elif ch in ("a", "d") and proposals:
                        sel = min(app.proposed_sel, len(proposals) - 1)
                        eid = proposals[sel]["id"]
                        kind = {"a": "apply", "d": "dismiss"}[ch]
                        app.refine_actions.append((kind, eid))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _proposal_entries(app: AppState) -> list:
    """Unapplied (proposed) fix entries for the current file."""
    fs = app.cur()
    if not fs:
        return []
    try:
        stack = PM.load_fix_stack(fs.stem)
    except Exception:
        return []
    return [e for e in stack.get("entries", []) if e.get("status") == "unapplied"]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def colorize_diff(text: str, max_lines: int = 40) -> Text:
    lines = text.splitlines()
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


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _tab_bar(app: AppState) -> Text:
    bar = Text()
    for i, name in enumerate(TAB_NAMES):
        style = _tab_active_key(app) if i == app.active_tab else _tab_idle_key(app)
        bar.append(f" {i + 1} {name} ", style=style)
        bar.append(" ")
    return bar


def _counts_table(fs: FileState) -> Table:
    t = Table.grid(padding=(0, 2))
    t.add_row(
        Text(f"alive {fs.alive}/{N_TRUE_POSITIVE}", style="stat.alive"),
        Text(f"total {fs.total}/{M_TOTAL}", style="stat.total"),
        Text(f"killed {fs.killed}", style="stat.killed"),
        Text(f"equiv {fs.equivalent}", style="stat.equiv"),
        Text(f"timeout {fs.timed_out}", style="stat.timeout"),
        Text(f"invalid {fs.invalid}", style="stat.invalid"),
    )
    return t


def _mutdafny_activity(app: AppState):
    """Animated 'what run.sh is waiting on + for how long' line, from the status
    side-channel. Because run.sh batches per-target stdout, this is the only live
    signal during a slow verify; the ticking timer distinguishes slow from hung."""
    with app.lock:
        active = app.run_active
        status = app.mutdafny_status
        ts = app.mutdafny_status_ts
    if not active or not status or status == "done":
        return None
    frame = SPINNER_FRAMES[int(time.monotonic() * 10) % len(SPINNER_FRAMES)]
    elapsed = (time.time() - ts) if ts else 0.0
    # Past the hard backstop the call should already have been killed — flag it red.
    elapsed_style = "stat.killed" if elapsed > DAFNY_HARD_TIMEOUT else "stat.timeout"
    line = Text()
    line.append(f" {frame} ", style="status.running")
    line.append("waiting on ", style="app.dim")
    line.append(status, style="status.running")
    line.append("  ·  ", style="app.dim")
    line.append(f"{elapsed:.1f}s", style=elapsed_style)
    return line


def _mutdafny_body(app: AppState) -> Group:
    fs = app.cur()
    head = Table.grid(padding=(0, 1))
    if fs:
        head.add_row(Text(f"▶ {fs.path.name}", style="app.title"),
                     Text(f"[phase: {fs.phase}]", style="status.running"))
        head.add_row(_counts_table(fs))
    else:
        head.add_row(Text("(waiting…)", style="app.dim"))
    activity = _mutdafny_activity(app)
    with app.lock:
        lines = list(app.mutdafny_lines)[-MUTDAFNY_SCROLLBACK:]
    body = Text("\n".join(lines) if lines else "(no output yet)", style="grey74")
    parts: list = [head]
    if activity is not None:
        parts.append(activity)
    parts += [Text(""), body]
    return Group(*parts)


def render_mutdafny(app: AppState) -> Panel:
    return Panel(_mutdafny_body(app), title="MutDafny · run.sh",
                 border_style="border", title_align="left")


def _problems_renderable(cluster_state: dict) -> Group:
    # Snapshot items: a background analysis thread may mutate clusters concurrently.
    clusters = list(cluster_state.get("clusters", {}).items())
    rows = [Text(f"SPEC PROBLEMS ({len(clusters)})", style="app.title")]
    if not clusters:
        rows.append(Text("  (none identified yet)", style="app.dim"))
        return Group(*rows)
    for cid, c in clusters:
        cnt = c.get("mutant_count", 0)
        title = c.get("title", "")
        header = Text()
        header.append(f"  [{cid}] ", style="problem.id")
        header.append(title, style="problem.title")
        header.append(f"   ({cnt} mut)", style="app.dim")
        if c.get("status") == "unresolved":
            header.append("  ⚑ unresolved", style="stat.timeout")
        rows.append(header)
        desc = c.get("description", "")
        if desc:
            # Full description, word-wrapped to the panel width (no truncation),
            # indented under the problem header via left padding.
            rows.append(Padding(Text(desc, style="problem.desc"), (0, 0, 0, 6)))
        oq = c.get("open_questions", "")
        if oq:
            rows.append(Padding(Text(f"unknown: {oq}", style="problem.unknown"), (0, 0, 0, 6)))
    return Group(*rows)


def _watch_stats(app: AppState, cs: dict) -> Table:
    total = cs.get("total_processed", 0)
    weak = cs.get("total_weak_spec", 0)
    equiv = cs.get("total_equivalent", 0)
    contextual = cs.get("total_contextual", 0)
    unverified = cs.get("total_unverified", 0)
    pct = f"{weak / total:.0%}" if total else "—"
    with app.lock:
        llm = "on" if app.llm_enabled else "off"
        mode = app.analysis_mode
        mutant_only = app.mutant_only
    if mutant_only:
        mode_label = "MUTANT-ONLY"
        mode_style = "app.title.yellow"
    elif mode == "mutant_first":
        mode_label = "MUTANT-FIRST"
        mode_style = "app.title.magenta"
    else:
        mode_label = "ON-THE-FLY"
        mode_style = "app.dim"
    cells = [
        Text(f"Processed {total}", style="stat.total"),
        Text(f"WEAK_SPEC {weak} ({pct})", style="stat.alive"),
        Text(f"EQUIV {equiv}", style="stat.equiv"),
    ]
    if contextual:
        cells.append(Text(f"CONTEXTUAL {contextual}", style="stat.timeout"))
    if unverified:
        cells.append(Text(f"UNKILLABLE {unverified}", style="stat.killed"))
    cells.append(Text(f"LLM {llm}  {mode_label}", style=mode_style))
    t = Table.grid(padding=(0, 3))
    t.add_row(*cells)
    return t


def _watch_problems_view(app: AppState, fs, cs: dict) -> Group:
    """Selectable problem list for the Watch tab (j/k to navigate)."""
    clusters = list(cs.get("clusters", {}).items())
    with app.lock:
        sel = min(app.watch_sel, max(0, len(clusters) - 1))
    rows: list = [Text(
        f"SPEC PROBLEMS ({len(clusters)})  ·  j/k select  ·  Enter: view diffs",
        style="app.title",
    )]
    if not clusters:
        rows.append(Text("  (none identified yet)", style="app.dim"))
        return Group(*rows)
    for idx, (cid, c) in enumerate(clusters):
        selected = (idx == sel)
        cnt = c.get("mutant_count", 0)
        marker = "▶ " if selected else "  "
        row_style = "row.sel" if selected else ""
        header = Text(style=row_style)
        header.append(f"{marker}[{cid}] ", style="problem.id")
        header.append(c.get("title", ""), style="problem.title")
        header.append(f"   ({cnt} mut)", style="app.dim")
        if c.get("status") == "unresolved":
            header.append("  ⚑ unresolved", style="stat.timeout")
        rows.append(header)
        desc = c.get("description", "")
        if desc:
            rows.append(Padding(Text(desc, style="problem.desc"), (0, 0, 0, 6)))
        oq = c.get("open_questions", "")
        if oq:
            rows.append(Padding(Text(f"unknown: {oq}", style="problem.unknown"), (0, 0, 0, 6)))
    return Group(*rows)


def _watch_diffs_view(app: AppState, fs, cs: dict) -> Group:
    """Concatenated mutant diffs for the selected problem (j/k or ↑/↓ to select)."""
    with app.lock:
        pid = app.watch_diff_pid
        sel = app.watch_diff_sel
    cluster = cs.get("clusters", {}).get(pid, {})
    title = Text()
    title.append(f"Diffs for [{pid}]: ", style="problem.id")
    title.append(cluster.get("title", ""), style="problem.title")
    title.append("   j/k select · ↑/↓ scroll · Esc/Backspace back", style="app.dim")
    parts: list = [title]
    mutants = cluster.get("mutants", [])
    if not mutants:
        parts.append(Text("  (no mutants in this cluster)", style="app.dim"))
    else:
        sel = max(0, min(sel, len(mutants) - 1))
        for idx, name in enumerate(mutants):
            selected = (idx == sel)
            marker = "▶ " if selected else "  "
            row_style = "row.sel" if selected else "app.dim"
            mut_head = Text()
            mut_head.append(f"\n{marker}{name}", style=row_style)
            parts.append(mut_head)
            dfile = PM.diffs_dir("alive", fs.stem) / (name[:-len(".dfy")] + ".diff")
            diff_text = dfile.read_text() if dfile.exists() else ""
            parts.append(Padding(colorize_diff(diff_text), (0, 0, 0, 2)))
    return Group(*parts)


def _unverified_methods_section(fs) -> Group:
    """Unkillable methods: no spec clauses, or marked {:verify false}."""
    cs = fs.problems if fs else {}
    uvm = cs.get("unverified_methods", {})
    total_unv = cs.get("total_unverified", 0)
    rows: list = [Text(f"UNKILLABLE METHODS ({total_unv} mutants)", style="app.title")]
    if not uvm:
        rows.append(Text("  (none)", style="app.dim"))
    else:
        no_spec = {k: v for k, v in uvm.items() if not v.get("verify_false")}
        vf = {k: v for k, v in uvm.items() if v.get("verify_false")}
        if no_spec:
            rows.append(Text(
                "  No postcondition — add ensures/decreases to enable spec fixes:",
                style="stat.killed",
            ))
            for name, info in no_spec.items():
                row = Text()
                row.append(f"  {name}", style="problem.title")
                row.append(f"   ({info.get('mutant_count', 0)} mutant(s))", style="app.dim")
                rows.append(row)
        if vf:
            rows.append(Text("  {:verify false} — Dafny skips verification:", style="stat.killed"))
            for _key, info in vf.items():
                row = Text()
                row.append(f"  {info.get('name', _key)}", style="problem.title")
                row.append(f"   ({info.get('mutant_count', 0)} mutant(s))", style="app.dim")
                rows.append(row)
    return Group(*rows)


def _watch_body(app: AppState) -> Group:
    fs = app.cur()
    cs = fs.problems if fs else {}
    stats = _watch_stats(app, cs)

    with app.lock:
        view = app.watch_view

    if view == "diffs" and fs:
        return Group(stats, Text(""), _watch_diffs_view(app, fs, cs))

    # Problems view
    with app.lock:
        pending = list(app.pending_names)
        pending_categ = list(app.pending_categ)
        equiv_dispatched = app.equiv_dispatched
        last = dict(app.last_batch)
    q1_total = len(pending) + equiv_dispatched
    pend_txt = Text(
        f"QUEUE 1 — equiv  ({q1_total} pending · {BATCH_SIZE} per batch)",
        style="app.title",
    )
    if equiv_dispatched and not pending:
        pend_body_str = f"  ({equiv_dispatched} in equiv analysis threads)"
    elif pending:
        pend_body_str = "\n".join("  " + n for n in pending[-8:])
    else:
        pend_body_str = "  (none — watching for new alive mutants)"
    pend_body = Text(pend_body_str, style="grey66")
    parts: list = [stats, Text(""), _watch_problems_view(app, fs, cs),
                   Text(""), _unverified_methods_section(fs),
                   Text(""), pend_txt, pend_body]
    if pending_categ:
        parts += [
            Text(""),
            Text(f"QUEUE 2 — categ ({len(pending_categ)} WEAK_SPEC awaiting problem assignment)",
                 style="stat.alive"),
        ]
    if last:
        ctx_note = f"  +{last.get('contextual', 0)} CONTEXTUAL" if last.get("contextual") else ""
        weak_note = f"  +{last.get('weak', 0)} WEAK" if "weak" in last else ""
        equiv_note = f"  +{last.get('equiv', 0)} EQUIV" if "equiv" in last else ""
        prob_note = (f"  {last.get('new_problems', 0)} new problem(s)"
                     f"  {last.get('merges', 0)} merge(s)" if "new_problems" in last else "")
        parts += [Text(""), Text(
            f"LAST BATCH{weak_note}{equiv_note}{ctx_note}{prob_note}",
            style="app.dim")]
    parts += [Text(""), _fix_history(fs), Text(""), _fixed_problems(fs)]
    return Group(*parts)


def render_watch(app: AppState) -> Panel:
    return Panel(_watch_body(app), title="Watch · mutant analysis + fix history",
                 border_style="border", title_align="left")


_FIX_STATUS_STYLE = {
    "applied": "fix.applied",
    "unapplied": "fix.proposed",
    "reverted": "fix.reverted",
    "dismissed": "fix.dismissed",
}


def _fix_delta(entry: dict) -> Group:
    """Full anchor→clause delta for one fix, with enclosing function and line number."""
    rows = []
    func = entry.get("function", "")
    line = entry.get("line", 0)
    if func or line:
        loc = Text()
        loc.append("    in ", style="app.dim")
        loc.append(func or "?", style="problem.title")
        if line:
            loc.append(f"  ·  line {line}", style="app.dim")
        edits = entry.get("edits", 0)
        if edits:
            loc.append(f"  ·  edited {edits}×", style="stat.timeout")
        rows.append(loc)
    anchor = entry.get("anchor_line", "")
    if anchor:
        ctx = Text()
        ctx.append("    @ ", style="app.dim")
        ctx.append(anchor, style="app.dim")
        rows.append(ctx)
    add = Text()
    add.append("    + ", style="stat.alive")
    add.append(entry.get("clause", ""), style="stat.alive")
    rows.append(add)
    return Group(*rows)


def _fix_history(fs: FileState) -> Group:
    """Applied + reverted fix history for the active file (read from the fix stack), each
    shown as a full anchor→clause delta (no truncation)."""
    entries = []
    if fs:
        try:
            entries = PM.load_fix_stack(fs.stem).get("entries", [])
        except Exception:
            entries = []
    shown = [e for e in entries if e.get("status") in ("applied", "reverted")]
    rows = [Text(f"FIX HISTORY ({len(shown)})", style="app.title")]
    if not shown:
        rows.append(Text("  (no fixes applied yet)", style="app.dim"))
        return Group(*rows)
    for e in shown:
        style = _FIX_STATUS_STYLE.get(e.get("status", ""), "app.dim")
        head = Text()
        head.append(f"  {e['id']} ", style="problem.id")
        head.append(f"[{e.get('status', '')}] ", style=style)
        head.append(f"{e.get('problem_id', '')} ", style="app.dim")
        head.append(e.get("kind", ""), style="problem.title")
        rows.append(head)
        rows.append(_fix_delta(e))
    return Group(*rows)


def _fixed_problems(fs: FileState) -> Group:
    """Problems that a fix resolved. Their clusters are emptied/removed once fixed, so we
    keep their descriptions here rather than letting them disappear."""
    resolved = fs.problems.get("resolved", {}) if fs else {}
    items = list(resolved.items())
    rows = [Text(f"FIXED PROBLEMS ({len(items)})", style="app.title")]
    if not items:
        rows.append(Text("  (none resolved yet)", style="app.dim"))
        return Group(*rows)
    for cid, c in items:
        header = Text()
        header.append(f"  [{cid}] ", style="problem.id")
        header.append(c.get("title", ""), style="problem.title")
        by = ", ".join(c.get("resolved_by", [])) or "—"
        tag = "byproduct" if c.get("byproduct") else "fixed"
        header.append(f"   ✓ {tag} by {by}", style="fix.applied")
        rows.append(header)
        desc = c.get("description", "")
        if desc:
            rows.append(Padding(Text(desc, style="problem.desc"), (0, 0, 0, 6)))
        oq = c.get("open_questions", "")
        if oq:
            rows.append(Padding(Text(f"unknown: {oq}", style="problem.unknown"), (0, 0, 0, 6)))
    return Group(*rows)


def _proposed_body(app: AppState) -> Group:
    fs = app.cur()
    proposals = _proposal_entries(app)
    mode = "HUMAN" if app.human_mode else "AUTO"
    head = Text()
    head.append(f"Proposed fixes  ·  {mode} mode", style="app.title")
    if not app.human_mode:
        head.append("   (auto mode applies fixes itself; this queue stays empty)", style="app.dim")
    rows: list = [head, Text("")]
    if not proposals:
        rows.append(Text("  (no proposals awaiting approval)", style="app.dim"))
    else:
        sel = min(app.proposed_sel, len(proposals) - 1)
        for i, e in enumerate(proposals):
            marker = "▶ " if i == sel else "  "
            row_style = "row.sel" if i == sel else ""
            head_line = Text(style=row_style)
            head_line.append(f"{marker}{e['id']} ", style="problem.id")
            head_line.append(f"{e.get('problem_id', '')} ", style="app.dim")
            head_line.append(e.get("kind", ""), style="fix.proposed")
            rows.append(head_line)
            desc = e.get("description", "")
            if desc:
                rows.append(Padding(Text(desc, style="problem.desc"), (0, 0, 0, 4)))
            rows.append(_fix_delta(e))

    # CONTEXTUAL section
    cs = fs.problems if fs else {}
    contextual = cs.get("contextual", [])
    rows.append(Text(""))
    ctx_head = Text(f"CONTEXTUAL ({len(contextual)})", style="app.title")
    ctx_head.append(
        "  — wrong but unspeccable (efficiency / algorithm-level bugs)", style="app.dim"
    )
    rows.append(ctx_head)
    if not contextual:
        rows.append(Text("  (none identified)", style="app.dim"))
    else:
        for item in contextual:
            mut_row = Text()
            mut_row.append(f"  {item.get('mutant', '')}", style="app.dim")
            rows.append(mut_row)
            just = item.get("justification", "")
            if just:
                rows.append(Padding(Text(just, style="problem.desc"), (0, 0, 0, 4)))

    # UNRESOLVABLE problems section
    unresolvable = [
        (cid, c) for cid, c in cs.get("clusters", {}).items()
        if c.get("status") == "unresolved" and c.get("unresolved_reason")
    ]
    rows.append(Text(""))
    unr_head = Text(f"UNRESOLVABLE ({len(unresolvable)})", style="app.title")
    unr_head.append(
        "  — require human input to narrow intended behaviour", style="app.dim"
    )
    rows.append(unr_head)
    if not unresolvable:
        rows.append(Text("  (none)", style="app.dim"))
    else:
        for cid, c in unresolvable:
            row = Text()
            row.append(f"  [{cid}] ", style="problem.id")
            row.append(c.get("title", ""), style="problem.title")
            rows.append(row)
            reason = c.get("unresolved_reason", "")
            if reason:
                rows.append(Padding(Text(reason, style="stat.timeout"), (0, 0, 0, 4)))

    return Group(*rows)


def render_proposed(app: AppState) -> Panel:
    return Panel(_proposed_body(app), title="Proposed · unapproved fixes",
                 border_style="border", title_align="left")


def _summary_body(app: AppState) -> Group:
    done = sum(1 for f in app.files if f.status == "done")
    table = Table(expand=True, border_style="border.dim", header_style="app.title")
    table.add_column("#", justify="right", width=3)
    table.add_column("File", overflow="fold")
    table.add_column("Status", width=9)
    table.add_column("alive", justify="right", width=6, style="stat.alive")
    table.add_column("killed", justify="right", width=6, style="stat.killed")
    table.add_column("total", justify="right", width=6, style="stat.total")
    table.add_column("weak", justify="right", width=5)
    table.add_column("probs", justify="right", width=5)
    table.add_column("fixes", justify="right", width=5, style="fix.applied")
    for i, f in enumerate(app.files):
        marker = "▶ " if i == app.current else "  "
        table.add_row(
            str(i + 1),
            marker + f.path.name,
            Text(f.status, style=f"status.{f.status}"),
            str(f.alive), str(f.killed), str(f.total), str(f.weak),
            str(len(f.problems.get("clusters", {}))),
            str(f.fixes_applied),
        )
    parts: list = [Text(f"Progress: {done}/{len(app.files)} files complete",
                        style="app.title"), table]
    fs = app.cur()
    if fs and fs.problems.get("clusters"):
        parts += [Text(""), Text(f"Problems in {fs.path.name}:", style="app.title"),
                  _problems_renderable(fs.problems)]
    return Group(*parts)


def render_summary(app: AppState) -> Panel:
    return Panel(_summary_body(app), title="Summary · problems & progress",
                 border_style="border", title_align="left")


SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _analysis_indicator(app: AppState):
    """Animated line shown while a Gemini call is in flight (queue + elapsed)."""
    with app.lock:
        inflight = app.analysis_inflight
        queued = app.analysis_queued
        started = app.analysis_started
        pending = len(app.pending_names)
    if inflight <= 0:
        return None
    frame = SPINNER_FRAMES[int(time.monotonic() * 10) % len(SPINNER_FRAMES)]
    elapsed = (time.monotonic() - started) if started else 0.0
    waiting = queued + pending
    line = Text()
    line.append(f" {frame} ", style="status.running")
    line.append("Gemini analysing ", style="status.running")
    line.append(str(inflight), style="stat.total")
    line.append(" example(s)", style="app.dim")
    if waiting:
        line.append("  ·  ", style="app.dim")
        line.append(f"{waiting}", style="stat.timeout")
        line.append(" queued", style="app.dim")
    line.append("  ·  ", style="app.dim")
    line.append(f"{elapsed:.1f}s", style="stat.timeout")
    return line


def _debug_body(app: AppState) -> Group:
    es = debug_log.entries()
    with app.lock:
        sel = app.debug_sel
        expanded = app.debug_expanded
    n = len(es)
    head = Text(
        f"debug log · {n} entr{'y' if n == 1 else 'ies'} · "
        "j/k navigate · Enter expand/collapse",
        style="app.dim",
    )
    if not es:
        return Group(head, Text(""), Text("(no debug output yet)", style="grey74"))
    sel = max(0, min(sel, n - 1))
    rows: list = [head, Text("")]
    for i, e in enumerate(es):
        is_sel = (i == sel)
        is_exp = e["id"] in expanded
        tag_line = f"[{e['ts']}] {e['tag']}"
        if not is_exp:
            tag_line = tag_line[:120]
        style = "row.sel" if is_sel else "grey74"
        rows.append(Text(tag_line, style=style))
        if is_exp and e["body"]:
            for bl in e["body"].splitlines():
                rows.append(Text("    " + bl, style="grey58"))
    return Group(*rows)


def render_debug(app: AppState) -> Panel:
    return Panel(_debug_body(app), title="Debug · log + Gemini traffic",
                 border_style="border", title_align="left")


def _refine_indicator(app: AppState):
    with app.lock:
        status = app.refine_status
    if not status:
        return None
    frame = SPINNER_FRAMES[int(time.monotonic() * 10) % len(SPINNER_FRAMES)]
    line = Text()
    line.append(f" {frame} ", style="status.running")
    line.append("refining spec  ·  ", style="status.running")
    line.append(status, style="problem.title")
    return line


def _mode_switch_indicator(app: AppState):
    """Shows the 5-second countdown after switching away from mutant-first mode."""
    with app.lock:
        ts = app.mode_switch_ts
    if ts == 0.0:
        return None
    remaining = 5.0 - (time.monotonic() - ts)
    if remaining <= 0:
        return None
    line = Text()
    line.append(" ⏱ ", style="stat.timeout")
    line.append(f"analysis in {remaining:.1f}s  ", style="stat.timeout")
    line.append("Shift+Tab to cancel", style="app.dim")
    return line


_PANEL_CONFIGS = [
    ("MutDafny · run.sh",                   _mutdafny_body),
    ("Watch · mutant analysis + fix history", _watch_body),
    ("Proposed · unapproved fixes",           _proposed_body),
    ("Summary · problems & progress",         _summary_body),
    ("Debug · log + Gemini traffic",          _debug_body),
]


class _WindowedLines:
    """Renders a pre-sliced list of segment-lines (a scroll window) plus an optional
    status line — the less-style viewport over a tall frame."""

    def __init__(self, lines, footer: str | None = None, footer_style: Style | None = None):
        self._lines = lines
        self._footer = footer
        self._footer_style = footer_style

    def __rich_console__(self, console, options):
        for ln in self._lines:
            yield from ln
            yield Segment("\n")
        if self._footer is not None:
            yield Segment(self._footer, self._footer_style)
            yield Segment("\n")


def _paginate(console: Console, renderable, app: AppState):
    """Legacy whole-frame paginator — kept for render() compatibility but no longer
    called from the main loop (replaced by _render_paginated)."""
    opts = console.options.update(height=None)
    lines = console.render_lines(renderable, opts, pad=True)
    total = len(lines)
    viewport = max(1, console.size.height - 5)
    if total <= viewport:
        with app.lock:
            app.scroll = 0
        return renderable
    content_rows = viewport - 1
    max_scroll = max(0, total - content_rows)
    with app.lock:
        app.scroll = max(0, min(app.scroll, max_scroll))
        s = app.scroll
    window = lines[s:s + content_rows]
    at_top = "TOP" if s == 0 else " ↑ "
    at_bot = "END" if s >= max_scroll else " ↓ "
    footer = f" {at_top} lines {s + 1}-{s + len(window)}/{total} {at_bot}  ↑/↓ scroll "
    return _WindowedLines(window, footer, Style(color="black", bgcolor="bright_cyan"))


def _render_paginated(console: Console, app: AppState) -> Group:
    """Full TUI frame with panel-interior-only paging.

    The title, tab bar, indicators, and footer are fixed. Only the panel body is
    windowed; the line counter appears in the panel subtitle (inside the border).
    Colors shift to magenta when analysis_mode == 'mutant_first'."""
    title = Text("MutDafny Launcher", style=_title_key(app))
    if app.human_mode:
        title.append("  (human)", style="status.running")
    bar = _tab_bar(app)

    with app.lock:
        active_tab = app.active_tab
        watch_view = app.watch_view
    panel_title, body_fn = _PANEL_CONFIGS[active_tab]
    body = body_fn(app)
    border = _border_key(app)

    # Render body at panel interior width (console width minus 2 border chars)
    interior_w = max(10, console.size.width - 2)
    opts = console.options.update(width=interior_w, height=None)
    body_lines = console.render_lines(body, opts, pad=True)
    total = len(body_lines)

    # viewport = terminal height - title(1) - tabbar(1) - panel_border×2(2) - footer(1)
    # subtract 2 more to absorb optional indicator lines without flicker
    viewport = max(4, console.size.height - 7)
    with app.lock:
        app.viewport = viewport

    if total <= viewport:
        with app.lock:
            app.scroll = 0
        panel = Panel(body, title=panel_title, border_style=border, title_align="left")
    else:
        max_scroll = max(0, total - viewport)
        with app.lock:
            app.scroll = max(0, min(app.scroll, max_scroll))
            s = app.scroll
        window = body_lines[s:s + viewport]
        at_top = "TOP" if s == 0 else "↑"
        at_bot = "END" if s >= max_scroll else "↓"
        sub = f" {at_top} {s + 1}–{s + len(window)}/{total} {at_bot}  ↑↓ scroll  ⇧↑↓ page "
        panel = Panel(
            _WindowedLines(window),
            title=panel_title,
            subtitle=sub,
            subtitle_align="right",
            border_style=border,
            title_align="left",
        )

    # Footer — hints vary by tab / sub-view
    foot = Text()
    foot.append(" 1-5 ", style="app.key"); foot.append(" tab  ", style="app.dim")
    foot.append(" Tab ", style="app.key"); foot.append(" cycle  ", style="app.dim")
    if active_tab == 1:   # Watch
        if watch_view == "problems":
            foot.append(" j/k ", style="app.key"); foot.append(" select  ", style="app.dim")
            foot.append(" Enter ", style="app.key"); foot.append(" diffs  ", style="app.dim")
        else:
            foot.append(" j/k ", style="app.key"); foot.append(" select mutant  ", style="app.dim")
            foot.append(" Esc ", style="app.key"); foot.append(" back  ", style="app.dim")
    elif active_tab == 2:  # Proposed
        foot.append(" j/k ", style="app.key"); foot.append(" select  ", style="app.dim")
        foot.append(" a ", style="app.key"); foot.append(" apply  ", style="app.dim")
        foot.append(" d ", style="app.key"); foot.append(" dismiss  ", style="app.dim")
    foot.append(" ⇧↑/↓ ", style="app.key"); foot.append(" page  ", style="app.dim")
    foot.append(" Alt+M ", style="app.key"); foot.append(" mode  ", style="app.dim")
    foot.append(" q ", style="app.key"); foot.append(" quit", style="app.dim")
    if app.finished:
        foot.append("    ALL FILES COMPLETE — press q", style="status.done")

    parts: list = [title, bar, panel]
    indicator = _analysis_indicator(app)
    if indicator is not None:
        parts.append(indicator)
    refine = _refine_indicator(app)
    if refine is not None:
        parts.append(refine)
    mode_sw = _mode_switch_indicator(app)
    if mode_sw is not None:
        parts.append(mode_sw)
    parts.append(foot)
    return Group(*parts)


def render(app: AppState) -> Group:
    """Non-paginated render — used only for the initial Live frame before _render_paginated
    takes over in the update loop."""
    title = Text("MutDafny Launcher", style="app.title")
    if app.human_mode:
        title.append("  (human)", style="status.running")
    bar = _tab_bar(app)
    _tab_panels = [render_mutdafny, render_watch, render_proposed, render_summary, render_debug]
    panel = _tab_panels[app.active_tab](app)
    foot = Text()
    foot.append(" 1-5 ", style="app.key"); foot.append(" tab  ", style="app.dim")
    foot.append(" Tab ", style="app.key"); foot.append(" cycle  ", style="app.dim")
    foot.append(" q ", style="app.key"); foot.append(" quit", style="app.dim")
    if app.finished:
        foot.append("    ALL FILES COMPLETE — press q", style="status.done")
    parts = [title, bar, panel]
    indicator = _analysis_indicator(app)
    if indicator is not None:
        parts.append(indicator)
    refine = _refine_indicator(app)
    if refine is not None:
        parts.append(refine)
    parts.append(foot)
    return Group(*parts)


# ---------------------------------------------------------------------------
# Input expansion
# ---------------------------------------------------------------------------

def expand_inputs(args: list) -> list:
    files: list = []
    seen: set = set()
    for a in args:
        p = Path(a)
        if p.is_dir():
            candidates = sorted(p.rglob("*.dfy"))
        elif p.is_file() and p.suffix == ".dfy":
            candidates = [p]
        else:
            print(f"[warn] skipping {a} (not a .dfy file or directory)", file=sys.stderr)
            continue
        for c in candidates:
            key = str(c.resolve())
            if key not in seen:
                seen.add(key)
                files.append(FileState(path=c, stem=c.stem))
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    argv = sys.argv[1:]
    human_mode = "--human" in argv
    mutant_only = "--mutant-only" in argv
    allow_overconstrain = "--allow-overconstrain" in argv
    max_files = 0
    flags_to_strip = {"--human", "--mutant-only", "--allow-overconstrain"}
    remaining = []
    for a in argv:
        if a in flags_to_strip:
            continue
        if a.startswith("-N") and a[2:].isdigit():
            max_files = int(a[2:])
        else:
            remaining.append(a)
    args = remaining
    if not args:
        sys.exit(
            "usage: launcher.py [--human] [--mutant-only] [--allow-overconstrain] "
            "[-N<max>] <file_or_dir> [...]"
        )
    if not RUN_SH.exists():
        sys.exit(f"[error] run.sh not found at {RUN_SH}")

    files = expand_inputs(args)
    if not files:
        sys.exit("[error] no .dfy inputs found")

    MUT_DIR.mkdir(parents=True, exist_ok=True)
    GTS_DIR.mkdir(parents=True, exist_ok=True)

    app = AppState(files=files, human_mode=human_mode, mutant_only=mutant_only,
                   max_files=max_files, allow_overconstrain=allow_overconstrain)
    pipeline = Pipeline(app)

    worker = threading.Thread(target=pipeline.run, daemon=True)
    keys = threading.Thread(target=keyboard_thread, args=(app,), daemon=True)
    worker.start()
    keys.start()

    console = Console(theme=THEME)
    try:
        with Live(render(app), console=console, screen=True,
                  refresh_per_second=REFRESH_HZ, auto_refresh=False) as live:
            while not app.quit:
                live.update(_render_paginated(console, app))
                live.refresh()
                if app.finished and app.quit:
                    break
                time.sleep(1 / REFRESH_HZ)
                if app.finished and not worker.is_alive():
                    # keep rendering until the user presses q
                    if app.quit:
                        break
    except KeyboardInterrupt:
        app.quit = True

    app.quit = True
    # Best-effort: ensure no stray run.sh children survive
    clean_stray_files()
    console.print("[app.dim]launcher exited.[/]")


if __name__ == "__main__":
    main()
