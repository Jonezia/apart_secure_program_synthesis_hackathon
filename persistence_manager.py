#!/usr/bin/env python3
"""
persistence_manager.py — single source of truth for MutDafny's on-disk cache.

Layout (all under mutdafny/mutants/):

    mutants/
      alive/<stem>/         <stem>__<pos>_<op>[_<arg>].dfy   (mutant files)
                  diffs/    <mutant_stem>.diff
                  analysis/ <stem>_analysis.jsonl
                  problems.json
                  equiv-checks/  <mutant>.equiv.txt
      equivalent/<stem>/    <mutant>.dfy + <mutant>.equiv.dfy/.equiv.txt
                  diffs/
      killed/<stem>/        <mutant>.dfy
      timed-out/<stem>/     <mutant>.dfy
      invalid/<stem>/       <mutant>.dfy
      gts/                  (ground truths, shared)
      <stem>.manifest.json  (per-file manifest, in the mutants root)
      .state/<stem>/        (transient run bookkeeping, e.g. consumed.log)

Warm start is two-sided and both sides resume from the cache:
  * Mutation (run.sh): the manager scans once and stores the full target list in
    the manifest. Before each run it writes a targets.csv of only the REMAINING
    targets (full - consumed) and tells run.sh to skip its own scan
    (MUT_PRELOADED=1). run.sh appends each processed target to MUT_CONSUMED_LOG;
    finalize_run merges that back into the manifest, so the next run continues
    exactly where the last one stopped — even for targets that produced no mutant.
  * Analysis (watcher): diffs/analysis live beside the alive mutants. The set of
    already-diffed / already-analyzed mutants is read straight off disk, so the
    watcher only processes what the analysis cache hasn't covered yet.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

CATEGORIES = ["alive", "killed", "timed-out", "invalid", "equivalent"]

# Matches Dafny declaration lines to locate the enclosing function/method/etc.
_DECL_RE = re.compile(
    r"^\s*(?:(?:ghost|static|twostate|greatest|least)\s+)*"
    r"(?:method|function|lemma|predicate|constructor|copredicate|colemma)"
    r"\s+([A-Za-z_][A-Za-z0-9_']*)"
)
DIFF_CATEGORIES = ["alive", "equivalent"]   # categories that keep a diffs/ folder
ANALYSIS_CATEGORY = "alive"                  # only alive keeps an analysis/ folder
FIXES_SUBDIR = "fixes"                        # alive/<stem>/fixes/ — independent purge class


def _target_key(pos: str, op: str, arg: str) -> str:
    return f"{pos},{op},{arg}"


def stem_of(source) -> str:
    return Path(source).stem


# ---------------------------------------------------------------------------
# Spec-clause insertion + offset remapping (spec-only fixes shift code offsets)
# ---------------------------------------------------------------------------

def apply_clause_to_text(text: str, anchor_line: str, clause: str) -> tuple[str, bool]:
    """Insert `clause` on its own line immediately after the first line equal to
    `anchor_line` (compared stripped). Returns (new_text, applied?)."""
    anchor = anchor_line.strip()
    if not anchor:
        return text, False
    lines = text.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if ln.strip() == anchor:
            indent = ln[: len(ln) - len(ln.lstrip())]
            if not lines[i].endswith("\n"):
                lines[i] = lines[i] + "\n"
            lines.insert(i + 1, f"{indent}{clause.strip()}\n")
            return "".join(lines), True
    return text, False


def _offset_map(src: str, dst: str):
    """Map a character offset in `src` to the corresponding offset in `dst`,
    assuming `dst` is `src` with insertions (the spec-only fix case)."""
    blocks = difflib.SequenceMatcher(a=src, b=dst, autojunk=False).get_matching_blocks()

    def remap(p: int) -> int:
        best = 0
        for i, j, n in blocks:
            if p < i:
                break
            if i <= p < i + n:
                return j + (p - i)
            best = j + n
        return best

    return remap


def _remap_pos(pos: str, remap) -> str:
    if pos in ("", "-"):
        return pos
    try:
        if "-" in pos:
            a, b = pos.split("-", 1)
            return f"{remap(int(a))}-{remap(int(b))}"
        return str(remap(int(pos)))
    except ValueError:
        return pos


def _remap_key(key: str, remap) -> str:
    parts = key.split(",")
    parts[0] = _remap_pos(parts[0], remap)
    return ",".join(parts)


def _locate_anchor(text: str, anchor_line: str) -> tuple[str, int]:
    """Return (enclosing_function_name, 1-based line_number) for `anchor_line` in `text`.
    Scans upward from the matching line to find the nearest Dafny declaration."""
    anchor = anchor_line.strip()
    if not anchor:
        return ("", 0)
    lines = text.splitlines()
    anchor_idx = None
    for i, ln in enumerate(lines):
        if ln.strip() == anchor:
            anchor_idx = i
            break
    if anchor_idx is None:
        return ("", 0)
    line_no = anchor_idx + 1
    for i in range(anchor_idx, -1, -1):
        m = _DECL_RE.match(lines[i])
        if m:
            return (m.group(1), line_no)
    return ("", line_no)


class PersistenceManager:
    """Owns the cache layout, manifests, warm-start state, and purging."""

    def __init__(self, mutdafny_dir: Path):
        self.mutdafny = Path(mutdafny_dir).resolve()
        self.mutants = self.mutdafny / "mutants"
        self.gts = self.mutants / "gts"
        # Absolute, so they resolve correctly when subprocesses run with cwd=mutdafny.
        self.dafny = self.mutdafny / "dafny" / "Binaries" / "Dafny.dll"
        self.z3 = self.mutdafny / "dafny" / "Binaries" / "z3"
        self.plugin = self.mutdafny / "mutdafny" / "bin" / "Debug" / "net8.0" / "mutdafny.dll"

    # ------------------------------------------------------------------ paths
    def cat_dir(self, category: str, stem: str) -> Path:
        return self.mutants / category / stem

    def diffs_dir(self, category: str, stem: str) -> Path:
        return self.cat_dir(category, stem) / "diffs"

    def analysis_dir(self, stem: str) -> Path:
        return self.cat_dir(ANALYSIS_CATEGORY, stem) / "analysis"

    def problems_path(self, stem: str) -> Path:
        return self.cat_dir(ANALYSIS_CATEGORY, stem) / "problems.json"

    def manifest_path(self, stem: str) -> Path:
        return self.mutants / f"{stem}.manifest.json"

    def _state_dir(self, stem: str) -> Path:
        return self.mutants / ".state" / stem

    def consumed_log(self, stem: str) -> Path:
        return self._state_dir(stem) / "consumed.log"

    def active_targets_file(self) -> Path:
        # run.sh reads "targets.csv" from its cwd (the mutdafny dir).
        return self.mutdafny / "targets.csv"

    def keymap_path(self, stem: str) -> Path:
        # Transient current-coords -> original-coords target key map for a fix run.
        return self._state_dir(stem) / "keymap.json"

    # ----------------------------------------------------------- fix cache
    def fixes_dir(self, stem: str) -> Path:
        return self.cat_dir(ANALYSIS_CATEGORY, stem) / FIXES_SUBDIR

    def fix_stack_path(self, stem: str) -> Path:
        return self.fixes_dir(stem) / "stack.json"

    def current_program_path(self, stem: str) -> Path:
        return self.fixes_dir(stem) / "current.dfy"

    # --------------------------------------------------------------- manifest
    def _new_manifest(self, stem: str, source) -> dict:
        return {
            "stem": stem,
            "source": str(Path(source).resolve()) if source else "",
            "scanned": False,
            "targets": [],       # full ordered list of "pos,op,arg"
            "consumed": [],      # subset already attempted (any order)
            "phase": "-",
            "thresholds": {},
            "updated": "",
        }

    def load_manifest(self, stem: str, source=None) -> dict:
        p = self.manifest_path(stem)
        if p.exists():
            try:
                m = json.loads(p.read_text())
                if source and not m.get("source"):
                    m["source"] = str(Path(source).resolve())
                return m
            except (json.JSONDecodeError, OSError):
                pass
        return self._new_manifest(stem, source)

    def save_manifest(self, manifest: dict) -> None:
        manifest["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        p = self.manifest_path(manifest["stem"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(manifest, indent=2))

    def all_stems(self) -> list[str]:
        """Every file with any cached footprint (manifest or mutant subdir)."""
        stems: set[str] = set()
        if self.mutants.exists():
            for mf in self.mutants.glob("*.manifest.json"):
                stems.add(mf.name[: -len(".manifest.json")])
            for cat in CATEGORIES:
                d = self.mutants / cat
                if d.exists():
                    for e in os.scandir(d):
                        if e.is_dir():
                            stems.add(e.name)
        return sorted(stems)

    # ------------------------------------------------------------------ scan
    def _parse_targets_csv(self, path: Path) -> list[str]:
        out: list[str] = []
        if not path.exists():
            return out
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            pos, op = parts[0], parts[1]
            arg = parts[2] if len(parts) > 2 else ""
            out.append(_target_key(pos, op, arg))
        return out

    def _run_scan(self, source) -> list[str]:
        """Resolve the program with the scan plugin to enumerate mutation targets.
        `resolve` (not `verify`) runs the Pre/PostResolve scanners without Z3."""
        active = self.active_targets_file()
        try:
            active.unlink()   # the scanner appends to any existing targets.csv
        except OSError:
            pass
        cmd = [
            "dotnet", str(self.dafny), "resolve", str(Path(source).resolve()),
            "--allow-warnings",
            "--plugin", f"{self.plugin},scan",
        ]
        try:
            subprocess.run(
                cmd, cwd=str(self.mutdafny),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=180,
            )
        except (subprocess.TimeoutExpired, OSError):
            return []
        targets = self._parse_targets_csv(active)
        try:
            active.unlink()
        except OSError:
            pass
        return targets

    def ensure_scanned(self, source) -> dict:
        """Scan once (cached in the manifest). Returns the manifest."""
        stem = stem_of(source)
        m = self.load_manifest(stem, source)
        if not m.get("scanned"):
            targets = self._run_scan(source)
            m["targets"] = targets
            m["scanned"] = True
            self.save_manifest(m)
        return m

    # ----------------------------------------------------------- warm start
    def _consumed_log_keys(self, stem: str) -> set:
        """Targets recorded by run.sh but not yet folded into the manifest. This is
        the source of truth between checkpoints, so warm-start survives an
        interrupted run that never reached finalize_run()."""
        return set(self._parse_targets_csv(self.consumed_log(stem)))

    def consumed_set(self, stem: str) -> set:
        m = self.load_manifest(stem)
        return set(m.get("consumed", [])) | self._consumed_log_keys(stem)

    def remaining_targets(self, stem: str) -> list[str]:
        m = self.load_manifest(stem)
        consumed = self.consumed_set(stem)
        return [t for t in m.get("targets", []) if t not in consumed]

    def prepare_run(self, source, multi: bool = False) -> dict:
        """Materialise the remaining targets into the active targets.csv and return
        the environment overrides run.sh needs. Call finalize_run() afterwards.

        The consumed log is append-only and persistent (NOT truncated here): it is
        the authoritative record of progress until finalize_run folds it into the
        manifest, so an interrupted run still resumes correctly next time."""
        stem = stem_of(source)
        self.ensure_scanned(source)
        remaining = self.remaining_targets(stem)   # canonical (original) coordinates

        # When spec fixes are applied, run.sh mutates current.dfy, whose code offsets are
        # shifted by the inserted clauses. Remap the remaining target positions into
        # current.dfy coordinates and record the exact current->original key map so
        # finalize_run can fold run.sh's (current-coords) consumed log back precisely.
        km = self.keymap_path(stem)
        try:
            km.unlink()
        except OSError:
            pass
        cur_path = self.current_program_path(stem)
        if self.has_applied_fixes(stem) and cur_path.exists():
            remap = _offset_map(Path(source).read_text(), cur_path.read_text())
            keymap = {}
            mapped = []
            for t in remaining:
                ck = _remap_key(t, remap)
                keymap[ck] = t
                mapped.append(ck)
            remaining = mapped
            km.parent.mkdir(parents=True, exist_ok=True)
            km.write_text(json.dumps(keymap))

        self.active_targets_file().write_text("".join(t + "\n" for t in remaining))

        env = {"MUT_PRELOADED": "1"}
        if not multi:
            self.consumed_log(stem).parent.mkdir(parents=True, exist_ok=True)
            env["MUT_CONSUMED_LOG"] = str(self.consumed_log(stem))
        return env

    def finalize_run(self, source, multi: bool = False) -> None:
        """Checkpoint: fold this run's consumption into the manifest and clear the
        log. Single mode reads run.sh's consumed log; multi mode derives consumption
        from the leftover (plugin-shrunk) targets.csv. Both union with whatever was
        already consumed, so nothing is lost."""
        stem = stem_of(source)
        m = self.load_manifest(stem, source)

        # If this was a fix run, run.sh logged consumed keys in current.dfy coordinates;
        # map them back to canonical (original) keys via the keymap written by prepare_run.
        keymap = {}
        km = self.keymap_path(stem)
        if km.exists():
            try:
                keymap = json.loads(km.read_text())
            except (json.JSONDecodeError, OSError):
                keymap = {}

        def to_orig(k: str) -> str:
            return keymap.get(k, k)

        consumed = set(m.get("consumed", []))
        consumed |= {to_orig(k) for k in self._consumed_log_keys(stem)}

        if multi:
            leftover = {to_orig(k) for k in self._parse_targets_csv(self.active_targets_file())}
            consumed |= {t for t in m.get("targets", []) if t not in leftover}

        m["consumed"] = [t for t in m.get("targets", []) if t in consumed]
        m["phase"] = "multi" if multi else "single"
        self.save_manifest(m)
        for p in (self.consumed_log(stem), self.active_targets_file(), km):
            try:
                p.unlink()
            except OSError:
                pass

    # ------------------------------------------------------- mutant counting
    def list_mutants(self, category: str, stem: str) -> list[str]:
        d = self.cat_dir(category, stem)
        if not d.exists():
            return []
        return sorted(
            e.name for e in os.scandir(d)
            if e.is_file() and e.name.endswith(".dfy") and not e.name.endswith(".equiv.dfy")
        )

    def counts(self, stem: str) -> dict:
        c = {cat: len(self.list_mutants(cat, stem)) for cat in CATEGORIES}
        c["total"] = sum(c[cat] for cat in CATEGORIES)
        return c

    # --------------------------------------------------- analysis warm start
    def diffed_mutants(self, stem: str) -> set:
        """Alive mutants that already have a diff (mutant filename, with .dfy)."""
        d = self.diffs_dir("alive", stem)
        if not d.exists():
            return set()
        return {
            e.name[: -len(".diff")] + ".dfy"
            for e in os.scandir(d)
            if e.is_file() and e.name.endswith(".diff")
        }

    def analyzed_mutants(self, stem: str) -> set:
        """Mutants already classified by the LLM (from the analysis jsonl)."""
        d = self.analysis_dir(stem)
        names: set = set()
        if not d.exists():
            return names
        for e in os.scandir(d):
            if not (e.is_file() and e.name.endswith(".jsonl")):
                continue
            try:
                for line in Path(e.path).read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if "mutant" in rec:
                        names.add(rec["mutant"])
            except (json.JSONDecodeError, OSError):
                continue
        return names

    def pending_alive(self, stem: str) -> list[str]:
        """Alive mutants not yet diffed."""
        done = self.diffed_mutants(stem)
        return [m for m in self.list_mutants("alive", stem) if m not in done]

    # --------------------------------------------------------------- problems
    def load_problems(self, stem: str) -> dict:
        p = self.problems_path(stem)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "clusters": {}, "total_processed": 0, "total_weak_spec": 0,
            "total_equivalent": 0, "total_contextual": 0, "contextual": [],
            "unverified_methods": {}, "total_unverified": 0,
            "next_cluster_id": 1,
        }

    def save_problems(self, stem: str, cluster_state: dict) -> None:
        p = self.problems_path(stem)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cluster_state, indent=2))

    # ------------------------------------------------------- fix stack ops
    def load_fix_stack(self, stem: str) -> dict:
        p = self.fix_stack_path(stem)
        if p.exists():
            try:
                s = json.loads(p.read_text())
                s.setdefault("entries", [])
                s.setdefault("next_id", len(s["entries"]) + 1)
                return s
            except (json.JSONDecodeError, OSError):
                pass
        return {"entries": [], "next_id": 1}

    def save_fix_stack(self, stem: str, stack: dict) -> None:
        p = self.fix_stack_path(stem)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(stack, indent=2))

    def applied_fixes(self, stem: str) -> list:
        return [e for e in self.load_fix_stack(stem)["entries"] if e.get("status") == "applied"]

    def has_applied_fixes(self, stem: str) -> bool:
        return any(e.get("status") == "applied" for e in self.load_fix_stack(stem)["entries"])

    def add_fix(self, stem: str, problem_id: str, snippet: dict, attempt: int = 1,
                status: str = "applied", original=None) -> str:
        """Append a fix entry. status: applied | unapplied (proposed) | reverted.
        Rebuilds current.dfy when applied and records function/line metadata."""
        stack = self.load_fix_stack(stem)
        eid = stack.get("next_id", 1)
        stack["next_id"] = eid + 1
        entry = {
            "id": f"F{eid:03d}",
            "problem_id": problem_id,
            "attempt": attempt,
            "edits": 0,
            "status": status,
            "anchor_line": snippet.get("anchor_line", ""),
            "kind": snippet.get("kind", ""),
            "clause": snippet.get("clause", ""),
            "description": snippet.get("description", ""),
            "function": "",
            "line": 0,
            "killed": [],
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        stack["entries"].append(entry)
        self.save_fix_stack(stem, stack)
        if status == "applied" and original is not None:
            self.rebuild_current_program(stem, original)
            cur = self.current_program_path(stem)
            loc_text = cur.read_text() if cur.exists() else Path(original).read_text()
            fn, ln = _locate_anchor(loc_text, entry["anchor_line"])
            entry["function"] = fn
            entry["line"] = ln
            self.save_fix_stack(stem, stack)
        return entry["id"]

    def set_fix_status(self, stem: str, entry_id: str, status: str, original=None) -> None:
        stack = self.load_fix_stack(stem)
        for e in stack["entries"]:
            if e["id"] == entry_id:
                e["status"] = status
        self.save_fix_stack(stem, stack)
        if original is not None:
            self.rebuild_current_program(stem, original)

    def unapply_last(self, stem: str, original) -> str | None:
        """Undo the most-recently-applied fix (stack-style). Returns its id or None."""
        stack = self.load_fix_stack(stem)
        for e in reversed(stack["entries"]):
            if e.get("status") == "applied":
                e["status"] = "unapplied"
                self.save_fix_stack(stem, stack)
                self.rebuild_current_program(stem, original)
                return e["id"]
        return None

    def edit_fix(self, stem: str, entry_id: str, snippet: dict, original) -> None:
        """Destructively overwrite a fix entry's clause. Bumps the edits counter,
        rebuilds current.dfy, and refreshes function/line metadata."""
        stack = self.load_fix_stack(stem)
        for e in stack["entries"]:
            if e["id"] == entry_id:
                e["anchor_line"] = snippet.get("anchor_line", e["anchor_line"])
                e["kind"] = snippet.get("kind", e["kind"])
                e["clause"] = snippet.get("clause", e["clause"])
                e["description"] = snippet.get("description", e["description"])
                e["edits"] = e.get("edits", 0) + 1
                self.save_fix_stack(stem, stack)
                if e.get("status") == "applied":
                    self.rebuild_current_program(stem, original)
                    cur = self.current_program_path(stem)
                    loc_text = cur.read_text() if cur.exists() else Path(original).read_text()
                    fn, ln = _locate_anchor(loc_text, e["anchor_line"])
                    e["function"] = fn
                    e["line"] = ln
                    self.save_fix_stack(stem, stack)
                return

    def revert_problem(self, stem: str, problem_id: str, original) -> None:
        """Mark all applied fixes for a problem as reverted (kept in history)."""
        stack = self.load_fix_stack(stem)
        for e in stack["entries"]:
            if e.get("problem_id") == problem_id and e.get("status") == "applied":
                e["status"] = "reverted"
        self.save_fix_stack(stem, stack)
        self.rebuild_current_program(stem, original)

    def record_fix_kills(self, stem: str, entry_id: str, killed_names) -> None:
        stack = self.load_fix_stack(stem)
        for e in stack["entries"]:
            if e["id"] == entry_id:
                e["killed"] = sorted(set(e.get("killed", [])) | set(killed_names))
        self.save_fix_stack(stem, stack)

    def rebuild_current_program(self, stem: str, original) -> None:
        """Replay the applied fix clauses onto the original source -> current.dfy.
        Removes current.dfy when no fix is applied."""
        cur = self.current_program_path(stem)
        applied = self.applied_fixes(stem)
        if not applied:
            try:
                cur.unlink()
            except OSError:
                pass
            return
        text = Path(original).read_text()
        for e in applied:
            text, _ok = apply_clause_to_text(text, e["anchor_line"], e["clause"])
        cur.parent.mkdir(parents=True, exist_ok=True)
        cur.write_text(text)

    def active_source(self, stem: str, original) -> Path:
        """The program run.sh should mutate: current.dfy if fixes are applied, else
        the original source."""
        cur = self.current_program_path(stem)
        if self.has_applied_fixes(stem) and cur.exists():
            return cur
        return Path(original)

    def move_mutant(self, stem: str, name: str, from_cat: str, to_cat: str) -> bool:
        src = self.cat_dir(from_cat, stem) / name
        if not src.exists():
            return False
        dst_dir = self.cat_dir(to_cat, stem)
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst_dir / name))
        return True

    # ----------------------------------------------------------------- purge
    def purge_mutants(self, stem: str) -> None:
        """Delete cached mutants AND their analysis (diffs/analysis/problems) and reset
        the mutation warm-start. The analysis is derived from the mutants, so it cannot
        outlive them. The fix cache (alive/<stem>/fixes) is independent and preserved."""
        for cat in CATEGORIES:
            d = self.cat_dir(cat, stem)
            if not d.exists():
                continue
            for e in os.scandir(d):
                if e.is_dir() and e.name == FIXES_SUBDIR:  # only alive has it
                    continue
                if e.is_dir():
                    shutil.rmtree(e.path, ignore_errors=True)
                else:
                    try:
                        os.unlink(e.path)
                    except OSError:
                        pass
            # If only the (preserved) fix cache remains, keep the dir; else drop if empty.
            if not any(os.scandir(d)):
                shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(self._state_dir(stem), ignore_errors=True)
        m = self.load_manifest(stem)
        m["consumed"] = []
        m["phase"] = "-"
        self.save_manifest(m)

    def purge_analysis(self, stem: str) -> None:
        """Delete diffs/analysis/problems but keep the mutant cache and the fix cache."""
        for cat in DIFF_CATEGORIES:
            shutil.rmtree(self.diffs_dir(cat, stem), ignore_errors=True)
        shutil.rmtree(self.analysis_dir(stem), ignore_errors=True)
        try:
            self.problems_path(stem).unlink()
        except OSError:
            pass

    def purge_fixes(self, stem: str) -> None:
        """Delete the fix cache (stack + current.dfy) only — touches nothing else."""
        shutil.rmtree(self.fixes_dir(stem), ignore_errors=True)

    def purge_all(self, stem: str) -> None:
        for cat in CATEGORIES:
            shutil.rmtree(self.cat_dir(cat, stem), ignore_errors=True)
        shutil.rmtree(self._state_dir(stem), ignore_errors=True)
        try:
            self.manifest_path(stem).unlink()
        except OSError:
            pass

    # --------------------------------------------------------------- summary
    @staticmethod
    def _dir_size(path: Path) -> int:
        total = 0
        if not path.exists():
            return 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total

    def _count_files(self, path: Path, suffix: str) -> int:
        if not path.exists():
            return 0
        return sum(1 for e in os.scandir(path) if e.is_file() and e.name.endswith(suffix))

    def summary(self, stem: str) -> dict:
        m = self.load_manifest(stem)
        counts = self.counts(stem)
        targets_total = len(m.get("targets", []))
        target_set = set(m.get("targets", []))
        consumed = len(self.consumed_set(stem) & target_set)
        diff_count = sum(self._count_files(self.diffs_dir(c, stem), ".diff") for c in DIFF_CATEGORIES)
        analyzed = len(self.analyzed_mutants(stem))
        problems = len(self.load_problems(stem).get("clusters", {})) if self.problems_path(stem).exists() else 0
        mutant_bytes = sum(self._dir_size(self.cat_dir(c, stem)) for c in CATEGORIES)
        stack = self.load_fix_stack(stem)
        fixes_total = len(stack.get("entries", []))
        fixes_applied = sum(1 for e in stack.get("entries", []) if e.get("status") == "applied")
        return {
            "stem": stem,
            "source": m.get("source", ""),
            "scanned": m.get("scanned", False),
            "targets_total": targets_total,
            "consumed": consumed,
            "remaining": max(targets_total - consumed, 0),
            "counts": counts,
            "diffs": diff_count,
            "analyzed": analyzed,
            "problems": problems,
            "mutant_bytes": mutant_bytes,
            "fixes": fixes_total,
            "fixes_applied": fixes_applied,
            "phase": m.get("phase", "-"),
            "updated": m.get("updated", ""),
        }

    def summarize_all(self) -> list[dict]:
        return [self.summary(s) for s in self.all_stems()]
