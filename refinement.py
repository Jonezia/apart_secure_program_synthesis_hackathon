#!/usr/bin/env python3
"""
refinement.py — iterative spec-refinement engine for MutDafny.

Once a specification problem is well-evidenced (a cluster of WEAK_SPEC mutants), the
engine asks a stronger Gemini model for a spec-only patch (extra requires/ensures/…),
applies it to a growing `current.dfy`, and re-verifies the WEAK_SPEC mutants against
the strengthened spec — looping until the problem's mutants are all killed or an attempt
cap is hit (then the problem's fixes are reverted and it is marked unresolved).

Re-testing never re-mutates: each WEAK_SPEC mutant file (original code + one mutation)
has the applied spec clauses inserted textually (matching an anchor line), then is
verified with plain Dafny. A mutant that no longer verifies is "killed by the fix".

Spec fixes only touch the spec; mutations only touch code — so the mutation target list
is stable across fixes (positions are remapped by PersistenceManager when run.sh resumes
against current.dfy).
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from persistence_manager import apply_clause_to_text

MAX_FIX_ATTEMPTS = int(os.getenv("LAUNCHER_MAX_FIX_ATTEMPTS", "4"))
FIX_EXAMPLES = int(os.getenv("LAUNCHER_FIX_EXAMPLES", "8"))
MUT_VERIFY_TIME_LIMIT = int(os.getenv("MUT_VERIFY_TIME_LIMIT", "10"))
DAFNY_HARD_TIMEOUT = int(os.getenv("DAFNY_HARD_TIMEOUT", "20"))
N_PER_PROBLEM = int(os.getenv("LAUNCHER_N_PER_PROBLEM", "5"))


def _parse_op(name: str) -> str:
    """Mutation operator from a mutant filename <stem>__<pos>_<op>[_<arg>].dfy."""
    stem = name[: -len(".dfy")] if name.endswith(".dfy") else name
    parts = stem.split("__", 1)
    if len(parts) < 2:
        return ""
    sub = parts[1].split("_", 2)
    return sub[1] if len(sub) > 1 else ""


class RefinementEngine:
    def __init__(self, pm, fixer, analyser, gts_dir: Path, log=None,
                 allow_overconstrain: bool = False):
        self.pm = pm
        self.fixer = fixer            # analyze_mutants.SpecFixer (or None)
        self.analyser = analyser      # analyze_mutants.MutantAnalyser (or None)
        self.gts_dir = Path(gts_dir)
        self._log = log or (lambda _m: None)
        self.allow_overconstrain = allow_overconstrain

    # ------------------------------------------------------------------ utils
    def log(self, msg: str):
        try:
            self._log(msg)
        except Exception:
            pass

    def _gt_path(self, fs) -> Path:
        gt = self.gts_dir / f"{fs.stem}.dfy"
        return gt if gt.exists() else Path(fs.path)

    def _patched_gt_text(self, fs) -> str | None:
        gt = self._gt_path(fs)
        if not gt.exists():
            return None
        text = gt.read_text()
        for e in self.pm.applied_fixes(fs.stem):
            text, _ = apply_clause_to_text(text, e["anchor_line"], e["clause"])
        return text

    def _gt_still_verifies(self, fs) -> bool:
        """Return True if the GT with all currently-applied fixes passes Dafny verification.
        Used to reject overconstrained spec proposals before they pollute the kill list."""
        text = self._patched_gt_text(fs)
        if text is None:
            return True  # can't check — optimistic
        tmp = self.pm.mutdafny / f".gt_check_{fs.stem}.dfy"
        try:
            tmp.write_text(text)
            return not self.verify_killed(tmp)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    def _diff_for(self, stem: str, name: str) -> dict:
        dfile = self.pm.diffs_dir("alive", stem) / (name[: -len(".dfy")] + ".diff")
        return {"name": name, "diff": dfile.read_text() if dfile.exists() else ""}

    def _examples(self, stem: str, names) -> list[dict]:
        """Up to FIX_EXAMPLES diffs, preferring operator diversity."""
        by_op: dict[str, list[str]] = {}
        for n in sorted(names):
            by_op.setdefault(_parse_op(n), []).append(n)
        picked: list[str] = []
        # round-robin across operators for diversity
        while len(picked) < FIX_EXAMPLES and any(by_op.values()):
            for op in list(by_op):
                if not by_op[op]:
                    continue
                picked.append(by_op[op].pop(0))
                if len(picked) >= FIX_EXAMPLES:
                    break
        return [self._diff_for(stem, n) for n in picked]

    # --------------------------------------------------------------- verify
    def verify_killed(self, program_path: Path) -> bool:
        """True iff the program does NOT verify cleanly (errors, or didn't finish).
        Mirrors run.sh's pass test ("finished with … 0 errors")."""
        cmd = [
            "dotnet", str(self.pm.dafny), "verify", str(program_path),
            "--solver-path", str(self.pm.z3), "--allow-warnings",
            "--verification-time-limit", str(MUT_VERIFY_TIME_LIMIT),
        ]
        try:
            res = subprocess.run(
                cmd, cwd=str(self.pm.mutdafny),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=DAFNY_HARD_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return True   # couldn't prove it under budget -> treat as killed
        except OSError:
            return False
        out = res.stdout or ""
        for line in out.splitlines():
            if "Dafny program verifier finished" in line and "0 errors" in line:
                return False   # verified cleanly -> still alive
        return True

    def _patched_mutant_text(self, stem: str, name: str) -> str | None:
        mfile = self.pm.cat_dir("alive", stem) / name
        if not mfile.exists():
            return None
        text = mfile.read_text()
        for e in self.pm.applied_fixes(stem):
            text, _ = apply_clause_to_text(text, e["anchor_line"], e["clause"])
        return text

    def _evaluate(self, fs, cids) -> dict:
        """Re-verify each named cluster's mutants against the applied spec.
        Returns {cid: {"killed": set, "survivors": set}}. Moves nothing."""
        stem = fs.stem
        clusters = fs.problems.get("clusters", {})
        out: dict[str, dict] = {}
        tmp = self.pm.mutdafny / f".retest_{stem}.dfy"
        for cid in cids:
            c = clusters.get(cid)
            if not c:
                continue
            killed, survivors = set(), set()
            for name in list(c.get("mutants", [])):
                text = self._patched_mutant_text(stem, name)
                if text is None:
                    continue
                tmp.write_text(text)
                (killed if self.verify_killed(tmp) else survivors).add(name)
            out[cid] = {"killed": killed, "survivors": survivors}
        try:
            tmp.unlink()
        except OSError:
            pass
        return out

    # --------------------------------------------------------------- refine
    def _apply_proposal(self, fs, pid: str, proposal: dict, attempt: int = 1) -> str | None:
        """Dispatch a proposal's action (add/edit/revert/unresolved) against the fix stack.
        Returns the affected entry id, or None on failure."""
        stem = fs.stem
        action = proposal.get("action", "add")
        target = proposal.get("target")

        if action == "unresolved":
            cluster = fs.problems.get("clusters", {}).get(pid)
            if cluster is not None:
                cluster["unresolved_reason"] = proposal.get("unresolved_reason", "")
            self._mark_unresolved(fs, pid)
            self.log(f"[refine] {pid} flagged unresolvable by fix proposer: "
                     f"{proposal.get('unresolved_reason', '')}")
            return None

        if action == "revert":
            if target:
                stack = self.pm.load_fix_stack(stem)
                valid = next(
                    (e for e in stack["entries"]
                     if e["id"] == target and e.get("status") == "applied"), None
                )
                if valid:
                    self.pm.set_fix_status(stem, target, "unapplied", original=fs.path)
                    return target
            return self.pm.unapply_last(stem, fs.path)

        if action == "edit" and target:
            stack = self.pm.load_fix_stack(stem)
            valid = next(
                (e for e in stack["entries"]
                 if e["id"] == target and e.get("status") == "applied"), None
            )
            if valid and proposal.get("clause"):
                self.pm.edit_fix(stem, target, proposal, fs.path)
                return target
            # Fall through to add

        return self.pm.add_fix(stem, pid, proposal, attempt=attempt,
                               status="applied", original=fs.path)

    def _reassess_and_split(self, fs, pid: str) -> list[str]:
        """Ask the analyser to re-describe or split cluster `pid` after a failed fix.
        Updates cluster_state in-place. Returns resulting cluster ids ([pid] if no split)."""
        if self.analyser is None:
            return [pid]
        stem = fs.stem
        clusters = fs.problems.get("clusters", {})
        cluster = clusters.get(pid)
        if not cluster:
            return [pid]
        diffs = [self._diff_for(stem, n) for n in cluster.get("mutants", [])]
        result = self.analyser.reassess_problem(cluster, self._gt_path(fs), diffs)
        sub_problems = result.get("problems", [])

        if len(sub_problems) <= 1:
            # Re-describe in place
            if sub_problems:
                sp = sub_problems[0]
                if sp.get("title"):
                    cluster["title"] = sp["title"]
                if sp.get("description"):
                    cluster["description"] = sp["description"]
                cluster["open_questions"] = sp.get("open_questions", "")
                if sp.get("mutants"):
                    valid = set(cluster.get("mutants", []))
                    cluster["mutants"] = [m for m in sp["mutants"] if m in valid]
                    cluster["mutant_count"] = len(cluster["mutants"])
            self.pm.save_problems(stem, fs.problems)
            return [pid]

        # Split: replace pid in-place with first sub-problem, add new clusters for rest
        existing = set(cluster.get("mutants", []))
        first = sub_problems[0]
        cluster["title"] = first.get("title") or cluster["title"]
        cluster["description"] = first.get("description") or cluster["description"]
        cluster["open_questions"] = first.get("open_questions", "")
        cluster["mutants"] = [m for m in first.get("mutants", []) if m in existing]
        cluster["mutant_count"] = len(cluster["mutants"])
        result_ids = [pid]

        for sp in sub_problems[1:]:
            new_id = f"C{fs.problems['next_cluster_id']:03d}"
            fs.problems["next_cluster_id"] += 1
            sp_mutants = [m for m in sp.get("mutants", []) if m in existing]
            clusters[new_id] = {
                "id": new_id,
                "title": sp.get("title", ""),
                "description": sp.get("description", ""),
                "open_questions": sp.get("open_questions", ""),
                "mutants": sp_mutants,
                "mutant_count": len(sp_mutants),
            }
            result_ids.append(new_id)

        self.pm.save_problems(stem, fs.problems)
        self.log(f"[refine] {pid} split → {result_ids}")
        return result_ids

    def _revert_lineage(self, fs, root_pid: str, split_ids: list[str]):
        """Revert all applied fixes for the root problem and any split sub-clusters."""
        stem = fs.stem
        self.pm.revert_problem(stem, root_pid, fs.path)
        self._mark_unresolved(fs, root_pid)
        for sid in split_ids:
            self.pm.revert_problem(stem, sid, fs.path)
            self._mark_unresolved(fs, sid)

    def refine_problem(self, fs, problem_id: str) -> bool:
        """Propose → apply → re-test loop for one problem (auto mode).

        After the first failed attempt, reassess+split once (free, no attempt cost).
        Each add/edit/revert counts as one attempt toward MAX_FIX_ATTEMPTS.
        Returns True if resolved, False if reverted/unresolved."""
        if self.fixer is None:
            return False
        stem = fs.stem
        cluster = fs.problems.get("clusters", {}).get(problem_id)
        if not cluster or not cluster.get("mutants"):
            return False

        active_pid = problem_id
        split_ids: list[str] = []
        reassessed = False

        for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
            cluster = fs.problems.get("clusters", {}).get(active_pid)
            if not cluster or not cluster.get("mutants"):
                self._commit(fs, active_pid)
                self.log(f"[refine] {active_pid} resolved (no mutants remaining)")
                return True
            res = self._evaluate(fs, [active_pid]).get(active_pid, {})
            survivors = res.get("survivors", set())
            if not survivors:
                self._commit(fs, active_pid)
                self.log(f"[refine] {active_pid} resolved after {attempt - 1} fix(es)")
                return True

            applied = self.pm.applied_fixes(stem)
            top_fixes = list(reversed(applied[-3:]))   # most recent first, up to 3
            program_text = self.pm.active_source(stem, fs.path).read_text()
            proposal = self.fixer.propose_fix(
                program_text, cluster, self._examples(stem, survivors),
                top_fixes=top_fixes,
            )
            if not proposal:
                self.log(f"[refine] {active_pid}: no proposal at attempt {attempt}")
                break

            # Fixer flagged as unresolvable — stop immediately
            if proposal.get("action") == "unresolved":
                self._apply_proposal(fs, active_pid, proposal, attempt=attempt)
                return False

            eid = self._apply_proposal(fs, active_pid, proposal, attempt=attempt)
            action = proposal.get("action", "add")
            self.log(f"[refine] {active_pid} attempt {attempt}: {action} → {eid}")

            # Reject fix if it overconstrained the GT
            if eid and action in ("add", "edit") and not self.allow_overconstrain:
                if not self._gt_still_verifies(fs):
                    self.log(
                        f"[refine] {active_pid} attempt {attempt}: REJECTED — "
                        f"fix overconstrained GT (patched GT fails Dafny), reverting"
                    )
                    self.pm.set_fix_status(stem, eid, "unapplied", original=fs.path)
                    continue

            # Re-evaluate after applying
            res = self._evaluate(fs, [active_pid]).get(active_pid, {})
            if not res.get("survivors"):
                self._commit(fs, active_pid)
                self.log(f"[refine] {active_pid} resolved after attempt {attempt}")
                return True

            # After the first failure: reassess + optionally split (free, no attempt cost)
            if not reassessed:
                reassessed = True
                result_ids = self._reassess_and_split(fs, active_pid)
                new_sids = [rid for rid in result_ids if rid != active_pid]
                split_ids.extend(new_sids)
                # Pick the biggest sub-cluster with at least one mutant
                best_pid, best_n = None, 0
                for rid in result_ids:
                    c = fs.problems.get("clusters", {}).get(rid)
                    if c:
                        cn = c.get("mutant_count", len(c.get("mutants", [])))
                        if cn > best_n:
                            best_n, best_pid = cn, rid
                if best_pid is None:
                    self._revert_lineage(fs, problem_id, split_ids)
                    self.log(
                        f"[refine] {problem_id}: all sub-problems below threshold "
                        f"after split — reverting"
                    )
                    return False
                active_pid = best_pid

        # Final check after loop
        res = self._evaluate(fs, [active_pid]).get(active_pid, {})
        if not res.get("survivors"):
            self._commit(fs, active_pid)
            self.log(f"[refine] {active_pid} resolved")
            return True
        self._revert_lineage(fs, problem_id, split_ids)
        self.log(f"[refine] {problem_id} unresolved after {MAX_FIX_ATTEMPTS} attempts — reverted")
        return False

    def propose_only(self, fs, problem_id: str) -> str | None:
        """Human mode: generate one proposal and store it unapplied. Returns entry id."""
        if self.fixer is None:
            return None
        cluster = fs.problems.get("clusters", {}).get(problem_id)
        if not cluster or not cluster.get("mutants"):
            return None
        res = self._evaluate(fs, [problem_id]).get(problem_id, {})
        survivors = res.get("survivors", set()) or set(cluster.get("mutants", []))
        program_text = self.pm.active_source(fs.stem, fs.path).read_text()
        snippet = self.fixer.propose_fix(program_text, cluster, self._examples(fs.stem, survivors))
        if not snippet:
            return None
        # Fixer flagged as unresolvable — mark immediately, no unapplied entry
        if snippet.get("action") == "unresolved":
            cluster["unresolved_reason"] = snippet.get("unresolved_reason", "")
            self._mark_unresolved(fs, problem_id)
            self.log(f"[refine] {problem_id} flagged unresolvable: "
                     f"{snippet.get('unresolved_reason', '')}")
            return None
        eid = self.pm.add_fix(fs.stem, problem_id, snippet, status="unapplied", original=fs.path)
        self.log(f"[refine] proposed {eid} for {problem_id} (awaiting apply)")
        return eid

    def apply_proposal(self, fs, entry_id: str) -> bool:
        """Human mode: apply a stored proposal, re-test, and queue the next proposal if
        the active problem still has survivors. Returns True if that problem is resolved."""
        stem = fs.stem
        stack = self.pm.load_fix_stack(stem)
        entry = next((e for e in stack["entries"] if e["id"] == entry_id), None)
        if entry is None:
            return False
        pid = entry.get("problem_id", "")
        self.pm.set_fix_status(stem, entry_id, "applied", original=fs.path)

        # Reject fix if it overconstrained the GT (unless --allow-overconstrain)
        if not self.allow_overconstrain and not self._gt_still_verifies(fs):
            self.log(
                f"[refine] {pid}: {entry_id} REJECTED — "
                f"fix overconstrained GT (patched GT fails Dafny), reverting"
            )
            self.pm.set_fix_status(stem, entry_id, "unapplied", original=fs.path)
            return False

        res = self._evaluate(fs, [pid]).get(pid, {})
        if not res.get("survivors"):
            self._commit(fs, pid)
            self.log(f"[refine] {pid} resolved (applied {entry_id})")
            return True
        self.propose_only(fs, pid)   # queue the next proposal for the human
        return False

    def dismiss_proposal(self, fs, entry_id: str) -> None:
        self.pm.set_fix_status(fs.stem, entry_id, "dismissed", original=fs.path)

    # --------------------------------------------------------------- commit
    def _commit(self, fs, active_pid: str):
        """A problem is resolved: re-test ALL clusters, move killed mutants alive→killed,
        drop them from clusters, re-describe affected non-active clusters, prune empties."""
        stem = fs.stem
        clusters = fs.problems.get("clusters", {})
        results = self._evaluate(fs, list(clusters))
        applied = self.pm.applied_fixes(stem)
        last_fix = applied[-1]["id"] if applied else None
        all_killed: list[str] = []

        for cid, r in results.items():
            killed = r["killed"]
            if not killed:
                continue
            for name in killed:
                self.pm.move_mutant(stem, name, "alive", "killed")
                all_killed.append(name)
            c = clusters[cid]
            c["mutants"] = [m for m in c.get("mutants", []) if m not in killed]
            c["mutant_count"] = len(c["mutants"])

        if last_fix and all_killed:
            self.pm.record_fix_kills(stem, last_fix, all_killed)

        # Re-describe affected non-active clusters from their survivors; move resolved
        # (now-empty) clusters into the persistent "resolved" record so their problem
        # descriptions survive after the fix kills every witnessing mutant.
        for cid in list(clusters):
            killed = results.get(cid, {}).get("killed", set())
            c = clusters[cid]
            if not c.get("mutants"):
                self._record_resolved(fs, cid, c, killed, byproduct=(cid != active_pid))
                del clusters[cid]
                continue
            if cid != active_pid and killed and self.analyser is not None:
                diffs = [self._diff_for(stem, n) for n in c["mutants"]]
                fresh = self.analyser.reevaluate_problem(c, self._gt_path(fs), diffs)
                c["title"] = fresh.get("title", c["title"])
                c["description"] = fresh.get("description", c["description"])
                c["open_questions"] = fresh.get("open_questions", c.get("open_questions", ""))
                self.log(f"[refine] re-described {cid} after byproduct kills")

        self.pm.save_problems(stem, fs.problems)

    def _record_resolved(self, fs, cid: str, cluster: dict, killed, byproduct: bool):
        """Preserve a resolved problem (its description + which fixes closed it) after its
        cluster is emptied, so the Watch tab can still show what was fixed."""
        stem = fs.stem
        by = [e["id"] for e in self.pm.load_fix_stack(stem).get("entries", [])
              if e.get("problem_id") == cid and e.get("status") == "applied"]
        resolved = fs.problems.setdefault("resolved", {})
        prev = resolved.get(cid, {})
        resolved[cid] = {
            "id": cid,
            "title": cluster.get("title", ""),
            "description": cluster.get("description", ""),
            "open_questions": cluster.get("open_questions", ""),
            "resolved_by": by,
            "killed_count": prev.get("killed_count", 0) + len(killed),
            "byproduct": byproduct,
            "resolved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.log(f"[refine] {cid} fixed ({'byproduct' if byproduct else 'active'}) — "
                 f"kept in resolved record")

    def _mark_unresolved(self, fs, problem_id: str):
        c = fs.problems.get("clusters", {}).get(problem_id)
        if c is not None:
            c["status"] = "unresolved"
            self.pm.save_problems(fs.stem, fs.problems)

    # --------------------------------------------------------------- trigger
    def qualifying_problem(self, fs, n_threshold: int) -> str | None:
        """The unresolved cluster with the most WEAK_SPEC mutants at/over the threshold,
        or None. (Used by the launcher's default-mode trigger.)"""
        best, best_n = None, n_threshold - 1
        for cid, c in fs.problems.get("clusters", {}).items():
            if c.get("status") == "unresolved":
                continue
            n = c.get("mutant_count", len(c.get("mutants", [])))
            if n > best_n:
                best, best_n = cid, n
        return best

    def pending_problems(self, fs, n_threshold: int) -> list[str]:
        """All unresolved clusters at/over threshold, most-mutants-first (end-of-targets)."""
        items = []
        for cid, c in fs.problems.get("clusters", {}).items():
            if c.get("status") == "unresolved":
                continue
            n = c.get("mutant_count", len(c.get("mutants", [])))
            if n >= n_threshold and c.get("mutants"):
                items.append((n, cid))
        return [cid for _n, cid in sorted(items, reverse=True)]
