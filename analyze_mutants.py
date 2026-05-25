#!/usr/bin/env python3
"""
LLM-based behavioural equivalence classifier for Dafny mutants.

Called by watch_mutants.py every N new mutants. Groups the batch by ground
truth file, creates (or reuses) a Gemini context cache for each GT, then asks
Gemini to:
  - classify each mutant as EQUIVALENT or WEAK_SPEC
  - assign it to an existing problem cluster or propose a new one
  - propose cluster merges when two clusters turn out to share a root cause

Cluster state is persisted to analysis/cluster_state.json and updated after
every batch.  The cache contains only the static system instruction + GT file;
cluster context travels in each per-request user prompt so the cache stays
valid as clusters evolve.
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

import debug_log
from persistence_manager import _DECL_RE

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
# A step up from the streaming-analysis model, used only for spec-fix proposal.
FIX_MODEL = os.getenv("GEMINI_FIX_MODEL", "gemini-3.5-flash")
CACHE_TTL = "1800s"  # 30 minutes
# Per-request wall-clock timeout in milliseconds (HttpOptions.timeout is ms;
# the SDK divides by 1000 before passing to httpx). Prevents indefinite hangs
# when the API stalls on a rate-limit retry.
API_TIMEOUT_MS = int(os.getenv("GEMINI_API_TIMEOUT_MS", "120000"))  # 120 s


def _make_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment or .env file")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=API_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=2, max_delay=8.0),
        ),
    )

# ---------------------------------------------------------------------------
# Mutation operator descriptions
# ---------------------------------------------------------------------------

_OP_TEMPLATES: dict[str, str] = {
    "AOR": "Arithmetic Operator Replacement — replaced an arithmetic operator with `{arg}`",
    "ROR": "Relational Operator Replacement — replaced a relational operator with `{arg}`",
    "COR": "Conditional Operator Replacement — replaced a conditional operator with `{arg}`",
    "LOR": "Logical Operator Replacement — replaced a logical operator with `{arg}`",
    "SOR": "Shift Operator Replacement — replaced a shift operator with `{arg}`",
    "BBR": "Boolean-Binary Expression Replacement — replaced a relational/conditional expression with `{arg}`",
    "AOI": "Arithmetic Operator Insertion — inserted a unary minus before an arithmetic expression",
    "COI": "Conditional Operator Insertion — inserted `!` before a conditional expression",
    "LOI": "Logical Operator Insertion — inserted `!` before a logical expression",
    "AOD": "Arithmetic Operator Deletion — removed a unary minus from an arithmetic expression",
    "COD": "Conditional Operator Deletion — removed `!` from a conditional expression",
    "LOD": "Logical Operator Deletion — removed `!` from a logical expression",
    "LVR": "Literal Value Replacement — replaced a literal value with `{arg}`",
    "EVR": "Expression Value Replacement — replaced an expression with the default literal for type `{arg}`",
    "VER": "Variable Expression Replacement — replaced a variable with `{arg}` (same type)",
    "LSR": "Loop Statement Replacement — replaced a loop control statement with `{arg}`",
    "LBI": "Loop Break Insertion — inserted a `break` at the start of a loop body",
    "MRR": "Method Return Value Replacement — replaced a method call result with the default literal for type `{arg}`",
    "MAP": "Method Argument Propagation — replaced a method call result with its argument at index `{arg}`",
    "MNR": "Method Naked Receiver — deleted a class method call, keeping only its receiver",
    "MCR": "Method Call Replacement — replaced a method call with `{arg}` (same signature)",
    "MVR": "Method-Variable Replacement — replaced a method call result with variable `{arg}` (same type)",
    "SAR": "Swap Argument — swapped a method call argument with the one at position `{arg}`",
    "CIR": "Collection Initialization Replacement — replaced a collection initializer with an empty/default one",
    "CBR": "Case Block Replacement — replaced a match case block with the default case",
    "CBE": "Case Block Extraction — extracted one branch of an if-statement, deleting the rest",
    "TAR": "Tuple Access Replacement — replaced a tuple element index with `{arg}`",
    "DCR": "Datatype Constructor Replacement — replaced a datatype constructor with `{arg}`",
    "FAR": "Field Access Replacement — replaced a field access with `{arg}` (same class)",
    "SDL": "Statement Deletion — deleted a statement or entire code block",
    "VDL": "Variable Deletion — deleted all occurrences of a variable",
    "SLD": "Subsequence Limit Deletion — deleted the bottom or top limit of a subsequence expression",
    "ODL": "Operator Deletion — deleted a binary operator and one of its arguments",
    "THI": "This Keyword Insertion — inserted `this` before a parameter shadowing a class field",
    "THD": "This Keyword Deletion — removed `this` from a class field access",
    "AMR": "Accessor Method Replacement — replaced a getter method body with the one at position `{arg}`",
    "MMR": "Modifier Method Replacement — replaced a setter method body with the one at position `{arg}`",
    "PRV": "Polymorphic Reference Replacement — replaced a child reference with `{arg}`",
    "SWS": "Swap Statement — swapped a statement with the one immediately above or below it",
    "SWV": "Swap Variable Declaration — swapped a variable declaration's RHS with the one at position `{arg}`",
}


def describe_operator(op: str, arg: str) -> str:
    template = _OP_TEMPLATES.get(op, f"Unknown operator `{op}`")
    return template.format(arg=arg if arg else "N/A")


def parse_mutant_filename(filename: str) -> dict:
    """Return {gt_name, pos, op, arg} parsed from a mutant filename."""
    stem = filename.removesuffix(".dfy")
    parts = stem.split("__", 1)
    result = {"gt_name": parts[0] + ".dfy", "pos": "", "op": "", "arg": ""}
    if len(parts) < 2:
        return result
    sub = parts[1].split("_", 2)
    result["pos"] = sub[0] if len(sub) > 0 else ""
    result["op"] = sub[1] if len(sub) > 1 else ""
    result["arg"] = sub[2].rstrip("_") if len(sub) > 2 else ""
    return result


# ---------------------------------------------------------------------------
# Unverified-method detection (no ensures/decreases/increases → can't be killed)
# ---------------------------------------------------------------------------

_SPEC_KEYWORDS = frozenset({"ensures", "decreases", "increases"})


def _find_method_at_offset(gt_text: str, char_offset: int) -> str | None:
    """Return the name of the method/function/lemma that contains `char_offset`,
    or None if no enclosing declaration is found."""
    lines = gt_text.splitlines(keepends=True)
    offset = 0
    line_idx = len(lines) - 1
    for i, line in enumerate(lines):
        if offset + len(line) > char_offset:
            line_idx = i
            break
        offset += len(line)
    for i in range(line_idx, -1, -1):
        m = _DECL_RE.match(lines[i])
        if m:
            return m.group(1)
    return None


def _method_has_spec(gt_text: str, method_name: str) -> bool:
    """True if `method_name` has at least one ensures/decreases/increases clause."""
    in_decl = False
    for line in gt_text.splitlines():
        stripped = line.strip()
        if not in_decl:
            m = _DECL_RE.match(line)
            if m and m.group(1) == method_name:
                in_decl = True
        else:
            first_word = stripped.split()[0] if stripped else ""
            if first_word in _SPEC_KEYWORDS:
                return True
            # Stop scanning at the opening brace (body) or another declaration
            if stripped.startswith("{") or (_DECL_RE.match(line) and stripped):
                break
    return False


def _method_has_verify_false(gt_text: str, method_name: str) -> bool:
    """True if `method_name`'s declaration contains {:verify false}."""
    in_decl = False
    for line in gt_text.splitlines():
        stripped = line.strip()
        if not in_decl:
            dm = _DECL_RE.match(line)
            if dm and dm.group(1) == method_name:
                in_decl = True
        if in_decl:
            if "{:verify false}" in line:
                return True
            # Stop at the opening brace of the body
            if stripped.startswith("{"):
                break
            # Stop at another declaration
            dm2 = _DECL_RE.match(line)
            if dm2 and dm2.group(1) != method_name:
                break
    return False


_METHOD_SPEC_KEYWORDS = frozenset({"requires", "ensures", "decreases", "increases",
                                    "reads", "modifies"})


def _collect_covering_conditions(gt_text: str, char_offset: int) -> dict:
    """Return spec clauses and loop invariants that cover `char_offset`.

    Returns {"method": str|None, "spec": [str], "loop_invariants": [str]}
    where spec = method-level clause lines and loop_invariants = invariant lines
    from the closest enclosing while loop.
    """
    lines = gt_text.splitlines(keepends=True)
    # Find line index containing char_offset
    offset = 0
    mut_line = len(lines) - 1
    for i, line in enumerate(lines):
        if offset + len(line) > char_offset:
            mut_line = i
            break
        offset += len(line)

    # --- Method-level spec ---
    method_name: str | None = None
    method_line: int | None = None
    spec: list[str] = []
    for i in range(mut_line, -1, -1):
        m = _DECL_RE.match(lines[i])
        if m:
            method_name = m.group(1)
            method_line = i
            break
    if method_line is not None:
        in_spec = False
        for i in range(method_line + 1, len(lines)):
            stripped = lines[i].strip()
            if not stripped:
                continue
            if stripped.startswith("{"):
                break
            if _DECL_RE.match(lines[i]):
                break
            first = stripped.split()[0]
            if first in _METHOD_SPEC_KEYWORDS:
                in_spec = True
                spec.append(stripped)
            elif in_spec:
                # continuation line of a multi-line clause
                spec.append(stripped)

    # --- Loop invariants from enclosing while ---
    loop_invariants: list[str] = []
    # Scan backward from mut_line for a 'while' that opens before our line
    for i in range(mut_line, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith("while ") or stripped == "while":
            # Collect invariant lines between this while and its opening {
            for j in range(i + 1, len(lines)):
                s = lines[j].strip()
                if not s:
                    continue
                if s.startswith("{"):
                    break
                if s.startswith("invariant "):
                    loop_invariants.append(s)
                elif s.startswith("decreases ") or s.startswith("modifies "):
                    pass  # skip non-invariant loop specs
                else:
                    break
            break

    return {"method": method_name, "spec": spec, "loop_invariants": loop_invariants}


# ---------------------------------------------------------------------------
# Cluster state persistence
# ---------------------------------------------------------------------------

def empty_cluster_state() -> dict:
    return {
        "clusters": {},
        "total_processed": 0,
        "total_weak_spec": 0,
        "total_equivalent": 0,
        "total_contextual": 0,
        "contextual": [],
        "unverified_methods": {},
        "total_unverified": 0,
        "next_cluster_id": 1,
    }


def load_cluster_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return empty_cluster_state()


def save_cluster_state(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def _format_cluster_context(cluster_state: dict) -> str:
    """Build the known-problems listing prepended to each batch request."""
    clusters = cluster_state.get("clusters", {})
    if not clusters:
        return (
            "No specification problems have been identified yet. "
            "For each WEAK_SPEC mutant, propose a new problem using a temporary id NEW_1, NEW_2, …"
        )
    lines = [
        "Known specification problems — assign each WEAK_SPEC mutant to the problem "
        "whose root cause it is a symptom of, or propose a new problem:"
    ]
    for cid, c in clusters.items():
        lines.append(f"[{cid}] {c['title']}")
        lines.append(f"     {c['description']}")
        oq = c.get("open_questions", "")
        if oq:
            lines.append(f"     (still unknown: {oq})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_EQUIV_SYSTEM_INSTRUCTION = """\
You are a Dafny formal verification specialist analysing mutation testing results.

Your task: for each surviving mutant (one that still verifies against the spec), \
determine whether it survived because of a real behavioural change or not.

For each mutant:
1. First, write a 1–2 sentence **description** of the actual change made — what \
   specifically changed in observable behaviour or computational meaning, not just \
   a paraphrase of the diff syntax.
2. Then assign a **verdict**:

   - **EQUIVALENT** — the mutant is behaviourally equivalent to the original; no \
     valid input can distinguish them, so the spec cannot and should not detect it.
   - **WEAK_SPEC** — the spec is underconstrained; it should logically catch this \
     behavioural change but doesn't, indicating a gap in the formal specification.
   - **CONTEXTUAL** — the mutant is wrong and not behaviourally equivalent, but the \
     difference is an implementation-level detail (efficiency, algorithmic strategy, \
     or an internal invariant) that cannot be captured by a sound \
     `requires`/`ensures`/`invariant` without over-constraining the spec or encoding \
     implementation choices. Use this only when certain no spec clause could \
     reasonably detect the mutation.

Critically consider whether the diff proposes a **real behavioural change** or \
merely **rearranges equivalent behaviour**. Examples of EQUIVALENT patterns: \
operator variants that coincide in the reachable value range (e.g. `>` vs `!=` when \
the branch structure guarantees they agree), cosmetic rewrites, unreachable-code \
mutations, and swaps of independent declarations where evaluation order is irrelevant.

Guidance:
- Infer algorithmic intent from the code structure — there is no separate natural-\
  language spec.
- Be sceptical of EQUIVALENT for mutations that alter loop invariants, \
  postcondition-relevant expressions, or boundary conditions.

Respond with exactly this JSON and nothing else:
{
  "results": [
    {
      "mutant": "<filename>",
      "description": "<1–2 sentences: what actually changed in behaviour or meaning>",
      "verdict": "EQUIVALENT" | "WEAK_SPEC" | "CONTEXTUAL",
      "confidence": <0.0–1.0>,
      "justification": "<1–2 sentences explaining the verdict>"
    }
  ]
}\
"""


_CATEG_SYSTEM_INSTRUCTION = """\
You are a Dafny formal verification specialist analysing mutation testing results.

You are given a set of **WEAK_SPEC** mutants — mutations that survive verification \
because the specification is underconstrained. Each mutant is already classified; \
your job is only to assign each one to the correct specification problem.

A **problem** is a specific gap in the formal specification — a missing or \
underconstrained clause that allows a class of behavioural deviations to \
verify. A WEAK_SPEC mutant is a *symptom* of a problem: it is a concrete \
witness that the gap exists, but the gap itself is the root cause. Many \
distinct mutants can be symptoms of the same problem. Conversely, two \
superficially similar mutations may stem from different problems if the spec \
gaps that permit them are independent.

Respond with exactly this JSON and nothing else:
{
  "results": [
    {
      "mutant": "<filename>",
      "problem_id": "<existing-id> | <NEW_x> | null"
    }
  ],
  "new_problems": [
    {
      "id": "<NEW_x>",
      "title": "<short title (≤8 words)>",
      "description": "<3–5 sentences: name the specific missing or underconstrained clause, explain the class of behavioural deviations it permits, and why the spec fails to catch them>",
      "open_questions": "<one short sentence on what is still unknown about this problem — e.g. which declaration the missing clause belongs on, or whether two symptoms share a root cause. Empty string if nothing material is unknown.>"
    }
  ],
  "problem_merges": [
    {"from": "<existing-id>", "into": "<existing-id>", "reason": "<1 sentence>"}
  ]
}

Assignment rules:
- WEAK_SPEC that is a symptom of an existing problem → use its id exactly
- WEAK_SPEC revealing a new spec gap not covered by any existing problem → use a \
  temporary id NEW_1, NEW_2, … and list it in new_problems. Multiple mutants in \
  this batch that are symptoms of the same new root cause share the same NEW_x id.
- problem_merges → only when two *existing* problems are actually the same spec gap \
  described differently; may be an empty list.
- new_problems and problem_merges may both be empty lists.

Granularity guidance — err on the side of fewer, broader problems:
- There is no correct number of problems. If every mutant in this batch looks like \
  a symptom of the same underlying spec gap, emit exactly one problem.
- Prefer one coarse problem over two fine-grained ones whenever the distinction is \
  uncertain. The refinement stage can split a problem later if evidence warrants it; \
  over-splitting on the first pass creates noise that is hard to undo.
- Only create a separate problem when you are confident the two groups require \
  *independent* spec clauses to fix — i.e. fixing one would not automatically fix \
  the other.\
"""


def _strip_diff_header_lines(diff: str) -> str:
    """Remove unified-diff filename header lines (--- path and +++ path)."""
    return "\n".join(
        line for line in diff.splitlines()
        if not (line.startswith("--- ") or line.startswith("+++ "))
    )


def _build_equiv_prompt(gt_name: str, mutants: list[dict]) -> str:
    """Pass-1 prompt: classify equivalence + generate change description. No cluster context."""
    lines = [f"Analyse the following {len(mutants)} mutant(s) of `{gt_name}`:", ""]
    for i, m in enumerate(mutants, 1):
        info = parse_mutant_filename(m["name"])
        op_desc = describe_operator(info["op"], info["arg"])
        stripped_diff = _strip_diff_header_lines(m["diff"])
        lines += [
            f"### Mutant {i}: `{m['name']}`",
            f"**Mutation applied:** {op_desc}",
            "**Diff** (ground truth → mutant):",
            "```diff",
            stripped_diff.rstrip(),
            "```",
            "",
        ]
    return "\n".join(lines)


def _build_categ_prompt(gt_name: str, mutants: list[dict], cluster_state: dict) -> str:
    """Pass-2 prompt: assign WEAK_SPEC mutants to problems. Uses descriptions, not diffs."""
    cluster_ctx = _format_cluster_context(cluster_state)
    lines = [
        cluster_ctx,
        "",
        f"Assign the following {len(mutants)} WEAK_SPEC mutant(s) of `{gt_name}` to problems:",
        "",
    ]
    for i, m in enumerate(mutants, 1):
        # Results from classify_equiv use "mutant" key; original input dicts use "name"
        mutant_name = m.get("mutant") or m.get("name", "")
        info = parse_mutant_filename(mutant_name)
        op_desc = describe_operator(info["op"], info["arg"])
        description = m.get("description", "(no description available)")
        lines += [
            f"### Mutant {i}: `{mutant_name}`",
            f"**Mutation applied:** {op_desc}",
            f"**Change:** {description}",
        ]
        cond = m.get("conditions")
        if cond and (cond.get("spec") or cond.get("loop_invariants")):
            lines.append("**Conditions currently covering this mutation point:**")
            if cond.get("spec"):
                lines.append("  method spec: " + "; ".join(cond["spec"]))
            if cond.get("loop_invariants"):
                lines.append("  loop invariants: " + "; ".join(cond["loop_invariants"]))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-GT analyser with context cache lifecycle management
# ---------------------------------------------------------------------------

class _GTAnalyser:
    """Manages one Gemini context cache for a single ground truth file."""

    def __init__(self, gt_path: Path, client: genai.Client):
        self._gt_path = gt_path
        self._client = client
        self._cache = None
        self._cache_born: float = 0.0
        self._equiv_cache = None
        self._equiv_cache_born: float = 0.0
        self._ttl_secs = 1800

    def _gt_content_block(self) -> str:
        text = self._gt_path.read_text()
        return f"Ground truth file `{self._gt_path.name}`:\n```dafny\n{text}\n```"

    def _ensure_cache(self) -> bool:
        age = time.monotonic() - self._cache_born
        if self._cache is not None and age < self._ttl_secs * 0.8:
            return True
        try:
            self._cache = self._client.caches.create(
                model=MODEL,
                config=types.CreateCachedContentConfig(
                    system_instruction=_CATEG_SYSTEM_INSTRUCTION,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[types.Part(text=self._gt_content_block())],
                        )
                    ],
                    ttl=CACHE_TTL,
                ),
            )
            self._cache_born = time.monotonic()
            print(
                f"  [llm]  KV cache (categ) created for {self._gt_path.name}: {self._cache.name}",
                file=sys.stderr,
            )
            debug_log.log(f"CACHE {MODEL} (categ) created for {self._gt_path.name}",
                          self._cache.name)
            return True
        except Exception as exc:
            print(
                f"  [llm]  context cache unavailable for {self._gt_path.name} "
                f"({exc}); inlining GT",
                file=sys.stderr,
            )
            debug_log.log(f"CACHE {MODEL} unavailable for {self._gt_path.name}", str(exc))
            self._cache = None
            return False

    def _ensure_equiv_cache(self) -> bool:
        age = time.monotonic() - self._equiv_cache_born
        if self._equiv_cache is not None and age < self._ttl_secs * 0.8:
            return True
        try:
            self._equiv_cache = self._client.caches.create(
                model=MODEL,
                config=types.CreateCachedContentConfig(
                    system_instruction=_EQUIV_SYSTEM_INSTRUCTION,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[types.Part(text=self._gt_content_block())],
                        )
                    ],
                    ttl=CACHE_TTL,
                ),
            )
            self._equiv_cache_born = time.monotonic()
            print(
                f"  [llm]  KV cache (equiv) created for {self._gt_path.name}: "
                f"{self._equiv_cache.name}",
                file=sys.stderr,
            )
            debug_log.log(f"CACHE {MODEL} (equiv) created for {self._gt_path.name}",
                          self._equiv_cache.name)
            return True
        except Exception as exc:
            print(
                f"  [llm]  equiv cache unavailable for {self._gt_path.name} "
                f"({exc}); inlining GT",
                file=sys.stderr,
            )
            debug_log.log(f"CACHE {MODEL} (equiv) unavailable for {self._gt_path.name}",
                          str(exc))
            self._equiv_cache = None
            return False

    def generate(self, prompt: str) -> str:
        cache_ok = self._ensure_cache()
        debug_log.log(f"REQUEST {MODEL} (categ · {self._gt_path.name} · "
                      f"cache={'hit' if cache_ok else 'inline'})", prompt)

        if cache_ok and self._cache is not None:
            response = self._client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    cached_content=self._cache.name,
                    response_mime_type="application/json",
                ),
            )
        else:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part(text=self._gt_content_block())],
                ),
                types.Content(
                    role="model",
                    parts=[types.Part(text="Understood. I have the ground truth file.")],
                ),
                types.Content(
                    role="user",
                    parts=[types.Part(text=prompt)],
                ),
            ]
            response = self._client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=_CATEG_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                ),
            )
        text = response.text.strip()
        debug_log.log(f"RESPONSE {MODEL} (categ)", text)
        return text

    def generate_equiv(self, prompt: str) -> str:
        cache_ok = self._ensure_equiv_cache()
        debug_log.log(f"REQUEST {MODEL} (equiv · {self._gt_path.name} · "
                      f"cache={'hit' if cache_ok else 'inline'})", prompt)

        if cache_ok and self._equiv_cache is not None:
            response = self._client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    cached_content=self._equiv_cache.name,
                    response_mime_type="application/json",
                ),
            )
        else:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part(text=self._gt_content_block())],
                ),
                types.Content(
                    role="model",
                    parts=[types.Part(text="Understood. I have the ground truth file.")],
                ),
                types.Content(
                    role="user",
                    parts=[types.Part(text=prompt)],
                ),
            ]
            response = self._client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=_EQUIV_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                ),
            )
        text = response.text.strip()
        debug_log.log(f"RESPONSE {MODEL} (equiv)", text)
        return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MutantAnalyser:
    """
    Stateful analyser. Two-stage pipeline:
      classify_equiv() — Pass 1, batches ≤5, classifies EQUIV/WEAK_SPEC/CONTEXTUAL
      assign_problems() — Pass 2, batches ≤50, assigns WEAK_SPEC to problem clusters

    Pass 2 must only be called after ALL Pass-1 work is complete.
    """

    def __init__(self):
        self._client = _make_client()
        self._gt_analysers: dict[str, _GTAnalyser] = {}

    def _gt_analyser(self, gt_path: Path) -> _GTAnalyser:
        key = str(gt_path)
        if key not in self._gt_analysers:
            self._gt_analysers[key] = _GTAnalyser(gt_path, self._client)
        return self._gt_analysers[key]

    def classify_equiv(
        self,
        mutants: list[dict],
        gts_dir: Path,
        output_dir: Path,
        cluster_state: dict,
    ) -> tuple[list[dict], dict]:
        """Pass 1: classify each mutant as EQUIV/WEAK_SPEC/CONTEXTUAL.
        Updates cluster_state for EQUIV and CONTEXTUAL (not WEAK_SPEC — that is Pass 2).
        Writes all verdicts to JSONL immediately (WEAK_SPEC with problem_id=null).
        Returns (all_results, summary). WEAK_SPEC results carry 'description' and
        'conditions' for use by assign_problems.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        groups: dict[str, list[dict]] = {}
        for m in mutants:
            gt_name = parse_mutant_filename(m["name"])["gt_name"]
            groups.setdefault(gt_name, []).append(m)

        all_results: list[dict] = []
        total_weak = total_equiv = total_contextual = 0

        for gt_name, group in groups.items():
            gt_path = gts_dir / gt_name
            if not gt_path.exists():
                print(
                    f"  [llm]  GT missing for {gt_name}, skipping {len(group)} mutant(s)",
                    file=sys.stderr,
                )
                continue

            analyser = self._gt_analyser(gt_path)

            # Equivalence detection in sub-batches of ≤5
            desc_map: dict[str, dict] = {}
            for i in range(0, len(group), 5):
                sub = group[i:i + 5]
                try:
                    raw = analyser.generate_equiv(_build_equiv_prompt(gt_name, sub))
                    data1 = json.loads(raw)
                except Exception as exc:
                    print(f"  [llm]  equiv pass error: {exc}", file=sys.stderr)
                    continue
                for r in data1.get("results", []):
                    desc_map[r["mutant"]] = r

            # Default EQUIVALENT for any parse/API failures
            for m in group:
                if m["name"] not in desc_map:
                    desc_map[m["name"]] = {"verdict": "EQUIVALENT", "description": "",
                                           "justification": "", "confidence": 0.0}

            results: list[dict] = []
            for m in group:
                name = m["name"]
                er = desc_map[name]
                verdict = er.get("verdict", "EQUIVALENT")
                r: dict = {
                    "mutant": name,
                    "verdict": verdict,
                    "description": er.get("description", ""),
                    "justification": er.get("justification", ""),
                    "confidence": er.get("confidence", 0.0),
                    "problem_id": None,
                }
                if verdict == "WEAK_SPEC":
                    # Pass through conditions for assign_problems prompt building
                    r["conditions"] = m.get("conditions")
                    total_weak += 1
                elif verdict == "CONTEXTUAL":
                    cluster_state.setdefault("contextual", [])
                    cluster_state.setdefault("total_contextual", 0)
                    cluster_state["contextual"].append({
                        "mutant": name,
                        "justification": er.get("justification", ""),
                    })
                    cluster_state["total_contextual"] += 1
                    cluster_state["total_processed"] += 1
                    total_contextual += 1
                else:
                    cluster_state["total_equivalent"] += 1
                    cluster_state["total_processed"] += 1
                    total_equiv += 1
                results.append(r)

            # Write all verdicts to JSONL (WEAK_SPEC with problem_id=null).
            # This marks them as analysed for warm-start purposes.
            gt_stem = gt_name.removesuffix(".dfy")
            out_file = output_dir / f"{gt_stem}_analysis.jsonl"
            with out_file.open("a") as fh:
                for r in results:
                    # Strip internal-only fields before writing
                    fh.write(json.dumps({
                        k: v for k, v in r.items() if k != "conditions"
                    }) + "\n")

            all_results.extend(results)

        summary = {"weak": total_weak, "equiv": total_equiv, "contextual": total_contextual}
        return all_results, summary

    def assign_problems(
        self,
        weak_spec_mutants: list[dict],
        gts_dir: Path,
        output_dir: Path,
        cluster_state: dict,
    ) -> tuple[list[dict], dict]:
        """Pass 2: assign WEAK_SPEC mutants to problem clusters.
        Mutants must already have 'description' set (from classify_equiv).
        Updates cluster_state in-place. Does NOT write to JSONL (already written by Pass 1).
        Returns (all_results, summary).
        """
        if not weak_spec_mutants:
            return [], {"new_problems": 0, "merges": 0}

        output_dir.mkdir(parents=True, exist_ok=True)

        groups: dict[str, list[dict]] = {}
        for m in weak_spec_mutants:
            mutant_name = m.get("mutant") or m.get("name", "")
            gt_name = parse_mutant_filename(mutant_name)["gt_name"]
            groups.setdefault(gt_name, []).append(m)

        all_results: list[dict] = []
        total_new_problems = total_merges = 0

        for gt_name, group in groups.items():
            gt_path = gts_dir / gt_name
            if not gt_path.exists():
                continue

            analyser = self._gt_analyser(gt_path)

            categ_result_map: dict[str, dict] = {}
            id_map: dict[str, str] = {}

            for i in range(0, len(group), 50):
                sub = group[i:i + 50]
                try:
                    raw = analyser.generate(
                        _build_categ_prompt(gt_name, sub, cluster_state)
                    )
                    data2 = json.loads(raw)
                except Exception as exc:
                    print(f"  [llm]  categ pass error: {exc}", file=sys.stderr)
                    continue

                new_problems: list[dict] = data2.get("new_problems", [])
                problem_merges: list[dict] = data2.get("problem_merges", [])

                # Remap NEW_x → real sequential IDs
                for np in new_problems:
                    tmp_id = np.get("id", "")
                    real_id = f"C{cluster_state['next_cluster_id']:03d}"
                    cluster_state["next_cluster_id"] += 1
                    id_map[tmp_id] = real_id
                    cluster_state["clusters"][real_id] = {
                        "id": real_id,
                        "title": np.get("title", "Untitled"),
                        "description": np.get("description", ""),
                        "open_questions": np.get("open_questions", ""),
                        "mutants": [],
                        "mutant_count": 0,
                    }
                    total_new_problems += 1

                # Apply problem merges
                for merge in problem_merges:
                    from_id = merge.get("from", "")
                    into_id = merge.get("into", "")
                    if (
                        from_id in cluster_state["clusters"]
                        and into_id in cluster_state["clusters"]
                        and from_id != into_id
                    ):
                        src = cluster_state["clusters"].pop(from_id)
                        dst = cluster_state["clusters"][into_id]
                        dst["mutants"].extend(src["mutants"])
                        dst["mutant_count"] += src["mutant_count"]
                        id_map[from_id] = into_id
                        total_merges += 1

                for r in data2.get("results", []):
                    raw_pid = r.get("problem_id")
                    real_pid = id_map.get(raw_pid, raw_pid) if raw_pid else None
                    categ_result_map[r["mutant"]] = {"problem_id": real_pid}

            # Update cluster_state and build result records
            results: list[dict] = []
            for m in group:
                name = m.get("mutant") or m.get("name", "")
                pid = categ_result_map.get(name, {}).get("problem_id")
                if pid and pid in cluster_state["clusters"]:
                    cluster_state["clusters"][pid]["mutants"].append(name)
                    cluster_state["clusters"][pid]["mutant_count"] += 1
                cluster_state["total_weak_spec"] += 1
                cluster_state["total_processed"] += 1
                results.append({
                    "mutant": name,
                    "verdict": "WEAK_SPEC",
                    "description": m.get("description", ""),
                    "justification": m.get("justification", ""),
                    "confidence": m.get("confidence", 0.0),
                    "problem_id": pid,
                })
            all_results.extend(results)

        summary = {"new_problems": total_new_problems, "merges": total_merges}
        return all_results, summary

    def reevaluate_problem(
        self, cluster: dict, gt_path: Path, surviving_diffs: list[dict]
    ) -> dict:
        """Refresh a cluster's wording from its remaining (still-alive) mutants after
        some of its symptoms were killed by an accepted spec fix. Uses the flash-lite
        model. Returns {title, description, open_questions}; falls back to the existing
        wording on any error."""
        fallback = {
            "title": cluster.get("title", ""),
            "description": cluster.get("description", ""),
            "open_questions": cluster.get("open_questions", ""),
        }
        gt_text = gt_path.read_text() if gt_path.exists() else ""
        examples = []
        for i, m in enumerate(surviving_diffs, 1):
            info = parse_mutant_filename(m["name"])
            examples += [
                f"### Surviving mutant {i}: `{m['name']}`",
                f"**Mutation applied:** {describe_operator(info['op'], info['arg'])}",
                "```diff",
                m["diff"].rstrip(),
                "```",
                "",
            ]
        prompt = "\n".join(
            [
                "A specification problem you previously identified has had some of its "
                "witnessing mutants killed by a newly-strengthened spec. The remaining "
                "mutants below are the ones the current spec still fails to catch.",
                "",
                f"Previous title: {cluster.get('title', '')}",
                f"Previous description: {cluster.get('description', '')}",
                f"Previous open questions: {cluster.get('open_questions', '')}",
                "",
                f"Ground truth `{gt_path.name}`:",
                "```dafny",
                gt_text,
                "```",
                "",
                "Remaining symptoms:",
                "",
                *examples,
                "Re-describe the problem as it now stands. Respond with exactly this JSON "
                "and nothing else:",
                '{ "title": "<≤8 words>", '
                '"description": "<3–5 sentences explaining the still-unaddressed gap>", '
                '"open_questions": "<one short sentence on what is still unknown, or empty>" }',
            ]
        )
        debug_log.log(f"REQUEST {MODEL} (re-evaluate {cluster.get('id', '?')})", prompt)
        try:
            response = self._client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            text = response.text.strip()
            debug_log.log(f"RESPONSE {MODEL} (re-evaluate)", text)
            data = json.loads(text)
            return {
                "title": data.get("title", fallback["title"]) or fallback["title"],
                "description": data.get("description", fallback["description"]) or fallback["description"],
                "open_questions": data.get("open_questions", ""),
            }
        except Exception as exc:
            print(f"  [llm]  problem re-evaluation failed: {exc}", file=sys.stderr)
            debug_log.log(f"ERROR {MODEL} (re-evaluate)", str(exc))
            return fallback

    def reassess_problem(
        self, cluster: dict, gt_path: Path, diffs: list[dict]
    ) -> dict:
        """Ask the fix model to re-describe or SPLIT a problem cluster after a failed fix.

        Returns {"problems": [{title, description, open_questions, mutants:[names]}]}.
        One entry means re-describe; multiple entries mean split. Falls back to a
        single-entry result preserving the original wording on any error."""
        mutant_names = [m["name"] for m in diffs]
        fallback = {
            "problems": [{
                "title": cluster.get("title", ""),
                "description": cluster.get("description", ""),
                "open_questions": cluster.get("open_questions", ""),
                "mutants": mutant_names,
            }]
        }
        gt_text = gt_path.read_text() if gt_path.exists() else ""
        examples: list[str] = []
        for i, m in enumerate(diffs, 1):
            info = parse_mutant_filename(m["name"])
            examples += [
                f"### Mutant {i}: `{m['name']}`",
                f"**Mutation applied:** {describe_operator(info['op'], info['arg'])}",
                "```diff",
                m["diff"].rstrip(),
                "```",
                "",
            ]
        prompt = "\n".join([
            "A spec fix attempt for the following problem FAILED — all mutants below "
            "still verify against the strengthened spec.",
            "Carefully reconsider whether these mutants all stem from the SAME root cause,",
            "or from MULTIPLE INDEPENDENT spec gaps that should be addressed separately.",
            "",
            f"Current problem: [{cluster.get('id', '?')}] {cluster.get('title', '')}",
            cluster.get("description", ""),
            (f"Previously open: {cluster.get('open_questions', '')}"
             if cluster.get("open_questions") else ""),
            "",
            f"Ground truth `{gt_path.name}`:",
            "```dafny",
            gt_text,
            "```",
            "",
            "Surviving mutants (still verify after the failed fix):",
            "",
            *examples,
            "If all mutants reveal the SAME spec gap, re-describe the problem more precisely.",
            "If they reveal DISTINCT spec gaps, SPLIT them into independent sub-problems.",
            "",
            f"Every mutant name must appear in exactly one sub-problem: {json.dumps(mutant_names)}",
            "",
            "Respond with exactly this JSON and nothing else:",
            '{"problems": [{"title": "<≤8 words>", "description": "<3–5 sentences>", '
            '"open_questions": "<one sentence or empty>", "mutants": ["<filename>", ...]}]}',
            "One entry = re-describe; multiple entries = split.",
        ])
        debug_log.log(f"REQUEST {FIX_MODEL} (reassess {cluster.get('id', '?')})", prompt)
        try:
            response = self._client.models.generate_content(
                model=FIX_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            text = response.text.strip()
            debug_log.log(f"RESPONSE {FIX_MODEL} (reassess)", text)
            data = json.loads(text)
            problems = data.get("problems", [])
            if not problems:
                return fallback
            valid_names = set(mutant_names)
            assigned: set[str] = set()
            for p in problems:
                p["mutants"] = [m for m in p.get("mutants", []) if m in valid_names]
                assigned.update(p["mutants"])
            # Any unassigned mutants go to the first sub-problem
            unassigned = valid_names - assigned
            if unassigned and problems:
                problems[0]["mutants"] = list(problems[0]["mutants"]) + sorted(unassigned)
            return {"problems": problems}
        except Exception as exc:
            print(f"  [llm]  problem reassessment failed: {exc}", file=sys.stderr)
            debug_log.log(f"ERROR {FIX_MODEL} (reassess)", str(exc))
            return fallback


# ---------------------------------------------------------------------------
# Spec-fix proposer (stronger model, spec clauses only)
# ---------------------------------------------------------------------------

_FIX_SYSTEM_INSTRUCTION = """\
You are a Dafny formal verification specialist. You are given a Dafny program whose \
specification is too weak: a class of behavioural mutations still verifies against it. \
Your job is to propose a specification action that tightens the spec so those mutations \
would no longer verify.

Hard constraints:
- You may ONLY modify spec clauses: `requires`, `ensures`, `invariant`, `decreases`, \
  `reads`, or `modifies`. NEVER modify executable code, method bodies, or expressions.
- Any clause you ADD or EDIT must be sound: it must hold for the original (correct) \
  program, so it rejects the mutants without rejecting correct behaviour.
- Identify the attachment point by quoting an EXISTING source line verbatim — the \
  declaration's signature line, or an existing clause — that the new clause should be \
  inserted immediately AFTER.
- The surviving mutants list may include "Conditions currently covering this point" — \
  use these to decide whether a method-level `ensures`/`requires` or a loop-level \
  `invariant` is the appropriate fix target.

Choose one of four actions:
- **add**: Insert a new spec clause after `anchor_line`.
- **edit**: Destructively replace the clause in a previous fix (set `target` to its id). \
  Use this when the previous clause was wrong and should be replaced, not supplemented.
- **revert**: Undo a previous fix entirely (set `target` to its id). Use this when a fix \
  was counter-productive or incorrect.
- **unresolved**: Flag this problem as requiring human judgement to narrow intended \
  behaviour. Use this **immediately, before attempting any patch**, when closing the gap \
  would require encoding an algorithm-level implementation choice that cannot soundly be \
  expressed as a `requires`/`ensures`/`invariant` — for example, constraining an \
  algorithm's internal strategy (O(log n) invariant, specific data-structure shape) rather \
  than its observable input/output contract. Fill `unresolved_reason` with 1–2 sentences \
  explaining why.

Default to `add` unless prior fix attempts were clearly wrong or the problem is unresolvable.

Respond with exactly this JSON and nothing else:
{
  "action": "add" | "edit" | "revert" | "unresolved",
  "unresolved_reason": "<why this cannot be closed by spec clauses alone, or null>",
  "target": "<fix id to edit/revert, e.g. F001, or null>",
  "anchor_line": "<an existing source line, copied verbatim, to insert after (required for add/edit)>",
  "kind": "requires" | "ensures" | "invariant" | "decreases" | "reads" | "modifies",
  "clause": "<the full clause text (required for add/edit)>",
  "description": "<1–2 sentences: what this clause asserts and why it closes the gap>"
}\
"""


class SpecFixer:
    """Stateful spec-fix proposer backed by the stronger FIX_MODEL."""

    def __init__(self):
        self._client = _make_client()

    def propose_fix(
        self,
        program_text: str,
        problem: dict,
        example_diffs: list[dict],
        top_fixes: list | None = None,
        conditions: dict | None = None,
    ) -> dict | None:
        """Ask for a spec action (add/edit/revert) that should kill `problem`'s mutants.
        top_fixes: up to 3 most-recently-applied fix entries (for edit/revert context).
        conditions: {"spec": [...], "loop_invariants": [...]} at the mutation point.
        Returns {action, target, anchor_line, kind, clause, description} or None on failure."""
        examples = []
        for i, m in enumerate(example_diffs, 1):
            info = parse_mutant_filename(m["name"])
            cond = m.get("conditions") or conditions
            cond_lines: list[str] = []
            if cond and (cond.get("spec") or cond.get("loop_invariants")):
                if cond.get("spec"):
                    cond_lines.append("  method spec: " + "; ".join(cond["spec"]))
                if cond.get("loop_invariants"):
                    cond_lines.append("  loop invariants: " + "; ".join(cond["loop_invariants"]))
            examples += [
                f"### Surviving mutant {i}: `{m['name']}`",
                f"**Mutation applied:** {describe_operator(info['op'], info['arg'])}",
                "```diff",
                m["diff"].rstrip(),
                "```",
            ]
            if cond_lines:
                examples.append("**Conditions currently covering this point:**")
                examples.extend(cond_lines)
            examples.append("")
        fixes_section: list[str] = []
        if top_fixes:
            fixes_section = [
                "Previously applied fixes (most recent first — you may edit or revert any):",
            ]
            for e in top_fixes:
                edits = e.get("edits", 0)
                edit_note = f"  (edited {edits}×)" if edits else ""
                fixes_section.append(
                    f"  {e['id']} · {e.get('kind', '')} {e.get('clause', '')[:80]}"
                    f"{edit_note}"
                )
            fixes_section.append("")
        prompt = "\n".join(
            [
                f"Specification problem: {problem.get('title', '')}",
                problem.get("description", ""),
                (f"Still unknown: {problem.get('open_questions', '')}"
                 if problem.get("open_questions") else ""),
                "",
                "Program (current spec):",
                "```dafny",
                program_text,
                "```",
                "",
                *fixes_section,
                f"These {len(example_diffs)} mutants still verify against the current spec "
                "(ground truth → mutant):",
                "",
                *examples,
                "Propose a specification action (add/edit/revert) that would make these "
                "mutants fail to verify.",
            ]
        )
        debug_log.log(f"REQUEST {FIX_MODEL} (fix · {problem.get('id', '?')})", prompt)
        try:
            response = self._client.models.generate_content(
                model=FIX_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_FIX_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                ),
            )
            text = response.text.strip()
            debug_log.log(f"RESPONSE {FIX_MODEL} (fix)", text)
            data = json.loads(text)
        except Exception as exc:
            print(f"  [llm]  fix proposal failed: {exc}", file=sys.stderr)
            debug_log.log(f"ERROR {FIX_MODEL} (fix)", str(exc))
            return None

        action = data.get("action", "add")

        # Unresolvable problem — flag immediately without attempting a patch
        if action == "unresolved":
            reason = (data.get("unresolved_reason") or "").strip()
            debug_log.log(f"UNRESOLVED {FIX_MODEL} (fix · {problem.get('id', '?')})",
                          reason)
            return {"action": "unresolved", "unresolved_reason": reason}

        target = (data.get("target") or "").strip() or None
        anchor = (data.get("anchor_line") or "").strip()
        clause = (data.get("clause") or "").strip()

        # Validate: edit/revert need a known target; add needs anchor+clause
        valid_ids = {e["id"] for e in (top_fixes or [])}
        if action in ("edit", "revert") and target not in valid_ids:
            action, target = "add", None   # fall back to add
        if action != "revert" and (not anchor or not clause):
            return None
        return {
            "action": action,
            "target": target,
            "anchor_line": anchor,
            "kind": data.get("kind", ""),
            "clause": clause,
            "description": data.get("description", ""),
        }
