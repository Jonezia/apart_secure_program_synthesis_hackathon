#!/usr/bin/env bash
# populate.sh <directory>
#
# Three-phase population strategy:
#   Phase 1: Harvest mutant cache for the first 10 files (no analysis).
#   Phase 2: Full analysis (LLM + refinement) for those same 10 files.
#   Phase 3: Harvest mutant cache for all remaining files (no analysis).
#
# The launcher exits automatically after each phase completes.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: populate.sh <directory>" >&2
    exit 1
fi

DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$SCRIPT_DIR/launcher.py"

if [[ ! -d "$DIR" ]]; then
    echo "error: '$DIR' is not a directory" >&2
    exit 1
fi

echo "=== Phase 1: mutant cache — first 10 files ==="
python3 "$LAUNCHER" --mutant-only -N10 "$DIR"

echo "=== Phase 2: full analysis — first 10 files ==="
python3 "$LAUNCHER" -N10 "$DIR"

echo "=== Phase 3: mutant cache — remaining files ==="
python3 "$LAUNCHER" --mutant-only "$DIR"

echo "=== populate.sh done ==="
