#!/usr/bin/env bash
# run_representatives.sh [--human] [--mutant-only] [extra launcher flags...]
#
# Invokes the launcher on every file listed in representative_scripts.txt.
# Comment/blank/hash lines in the txt file are ignored.
# All arguments are forwarded to launcher.py unchanged.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIST="$SCRIPT_DIR/representative_scripts.txt"
LAUNCHER="$SCRIPT_DIR/launcher.py"

if [[ ! -f "$LIST" ]]; then
    echo "error: $LIST not found" >&2
    exit 1
fi

# Collect non-comment, non-blank lines as file paths
files=()
while IFS= read -r line || [[ -n "$line" ]]; do
    # Strip inline comments and surrounding whitespace
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" ]] && continue
    # Resolve relative paths against the script directory
    if [[ "$line" != /* ]]; then
        line="$SCRIPT_DIR/$line"
    fi
    if [[ ! -f "$line" ]]; then
        echo "warn: $line not found, skipping" >&2
        continue
    fi
    files+=("$line")
done < "$LIST"

if [[ ${#files[@]} -eq 0 ]]; then
    echo "error: no valid files found in $LIST" >&2
    exit 1
fi

echo "Running launcher on ${#files[@]} representative file(s):"
for f in "${files[@]}"; do
    echo "  $f"
done
echo

python3 "$LAUNCHER" "$@" "${files[@]}"
