#!/usr/bin/env bash
#
# ------------------------------------------------------------------------------
# This script generates mutants of a given Dafny program. By default, one mutation
# is applied per program, following the principles of mutation testing. We also
# provide an option to apply more than one mutation to the same program, which 
# can be useful for a variety of use cases.
#
# Usage:
# run.sh
#   <full path to the program under test, e.g., $SCRIPT_DIR/../DafnyBench/DafnyBench/dataset/ground_truth/630-dafny_tmp_tmpz2kokaiq_Solution.dfy> 
#   [--num_mutations <the number of mutations to apply to the input program, e.g., 1 (by default)>]
#   [help]
# ------------------------------------------------------------------------------ General utils

die() {
  echo "$@" >&2
  exit 1
}

# ------------------------------------------------------------------------------ Args

USAGE="Usage: ${BASH_SOURCE[0]}
   <full path to the program under test, e.g., $SCRIPT_DIR/../DafnyBench/DafnyBench/dataset/ground_truth/630-dafny_tmp_tmpz2kokaiq_Solution.dfy>
   [--num_mutations <the number of mutations to apply to the input program, e.g., 1 (by default)>]
   [help]"
if [ "$#" -ne "1" ] && [ "$#" -ne "3" ]; then
  die "$USAGE"
fi

if [ "$#" -eq "1" ] && [ "$1" = "--help" ]; then
    echo "$USAGE"
    exit 0
fi

PROGRAM=$1
NUM_MUTS=1
if [ "$#" -eq "3" ]; then
    NUM_MUTS=$3
fi

# Source file stem — mutants for this program are cached under mutants/<cat>/$STEM/
# so the persistence_manager can track each file independently.
STEM=$(basename "$PROGRAM" .dfy)

# ------------------------------------------------------------------------------ Config

# Tunable knobs for the formal equivalent-mutant filter (see equiv_check below).
[ -f equiv.env ] && . ./equiv.env

# Per-mutant verification wall-clock budget (seconds, per procedure). Without it a
# mutant that makes Z3 spin would hang the whole run indefinitely; with it such a
# mutant is reported as "time out" and routed to mutants/timed-out. Override via the
# MUT_VERIFY_TIME_LIMIT environment variable (0 = no limit, the old hanging behaviour).
MUT_VERIFY_TIME_LIMIT="${MUT_VERIFY_TIME_LIMIT:-10}"

# Hard wall-clock backstop (seconds) around every Dafny call, via coreutils `timeout`.
# This is the ONLY bound on the plugin's mutation/PreResolve phase: --verification-time-limit
# governs Z3 only, so a mutator that stalls while generating a mutant (before verification)
# can only be stopped here. Keep it just above a normal mutate+verify (dotnet startup +
# mutation + a few parallel verify batches at MUT_VERIFY_TIME_LIMIT). 0 disables.
DAFNY_HARD_TIMEOUT="${DAFNY_HARD_TIMEOUT:-20}"    # whole mut+verify / equiv verify call
EQUIV_GEN_TIMEOUT="${EQUIV_GEN_TIMEOUT:-15}"      # equiv harness generation (resolve)
_TIMEOUT_BIN="$(command -v timeout || true)"

# with_timeout <seconds> <cmd...> — run cmd under a hard SIGKILL timeout if available.
with_timeout() {
    local secs="$1"; shift
    if [ -n "$_TIMEOUT_BIN" ] && [ "$secs" != "0" ]; then
        "$_TIMEOUT_BIN" --signal=KILL "$secs" "$@"
    else
        "$@"
    fi
}

# Progress side-channel. The main loop captures each target's stdout via command
# substitution (output=$(single_mutation ...)), so nothing reaches a watching TUI
# until a target finishes — during a slow verify the screen looks frozen. set_status
# writes the current activity to a file directly (a redirect still fires inside command
# substitution), and the launcher polls it + the file mtime to show "what we're waiting
# on" and for how long. Override path via MUTDAFNY_STATUS_FILE; empty disables.
MUTDAFNY_STATUS_FILE="${MUTDAFNY_STATUS_FILE:-mutants/.mutdafny_status}"
set_status() {
    [ -n "$MUTDAFNY_STATUS_FILE" ] && printf '%s' "$*" > "$MUTDAFNY_STATUS_FILE" 2>/dev/null
}

# ------------------------------------------------------------------------------ Cleanup

# When MUT_PRELOADED is set the persistence_manager has already written targets.csv
# (only the not-yet-consumed targets) and we must NOT scan or clobber it.
[ -z "${MUT_PRELOADED}" ] && rm -rf targets.csv
mkdir -p mutants
mkdir -p mutants/alive
mkdir -p mutants/timed-out
mkdir -p mutants/killed
mkdir -p mutants/invalid
mkdir -p mutants/equivalent
mkdir -p "mutants/alive/$STEM" "mutants/killed/$STEM" "mutants/timed-out/$STEM" \
         "mutants/invalid/$STEM" "mutants/equivalent/$STEM"

# ------------------------------------------------------------------------------ MutDafny utils

scan_program() {
    echo Scanning $PROGRAM for mutation targets
    set_status "scanning for mutation targets"
    with_timeout "${DAFNY_HARD_TIMEOUT}" \
        dotnet ./dafny/Binaries/Dafny.dll verify $PROGRAM \
        --solver-path ./dafny/Binaries/z3 --allow-warnings \
        --plugin ./mutdafny/bin/Debug/net8.0/mutdafny.dll,scan > /dev/null
}

single_mutation() {
    local pos="$1"
    local op="$2"
    local arg="$3"

    local plugin_arg="mut $pos $op"
    [ -n "$arg" ] && plugin_arg="mut $pos $op $arg"
    if [[ -z $arg ]]; then
        echo Mutating position $pos: operator $op
    else
        echo Mutating position $pos: operator $op, argument $arg
    fi
    output=$(with_timeout "${DAFNY_HARD_TIMEOUT}" \
        dotnet ./dafny/Binaries/Dafny.dll verify $PROGRAM \
        --solver-path ./dafny/Binaries/z3 --allow-warnings \
        --verification-time-limit "$MUT_VERIFY_TIME_LIMIT" \
        --plugin ./mutdafny/bin/Debug/net8.0/mutdafny.dll,"$plugin_arg" 2>/dev/null)
    local rc=$?
    echo $output
    # `timeout` exits 124/137 when it kills a stalled call. This covers BOTH the
    # mutation/PreResolve phase (no Z3, no "time out" string) and a verification
    # phase that the soft --verification-time-limit didn't catch in time.
    if [ "$rc" = "124" ] || [ "$rc" = "137" ]; then echo "__MUTDAFNY_TIMEOUT__"; fi
}

multiple_mutation() {
    echo Applying $NUM_MUTS mutations to the program
    output=$(with_timeout "${DAFNY_HARD_TIMEOUT}" \
        dotnet ./dafny/Binaries/Dafny.dll verify $PROGRAM \
        --solver-path ./dafny/Binaries/z3 --allow-warnings \
        --verification-time-limit "$MUT_VERIFY_TIME_LIMIT" \
        --plugin ./mutdafny/bin/Debug/net8.0/mutdafny.dll,"mut $NUM_MUTS" 2>/dev/null)
    local rc=$?
    echo $output
    if [ "$rc" = "124" ] || [ "$rc" = "137" ]; then echo "__MUTDAFNY_TIMEOUT__"; fi
}

# Formal behavioural-equivalence gate for an alive mutant.
# Generates an equivalence harness (original vs mutated changed declaration) and
# verifies only the EquivCheck__ lemma. Echoes "equivalent" if equivalence is
# *proved* (false positive -> discard), otherwise "alive".
# Args: <mutant_file> <pos> <op> <arg>
equiv_check() {
    local mutant="$1" pos="$2" op="$3" arg="$4"
    if [ "${EQUIV_ENABLED:-0}" != "1" ]; then echo alive; return; fi

    local base="${mutant%.dfy}"
    local harness="${base}.equiv.dfy"
    rm -f "$harness" "${base}.equiv.skip"

    # Build the harness from the original program + this single mutation.
    local plugin_arg="equiv $pos $op"
    [ -n "$arg" ] && plugin_arg="equiv $pos $op $arg"
    set_status "equiv-gen: ${pos} ${op} ${arg}"
    with_timeout "${EQUIV_GEN_TIMEOUT}" \
        dotnet ./dafny/Binaries/Dafny.dll resolve "$PROGRAM" \
        --allow-warnings \
        --plugin ./mutdafny/bin/Debug/net8.0/mutdafny.dll,"$plugin_arg" >/dev/null 2>&1

    if [ ! -f "$harness" ]; then
        # Unsupported shape, generation failure, or a hung/killed resolve (timeout):
        # conservatively keep the mutant alive.
        rm -f "${base}.equiv.skip"
        echo alive; return
    fi

    # Verify only the equivalence lemma, with the configured budget.
    local rlimit_flag=""
    [ "${EQUIV_RLIMIT:-0}" != "0" ] && rlimit_flag="--resource-limit ${EQUIV_RLIMIT}"
    local eout
    set_status "equiv-verify: ${pos} ${op} ${arg}"
    eout=$(with_timeout "${DAFNY_HARD_TIMEOUT}" \
        dotnet ./dafny/Binaries/Dafny.dll verify "$harness" \
        --solver-path ./dafny/Binaries/z3 --allow-warnings \
        --filter-symbol "${EQUIV_FILTER_SYMBOL:-EquivCheck__}" \
        --verification-time-limit "${EQUIV_TIME_LIMIT:-15}" \
        --cores "${EQUIV_CORES:-4}" $rlimit_flag 2>&1)

    # Proved equivalent iff a positive number of symbols verified with no errors.
    # ("0 verified, 0 errors" from a bad filter must NOT count as a proof.)
    if echo "$eout" | grep -qE "finished with [1-9][0-9]* verified, 0 error"; then
        mkdir -p "mutants/equivalent/$STEM"
        echo "$eout" > "mutants/equivalent/$STEM/${base}.equiv.txt"
        mv "$harness" "mutants/equivalent/$STEM"/ 2>/dev/null
        echo equivalent; return
    fi

    # Not proved (counterexample / timeout / inconclusive): keep alive, save the evidence.
    mkdir -p "mutants/alive/$STEM/equiv-checks"
    echo "$eout" > "mutants/alive/$STEM/equiv-checks/${base}.equiv.txt"
    rm -f "$harness"
    echo alive
}

process_output() {
    local output="$1"

    verification_finished=$(echo $output | grep "Dafny program verifier finished")
    verified=$(echo $output | grep "Dafny program verifier finished.*0 errors")
    # Timeouts come in two flavours, both routed to mutants/timed-out:
    #  - verification (soft): Dafny prints "... time out" and finishes normally.
    #  - mutation OR verification (hard): the wall-clock `timeout` killed the call,
    #    which single_mutation/multiple_mutation flag with __MUTDAFNY_TIMEOUT__.
    soft_timeout=$(echo $output | grep "Dafny program verifier finished.*time out")
    hard_timeout=$(echo $output | grep "__MUTDAFNY_TIMEOUT__")
    timed_out="${soft_timeout}${hard_timeout}"
    output=$(echo $output | tail -1)

    COLOR='\033[0;31m'; if [[ -n $verified ]]; then COLOR='\033[0m'; fi
    if [[ -n $timed_out ]]; then
        echo Mutation or verification timed out
        mkdir -p "mutants/timed-out/$STEM"
        if [ -f *.dfy ]; then
            mv *.dfy "mutants/timed-out/$STEM"/
        else
            # Hard timeout during the mutation phase: no mutant was generated.
            # Record a marker so the timeout is still counted (and traceable).
            local marker="${STEM}__timeout_$$_${RANDOM}"
            [ -n "$pos" ] && marker="${STEM}__${pos}_${op}_${arg}"
            echo "// timed out after ${DAFNY_HARD_TIMEOUT}s (no mutant produced)" \
                > "mutants/timed-out/$STEM/${marker}.dfy"
        fi
    elif [[ -z $verification_finished ]]; then # verification did not finish due to invalid program

        if [ -f *.dfy ]; then
            echo Error: mutant is invalid
            mkdir -p "mutants/invalid/$STEM"
            mv *.dfy "mutants/invalid/$STEM"/
        else
            echo Could not apply $NUM_MUTS mutations to the program
        fi

    elif [ -f *.dfy ]; then
        echo -e "${COLOR}${output}\033[0m"
        output_dir=""
        if [[ -n $verified ]]; then
            mutant_file=$(ls *.dfy 2>/dev/null | head -1)
            verdict=alive
            # Only single-mutation runs expose one changed declaration to compare.
            if [[ -n "$pos" && -n "$mutant_file" ]]; then
                verdict=$(equiv_check "$mutant_file" "$pos" "$op" "$arg")
            fi
            if [ "$verdict" = "equivalent" ]; then
                echo Verification succeeded but mutant is a proved behavioural equivalent: discarding to mutants/equivalent
                output_dir="mutants/equivalent/$STEM"
            else
                echo Verification succeeded: mutant is alive
                output_dir="mutants/alive/$STEM"
            fi
        else
            echo Verification failed: mutant was killed
            output_dir="mutants/killed/$STEM"
        fi

        mkdir -p "$output_dir"
        mv *.dfy "$output_dir"/
    else
        echo Could not apply $NUM_MUTS mutations to the program
    fi
}

# ------------------------------------------------------------------------------ Main

# Skip scanning when the manager preloaded targets.csv (warm start).
[ -z "${MUT_PRELOADED}" ] && scan_program

IFS=','
if [ "$#" -eq "1" ] || [ $NUM_MUTS -eq "1" ]; then
    TARGET_TOTAL=$(wc -l < targets.csv 2>/dev/null || echo 0)
    TARGET_IDX=0
    while read pos op arg;
    do
        TARGET_IDX=$((TARGET_IDX + 1))

        # A previous Dafny call killed by the hard timeout can leave a stale mutant
        # in the root; if it lingers, the `*.dfy` glob matches two files and
        # process_output misfires as "could not apply". Clear it before generating.
        rm -f ./*.dfy 2>/dev/null

        set_status "verify ${TARGET_IDX}/${TARGET_TOTAL}: ${pos} ${op} ${arg}"
        output=$(single_mutation $pos $op $arg)
        mutant_type_msg=$(echo $output | head -n 1)
        echo $mutant_type_msg
        mutant_outcome_msg=$(process_output "$output")
        echo $mutant_outcome_msg
        echo
        rm -f elapsed-time.csv

        # Record this target as consumed so the manager can resume past it next time.
        [ -n "${MUT_CONSUMED_LOG}" ] && printf '%s,%s,%s\n' "$pos" "$op" "$arg" >> "${MUT_CONSUMED_LOG}"

    done < targets.csv
    set_status "done"
else
    num_targets=$(wc -l < targets.csv)
    MAX_TRIES=$(($num_targets / $NUM_MUTS * 5)) # 5 tries per mutant
    NUM_TRIES=0
    while [ $(wc -l < targets.csv 2>/dev/null || echo 0) -ge $NUM_MUTS ] && [ $NUM_TRIES -lt $MAX_TRIES ];
    do

        rm -f ./*.dfy 2>/dev/null

        set_status "verify (multi ${NUM_MUTS}x): $(wc -l < targets.csv 2>/dev/null || echo 0) targets left"
        output=$(multiple_mutation)
        mutant_type_msg=$(echo $output | head -n 1)
        echo $mutant_type_msg
        mutant_outcome_msg=$(process_output "$output")
        echo $mutant_outcome_msg
        echo

        could_not_apply_all_muts=$(echo $mutant_outcome_msg | grep "Could not apply $NUM_MUTS mutations to the program")
        if [[ -n $could_not_apply_all_muts ]]; then
            NUM_TRIES=$((NUM_TRIES+1))
        else
            NUM_TRIES=0
            num_targets=$(wc -l < targets.csv)
            MAX_TRIES=$(($num_targets / $NUM_MUTS * 5))
        fi
        rm -f elapsed-time.csv

    done

    line_count=$(wc -l < targets.csv 2>/dev/null || echo 0)
    if [ "$line_count" -lt "$NUM_MUTS" ]; then
        echo "Consumed all targets"
    else
        echo "Reached max combination tries"
    fi
    set_status "done"
fi

# Leave a preloaded targets.csv in place: the manager reads the leftover (in multi
# mode the plugin shrinks it) to update consumed state, and removes it in finalize_run.
[ -z "${MUT_PRELOADED}" ] && rm -f targets.csv