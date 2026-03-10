#!/bin/bash
# run_jit_tests.sh — run each *.ini test twice (JIT on/off), diff register state.
#
# Usage: ./run_jit_tests.sh [path/to/vax-binary] [test-dir]
#
# Each .ini file contains:
#   - register/memory deposits (initial state)
#   - step / go commands to execute instruction(s)
#   - examine commands for every register to snapshot
# This script prepends "set cpu jit" or "set cpu nojit" before running.
#
# PASS: JIT and interpreter produce identical register+PSL output.
# FAIL: any difference between the two runs.

SIMH="${1:-./BIN/vax}"
TESTDIR="${2:-VAX/tests/jit}"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

PASS=0
FAIL=0

run_test() {
    local ini="$1"
    local mode="$2"      # jit | nojit
    local out="$3"

    # Inject the jit flag as the very first command, then append the test body.
    { echo "set cpu $mode"; cat "$ini"; } \
        | "$SIMH" /dev/null 2>&1 \
        | grep -E '^R[0-9]+:[[:space:]]|^PSL:[[:space:]]' \
        | awk '{print $1, $2}' \
        > "$out"
}

for ini in "$TESTDIR"/*.ini; do
    name="$(basename "$ini" .ini)"
    jit_out="$TMPDIR/${name}.jit"
    nonjit_out="$TMPDIR/${name}.nonjit"

    run_test "$ini" "jit"   "$jit_out"
    run_test "$ini" "nojit" "$nonjit_out"

    if diff -q "$jit_out" "$nonjit_out" > /dev/null 2>&1; then
        echo "PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL  $name"
        diff "$nonjit_out" "$jit_out" | sed 's/^/      /'
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
