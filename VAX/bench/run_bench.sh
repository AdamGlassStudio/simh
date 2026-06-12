#!/bin/bash
# run_bench.sh — wall-clock JIT vs interpreter on microbenchmarks.
#
# Each bench_*.ini is run twice (set cpu nojit, set cpu jit) under
# /usr/bin/time. After execution, "show cpu jitstats" is appended so the
# JIT-mode run reports blocks_run / cache_hits / insns_jit / insns_interp.
#
# Output: one summary table to stdout; per-run full simh output saved in
# the bench tmpdir for follow-up inspection.
#
# Usage: ./run_bench.sh [path/to/vax-binary] [bench-dir]

set -u

SIMH="${1:-./BIN/vax}"
BENCH_DIR="${2:-VAX/bench}"

if [ ! -x "$SIMH" ]; then
    echo "ERROR: vax binary not found or not executable: $SIMH" >&2
    exit 1
fi

if ! command -v /usr/bin/time >/dev/null 2>&1; then
    echo "ERROR: /usr/bin/time required (the shell builtin doesn't support -f)" >&2
    exit 1
fi

OUTDIR="$(mktemp -d -t vaxbench.XXXXXX)"
echo "# raw output kept in: $OUTDIR"
echo ""

printf "%-22s %-6s %10s %16s %16s %10s %10s\n" \
    "test" "mode" "wall_s" "insns_jit" "insns_interp" "blocks" "cache_hit"
printf "%-22s %-6s %10s %16s %16s %10s %10s\n" \
    "----" "----" "------" "---------" "------------" "------" "---------"

for ini in "$BENCH_DIR"/bench_*.ini; do
    [ -f "$ini" ] || continue
    name=$(basename "$ini" .ini)
    declare -A wall jit interp blocks hits

    for mode in nojit jit; do
        out="$OUTDIR/${name}.${mode}.out"
        timef="$OUTDIR/${name}.${mode}.time"

        # Inject `set cpu MODE` first, then test body, then jitstats query.
        { echo "set cpu $mode"; cat "$ini"; echo "show cpu jitstats"; } \
            | /usr/bin/time -f '%e' -o "$timef" "$SIMH" /dev/null > "$out" 2>&1

        wall[$mode]=$(cat "$timef" 2>/dev/null || echo "?")
        jit[$mode]=$(grep -oE 'Insns via JIT:[[:space:]]+[0-9]+' "$out" | awk '{print $NF}')
        interp[$mode]=$(grep -oE 'Insns interpreter:[[:space:]]+[0-9]+' "$out" | awk '{print $NF}')
        blocks[$mode]=$(grep -oE 'Blocks run:[[:space:]]+[0-9]+' "$out" | awk '{print $NF}')
        hits[$mode]=$(grep -oE 'Cache hits:[[:space:]]+[0-9]+' "$out" | awk '{print $NF}')

        printf "%-22s %-6s %10s %16s %16s %10s %10s\n" \
            "$name" "$mode" "${wall[$mode]}" \
            "${jit[$mode]:-0}" "${interp[$mode]:-0}" \
            "${blocks[$mode]:-0}" "${hits[$mode]:-0}"
    done

    # Speedup line (interpreter / jit), only if both numbers parseable.
    if [[ "${wall[nojit]}" =~ ^[0-9.]+$ ]] && [[ "${wall[jit]}" =~ ^[0-9.]+$ ]] \
       && [ "$(echo "${wall[jit]} > 0" | bc -l 2>/dev/null)" = "1" ]; then
        speedup=$(echo "scale=2; ${wall[nojit]} / ${wall[jit]}" | bc -l)
        printf "%-22s %-6s %10s\n" "$name" "speedup" "${speedup}x"
    fi
    echo ""
    unset wall jit interp blocks hits
done

echo "# raw simh output: $OUTDIR/*.out"
echo "# raw timings:     $OUTDIR/*.time"
