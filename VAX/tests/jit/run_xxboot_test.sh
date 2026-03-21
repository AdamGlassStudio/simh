#!/bin/bash
# run_xxboot_test.sh: run xxboot to the 'relocated' breakpoint under JIT and
# interpreter and verify register/PSL state is identical.
#
# Usage: run_xxboot_test.sh <vax-binary> [xxboot-bin]
#   vax-binary  : path to BIN/vax
#   xxboot-bin  : path to raw xxboot binary (default: /tmp/xxboot_gcc.bin)

set -euo pipefail

VAX="${1:-BIN/vax}"
XXBOOT="${2:-/tmp/xxboot_gcc.bin}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INI="$SCRIPT_DIR/xxboot_relocated.ini"

if [ ! -f "$VAX" ]; then
    echo "ERROR: vax binary not found: $VAX" >&2
    exit 1
fi

if [ ! -f "$XXBOOT" ]; then
    echo "SKIP  xxboot_relocated (xxboot binary not found: $XXBOOT)"
    exit 0
fi

run_sim() {
    local mode="$1"
    printf "set cpu %s\ndo %s\n" "$mode" "$INI" \
        | "$VAX" 2>/dev/null \
        | grep -E "^R[0-9]|^PSL:"
}

OUT_NOJIT=$(run_sim nojit)
OUT_JIT=$(run_sim jit)

if diff <(echo "$OUT_NOJIT") <(echo "$OUT_JIT") >/dev/null 2>&1; then
    echo "PASS  xxboot_relocated"
    echo "      PC=0x10024A (relocated), SP=0x${OUT_NOJIT##*R14:	}"
    exit 0
else
    echo "FAIL  xxboot_relocated"
    diff <(echo "$OUT_NOJIT") <(echo "$OUT_JIT") | sed 's/^/      /'
    exit 1
fi
