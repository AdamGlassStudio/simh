#!/usr/bin/env python3
"""Generator for multiblock_movl_addl2.ini — MOVL + ADDL2 multi-block test."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from vax_encode import (
    MOVL, ADDL2,
    imm_long, reg,
    ini_header, deposit_bytes, ini_setup, ini_run_examine,
    R0, R1,
)

# ── Encoding ──────────────────────────────────────────────────────────────────
#
#   0x00-0x06: MOVL #1,R0   D0 8F 01 00 00 00 50  (7 bytes)
#   0x07-0x0D: ADDL2 #2,R0  C0 8F 02 00 00 00 50  (7 bytes)
#   0x0E:      ← breakpoint

movl_code  = MOVL( imm_long(1), reg(R0))
addl2_code = ADDL2(imm_long(2), reg(R0))
code = movl_code + addl2_code
breakpoint_addr = len(code)  # 0x0E = 14

# ── Sanity checks ─────────────────────────────────────────────────────────────

assert movl_code  == bytes([0xD0, 0x8F, 0x01, 0x00, 0x00, 0x00, 0x50]), "MOVL wrong"
assert addl2_code == bytes([0xC0, 0x8F, 0x02, 0x00, 0x00, 0x00, 0x50]), "ADDL2 wrong"
assert breakpoint_addr == 0x0E, f"breakpoint should be 0x0E, got {breakpoint_addr:#x}"

# ── Output ────────────────────────────────────────────────────────────────────

lines = ini_header('multiblock_movl_addl2.ini', 'gen_multiblock_movl_addl2.py')
lines += [
    '; Multi-instruction block: MOVL #1,R0 then ADDL2 #2,R0 \u2192 R0=3',
    '; Uses a breakpoint at address 14 (after both instructions) so that',
    '; both JIT and interpreter modes stop at the same PC with R0=3.',
    ';',
    '; MOVL #1,R0   : D0 8F 01 00 00 00 50  (7 bytes, addr 0-6)',
    '; ADDL2 #2,R0  : C0 8F 02 00 00 00 50  (7 bytes, addr 7-13)',
    '; Breakpoint    : addr 14 = 0xe',
    '',
]

lines += ini_setup(regs_to_zero=[0, 1], pc=0)
lines += deposit_bytes(code, base_addr=0)
lines += ini_run_examine(breakpoint_addr=breakpoint_addr)

print('\n'.join(lines))
