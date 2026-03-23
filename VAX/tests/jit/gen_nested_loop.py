#!/usr/bin/env python3
"""Generator for nested_loop.ini — nested counted loops via BGTR back-edges."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from vax_encode import (
    CLRL, MOVL, INCL, DECL, BGTR,
    short_lit, reg,
    ini_header, deposit_bytes, ini_setup, ini_run_examine,
    R0, R1, R2, R3,
)

# ── Encoding (pc_counter approach) ────────────────────────────────────────────
#
# Branch displacement = target - (branch_instr_pc + 2)
# Because the 2-byte branch instruction advances PC past both opcode and disp.
#
#   0x00: D4 50       CLRL  R0
#   0x02: D0 05 52    MOVL  #5, R2           ← outer loop init
#   0x05: D0 03 53    MOVL  #3, R3           ← outer_top (inner reset)
#   0x08: D6 50       INCL  R0               ← inner_top
#   0x0A: D7 53       DECL  R3
#   0x0C: 14 FA       BGTR  inner_top        disp = 0x08 - 0x0E = -6 = 0xFA
#   0x0E: D7 52       DECL  R2
#   0x10: 14 F3       BGTR  outer_top        disp = 0x05 - 0x12 = -13 = 0xF3
#   0x12:             ← breakpoint

pc = 0
instructions = []

clrl_r0   = CLRL(reg(R0))          ; addr_clrl   = pc; pc += len(clrl_r0)
movl_5_r2 = MOVL(short_lit(5), reg(R2)); addr_outer_init = pc; pc += len(movl_5_r2)
outer_top = pc
movl_3_r3 = MOVL(short_lit(3), reg(R3)); pc += len(movl_3_r3)
inner_top = pc
incl_r0   = INCL(reg(R0))          ; pc += len(incl_r0)
decl_r3   = DECL(reg(R3))          ; pc += len(decl_r3)
bgtr_inner = BGTR(inner_top - (pc + 2)); pc += 2   # branch instr is always 2 bytes
decl_r2   = DECL(reg(R2))          ; pc += len(decl_r2)
bgtr_outer = BGTR(outer_top  - (pc + 2)); pc += 2
breakpoint_addr = pc

code = (
    clrl_r0 + movl_5_r2 + movl_3_r3 +
    incl_r0 + decl_r3 + bgtr_inner +
    decl_r2 + bgtr_outer
)

# ── Sanity checks ─────────────────────────────────────────────────────────────

assert code[0x00:0x02] == bytes([0xD4, 0x50]), "CLRL R0 wrong"
assert code[0x02:0x05] == bytes([0xD0, 0x05, 0x52]), "MOVL #5,R2 wrong"
assert code[0x05:0x08] == bytes([0xD0, 0x03, 0x53]), "MOVL #3,R3 wrong"
assert code[0x08:0x0A] == bytes([0xD6, 0x50]), "INCL R0 wrong"
assert code[0x0A:0x0C] == bytes([0xD7, 0x53]), "DECL R3 wrong"
assert code[0x0C:0x0E] == bytes([0x14, 0xFA]), "BGTR inner_top wrong"
assert code[0x0E:0x10] == bytes([0xD7, 0x52]), "DECL R2 wrong"
assert code[0x10:0x12] == bytes([0x14, 0xF3]), "BGTR outer_top wrong"
assert breakpoint_addr == 0x12, f"breakpoint should be 0x12, got {breakpoint_addr:#x}"

# ── Output ────────────────────────────────────────────────────────────────────

lines = ini_header('nested_loop.ini', 'gen_nested_loop.py')
lines += [
    '; nested_loop.ini — nested counted loops via BGTR back-edges',
    ';',
    '; Outer loop (R2): counts 5 \u2192 0  (5 iterations)',
    '; Inner loop (R3): counts 3 \u2192 0  (3 iterations per outer)',
    '; R0: incremented on every inner iteration \u2192 expected R0 = 5\xd73 = 15 = 0xF',
    ';',
    '; Layout (all at PC=0):',
    ';   0x00: D4 50       CLRL  R0            ; clear accumulator',
    ';   0x02: D0 05 52    MOVL  #5, R2        ; outer count = 5',
    ';   0x05: D0 03 53    MOVL  #3, R3        ; inner count = 3  \u2190 outer_top',
    ';   0x08: D6 50       INCL  R0            ;                  \u2190 inner_top',
    ';   0x0A: D7 53       DECL  R3',
    ';   0x0C: 14 FA       BGTR  inner_top     ; target=0x08, disp=8-14=-6=0xFA',
    ';   0x0E: D7 52       DECL  R2',
    ';   0x10: 14 F3       BGTR  outer_top     ; target=0x05, disp=5-18=-13=0xF3',
    ';   0x12:             \u2190 breakpoint',
    ';',
    '; Region JIT should capture all 8 instructions as one block with two',
    '; conditional back-edges (inner_top and outer_top).',
    '',
]

lines += ini_setup(regs_to_zero=[0, 1, 2, 3], pc=0)
lines += deposit_bytes(code, base_addr=0)
lines += ini_run_examine(breakpoint_addr=breakpoint_addr)

print('\n'.join(lines))
