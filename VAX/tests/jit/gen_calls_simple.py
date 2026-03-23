#!/usr/bin/env python3
"""Generator for calls_simple.ini — CALLS/RET frame setup and teardown test."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from vax_encode import (
    MOVL, CALLS, RET, entry_mask,
    short_lit, reg, w_disp_pc,
    ini_header, deposit_bytes, ini_setup, ini_run_examine,
    R0, R1, R6,
)

# ── Encoding ──────────────────────────────────────────────────────────────────
#
# Layout:
#   0x00: D0 2A 56     MOVL #42, R6          (short literal 42 = 0x2A, R6)
#   0x03: D0 05 51     MOVL #5, R1           (short literal 5, R1)
#   0x06: FB 00 CF     CALLS #0, W^0x20      (5 bytes total)
#         15 00        word disp = 0x15; PC_after = 0x0B; target = 0x0B+0x15 = 0x20 ✓
#   0x0B: ← breakpoint
#
#   0x20: 40 00        subroutine entry mask (save R6, bit 6 set)
#   0x22: D0 07 56     MOVL #7, R6           (modify R6 in callee → restored by RET)
#   0x25: D0 07 50     MOVL #7, R0           (return value)
#   0x28: 04           RET
#
# CALLS displacement: target=0x20, PC_after_operand=0x0B → disp = 0x20 - 0x0B = 0x15

CALLEE_ADDR = 0x20
CALLS_PC_AFTER = 0x0B   # byte after the 5-byte CALLS instruction (0x06 + 5 = 0x0B)
CALLS_DISP = CALLEE_ADDR - CALLS_PC_AFTER  # 0x15

main_code = (
    MOVL(short_lit(42), reg(R6)) +   # 0x00: D0 2A 56
    MOVL(short_lit(5),  reg(R1)) +   # 0x03: D0 05 51
    CALLS(short_lit(0), w_disp_pc(CALLS_DISP))  # 0x06: FB 00 CF 15 00
)

sub_code = (
    entry_mask(0x0040) +             # 0x20: 40 00  (save R6, bit 6)
    MOVL(short_lit(7), reg(R6)) +    # 0x22: D0 07 56
    MOVL(short_lit(7), reg(R0)) +    # 0x25: D0 07 50
    RET()                            # 0x28: 04
)

# ── Sanity checks ─────────────────────────────────────────────────────────────

assert main_code[0:3]  == bytes([0xD0, 0x2A, 0x56]), "MOVL #42,R6 wrong"
assert main_code[3:6]  == bytes([0xD0, 0x05, 0x51]), "MOVL #5,R1 wrong"
assert main_code[6:11] == bytes([0xFB, 0x00, 0xCF, 0x15, 0x00]), "CALLS encoding wrong"
assert len(main_code)  == 11, f"main_code should be 11 bytes, got {len(main_code)}"
assert sub_code[0:2]   == bytes([0x40, 0x00]), "entry mask wrong"
assert w_disp_pc(0x15) == bytes([0xCF, 0x15, 0x00]), "w_disp_pc wrong"

# ── Output ────────────────────────────────────────────────────────────────────

lines = ini_header('calls_simple.ini', 'gen_calls_simple.py')
lines += [
    '; calls_simple.ini — CALLS/RET frame setup and teardown',
    ';',
    '; Tests that CALLS correctly builds the VAX call frame and RET tears it down:',
    ';   - Entry mask (save R6) is read from callee_addr',
    ';   - R6 is saved in the frame and restored by RET (callee modifies it)',
    ';   - R0 set in callee (not saved by mask) survives the return',
    ';   - SP returns to its pre-CALLS value after RET',
    ';   - AP/FP updated correctly during CALLS, restored by RET',
    ';',
    '; Layout:',
    ';   0x00: D0 2A 56     MOVL #42, R6          ; R6 = 42 (short literal, saved/restored by RET)',
    ';   0x03: D0 05 51     MOVL #5, R1           ; R1 = 5 (untouched by call)',
    ';   0x06: FB 00 CF     CALLS #0, W^0x20      ; (5 bytes total: opcode+argcount+spec+disp2)',
    ';         15 00        ; word disp = 0x15 (target 0x20, pc_after = 0x0B)',
    ';   0x0B: \u2190 breakpoint',
    ';',
    ';   0x20: 40 00        ; subroutine entry mask (save R6, bit 6)',
    ';   0x22: D0 07 56     MOVL #7, R6           ; modify R6 in callee (should be restored to 42)',
    ';   0x25: D0 07 50     MOVL #7, R0           ; R0 = 7 (return value, not saved by mask)',
    ';   0x28: 04           RET',
    ';',
    '; Expected: R0=7, R1=5, R6=42 (0x2A, restored), R14=0x100 (SP restored)',
    ';',
    '; Initial:  SP = 0x100, all others = 0',
    '; Expected: R0 = 7, R1 = 5, R6 = 42 (0x2A, restored by RET), R14 = 0x100',
    '',
]

lines += ini_setup(regs_to_zero=list(range(14)), pc=0, sp=0x100)
lines += ['']
lines += ['; Main code at 0x00-0x0A']
lines += deposit_bytes(main_code, base_addr=0)
lines += ['']
lines += ['; Subroutine at 0x20']
lines += deposit_bytes(sub_code, base_addr=CALLEE_ADDR)
lines += ['']
lines += ini_run_examine(breakpoint_addr=0x0B)

print('\n'.join(lines))
