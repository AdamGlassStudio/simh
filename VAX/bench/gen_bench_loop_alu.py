#!/usr/bin/env python3
"""Generator for bench_loop_alu.ini — 5-insn integer ALU loop body.

Goal: measure JIT vs interpreter throughput on a body that does representative
integer ALU work, not just back-edge dispatch. Entirely region-JIT-able with
the current scanner.

Body (5 insns, 12 bytes):
  ADDL2 R3, R0    ; R0 += R3   (3 bytes)
  ADDL2 R3, R1    ; R1 += R3   (3 bytes)
  INCL  R4        ; R4++       (2 bytes)
  DECL  R2        ; counter--  (2 bytes)
  BGTR  loop_top  ; back-edge  (2 bytes)

Layout (all at PC=0):
  0x00: MOVL  #N, R2          ; loop counter (7 bytes)
  0x07: CLRL  R0              ; (2 bytes)
  0x09: MOVL  #3, R3          ; constant operand (3 bytes via short_lit)
  0x0C: ← loop_top
  0x0C-0x17: body (12 bytes)
  0x18: ← breakpoint

Iterations of body = N. Per-iteration insns = 5.
Total VAX insns executed = 3 (init) + 5N (body).

Default N = 100_000_000 → ~500M insns. Override via BENCH_N env var:
    BENCH_N=10000000 python3 gen_bench_loop_alu.py > bench_loop_alu.ini
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests', 'jit'))
from vax_encode import (
    MOVL, CLRL, ADDL2, INCL, DECL, BGTR,
    imm_long, short_lit, reg,
    ini_header, deposit_bytes, ini_setup, ini_run_examine,
    R0, R1, R2, R3, R4,
)

N = int(os.environ.get('BENCH_N', '100000000'))

pc = 0
init_movl_n  = MOVL(imm_long(N), reg(R2));      pc += len(init_movl_n)
init_clrl_r0 = CLRL(reg(R0));                    pc += len(init_clrl_r0)
init_movl_3  = MOVL(short_lit(3), reg(R3));      pc += len(init_movl_3)
loop_top = pc
addl2_a = ADDL2(reg(R3), reg(R0));               pc += len(addl2_a)
addl2_b = ADDL2(reg(R3), reg(R1));               pc += len(addl2_b)
incl_r4 = INCL(reg(R4));                          pc += len(incl_r4)
decl_r2 = DECL(reg(R2));                          pc += len(decl_r2)
bgtr    = BGTR(loop_top - (pc + 2));             pc += 2
breakpoint_addr = pc

code = init_movl_n + init_clrl_r0 + init_movl_3 + \
       addl2_a + addl2_b + incl_r4 + decl_r2 + bgtr

# Sanity
assert len(code) == breakpoint_addr, (len(code), breakpoint_addr)

lines = []
lines += ini_header('bench_loop_alu', os.path.basename(__file__))
lines.append(f'; Iterations N = {N:_}')
lines.append(f'; Expected VAX insns executed = 3 + 5*N = {3 + 5*N:_}')
lines.append(f'; Expected R0 = R1 = 3*N (mod 2^32); R4 = N')
lines.append('')
lines += ini_setup(regs_to_zero=list(range(14)), pc=0)
lines += deposit_bytes(code, base_addr=0)
lines += ini_run_examine(breakpoint_addr)

print('\n'.join(lines))
