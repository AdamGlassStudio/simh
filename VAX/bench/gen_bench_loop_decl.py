#!/usr/bin/env python3
"""Generator for bench_loop_decl.ini — minimal 2-insn back-edge loop.

Goal: measure steady-state JIT throughput on the smallest possible loop body
(DECL + BGTR). This isolates the cost of:
  - back-edge dispatch / sim_interval guard
  - cache-hit path (the whole loop is one tiny cached JIT block)
  - per-iteration JIT overhead with effectively zero real work

Layout (all at PC=0):
  0x00: D0 8F NN NN NN NN 52   MOVL  #N, R2     ; counter (7 bytes)
  0x07: D7 52                  DECL  R2         ; (2 bytes)
  0x09: 14 FC                  BGTR  -4         ; target=0x07, disp=0x07-0x0B=-4
  0x0B:                        ← breakpoint here

Iterations of body = N. Total VAX insns executed ≈ 2N + 1.

Default N = 250_000_000 → ~500M insns. On a modern host this should give
single-digit-second runtimes for the interpreter, leaving plenty of dynamic
range for any JIT speedup.

Override N via the BENCH_N env var when generating, e.g.:
    BENCH_N=10000000 python3 gen_bench_loop_decl.py > bench_loop_decl.ini
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests', 'jit'))
from vax_encode import (
    MOVL, DECL, BGTR, imm_long, reg,
    ini_header, deposit_bytes, ini_setup, ini_run_examine,
    R2,
)

N = int(os.environ.get('BENCH_N', '250000000'))

pc = 0
movl_init = MOVL(imm_long(N), reg(R2));   pc += len(movl_init)
loop_top  = pc
decl_r2   = DECL(reg(R2));                 pc += len(decl_r2)
# BGTR disp = loop_top - (current_pc + 2); branch instr is 2 bytes
bgtr      = BGTR(loop_top - (pc + 2));     pc += 2
breakpoint_addr = pc

code = movl_init + decl_r2 + bgtr

# Sanity
assert len(code) == breakpoint_addr, (len(code), breakpoint_addr)
assert breakpoint_addr == 0x0B, hex(breakpoint_addr)

lines = []
lines += ini_header('bench_loop_decl', os.path.basename(__file__))
lines.append(f'; Iterations N = {N:_}')
lines.append(f'; Expected VAX insns executed ≈ {2*N + 1:_}')
lines.append('')
lines += ini_setup(regs_to_zero=list(range(14)), pc=0)
lines += deposit_bytes(code, base_addr=0)
lines += ini_run_examine(breakpoint_addr)

print('\n'.join(lines))
