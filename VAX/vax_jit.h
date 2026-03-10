/* vax_jit.h: VAX JIT compiler interface

   The JIT is an optional accelerator that translates VAX instructions to
   native x86-64 via LLVM IR and ORC JIT.  It is enabled at runtime with
   SET CPU JIT / SET CPU NOJIT.

   The interpreter remains the correctness oracle throughout development.
   Instructions not yet handled by the JIT fall through to the interpreter
   unchanged.
*/

#ifndef VAX_JIT_H
#define VAX_JIT_H

#include "vax_defs.h"

/* Initialise / shut down the LLVM ORC JIT engine.
   Call vax_jit_init() once at simulator startup (cpu_reset).
   Call vax_jit_destroy() at simulator exit.                        */
int  vax_jit_init    (void);
void vax_jit_destroy (void);

/* How many specifier bytes the JIT needs to read before calling vax_jit_execute.
   Returns -1 if the opcode is not JIT-handled (skip the JIT entirely).
   Returns  0 for zero-operand instructions (NOP).
   Returns  N for N single-byte specifiers.
   The caller (vax_cpu.c) reads exactly this many bytes from the instruction
   stream and saves/restores the prefetch state on JIT miss.          */
int  vax_jit_nspecs  (int32 opc);

/* Attempt to JIT-execute one instruction.
   opc   - VAX opcode
   state - pointer to live VAXCPUState
   specs - specifier bytes already read from the instruction stream
           (vax_jit_nspecs(opc) of them); NULL for zero-specifier instructions
   nspecs - number of entries in specs[]

   Returns 1 if the JIT handled the instruction (caller should `continue`
   the main dispatch loop — interpreter must NOT also run it).
   Returns 0 if the JIT did not handle it (caller must restore the prefetch
   state to undo any specifier bytes that were consumed).             */
int  vax_jit_execute (int32 opc, VAXCPUState *state,
                      int32 *specs, int nspecs);

/* 1 when JIT is enabled (SET CPU JIT), 0 when disabled (SET CPU NOJIT). */
extern int vax_jit_enabled;

#endif /* VAX_JIT_H */
