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

/* Attempt to JIT-execute one instruction.
   opc   - VAX opcode (0x000–0x1FF)
   state - pointer to live VAXCPUState
   opnd  - decoded operand array (same layout as the interpreter's opnd[])
   nopnd - number of valid entries in opnd[]

   Returns 1 if the JIT handled the instruction (interpreter should skip it),
   returns 0 if the JIT did not handle it (interpreter should proceed normally).

   In the current implementation only NOP (opcode 0x01) is handled; all other
   opcodes return 0 immediately.                                     */
int  vax_jit_execute (int32 opc, VAXCPUState *state,
                      int32 *opnd, int nopnd);

/* 1 when JIT is enabled (SET CPU JIT), 0 when disabled (SET CPU NOJIT). */
extern int vax_jit_enabled;

#endif /* VAX_JIT_H */
