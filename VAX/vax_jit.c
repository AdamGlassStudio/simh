/* vax_jit.c: SIMH-facing glue for the VAX JIT
 *
 * This file includes SIMH/VAX headers and bridges them to the LLVM
 * engine in vax_jit_llvm.c, which must NOT include any SIMH headers
 * (SIMH defines short macros like BB, FP, SP that collide with LLVM
 * parameter names).
 */

#include "vax_jit.h"

/* Declarations from vax_jit_llvm.c (plain C types only) */
int  vax_jit_llvm_init    (void);
void vax_jit_llvm_destroy (void);
int  vax_jit_llvm_execute (int opc);

int vax_jit_enabled = 0;

int vax_jit_init(void)
{
    return vax_jit_llvm_init();
}

void vax_jit_destroy(void)
{
    vax_jit_llvm_destroy();
}

int vax_jit_execute(int32 opc, VAXCPUState *state,
                    int32 *opnd, int nopnd)
{
    if (!vax_jit_enabled)
        return 0;
    return vax_jit_llvm_execute((int)opc);
}
