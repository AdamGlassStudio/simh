/* vax_jit.c: SIMH-facing glue for the VAX JIT
 *
 * This file includes SIMH/VAX headers and bridges them to the LLVM
 * engine in vax_jit_llvm.c, which must NOT include any SIMH headers
 * (SIMH defines short macros like BB, FP, SP that collide with LLVM
 * parameter names).
 *
 * Spec-byte reading happens in vax_cpu.c (which has access to get_istr).
 * This file receives already-decoded specifier bytes and routes to the
 * appropriate LLVM function.
 */

#include "vax_jit.h"
#include <stdint.h>

/* vax_defs.h defines R and PSL as compat macros pointing to cpu_state.
   Undefine them here so we can access struct members via a pointer
   (state->R, state->PSL) without the macros mangling the expressions. */
#undef R
#undef PSL

/* Declarations from vax_jit_llvm.c (plain C types only) */
int  vax_jit_llvm_init    (void);
void vax_jit_llvm_destroy (void);
int  vax_jit_llvm_nop     (void);
int  vax_jit_llvm_addl2   (int32_t *regs, int32_t *psl, int src, int dst);

int vax_jit_enabled = 0;

int vax_jit_init(void)
{
    return vax_jit_llvm_init();
}

void vax_jit_destroy(void)
{
    vax_jit_llvm_destroy();
}

/* How many specifier bytes the caller must read before vax_jit_execute.
   -1 = opcode not JIT-handled (skip the JIT hook entirely).          */
int vax_jit_nspecs(int32 opc)
{
    switch (opc) {
    case NOP:   return 0;
    case ADDL2: return 2;
    default:    return -1;
    }
}

int vax_jit_execute(int32 opc, VAXCPUState *state,
                    int32 *specs, int nspecs)
{
    if (!vax_jit_enabled)
        return 0;

    switch (opc) {

    case NOP:
        return vax_jit_llvm_nop();

    case ADDL2: {
        int src_mode, dst_mode, src_reg, dst_reg;
        if (nspecs < 2) return 0;
        src_mode = (specs[0] >> 4) & 0xF;
        dst_mode = (specs[1] >> 4) & 0xF;
        src_reg  =  specs[0]       & 0xF;
        dst_reg  =  specs[1]       & 0xF;
        /* mode 5 = register direct; other modes fall back to interpreter */
        if (src_mode != 5 || dst_mode != 5) return 0;
        return vax_jit_llvm_addl2(state->R, &state->PSL, src_reg, dst_reg);
    }

    default:
        return 0;
    }
}
