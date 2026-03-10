/* vax_jit.c: SIMH-facing glue for the VAX JIT
 *
 * This file includes SIMH/VAX headers and bridges them to the LLVM
 * engine in vax_jit_llvm.c, which must NOT include any SIMH headers
 * (SIMH defines short macros like BB, FP, SP that collide with LLVM
 * parameter names).
 *
 * Operand reading happens in vax_cpu.c (which has access to get_istr).
 * This file owns: operand decode, instruction dispatch, LLVM routing.
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
int  vax_jit_llvm_addl2   (int32_t *regs, int32_t *psl,
                            int src_is_const, int32_t src_val, int dst_reg);

int vax_jit_enabled = 0;

int vax_jit_init(void)    { return vax_jit_llvm_init(); }
void vax_jit_destroy(void) { vax_jit_llvm_destroy(); }

/* ------------------------------------------------------------------ */
/* Operand decode                                                       */
/* ------------------------------------------------------------------ */

void vax_jit_decode_operand(int32 spec, int32 follow, VaxJITOperand *op)
{
    if (VAX_SPEC_IS_SHORT_LIT(spec)) {
        op->kind = JITOPK_LITERAL;
        op->reg  = 0;
        op->imm  = spec & 0x3F;
    } else if (VAX_SPEC_MODE(spec) == VAX_SPEC_REGISTER) {
        op->kind = JITOPK_REGISTER;
        op->reg  = VAX_SPEC_REG(spec);
        op->imm  = 0;
    } else if (VAX_SPEC_IS_IMMEDIATE(spec)) {
        op->kind = JITOPK_IMMEDIATE;
        op->reg  = 0;
        op->imm  = follow;
    } else {
        op->kind = JITOPK_UNSUPPORTED;
        op->reg  = 0;
        op->imm  = 0;
    }
}

int vax_jit_operand_needs_long(int32 spec)
{
    return VAX_SPEC_IS_IMMEDIATE(spec);
}

/* ------------------------------------------------------------------ */
/* Operand validity checks                                             */
/* ------------------------------------------------------------------ */

/* Any operand kind we can read a value from */
static int op_is_readable(const VaxJITOperand *op)
{
    return op->kind == JITOPK_REGISTER
        || op->kind == JITOPK_LITERAL
        || op->kind == JITOPK_IMMEDIATE;
}

/* Destination: register only, and not PC (would be a computed branch) */
static int op_is_writable(const VaxJITOperand *op)
{
    return op->kind == JITOPK_REGISTER && op->reg != nPC;
}

/* ------------------------------------------------------------------ */
/* Instruction count — use drom for the actual spec count              */
/* ------------------------------------------------------------------ */

int vax_jit_noperands(int32 opc)
{
    switch (opc) {
    /* Supported opcodes: return spec count from the decode ROM */
    case NOP:
    case ADDL2:
        return DR_GETNSP(drom[opc][0]);
    default:
        return -1;  /* not JIT-handled */
    }
}

/* ------------------------------------------------------------------ */
/* Dispatch                                                            */
/* ------------------------------------------------------------------ */

int vax_jit_execute(int32 opc, VAXCPUState *state,
                    VaxJITOperand *ops, int nops)
{
    if (!vax_jit_enabled)
        return 0;

    switch (opc) {

    case NOP:
        return vax_jit_llvm_nop();

    case ADDL2: {
        const VaxJITOperand *src = &ops[0];
        const VaxJITOperand *dst = &ops[1];
        if (!op_is_readable(src) || !op_is_writable(dst))
            return 0;
        return vax_jit_llvm_addl2(
            state->R, &state->PSL,
            src->kind != JITOPK_REGISTER,  /* is_const */
            src->kind == JITOPK_REGISTER ? (int32_t)src->reg : (int32_t)src->imm,
            dst->reg);
    }

    default:
        return 0;
    }
}
