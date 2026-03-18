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

/* ------------------------------------------------------------------ */
/* Integer ALU op kind (must match VaxIntOpL enum in vax_jit_llvm.c)  */
/* ------------------------------------------------------------------ */

typedef enum {
    VAX_INTL_ADD     = 0,
    VAX_INTL_SUB     = 1,
    VAX_INTL_OR      = 2,
    VAX_INTL_AND_NOT = 3,
    VAX_INTL_XOR     = 4,
    VAX_INTL_MOV     = 5,
    VAX_INTL_COM     = 6,
    VAX_INTL_NOPS    = 7
} VaxIntOpL;

/* ------------------------------------------------------------------ */
/* Declarations from vax_jit_llvm.c (plain C types only)              */
/* ------------------------------------------------------------------ */

int  vax_jit_llvm_init    (void);
void vax_jit_llvm_destroy (void);
void vax_jit_llvm_set_ir_dump (int enable);
int  vax_jit_llvm_nop     (void);
int  vax_jit_llvm_intl2   (int32_t *regs, int32_t *psl, int op,
                            int src_is_const, int32_t src_val, int dst_reg);
int  vax_jit_llvm_cmpl    (int32_t *regs, int32_t *psl,
                            int src1_is_const, int32_t src1_val,
                            int src2_is_const, int32_t src2_val);
int  vax_jit_llvm_tstl    (int32_t *regs, int32_t *psl,
                            int src_is_const, int32_t src_val);
int  vax_jit_llvm_exec_block (VaxJITBlock *blk, int32_t *regs,
                               int32_t *psl, int32_t *mem);

int vax_jit_enabled = 0;
int vax_jit_ir_dump = 0;

int vax_jit_init(void)
{
    vax_jit_llvm_set_ir_dump(vax_jit_ir_dump);
    return vax_jit_llvm_init();
}
void vax_jit_destroy(void) { vax_jit_llvm_destroy(); }

/* ------------------------------------------------------------------ */
/* Extension byte count                                                */
/* ------------------------------------------------------------------ */

int vax_jit_spec_ext_lnt(int32 spec, int32 op_lnt)
{
    int mode = VAX_SPEC_MODE(spec);
    int reg  = VAX_SPEC_REG(spec);
    if (mode <= VAX_SPEC_AUTOINCREMENT) {
        return (mode == VAX_SPEC_AUTOINCREMENT && reg == nPC) ? op_lnt : 0;
    }
    if (mode == VAX_SPEC_AUTOINC_DEF)
        return (reg == nPC) ? 4 : 0;
    /* modes A-F: byte/word/long displacement */
    {
        static const int lnt[6] = {1, 1, 2, 2, 4, 4};
        return lnt[mode - VAX_SPEC_BYTE_DISP];
    }
}

/* ------------------------------------------------------------------ */
/* Operand decode                                                       */
/* ------------------------------------------------------------------ */

void vax_jit_decode_operand(int32 spec, int32 follow, int32 pc_after,
                             VaxJITOperand *op)
{
    int mode = VAX_SPEC_MODE(spec);
    int reg  = VAX_SPEC_REG(spec);
    op->reg = 0; op->imm = 0;

    if (VAX_SPEC_IS_SHORT_LIT(spec)) {
        op->kind = JITOPK_LITERAL; op->imm = spec & 0x3F; return;
    }
    switch (mode) {
    case VAX_SPEC_REGISTER:
        op->kind = JITOPK_REGISTER; op->reg = reg; return;
    case VAX_SPEC_REG_DEFERRED:
        op->kind = JITOPK_REG_DEFERRED; op->reg = reg; return;
    case VAX_SPEC_AUTODECREMENT:
        op->kind = JITOPK_AUTODECREMENT; op->reg = reg; return;
    case VAX_SPEC_AUTOINCREMENT:
        if (reg == nPC) { op->kind = JITOPK_IMMEDIATE; op->imm = follow; }
        else            { op->kind = JITOPK_AUTOINCREMENT; op->reg = reg; }
        return;
    case VAX_SPEC_AUTOINC_DEF:
        if (reg == nPC) { op->kind = JITOPK_ABSOLUTE; op->imm = follow; }
        else            { op->kind = JITOPK_AUTOINC_DEF; op->reg = reg; }
        return;
    case VAX_SPEC_BYTE_DISP:
        if (reg == nPC) { op->kind = JITOPK_ABSOLUTE;
                          op->imm  = pc_after + (int32)(int8_t)(follow & 0xFF); }
        else            { op->kind = JITOPK_DISP; op->reg = reg;
                          op->imm  = (int32)(int8_t)(follow & 0xFF); }
        return;
    case VAX_SPEC_BYTE_DISP_DEF:
        if (reg == nPC) { op->kind = JITOPK_ABS_DEFERRED;
                          op->imm  = pc_after + (int32)(int8_t)(follow & 0xFF); }
        else            { op->kind = JITOPK_DISP_DEFERRED; op->reg = reg;
                          op->imm  = (int32)(int8_t)(follow & 0xFF); }
        return;
    case VAX_SPEC_WORD_DISP:
        if (reg == nPC) { op->kind = JITOPK_ABSOLUTE;
                          op->imm  = pc_after + (int32)(int16_t)(follow & 0xFFFF); }
        else            { op->kind = JITOPK_DISP; op->reg = reg;
                          op->imm  = (int32)(int16_t)(follow & 0xFFFF); }
        return;
    case VAX_SPEC_WORD_DISP_DEF:
        if (reg == nPC) { op->kind = JITOPK_ABS_DEFERRED;
                          op->imm  = pc_after + (int32)(int16_t)(follow & 0xFFFF); }
        else            { op->kind = JITOPK_DISP_DEFERRED; op->reg = reg;
                          op->imm  = (int32)(int16_t)(follow & 0xFFFF); }
        return;
    case VAX_SPEC_LONG_DISP:
        if (reg == nPC) { op->kind = JITOPK_ABSOLUTE; op->imm = pc_after + follow; }
        else            { op->kind = JITOPK_DISP; op->reg = reg; op->imm = follow; }
        return;
    case VAX_SPEC_LONG_DISP_DEF:
        if (reg == nPC) { op->kind = JITOPK_ABS_DEFERRED; op->imm = pc_after + follow; }
        else            { op->kind = JITOPK_DISP_DEFERRED; op->reg = reg; op->imm = follow; }
        return;
    default:
        op->kind = JITOPK_UNSUPPORTED; return;
    }
}

/* ------------------------------------------------------------------ */
/* Operand validity checks                                             */
/* ------------------------------------------------------------------ */

static int op_is_readable(const VaxJITOperand *op)
{
    return op->kind != JITOPK_UNSUPPORTED;
}

static int op_is_writable(const VaxJITOperand *op)
{
    /* literals/immediates can't be write targets */
    if (op->kind == JITOPK_LITERAL || op->kind == JITOPK_IMMEDIATE)
        return 0;
    /* PC writes mean computed branch - hand to interpreter */
    if (op->kind == JITOPK_REGISTER && op->reg == nPC)
        return 0;
    return op->kind != JITOPK_UNSUPPORTED;
}

static int op_is_mem(const VaxJITOperand *op)
{
    return op->kind >= JITOPK_REG_DEFERRED;
}

/* Extract (is_const, val) pair from a decoded readable operand.
   Used by the fast-path handlers (register/literal/immediate only). */
#define OP_SRC(op)  ((op)->kind != JITOPK_REGISTER), \
                    ((op)->kind == JITOPK_REGISTER ? (int32_t)(op)->reg \
                                                   : (int32_t)(op)->imm)

/* ------------------------------------------------------------------ */
/* Block path helpers                                                   */
/* ------------------------------------------------------------------ */

static VaxJITBlkOp to_blk_op(const VaxJITOperand *op)
{
    VaxJITBlkOp b;
    b.kind = (VaxJITBlkOpKind)op->kind;  /* enums are layout-compatible */
    b.reg  = op->reg;
    b.imm  = (int32_t)op->imm;
    return b;
}

static int needs_block_path(VaxJITOperand *ops, int nops)
{
    int i;
    for (i = 0; i < nops; i++)
        if (op_is_mem(&ops[i])) return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Instruction operand count (returns -1 if not JIT-handled)          */
/* ------------------------------------------------------------------ */

int vax_jit_noperands(int32 opc)
{
    switch (opc) {
    case NOP:
    case ADDL2: case SUBL2:
    case BISL2: case BICL2: case XORL2:
    case MOVL:  case CMPL:  case MCOML:
    case CLRL:  case TSTL:  case INCL:  case DECL:
        return DR_GETNSP(drom[opc][0]);
    default:
        return -1;
    }
}

/* ------------------------------------------------------------------ */
/* Dispatch                                                            */
/* ------------------------------------------------------------------ */

int vax_jit_execute(int32 opc, VAXCPUState *state,
                    VaxJITOperand *ops, int nops)
{
    int32_t *regs = state->R;
    int32_t *psl  = &state->PSL;
    int i;

    if (!vax_jit_enabled)
        return 0;

    /* --- Block path: any memory-mode operand ------------------------ */
    if (needs_block_path(ops, nops)) {
        VaxJITBlock blk;
        blk.opc   = opc;
        blk.n_ops = nops;
        for (i = 0; i < nops; i++)
            blk.ops[i] = to_blk_op(&ops[i]);
        return vax_jit_llvm_exec_block(&blk, regs, psl, (int32_t*)M);
    }

    /* --- Fast path: register/literal/immediate operands only -------- */
    switch (opc) {

    case NOP:
        return vax_jit_llvm_nop();

    /* --- Two-operand: src (RL), dst (ML) - result written to dst --- */
    case ADDL2: case SUBL2:
    case BISL2: case BICL2: case XORL2:
    case MOVL:  case MCOML: {
        static const int8_t op_map[] = {
            /* ADDL2=0xC0 */ VAX_INTL_ADD,
            /* ADDL3=0xC1 */ -1,
            /* SUBL2=0xC2 */ VAX_INTL_SUB,
            /* SUBL3=0xC3 */ -1,
            /* ... */        -1, -1, -1, -1,
            /* BISL2=0xC8 */ VAX_INTL_OR,
            /* BISL3=0xC9 */ -1,
            /* BICL2=0xCA */ VAX_INTL_AND_NOT,
            /* BICL3=0xCB */ -1,
            /* XORL2=0xCC */ VAX_INTL_XOR,
            /* XORL3=0xCD */ -1,
            /* MNEGL=0xCE */ -1,
            /* CASEL=0xCF */ -1,
            /* MOVL =0xD0 */ VAX_INTL_MOV,
            /* CMPL =0xD1 */ -1,    /* handled separately */
            /* MCOML=0xD2 */ VAX_INTL_COM,
        };
        int idx = opc - ADDL2;
        int intl_op = (idx >= 0 && idx < (int)(sizeof op_map))
                      ? op_map[idx] : -1;
        if (intl_op < 0) return 0;

        const VaxJITOperand *src = &ops[0];
        const VaxJITOperand *dst = &ops[1];
        if (!op_is_readable(src) || !op_is_writable(dst))
            return 0;
        /* Fast path: dst must be a register (memory handled by block path above) */
        if (dst->kind != JITOPK_REGISTER)
            return 0;
        return vax_jit_llvm_intl2(regs, psl, intl_op, OP_SRC(src), dst->reg);
    }

    /* --- CMPL: two read operands, no store --- */
    case CMPL: {
        const VaxJITOperand *s1 = &ops[0];
        const VaxJITOperand *s2 = &ops[1];
        if (!op_is_readable(s1) || !op_is_readable(s2))
            return 0;
        return vax_jit_llvm_cmpl(regs, psl, OP_SRC(s1), OP_SRC(s2));
    }

    /* --- TSTL: one read operand, no store --- */
    case TSTL: {
        const VaxJITOperand *src = &ops[0];
        if (!op_is_readable(src))
            return 0;
        return vax_jit_llvm_tstl(regs, psl, OP_SRC(src));
    }

    /* --- CLRL: one write operand, result = 0 --- */
    case CLRL: {
        const VaxJITOperand *dst = &ops[0];
        if (!op_is_writable(dst))
            return 0;
        if (dst->kind != JITOPK_REGISTER)
            return 0;
        return vax_jit_llvm_intl2(regs, psl, VAX_INTL_MOV,
                                   1, 0,        /* src = const 0 */
                                   dst->reg);
    }

    /* --- INCL: one modify operand, dst = dst + 1 --- */
    case INCL: {
        const VaxJITOperand *dst = &ops[0];
        if (!op_is_writable(dst))
            return 0;
        if (dst->kind != JITOPK_REGISTER)
            return 0;
        return vax_jit_llvm_intl2(regs, psl, VAX_INTL_ADD,
                                   1, 1,        /* src = const 1 */
                                   dst->reg);
    }

    /* --- DECL: one modify operand, dst = dst - 1 --- */
    case DECL: {
        const VaxJITOperand *dst = &ops[0];
        if (!op_is_writable(dst))
            return 0;
        if (dst->kind != JITOPK_REGISTER)
            return 0;
        return vax_jit_llvm_intl2(regs, psl, VAX_INTL_SUB,
                                   1, 1,        /* src = const 1 */
                                   dst->reg);
    }

    default:
        return 0;
    }
}
