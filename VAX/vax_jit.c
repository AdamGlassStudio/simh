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
#include <string.h>

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
                               int32_t *psl, int32_t *mem,
                               int32_t *sim_interval);

int vax_jit_enabled = 0;
int vax_jit_ir_dump = 0;

/* ------------------------------------------------------------------ */
/* Telemetry                                                            */
/* ------------------------------------------------------------------ */

VaxJITStats vax_jit_stats;

void vax_jit_stats_reset(void)
{
    memset(&vax_jit_stats, 0, sizeof vax_jit_stats);
}

/* Opcode name table for the 0x00-0xFF single-byte range (common ones) */
static const char *vax_opc_name(int opc)
{
    extern char const * const opcode[];
    if (opc >= 0 && opc < 256 && opcode[opc])
        return opcode[opc];
    return NULL;
}

t_stat vax_jit_stats_show(FILE *st, UNIT *uptr, int32 val, CONST void *desc)
{
    uint64_t total = vax_jit_stats.insns_jit + vax_jit_stats.insns_interp;
    double jit_pct = total ? 100.0 * vax_jit_stats.insns_jit / total : 0.0;
    int i;

    fprintf(st, "JIT statistics:\n");
    fprintf(st, "  Blocks run:         %llu\n",
            (unsigned long long)vax_jit_stats.blocks_run);
    fprintf(st, "  Insns via JIT:      %llu (%.1f%%)\n",
            (unsigned long long)vax_jit_stats.insns_jit, jit_pct);
    fprintf(st, "  Insns interpreter:  %llu\n",
            (unsigned long long)vax_jit_stats.insns_interp);
    fprintf(st, "  Scanner returned 0: %llu\n",
            (unsigned long long)vax_jit_stats.scan_empty);
    fprintf(st, "  Compile failures:   %llu\n",
            (unsigned long long)vax_jit_stats.compile_fail);

    if (vax_jit_stats.blocks_run > 0) {
        double avg = (double)vax_jit_stats.insns_jit / vax_jit_stats.blocks_run;
        uint32_t mn = 33, mx = 0;
        for (i = 1; i <= 32; i++) {
            if (vax_jit_stats.size_hist[i]) {
                if ((uint32_t)i < mn) mn = i;
                if ((uint32_t)i > mx) mx = i;
            }
        }
        fprintf(st, "  Block size avg/min/max: %.2f / %u / %u\n", avg, mn, mx);
        fprintf(st, "  Block size distribution:\n");
        for (i = 1; i <= 32; i++) {
            if (vax_jit_stats.size_hist[i])
                fprintf(st, "    %2d insns: %u blocks\n",
                        i, vax_jit_stats.size_hist[i]);
        }
    }

    /* Print top scan_empty blocking opcodes (sorted by count, top 16) */
    {
        int any = 0;
        for (i = 0; i < 256; i++)
            if (vax_jit_stats.scan_empty_opc[i]) { any = 1; break; }
        if (any) {
            /* simple selection sort for top 16 */
            int order[256], n = 0;
            for (i = 0; i < 256; i++)
                if (vax_jit_stats.scan_empty_opc[i]) order[n++] = i;
            /* bubble top entries to front */
            int j;
            for (i = 0; i < n && i < 16; i++) {
                int best = i;
                for (j = i+1; j < n; j++)
                    if (vax_jit_stats.scan_empty_opc[order[j]] >
                        vax_jit_stats.scan_empty_opc[order[best]])
                        best = j;
                int tmp = order[i]; order[i] = order[best]; order[best] = tmp;
            }
            fprintf(st, "  Top blocking opcodes (scan_empty cause):\n");
            for (i = 0; i < n && i < 16; i++) {
                int opc = order[i];
                const char *nm = vax_opc_name(opc);
                if (nm)
                    fprintf(st, "    0x%02X %-10s %u\n", opc, nm,
                            vax_jit_stats.scan_empty_opc[opc]);
                else
                    fprintf(st, "    0x%02X %-10s %u\n", opc, "?",
                            vax_jit_stats.scan_empty_opc[opc]);
            }
        }
    }
    return SCPE_OK;
}

t_stat vax_jit_stats_set(UNIT *uptr, int32 val, CONST char *cptr, void *desc)
{
    if (cptr && strcmp(cptr, "RESET") == 0) {
        vax_jit_stats_reset();
        return SCPE_OK;
    }
    return sim_messagef(SCPE_ARG, "Usage: SET CPU JITSTATS=RESET\n");
}

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
    int32 disp8, disp16;

    op->kind = JITOPK_UNSUPPORTED;
    op->reg  = 0;
    op->imm  = 0;

    /* Short literal: value 0-63 encoded in the specifier byte itself */
    if (VAX_SPEC_IS_SHORT_LIT(spec)) {
        op->kind = JITOPK_LITERAL;
        op->imm  = spec & 0x3F;
        return;
        }

    switch (mode) {

    case VAX_SPEC_REGISTER:
        op->kind = JITOPK_REGISTER;
        op->reg  = reg;
        break;

    case VAX_SPEC_REG_DEFERRED:
        op->kind = JITOPK_REG_DEFERRED;
        op->reg  = reg;
        break;

    case VAX_SPEC_AUTODECREMENT:
        op->kind = JITOPK_AUTODECREMENT;
        op->reg  = reg;
        break;

    case VAX_SPEC_AUTOINCREMENT:
        /* PC case: immediate (#imm) — follow holds the inline value */
        if (reg == nPC) {
            op->kind = JITOPK_IMMEDIATE;
            op->imm  = follow;
            }
        else {
            op->kind = JITOPK_AUTOINCREMENT;
            op->reg  = reg;
            }
        break;

    case VAX_SPEC_AUTOINC_DEF:
        /* PC case: absolute (@#addr) — follow holds the target address */
        if (reg == nPC) {
            op->kind = JITOPK_ABSOLUTE;
            op->imm  = follow;
            }
        else {
            op->kind = JITOPK_AUTOINC_DEF;
            op->reg  = reg;
            }
        break;

    case VAX_SPEC_BYTE_DISP:
        disp8 = (int32)(int8_t)(follow & 0xFF);
        if (reg == nPC) {
            op->kind = JITOPK_ABSOLUTE;         /* PC-relative → fold EA */
            op->imm  = pc_after + disp8;
            }
        else {
            op->kind = JITOPK_DISP;
            op->reg  = reg;
            op->imm  = disp8;
            }
        break;

    case VAX_SPEC_BYTE_DISP_DEF:
        disp8 = (int32)(int8_t)(follow & 0xFF);
        if (reg == nPC) {
            op->kind = JITOPK_ABS_DEFERRED;
            op->imm  = pc_after + disp8;
            }
        else {
            op->kind = JITOPK_DISP_DEFERRED;
            op->reg  = reg;
            op->imm  = disp8;
            }
        break;

    case VAX_SPEC_WORD_DISP:
        disp16 = (int32)(int16_t)(follow & 0xFFFF);
        if (reg == nPC) {
            op->kind = JITOPK_ABSOLUTE;
            op->imm  = pc_after + disp16;
            }
        else {
            op->kind = JITOPK_DISP;
            op->reg  = reg;
            op->imm  = disp16;
            }
        break;

    case VAX_SPEC_WORD_DISP_DEF:
        disp16 = (int32)(int16_t)(follow & 0xFFFF);
        if (reg == nPC) {
            op->kind = JITOPK_ABS_DEFERRED;
            op->imm  = pc_after + disp16;
            }
        else {
            op->kind = JITOPK_DISP_DEFERRED;
            op->reg  = reg;
            op->imm  = disp16;
            }
        break;

    case VAX_SPEC_LONG_DISP:
        if (reg == nPC) {
            op->kind = JITOPK_ABSOLUTE;
            op->imm  = pc_after + follow;
            }
        else {
            op->kind = JITOPK_DISP;
            op->reg  = reg;
            op->imm  = follow;
            }
        break;

    case VAX_SPEC_LONG_DISP_DEF:
        if (reg == nPC) {
            op->kind = JITOPK_ABS_DEFERRED;
            op->imm  = pc_after + follow;
            }
        else {
            op->kind = JITOPK_DISP_DEFERRED;
            op->reg  = reg;
            op->imm  = follow;
            }
        break;

    default:
        op->kind = JITOPK_UNSUPPORTED;
        break;
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
/* Block path helpers (used by both scanner and vax_jit_execute)      */
/* ------------------------------------------------------------------ */

static VaxJITBlkOp to_blk_op(const VaxJITOperand *op, uint8_t width)
{
    VaxJITBlkOp b;
    b.kind  = (VaxJITBlkOpKind)op->kind;
    b.reg   = op->reg;
    b.imm   = (int32_t)op->imm;
    b.width = width;
    return b;
}

static int needs_block_path(VaxJITOperand *ops, int nops)
{
    int i;
    for (i = 0; i < nops; i++)
        if (op_is_mem(&ops[i])) return 1;
    return 0;
}

static int always_block_path(int32 opc)
{
    return opc == MOVB || opc == MOVW || opc == MOVZBL || opc == MOVZWL || opc == PUSHL
        || opc == ASHL || opc == MOVQ
        || opc == MOVAB || opc == MOVAL || opc == PUSHAB || opc == PUSHAL
        || opc == ADDL3 || opc == SUBL3
        || opc == BISL3 || opc == BICL3 || opc == XORL3
        || opc == PUSHR || opc == MOVPSL;
}

/* ------------------------------------------------------------------ */
/* Scanner: direct memory read helpers                                 */
/* ------------------------------------------------------------------ */

static uint8_t scan_read_byte(int32_t addr, int32_t *mem)
{
    return (uint8_t)((mem[addr >> 2] >> ((addr & 3) << 3)) & 0xFF);
}

static int32_t scan_read_word(int32_t addr, int32_t *mem)
{
    return (int32_t)(int16_t)((uint16_t)scan_read_byte(addr, mem) |
                               ((uint16_t)scan_read_byte(addr + 1, mem) << 8));
}

static int32_t scan_read_long(int32_t addr, int32_t *mem)
{
    return (int32_t)((uint32_t)scan_read_byte(addr,     mem)        |
                     ((uint32_t)scan_read_byte(addr + 1, mem) << 8)  |
                     ((uint32_t)scan_read_byte(addr + 2, mem) << 16) |
                     ((uint32_t)scan_read_byte(addr + 3, mem) << 24));
}

/* Returns the access width (bytes) for operand op_idx of opcode opc. */
static uint8_t vax_jit_scan_op_lnt(int32_t opc, int op_idx)
{
    switch (opc) {
    case MOVB:   return 1;
    case MOVW:   return 2;
    case PUSHR:  return 2;   /* mask is a word */
    case MOVZBL: return (op_idx == 0) ? 1 : 4;
    case MOVZWL: return (op_idx == 0) ? 2 : 4;
    case ASHL:   return (op_idx == 0) ? 1 : 4;
    case MOVQ:   return 8;
    default:     return 4;
    }
}

/* Returns 1 if opc is a branch/jump/call/return that terminates a block.
   These are hard stops — not decodeable as simple displacement branches.   */
static int is_branch_opc(int32_t opc)
{
    switch (opc) {
    case 0x04: /* RET    */
    case 0x05: /* RSB    */
    case 0x10: /* BSBB   */
    case 0x11: /* BRB    */
    case 0x12: /* BNEQ   */
    case 0x13: /* BEQL   */
    case 0x14: /* BGTR   */
    case 0x15: /* BLEQ   */
    case 0x16: /* JSB    */
    case 0x17: /* JMP    */
    case 0x18: /* BGEQ   */
    case 0x19: /* BLSS   */
    case 0x1A: /* BGTRU  */
    case 0x1B: /* BLEQU  */
    case 0x1C: /* BVC    */
    case 0x1D: /* BVS    */
    case 0x1E: /* BGEQU  */
    case 0x1F: /* BLSSU  */
    case 0x30: /* BSBW   */
    case 0x31: /* BRW    */
    case 0xE0: /* BBS    */
    case 0xE1: /* BBC    */
    case 0xE2: /* BBSS   */
    case 0xE3: /* BBCS   */
    case 0xE4: /* BBSC   */
    case 0xE5: /* BBCC   */
    case 0xE8: /* BLBS   */
    case 0xE9: /* BLBC   */
    case 0xFA: /* CALLG  */
    case 0xFB: /* CALLS  */
        return 1;
    default:
        return 0;
    }
}

/* Returns 1 for call/return/indirect that cannot be decoded as a simple
   PC-relative displacement branch.  These always terminate the block.   */
static int is_hard_stop_opc(int32_t opc)
{
    switch (opc) {
    case 0x04: /* RET    */
    case 0x05: /* RSB    */
    case 0x10: /* BSBB   */
    case 0x16: /* JSB    */
    case 0x17: /* JMP    */
    case 0x30: /* BSBW   */
    case 0xE0: /* BBS    */
    case 0xE1: /* BBC    */
    case 0xE2: /* BBSS   */
    case 0xE3: /* BBCS   */
    case 0xE4: /* BBSC   */
    case 0xE5: /* BBCC   */
    case 0xE8: /* BLBS   */
    case 0xE9: /* BLBC   */
    case 0xFA: /* CALLG  */
    case 0xFB: /* CALLS  */
        return 1;
    default:
        return 0;
    }
}

/* Returns 1 for BRB, BRW, and conditional branches — all use a simple
   PC-relative displacement and can be decoded to a target PC.            */
static int is_decodeable_branch_opc(int32_t opc)
{
    if (opc == 0x11 || opc == 0x31) return 1;        /* BRB, BRW      */
    if (opc >= 0x12 && opc <= 0x1F) return 1;        /* cond branches */
    return 0;
}

/* Decode the displacement of a decodeable branch opcode.
   pc  = address of the first byte after the opcode byte.
   Advances *pc_inout past the displacement bytes.
   Sets *cond_out to the branch condition (VAX_BCOND_*).
   Returns the target guest PC.                                           */
static int32_t decode_branch_target(int32_t opc, int32_t *pc_inout,
                                     int32_t *mem, uint8_t *cond_out)
{
    int32_t pc = *pc_inout;
    int32_t target;

    if (opc == 0x31) { /* BRW: 2-byte signed displacement */
        int32_t disp = scan_read_word(pc, mem);
        target = (pc + 2) + disp;
        pc += 2;
        *cond_out = VAX_BCOND_UNCOND;
    } else {           /* BRB and all 1-byte conditional branches */
        int32_t disp = (int32_t)(int8_t)scan_read_byte(pc, mem);
        target = (pc + 1) + disp;
        pc += 1;
        switch (opc) {
        case 0x11: *cond_out = VAX_BCOND_UNCOND; break;
        case 0x12: *cond_out = VAX_BCOND_NEQ;    break;
        case 0x13: *cond_out = VAX_BCOND_EQL;    break;
        case 0x14: *cond_out = VAX_BCOND_GTR;    break;
        case 0x15: *cond_out = VAX_BCOND_LEQ;    break;
        case 0x18: *cond_out = VAX_BCOND_GEQ;    break;
        case 0x19: *cond_out = VAX_BCOND_LSS;    break;
        case 0x1A: *cond_out = VAX_BCOND_GTRU;   break;
        case 0x1B: *cond_out = VAX_BCOND_LEQU;   break;
        case 0x1C: *cond_out = VAX_BCOND_VC;     break;
        case 0x1D: *cond_out = VAX_BCOND_VS;     break;
        case 0x1E: *cond_out = VAX_BCOND_GEQU;   break;
        case 0x1F: *cond_out = VAX_BCOND_LSSU;   break;
        default:   *cond_out = VAX_BCOND_UNCOND; break;
        }
    }
    *pc_inout = pc;
    return target;
}


/* Scan VAX instructions from start_pc in mem[], filling blk.
   Stops at: branch/call/return opcodes, unknown/unsupported opcodes,
   unsupported operand modes, or VAX_JIT_MAX_INSNS instructions.
   Records the PC after the last accepted instruction as fallthrough_pc. */
void vax_jit_scan_block(int32_t start_pc, VaxJITBlock *blk, int32_t *mem)
{
    int32_t pc = start_pc;
    int i;
    int32_t first_stop_opc = -1;
    int32_t scan_end = start_pc;  /* extends rightward as forward branches are found */
    blk->n_insns          = 0;
    blk->fallthrough_pc   = start_pc;
    blk->region_start_pc  = start_pc;
    blk->has_back_edge    = 0;

    while (blk->n_insns < VAX_JIT_MAX_INSNS) {
        int32_t insn_start_pc = pc;
        int32_t opc = (int32_t)(uint8_t)scan_read_byte(pc, mem);
        pc++;

        /* Extended opcode (0xFD prefix) */
        if (opc == 0xFD) {
            opc = 0x100 | (int32_t)(uint8_t)scan_read_byte(pc, mem);
            pc++;
        }

        /* Hard stop: call/return/indirect jump */
        if (is_hard_stop_opc(opc)) {
            if (first_stop_opc < 0) first_stop_opc = opc;
            break;
        }

        /* Decodeable branch: decode displacement, classify, add to block */
        if (is_decodeable_branch_opc(opc)) {
            uint8_t cond;
            int32_t target = decode_branch_target(opc, &pc, mem, &cond);
            /* pc now points past the displacement (fall-through address) */

            /* Backward escape: target escapes before the region → hard stop */
            if (target < start_pc) {
                if (first_stop_opc < 0) first_stop_opc = opc;
                break;
            }

            /* Back-edge: target is behind the current instruction */
            if (target < insn_start_pc)
                blk->has_back_edge = 1;

            /* Extend scan window for forward branches */
            if (target > scan_end)
                scan_end = target;

            VaxJITBlkInsn br_insn;
            br_insn.opc           = opc;
            br_insn.n_ops         = 0;
            br_insn.insn_pc       = insn_start_pc;
            br_insn.branch_target = target;
            br_insn.branch_cond   = cond;
            blk->insns[blk->n_insns++] = br_insn;
            blk->fallthrough_pc = pc;
            continue;
        }

        /* Stop if opcode is not JIT-handled */
        int nops = vax_jit_noperands(opc);
        if (nops < 0) {
            if (first_stop_opc < 0) first_stop_opc = opc;
            break;
        }

        /* Decode operands */
        VaxJITBlkInsn insn;
        insn.opc           = opc;
        insn.n_ops         = nops;
        insn.insn_pc       = insn_start_pc;
        insn.branch_target = -1;
        insn.branch_cond   = VAX_BCOND_NONE;
        int ok = 1;

        for (i = 0; i < nops; i++) {
            int32_t spec    = (int32_t)(uint8_t)scan_read_byte(pc, mem);
            pc++;
            uint8_t op_lnt  = vax_jit_scan_op_lnt(opc, i);
            int     ext_lnt = vax_jit_spec_ext_lnt(spec, (int32)op_lnt);

            /* Can't represent ext_lnt > 4 in a single int32_t follow value */
            if (ext_lnt > 4) { ok = 0; if (first_stop_opc < 0) first_stop_opc = opc; break; }

            int32_t follow = 0;
            if (ext_lnt == 1)
                follow = (int32_t)(int8_t)scan_read_byte(pc, mem);
            else if (ext_lnt == 2)
                follow = scan_read_word(pc, mem);
            else if (ext_lnt == 4)
                follow = scan_read_long(pc, mem);
            pc += ext_lnt;

            int32_t pc_after = pc;

            VaxJITOperand op;
            vax_jit_decode_operand(spec, follow, pc_after, &op);
            if (op.kind == JITOPK_UNSUPPORTED) { ok = 0; if (first_stop_opc < 0) first_stop_opc = opc; break; }

            insn.ops[i] = to_blk_op(&op, op_lnt);
        }

        /* PUSHR: mask must be a compile-time constant (literal or immediate).
           A register-mode mask requires a runtime loop we don't generate. */
        if (ok && opc == PUSHR && insn.ops[0].kind == JITBLK_REGISTER) {
            if (first_stop_opc < 0) first_stop_opc = opc;
            ok = 0;
        }

        if (!ok)
            break;

        blk->insns[blk->n_insns++] = insn;
        blk->fallthrough_pc = pc;
    }

    /* Record the blocking opcode when the very first instruction was rejected */
    if (blk->n_insns == 0 && first_stop_opc >= 0 && first_stop_opc < 256)
        vax_jit_stats.scan_empty_opc[first_stop_opc]++;
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
    case ADDL3: case SUBL3:
    case BISL3: case BICL3: case XORL3:
        return DR_GETNSP(drom[opc][0]);
    case MOVB: case MOVW:
    case MOVZBL: case MOVZWL:
    case PUSHL:
    case ASHL: case MOVQ:
    case MOVAB: case MOVAL:
    case PUSHAB: case PUSHAL:
        return DR_GETNSP(drom[opc][0]);
    case PUSHR:
    case MOVPSL:
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

    /* --- Block path: any memory-mode operand, or width-sensitive ops --- */
    if (needs_block_path(ops, nops) || always_block_path(opc)) {
        VaxJITBlock blk;
        VaxJITBlkInsn *insn = &blk.insns[0];
        blk.n_insns        = 1;
        blk.fallthrough_pc = state->R[nPC];  /* PC already advanced past instruction */
        insn->opc   = opc;
        insn->n_ops = nops;
        /* determine per-operand widths */
        uint8_t w0 = 4, w1 = 4;
        switch (opc) {
        case MOVB:   w0 = 1; w1 = 1; break;
        case MOVW:   w0 = 2; w1 = 2; break;
        case MOVZBL: w0 = 1; w1 = 4; break;
        case MOVZWL: w0 = 2; w1 = 4; break;
        case ASHL:   w0 = 1; w1 = 4; break;  /* ops[2] gets default 4 via loop */
        case MOVQ:   w0 = 8; w1 = 8; break;
        case MOVAB: case MOVAL:
        case PUSHAB: case PUSHAL: w0 = 4; w1 = 4; break;
        default:     break;
        }
        insn->ops[0] = (nops > 0) ? to_blk_op(&ops[0], w0) : (VaxJITBlkOp){0};
        insn->ops[1] = (nops > 1) ? to_blk_op(&ops[1], w1) : (VaxJITBlkOp){0};
        for (i = 2; i < nops; i++)
            insn->ops[i] = to_blk_op(&ops[i], 4);
        return vax_jit_llvm_exec_block(&blk, regs, psl, (int32_t*)M, &sim_interval);
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
