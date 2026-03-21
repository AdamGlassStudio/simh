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
#include "vax_jit_block.h"

/* ------------------------------------------------------------------
   SIMH unit flags for the JIT.  Defined here (rather than vax_cpu.c)
   so that vax_jit.c can read them from cpu_unit.flags at init time.
   Must not overlap with UNIT_V_UF+0 (CONH) or UNIT_V_UF+1 (MSIZE).
   ------------------------------------------------------------------ */
#define UNIT_V_JIT      (UNIT_V_UF + 2)    /* JIT enabled               */
#define UNIT_V_JITDUMP  (UNIT_V_UF + 3)    /* dump LLVM IR to stderr    */
#define UNIT_JIT        (1u << UNIT_V_JIT)
#define UNIT_JITDUMP    (1u << UNIT_V_JITDUMP)

/* ------------------------------------------------------------------
   VAX specifier mode nibble constants (high nibble of each spec byte).
   The full VAX Architecture Reference Manual defines these (ch. 3).
   ------------------------------------------------------------------ */
#define VAX_SPEC_SHORT_LIT_MAX  0x03    /* modes 0-3: short literal       */
#define VAX_SPEC_INDEX          0x04
#define VAX_SPEC_REGISTER       0x05    /* mode 5: register direct        */
#define VAX_SPEC_REG_DEFERRED   0x06
#define VAX_SPEC_AUTODECREMENT  0x07
#define VAX_SPEC_AUTOINCREMENT  0x08    /* +PC (reg=15): immediate        */
#define VAX_SPEC_AUTOINC_DEF    0x09    /* +PC: absolute                  */
#define VAX_SPEC_BYTE_DISP      0x0A    /* +PC: byte PC-relative          */
#define VAX_SPEC_BYTE_DISP_DEF  0x0B
#define VAX_SPEC_WORD_DISP      0x0C    /* +PC: word PC-relative          */
#define VAX_SPEC_WORD_DISP_DEF  0x0D
#define VAX_SPEC_LONG_DISP      0x0E    /* +PC: longword PC-relative      */
#define VAX_SPEC_LONG_DISP_DEF  0x0F

/* Convenience macros on a raw specifier byte */
#define VAX_SPEC_MODE(s)    (((s) >> 4) & 0xF)
#define VAX_SPEC_REG(s)     ((s) & 0xF)
#define VAX_SPEC_IS_SHORT_LIT(s)  (VAX_SPEC_MODE(s) <= VAX_SPEC_SHORT_LIT_MAX)
#define VAX_SPEC_IS_IMMEDIATE(s)  (VAX_SPEC_MODE(s) == VAX_SPEC_AUTOINCREMENT \
                                   && VAX_SPEC_REG(s) == nPC)

/* ------------------------------------------------------------------
   Decoded operand: built by vax_jit_decode_operand() in vax_jit.c.
   VaxJITOpKind values MUST stay in sync with VaxJITBlkOpKind in
   vax_jit_block.h (same ordinal values; to_blk_op() casts between them).
   ------------------------------------------------------------------ */

typedef enum {
    JITOPK_UNSUPPORTED = 0, /* mode not yet handled - fall to interpreter */
    JITOPK_REGISTER,        /* R[reg]                                      */
    JITOPK_LITERAL,         /* short literal 0-63, value in .imm           */
    JITOPK_IMMEDIATE,       /* longword immediate, value in .imm           */
    JITOPK_REG_DEFERRED,    /* (Rn)    - M[R[n]]         reg=n            */
    JITOPK_AUTODECREMENT,   /* -(Rn)   - M[--R[n]]       reg=n            */
    JITOPK_AUTOINCREMENT,   /* (Rn)+   - M[R[n]++]       reg=n            */
    JITOPK_AUTOINC_DEF,     /* @(Rn)+  - M[M[R[n]++]]   reg=n            */
    JITOPK_ABSOLUTE,        /* @#addr  - M[addr]          imm=addr         */
    JITOPK_ABS_DEFERRED,    /* @(PCrel-deferred) - M[M[addr]] imm=addr    */
    JITOPK_DISP,            /* d(Rn)   - M[R[n]+d]       reg=n, imm=d    */
    JITOPK_DISP_DEFERRED,   /* @d(Rn)  - M[M[R[n]+d]]   reg=n, imm=d    */
} VaxJITOpKind;

typedef struct {
    VaxJITOpKind kind;
    int          reg;   /* register index                                  */
    int32        imm;   /* value for literal/immediate/displacement/addr   */
} VaxJITOperand;

/* Decode one operand from a specifier byte and an optional follow value.
   pc_after: PC value after consuming all extension bytes for this operand
   (used to resolve PC-relative addresses to absolute addresses).         */
void vax_jit_decode_operand (int32 spec, int32 follow, int32 pc_after,
                              VaxJITOperand *op);

/* Returns the number of extension bytes that follow the spec byte.
   op_lnt: the logical operand size (for immediate mode).                 */
int  vax_jit_spec_ext_lnt (int32 spec, int32 op_lnt);

/* How many operands the JIT expects for this opcode.
   -1 = opcode not JIT-handled (skip hook entirely).
    0 = no operands (NOP).
    N = N operands to decode.                                             */
int  vax_jit_noperands (int32 opc);

/* Attempt to JIT-execute one instruction.
   Returns 1 if handled (caller continues dispatch loop, interpreter skipped).
   Returns 0 if not handled (caller must restore prefetch state).         */
int  vax_jit_execute (int32 opc, VAXCPUState *state,
                      VaxJITOperand *ops, int nops);

/* Scan VAX instructions from start_pc in mem[], building a multi-instruction
   block.  Stops at branch/call/return opcodes, unknown opcodes, unsupported
   operand modes, or VAX_JIT_MAX_INSNS instructions.
   fallthrough_pc is set to the PC after the last accepted instruction.   */
void vax_jit_scan_block (int32_t start_pc, VaxJITBlock *blk, int32_t *mem);

/* Compile and execute a multi-instruction block.
   Writes fallthrough_pc into regs[15] before returning.
   Returns 1 on success, 0 on compile failure.                           */
int  vax_jit_llvm_exec_block (VaxJITBlock *blk, int32_t *regs,
                               int32_t *psl, int32_t *mem);

/* ------------------------------------------------------------------
   JIT telemetry: counters updated by vax_cpu.c, read by SHOW CPU JITSTATS
   ------------------------------------------------------------------ */
typedef struct {
    uint64_t blocks_run;       /* JIT blocks executed successfully        */
    uint64_t insns_jit;        /* instructions executed via JIT           */
    uint64_t insns_interp;     /* instructions executed via interpreter   */
    uint64_t scan_empty;       /* scanner found 0 JIT-able instructions   */
    uint64_t compile_fail;     /* compile_block returned NULL             */
    uint32_t size_hist[33];    /* size_hist[n]: blocks of exactly n insns */
} VaxJITStats;

extern VaxJITStats vax_jit_stats;

/* Reset all counters to zero. */
void vax_jit_stats_reset (void);

/* Print a formatted stats report to st. */
t_stat vax_jit_stats_show (FILE *st, UNIT *uptr, int32 val, CONST void *desc);

/* SIMH SET handler: SET CPU JITSTATS=RESET clears counters. */
t_stat vax_jit_stats_set  (UNIT *uptr, int32 val, CONST char *cptr, void *desc);

/* Initialise / shut down the LLVM ORC JIT engine.                       */
int  vax_jit_init    (void);
void vax_jit_destroy (void);

/* 1 when JIT is enabled (SET CPU JIT), 0 when disabled (SET CPU NOJIT). */
extern int vax_jit_enabled;

/* 1 when IR dump is enabled (SET CPU JITDUMP). */
extern int vax_jit_ir_dump;

#endif /* VAX_JIT_H */
