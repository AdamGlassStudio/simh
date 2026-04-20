/* VAX/vax_jit_block.h — plain-C block descriptor (no SIMH dependencies)
 *
 * This header is shared between vax_jit.c (SIMH-facing) and
 * vax_jit_llvm.c (LLVM-facing).  It must include only <stdint.h>
 * so that LLVM and SIMH headers never mix in the same translation unit.
 */
#ifndef VAX_JIT_BLOCK_H
#define VAX_JIT_BLOCK_H
#include <stdint.h>
#define VAX_JIT_MAX_OPS 6

/* Branch condition codes for intra-region branches.
   VAX_BCOND_NONE = not a branch instruction.
   VAX_BCOND_UNCOND = unconditional (BRB, BRW).
   Others correspond to the named VAX conditional branch instructions. */
typedef enum {
    VAX_BCOND_NONE  = 0,  /* not a branch */
    VAX_BCOND_UNCOND,     /* BRB(0x11), BRW(0x31) — unconditional */
    VAX_BCOND_NEQ,        /* BNEQ(0x12): Z=0 */
    VAX_BCOND_EQL,        /* BEQL(0x13): Z=1 */
    VAX_BCOND_GTR,        /* BGTR(0x14): !Z && N=V */
    VAX_BCOND_LEQ,        /* BLEQ(0x15):  Z || N!=V */
    VAX_BCOND_GEQ,        /* BGEQ(0x18): N=V */
    VAX_BCOND_LSS,        /* BLSS(0x19): N!=V */
    VAX_BCOND_GTRU,       /* BGTRU(0x1A): !C && !Z */
    VAX_BCOND_LEQU,       /* BLEQU(0x1B):  C || Z */
    VAX_BCOND_VC,         /* BVC(0x1C): !V */
    VAX_BCOND_VS,         /* BVS(0x1D):  V */
    VAX_BCOND_GEQU,       /* BGEQU(0x1E): !C */
    VAX_BCOND_LSSU,       /* BLSSU(0x1F):  C */
} VaxBCond;

typedef enum {
    JITBLK_UNSUPPORTED = 0,
    JITBLK_REGISTER,
    JITBLK_LITERAL,
    JITBLK_IMMEDIATE,
    JITBLK_REG_DEFERRED,
    JITBLK_AUTODECREMENT,
    JITBLK_AUTOINCREMENT,
    JITBLK_AUTOINC_DEF,
    JITBLK_ABSOLUTE,
    JITBLK_ABS_DEFERRED,
    JITBLK_DISP,
    JITBLK_DISP_DEFERRED,
} VaxJITBlkOpKind;

typedef struct {
    VaxJITBlkOpKind kind;
    int             reg;
    int32_t         imm;
    uint8_t         width;   /* operand access width: 1, 2, or 4 */
} VaxJITBlkOp;

typedef struct {
    int32_t     opc;
    int         n_ops;
    VaxJITBlkOp ops[VAX_JIT_MAX_OPS];
    int32_t     insn_pc;       /* guest VA of the opcode byte */
    int32_t     branch_target; /* -1 = not a branch; else target guest PC */
    uint8_t     branch_cond;   /* VaxBCond value */
} VaxJITBlkInsn;

#define VAX_JIT_MAX_INSNS 64

typedef struct {
    int           n_insns;
    int32_t       fallthrough_pc;  /* PC after the last instruction */
    int32_t       region_start_pc; /* start_pc passed to scan_block */
    int           has_back_edge;   /* 1 if any intra-region backward branch found */
    VaxJITBlkInsn insns[VAX_JIT_MAX_INSNS];
} VaxJITBlock;

/* ---- Multi-segment compile result ---- */

/* One compiled function entry: an (entry_pc, fn, n_insns) tuple. */
typedef struct {
    int32_t entry_pc;
    void   *fn;
    int     n_insns;  /* instructions compiled into this segment */
} VaxJITCompileEntry;

/* Up to VAX_JIT_MAX_SEGMENTS entries per scan region (one per CALLS + 1). */
#define VAX_JIT_MAX_SEGMENTS 4

typedef struct {
    VaxJITCompileEntry entries[VAX_JIT_MAX_SEGMENTS];
    int                n_entries;
} VaxJITCompileResult;
#endif /* VAX_JIT_BLOCK_H */
