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
} VaxJITBlock;
#endif /* VAX_JIT_BLOCK_H */
