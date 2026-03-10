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
#define VAX_SPEC_SHORT_LIT_MAX  0x03    /* modes 0–3: short literal       */
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
   The Option A hook in vax_cpu.c reads the raw bytes; this struct
   carries the semantic result so all instruction handlers share one
   decode path.
   ------------------------------------------------------------------ */

typedef enum {
    JITOPK_UNSUPPORTED = 0, /* mode not yet handled — fall to interpreter */
    JITOPK_REGISTER,        /* R[reg]                                      */
    JITOPK_LITERAL,         /* short literal 0–63, value in .imm           */
    JITOPK_IMMEDIATE        /* longword immediate, value in .imm           */
} VaxJITOpKind;

typedef struct {
    VaxJITOpKind kind;
    int          reg;   /* register index (JITOPK_REGISTER only)          */
    int32        imm;   /* value (JITOPK_LITERAL / JITOPK_IMMEDIATE)      */
} VaxJITOperand;

/* Decode one operand from a specifier byte (already read from the stream)
   and an optional follow longword (only consumed for immediate mode).
   Result written into *op.                                               */
void vax_jit_decode_operand (int32 spec, int32 follow, VaxJITOperand *op);

/* Returns 1 if the specifier byte requires a following longword to be
   read from the instruction stream (immediate longword: 0x8F).
   The caller (vax_cpu.c) uses this to know whether to call
   get_istr(L_LONG) after the spec byte.                                  */
int  vax_jit_operand_needs_long (int32 spec);

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

/* Initialise / shut down the LLVM ORC JIT engine.                       */
int  vax_jit_init    (void);
void vax_jit_destroy (void);

/* 1 when JIT is enabled (SET CPU JIT), 0 when disabled (SET CPU NOJIT). */
extern int vax_jit_enabled;

/* 1 when IR dump is enabled (SET CPU JITDUMP) — prints LLVM IR for each
   compiled handler to stderr.  Useful for inspecting generated code.    */
extern int vax_jit_ir_dump;

#endif /* VAX_JIT_H */
