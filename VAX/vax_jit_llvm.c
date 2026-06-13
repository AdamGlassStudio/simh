/* vax_jit_llvm.c: LLVM ORC JIT engine for the VAX simulator
 *
 * This file is intentionally isolated from all SIMH and VAX headers.
 * It uses only standard C types so that LLVM's macro-heavy headers
 * never mix with SIMH's own short-name macros (BB, FP, SP, PC, etc.)
 * which would cause collisions.
 *
 * All JIT handlers are compiled once during vax_jit_llvm_init() and
 * cached as native function pointers.  Subsequent calls just invoke
 * the cached pointer — no per-call recompilation.
 *
 * Build requires: -I/usr/include/llvm-c-18
 *                 -I/usr/lib/llvm-18/include
 *                 -lLLVM-18
 */

#include <llvm-c/Core.h>
#include <llvm-c/Analysis.h>
#include <llvm-c/ErrorHandling.h>
#include <llvm-c/LLJIT.h>
#include <llvm-c/Orc.h>
#include <llvm-c/Target.h>
#include <llvm-c/TargetMachine.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "vax_jit_block.h"

/* ------------------------------------------------------------------ */
/* Integer ALU operation kinds                                          */
/* Keep in sync with the matching enum in vax_jit.c.                   */
/* ------------------------------------------------------------------ */

typedef enum {
    VAX_INTL_ADD     = 0,   /* dst + src,   CC_ADD                        */
    VAX_INTL_SUB     = 1,   /* dst - src,   CC_SUB                        */
    VAX_INTL_OR      = 2,   /* dst | src,   CC_LOG (V=0,C=0)              */
    VAX_INTL_AND_NOT = 3,   /* dst & ~src,  CC_LOG                        */
    VAX_INTL_XOR     = 4,   /* dst ^ src,   CC_LOG                        */
    VAX_INTL_MOV     = 5,   /* src,         CC_LOG (dst_old ignored)      */
    VAX_INTL_COM     = 6,   /* ~src,        CC_LOG (dst_old ignored)      */
    VAX_INTL_NOPS    = 7
} VaxIntOpL;

/* ------------------------------------------------------------------ */
/* LLVM / ORC state                                                     */
/* ------------------------------------------------------------------ */

static LLVMOrcLLJITRef jit = NULL;
static LLVMContextRef  ctx = NULL;
static int             ir_dump = 0;   /* set by vax_jit_llvm_set_ir_dump() */

static void llvm_fatal(const char *reason)
{
    fprintf(stderr, "LLVM fatal error: %s\n", reason);
    exit(1);
}

/* ------------------------------------------------------------------ */
/* Compiled handler cache                                               */
/* ------------------------------------------------------------------ */

/* intl2: void fn(regs, psl, src_is_const, src_val, dst_reg) */
typedef void (*IntL2Fn)(int32_t *, int32_t *, int32_t, int32_t, int32_t);
/* cmpl:  void fn(regs, psl, s1_is_const, s1_val, s2_is_const, s2_val) */
typedef void (*CmpLFn)(int32_t *, int32_t *, int32_t, int32_t, int32_t, int32_t);
/* tstl:  void fn(regs, psl, src_is_const, src_val) */
typedef void (*TstLFn)(int32_t *, int32_t *, int32_t, int32_t);
/* nop:   void fn(void) */
typedef void (*NopFn)(void);

static IntL2Fn fn_intl2[VAX_INTL_NOPS];
static CmpLFn  fn_cmpl;
static TstLFn  fn_tstl;
static NopFn   fn_nop;

/* Address of the guest "mapen" flag (see vax_jit_llvm_register_mmu_state).
   Baked into each compiled block as a constant pointer; JIT loads its
   value at runtime to branch between inline fast-path and helper call. */
static int32_t *mapen_ptr = NULL;

/* ------------------------------------------------------------------ */
/* Emitter context                                                     */
/* ------------------------------------------------------------------ */

/* Per-instruction emitter context. Threaded into memory-touching helpers
   so we have a single place to plumb additional state (PSL pointer,
   faulting PC, future fault-exit basic block, MMU helper pointers, etc.)
   as the inline fault-check work lands. Today it only carries cur_pc;
   helper bodies still use their existing positional parameters so this
   refactor is intentionally behavior-preserving. */
typedef struct RegShadow RegShadow;
typedef struct {
    uint32_t          cur_pc;       /* PC of instruction currently being emitted */
    LLVMValueRef      regs;         /* i32* register file param */
    LLVMValueRef      psl;          /* i32* PSL param */
    LLVMValueRef      fault_slot;   /* i32* alloca in entry BB; helper writes 0/1 */
    LLVMBasicBlockRef fault_exit_bb;/* shared per-block: phi cur_pc, store to R15, ret */
    LLVMValueRef      fault_pc_phi; /* phi i32 in fault_exit_bb */
    RegShadow        *shadow;       /* current shadow state (for spilling on slow path) */
} EmitCtx;

/* ------------------------------------------------------------------ */
/* IR build helpers                                                     */
/* ------------------------------------------------------------------ */

/* Resolve a "maybe-const" source operand.
   is_const != 0  → use val as a sign-extended literal.
   is_const == 0  → load from regs[val].
   A safe index (0) is used for the GEP when const so that val (which may
   be a literal up to 63) never indexes out of the 16-entry register file. */
static LLVMValueRef build_resolve(LLVMBuilderRef b, LLVMTypeRef i32,
                                   LLVMValueRef v_regs,
                                   LLVMValueRef v_ic, LLVMValueRef v_val)
{
    LLVMValueRef zero     = LLVMConstInt(i32, 0, 0);
    LLVMValueRef is_c     = LLVMBuildICmp(b, LLVMIntNE, v_ic, zero, "is_c");
    LLVMValueRef safe_idx = LLVMBuildSelect(b, is_c, zero, v_val, "sidx");
    LLVMValueRef ptr      = LLVMBuildGEP2(b, i32, v_regs, &safe_idx, 1, "rp");
    LLVMValueRef loaded   = LLVMBuildLoad2(b, i32, ptr, "rv");
    return LLVMBuildSelect(b, is_c, v_val, loaded, "src");
}

/* CC bits for addition (result = dst + src): N,Z,V,C */
static LLVMValueRef build_cc_add(LLVMBuilderRef b, LLVMTypeRef i32,
                                  LLVMValueRef src, LLVMValueRef dst_old,
                                  LLVMValueRef result,
                                  uint32_t mask, uint32_t sign_bit)
{
    LLVMValueRef z     = LLVMConstInt(i32, 0, 0);
    LLVMValueRef msign = LLVMConstInt(i32, sign_bit, 0);
    LLVMValueRef mmask = LLVMConstInt(i32, mask, 0);
    LLVMValueRef cn    = LLVMConstInt(i32, 0x08, 0);
    LLVMValueRef cz    = LLVMConstInt(i32, 0x04, 0);
    LLVMValueRef cv    = LLVMConstInt(i32, 0x02, 0);
    LLVMValueRef cc    = LLVMConstInt(i32, 0x01, 0);

    LLVMValueRef nb = LLVMBuildSelect(b,
        LLVMBuildICmp(b, LLVMIntNE,
            LLVMBuildAnd(b, result, msign, "sn"), z, "nc"), cn, z, "n");
    LLVMValueRef zb = LLVMBuildSelect(b,
        LLVMBuildICmp(b, LLVMIntEQ,
            LLVMBuildAnd(b, result, mmask, "zm"), z, "zc"), cz, z, "z");

    /* V: (~src ^ dst_old) & (src ^ result) has sign_bit set */
    LLVMValueRef t1 = LLVMBuildXor(b, LLVMBuildNot(b, src, "ns"), dst_old, "t1");
    LLVMValueRef t2 = LLVMBuildXor(b, src, result, "t2");
    LLVMValueRef vm = LLVMBuildAnd(b, LLVMBuildAnd(b, t1, t2, "t3"), msign, "vm");
    LLVMValueRef vb = LLVMBuildSelect(b,
        LLVMBuildICmp(b, LLVMIntNE, vm, z, "vc"), cv, z, "v");

    /* C: for longword, unsigned wrap: result < dst_old.
          for sub-word, no i32 wrap: result > mask. */
    LLVMValueRef cb;
    if (mask == 0xFFFFFFFFu) {
        cb = LLVMBuildSelect(b,
            LLVMBuildICmp(b, LLVMIntULT, result, dst_old, "cc_c"), cc, z, "c");
    } else {
        cb = LLVMBuildSelect(b,
            LLVMBuildICmp(b, LLVMIntUGT, result, mmask, "cc_c"), cc, z, "c");
    }

    return LLVMBuildOr(b,
               LLVMBuildOr(b, LLVMBuildOr(b, nb, zb, "nz"), vb, "nzv"),
               cb, "cc");
}

/* CC bits for subtraction (result = dst - src): N,Z,V,C */
static LLVMValueRef build_cc_sub(LLVMBuilderRef b, LLVMTypeRef i32,
                                  LLVMValueRef src, LLVMValueRef dst_old,
                                  LLVMValueRef result,
                                  uint32_t mask, uint32_t sign_bit)
{
    LLVMValueRef z     = LLVMConstInt(i32, 0, 0);
    LLVMValueRef msign = LLVMConstInt(i32, sign_bit, 0);
    LLVMValueRef mmask = LLVMConstInt(i32, mask, 0);
    LLVMValueRef cn    = LLVMConstInt(i32, 0x08, 0);
    LLVMValueRef cz    = LLVMConstInt(i32, 0x04, 0);
    LLVMValueRef cv    = LLVMConstInt(i32, 0x02, 0);
    LLVMValueRef cc    = LLVMConstInt(i32, 0x01, 0);

    LLVMValueRef nb = LLVMBuildSelect(b,
        LLVMBuildICmp(b, LLVMIntNE,
            LLVMBuildAnd(b, result, msign, "sn"), z, "nc"), cn, z, "n");
    LLVMValueRef zb = LLVMBuildSelect(b,
        LLVMBuildICmp(b, LLVMIntEQ,
            LLVMBuildAnd(b, result, mmask, "zm"), z, "zc"), cz, z, "z");

    /* V: (dst ^ src) & (dst ^ result) has sign_bit set */
    LLVMValueRef t1 = LLVMBuildAnd(b,
                          LLVMBuildXor(b, dst_old, src,    "t1"),
                          LLVMBuildXor(b, dst_old, result, "t2"), "t3");
    LLVMValueRef vm = LLVMBuildAnd(b, t1, msign, "vm");
    LLVMValueRef vb = LLVMBuildSelect(b,
        LLVMBuildICmp(b, LLVMIntNE, vm, z, "vc"), cv, z, "v");

    /* C: borrow = src > dst_old (unsigned) — works for all widths */
    LLVMValueRef cb = LLVMBuildSelect(b,
        LLVMBuildICmp(b, LLVMIntUGT, src, dst_old, "cc_c"), cc, z, "c");

    (void)mmask;

    return LLVMBuildOr(b,
               LLVMBuildOr(b, LLVMBuildOr(b, nb, zb, "nz"), vb, "nzv"),
               cb, "cc");
}

/* CC bits for logical ops: N,Z set; V=0,C=0 */
static LLVMValueRef build_cc_logical(LLVMBuilderRef b, LLVMTypeRef i32,
                                      LLVMValueRef result,
                                      uint32_t mask, uint32_t sign_bit)
{
    LLVMValueRef z    = LLVMConstInt(i32, 0, 0);
    LLVMValueRef cn   = LLVMConstInt(i32, 0x08, 0);
    LLVMValueRef cz   = LLVMConstInt(i32, 0x04, 0);
    LLVMValueRef msign = LLVMConstInt(i32, sign_bit, 0);
    LLVMValueRef mmask = LLVMConstInt(i32, mask, 0);
    LLVMValueRef nb   = LLVMBuildSelect(b,
        LLVMBuildICmp(b, LLVMIntNE,
            LLVMBuildAnd(b, result, msign, "sn"), z, "nc"), cn, z, "n");
    LLVMValueRef zb   = LLVMBuildSelect(b,
        LLVMBuildICmp(b, LLVMIntEQ,
            LLVMBuildAnd(b, result, mmask, "zm"), z, "zc"), cz, z, "z");
    return LLVMBuildOr(b, nb, zb, "cc");
}

/* PSL = (PSL & ~0xF) | cc_bits  — for arithmetic ops: replaces all 4 CC bits */
static void build_psl_update(LLVMBuilderRef b, LLVMTypeRef i32,
                              LLVMValueRef v_psl, LLVMValueRef cc)
{
    LLVMValueRef old  = LLVMBuildLoad2(b, i32, v_psl, "po");
    LLVMValueRef mask = LLVMConstInt(i32, 0xFFFFFFF0u, 0);
    LLVMBuildStore(b, LLVMBuildOr(b,
                       LLVMBuildAnd(b, old, mask, "pm"),
                       cc, "pn"), v_psl);
}

/* PSL = (PSL & ~0xE) | cc_bits  — for logical ops: preserves C (VAX spec) */
static void build_psl_update_logical(LLVMBuilderRef b, LLVMTypeRef i32,
                                     LLVMValueRef v_psl, LLVMValueRef cc)
{
    LLVMValueRef old  = LLVMBuildLoad2(b, i32, v_psl, "po");
    LLVMValueRef mask = LLVMConstInt(i32, 0xFFFFFFF1u, 0);  /* keep bit 0 (C) */
    LLVMBuildStore(b, LLVMBuildOr(b,
                       LLVMBuildAnd(b, old, mask, "pm"),
                       cc, "pn"), v_psl);
}

/* ------------------------------------------------------------------ */
/* Module helpers                                                       */
/* ------------------------------------------------------------------ */

static LLVMErrorRef jit_add_module(LLVMModuleRef mod)
{
    if (ir_dump) {
        char *ir = LLVMPrintModuleToString(mod);
        fprintf(stderr, "%s\n", ir);
        LLVMDisposeMessage(ir);
    }
    LLVMOrcThreadSafeModuleRef tsm =
        LLVMOrcCreateNewThreadSafeModule(mod,
            LLVMOrcCreateNewThreadSafeContext());
    return LLVMOrcLLJITAddLLVMIRModule(jit,
               LLVMOrcLLJITGetMainJITDylib(jit), tsm);
}

static void *jit_lookup(const char *name)
{
    LLVMOrcExecutorAddress addr = 0;
    LLVMErrorRef err = LLVMOrcLLJITLookup(jit, &addr, name);
    if (err) {
        char *msg = LLVMGetErrorMessage(err);
        fprintf(stderr, "vax_jit: lookup %s: %s\n", name, msg);
        LLVMDisposeErrorMessage(msg);
        return NULL;
    }
    return (void *)(uintptr_t)addr;
}

/* ------------------------------------------------------------------ */
/* Compile: NOP                                                         */
/* ------------------------------------------------------------------ */

static int compile_nop(void)
{
    LLVMModuleRef  mod = LLVMModuleCreateWithNameInContext("m_nop", ctx);
    LLVMValueRef   fn  = LLVMAddFunction(mod, "vax_nop",
                             LLVMFunctionType(LLVMVoidTypeInContext(ctx),
                                              NULL, 0, 0));
    LLVMBuilderRef b   = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(b,
        LLVMAppendBasicBlockInContext(ctx, fn, "entry"));
    LLVMBuildRetVoid(b);
    LLVMDisposeBuilder(b);

    LLVMErrorRef err = jit_add_module(mod);
    if (err) { char *m = LLVMGetErrorMessage(err);
               fprintf(stderr, "vax_jit: compile nop: %s\n", m);
               LLVMDisposeErrorMessage(m); return 0; }
    fn_nop = (NopFn)jit_lookup("vax_nop");
    return fn_nop != NULL;
}

/* ------------------------------------------------------------------ */
/* Compile: integer longword 2-op (intl2 family)                       */
/* void fn(ptr regs, ptr psl, i32 src_is_const, i32 src_val, i32 dst_reg) */
/* ------------------------------------------------------------------ */

static int compile_intl2(VaxIntOpL op, const char *sym)
{
    LLVMTypeRef i32     = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef ptr     = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef ps[5]   = { ptr, ptr, i32, i32, i32 };
    LLVMModuleRef mod   = LLVMModuleCreateWithNameInContext(sym, ctx);
    LLVMValueRef  fn    = LLVMAddFunction(mod, sym,
                              LLVMFunctionType(LLVMVoidTypeInContext(ctx),
                                               ps, 5, 0));
    LLVMBuilderRef b    = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(b,
        LLVMAppendBasicBlockInContext(ctx, fn, "entry"));

    LLVMValueRef v_regs = LLVMGetParam(fn, 0);
    LLVMValueRef v_psl  = LLVMGetParam(fn, 1);
    LLVMValueRef v_sic  = LLVMGetParam(fn, 2);
    LLVMValueRef v_sv   = LLVMGetParam(fn, 3);
    LLVMValueRef v_dr   = LLVMGetParam(fn, 4);

    LLVMValueRef src     = build_resolve(b, i32, v_regs, v_sic, v_sv);
    LLVMValueRef dst_ptr = LLVMBuildGEP2(b, i32, v_regs, &v_dr, 1, "dp");
    LLVMValueRef dst_old = LLVMBuildLoad2(b, i32, dst_ptr, "dv");

    LLVMValueRef result, cc;
    switch (op) {
    case VAX_INTL_ADD:
        result = LLVMBuildAdd(b, dst_old, src, "r");
        cc     = build_cc_add(b, i32, src, dst_old, result, 0xFFFFFFFFu, 0x80000000u);
        break;
    case VAX_INTL_SUB:
        result = LLVMBuildSub(b, dst_old, src, "r");
        cc     = build_cc_sub(b, i32, src, dst_old, result, 0xFFFFFFFFu, 0x80000000u);
        break;
    case VAX_INTL_OR:
        result = LLVMBuildOr(b, dst_old, src, "r");
        cc     = build_cc_logical(b, i32, result, 0xFFFFFFFFu, 0x80000000u);
        break;
    case VAX_INTL_AND_NOT: {
        LLVMValueRef ns = LLVMBuildNot(b, src, "ns");
        result = LLVMBuildAnd(b, dst_old, ns, "r");
        cc     = build_cc_logical(b, i32, result, 0xFFFFFFFFu, 0x80000000u);
        break;
    }
    case VAX_INTL_XOR:
        result = LLVMBuildXor(b, dst_old, src, "r");
        cc     = build_cc_logical(b, i32, result, 0xFFFFFFFFu, 0x80000000u);
        break;
    case VAX_INTL_MOV:
        result = src;
        cc     = build_cc_logical(b, i32, result, 0xFFFFFFFFu, 0x80000000u);
        break;
    case VAX_INTL_COM:
        result = LLVMBuildNot(b, src, "r");
        cc     = build_cc_logical(b, i32, result, 0xFFFFFFFFu, 0x80000000u);
        break;
    default:
        LLVMDisposeBuilder(b);
        LLVMDisposeModule(mod);
        return 0;
    }

    LLVMBuildStore(b, result, dst_ptr);
    build_psl_update(b, i32, v_psl, cc);
    LLVMBuildRetVoid(b);
    LLVMDisposeBuilder(b);

    LLVMErrorRef err = jit_add_module(mod);
    if (err) { char *m = LLVMGetErrorMessage(err);
               fprintf(stderr, "vax_jit: compile %s: %s\n", sym, m);
               LLVMDisposeErrorMessage(m); return 0; }
    fn_intl2[op] = (IntL2Fn)jit_lookup(sym);
    return fn_intl2[op] != NULL;
}

/* ------------------------------------------------------------------ */
/* Compile: CMPL  (src1 - src2, CC_SUB, no store)                      */
/* void fn(ptr regs, ptr psl,                                           */
/*         i32 s1_is_const, i32 s1_val, i32 s2_is_const, i32 s2_val)   */
/* ------------------------------------------------------------------ */

static int compile_cmpl(void)
{
    LLVMTypeRef i32   = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef ptr   = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef ps[6] = { ptr, ptr, i32, i32, i32, i32 };
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext("m_cmpl", ctx);
    LLVMValueRef  fn  = LLVMAddFunction(mod, "vax_cmpl",
                            LLVMFunctionType(LLVMVoidTypeInContext(ctx),
                                             ps, 6, 0));
    LLVMBuilderRef b  = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(b,
        LLVMAppendBasicBlockInContext(ctx, fn, "entry"));

    LLVMValueRef v_regs = LLVMGetParam(fn, 0);
    LLVMValueRef v_psl  = LLVMGetParam(fn, 1);

    /* src1 is the minuend (subtracted FROM), src2 is the subtrahend */
    LLVMValueRef src1 = build_resolve(b, i32, v_regs,
                                       LLVMGetParam(fn, 2), LLVMGetParam(fn, 3));
    LLVMValueRef src2 = build_resolve(b, i32, v_regs,
                                       LLVMGetParam(fn, 4), LLVMGetParam(fn, 5));

    LLVMValueRef result = LLVMBuildSub(b, src1, src2, "r");
    /* CC_SUB: subtrahend=src2, minuend=src1, result=src1-src2 */
    LLVMValueRef cc     = build_cc_sub(b, i32, src2, src1, result, 0xFFFFFFFFu, 0x80000000u);
    build_psl_update(b, i32, v_psl, cc);
    LLVMBuildRetVoid(b);
    LLVMDisposeBuilder(b);

    LLVMErrorRef err = jit_add_module(mod);
    if (err) { char *m = LLVMGetErrorMessage(err);
               fprintf(stderr, "vax_jit: compile cmpl: %s\n", m);
               LLVMDisposeErrorMessage(m); return 0; }
    fn_cmpl = (CmpLFn)jit_lookup("vax_cmpl");
    return fn_cmpl != NULL;
}

/* ------------------------------------------------------------------ */
/* Compile: TSTL  (test src, CC_LOGICAL, no store)                     */
/* void fn(ptr regs, ptr psl, i32 src_is_const, i32 src_val)           */
/* ------------------------------------------------------------------ */

static int compile_tstl(void)
{
    LLVMTypeRef i32   = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef ptr   = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef ps[4] = { ptr, ptr, i32, i32 };
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext("m_tstl", ctx);
    LLVMValueRef  fn  = LLVMAddFunction(mod, "vax_tstl",
                            LLVMFunctionType(LLVMVoidTypeInContext(ctx),
                                             ps, 4, 0));
    LLVMBuilderRef b  = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(b,
        LLVMAppendBasicBlockInContext(ctx, fn, "entry"));

    LLVMValueRef src = build_resolve(b, i32, LLVMGetParam(fn, 0),
                                      LLVMGetParam(fn, 2), LLVMGetParam(fn, 3));
    build_psl_update_logical(b, i32, LLVMGetParam(fn, 1),
                     build_cc_logical(b, i32, src, 0xFFFFFFFFu, 0x80000000u));
    LLVMBuildRetVoid(b);
    LLVMDisposeBuilder(b);

    LLVMErrorRef err = jit_add_module(mod);
    if (err) { char *m = LLVMGetErrorMessage(err);
               fprintf(stderr, "vax_jit: compile tstl: %s\n", m);
               LLVMDisposeErrorMessage(m); return 0; }
    fn_tstl = (TstLFn)jit_lookup("vax_tstl");
    return fn_tstl != NULL;
}

/* ------------------------------------------------------------------ */
/* VAX opcode values (numeric) - no SIMH include needed               */
/* ------------------------------------------------------------------ */

#define VAX_OPC_NOP   0x01
#define VAX_OPC_MOVZWL 0x3C
#define VAX_OPC_ADDL2 0xC0
#define VAX_OPC_ADDL3 0xC1
#define VAX_OPC_SUBL2 0xC2
#define VAX_OPC_SUBL3 0xC3
#define VAX_OPC_BISL2 0xC8
#define VAX_OPC_BISL3 0xC9
#define VAX_OPC_BICL2 0xCA
#define VAX_OPC_BICL3 0xCB
#define VAX_OPC_XORL2 0xCC
#define VAX_OPC_XORL3 0xCD
#define VAX_OPC_MOVL  0xD0
#define VAX_OPC_CMPL  0xD1
#define VAX_OPC_MCOML 0xD2
#define VAX_OPC_CLRL  0xD4
#define VAX_OPC_TSTL  0xD5
#define VAX_OPC_INCL  0xD6
#define VAX_OPC_DECL  0xD7
#define VAX_OPC_ASHL   0x78
#define VAX_OPC_MOVQ   0x7D
#define VAX_OPC_MOVB   0x90
#define VAX_OPC_MOVAB  0x9E
#define VAX_OPC_PUSHAB 0x9F
#define VAX_OPC_MOVZBL 0x9A
#define VAX_OPC_MOVW   0xB0
#define VAX_OPC_PUSHL  0xDD
#define VAX_OPC_MOVAL  0xDE
#define VAX_OPC_PUSHAL 0xDF
#define VAX_OPC_PUSHR  0xBB
#define VAX_OPC_MOVPSL 0xDC

#define VAX_OPC_MOVZBW 0x9B
#define VAX_OPC_MOVF   0x50
#define VAX_OPC_CLRQ   0x7C
#define VAX_OPC_ADWC   0xD8
#define VAX_OPC_SBWC   0xD9
#define VAX_OPC_ADAWI  0x58
#define VAX_OPC_POPR   0xBA
#define VAX_OPC_ASHQ   0x79
#define VAX_OPC_EMUL   0x7A
#define VAX_OPC_EDIV   0x7B
#define VAX_OPC_EXTZV  0xEF
#define VAX_OPC_INSV   0xF0
#define VAX_OPC_MOVC3  0x28

/* ------------------------------------------------------------------ */
/* Register shadow: tracks loaded/modified register values             */
/* NULL = not yet loaded; non-NULL = loaded (and possibly modified).   */
/* ------------------------------------------------------------------ */

typedef struct RegShadow {
    LLVMValueRef val[16];
} RegShadow;

static LLVMValueRef shadow_get(LLVMBuilderRef b, LLVMTypeRef i32,
                                RegShadow *s, int reg, LLVMValueRef regs)
{
    if (!s->val[reg]) {
        LLVMValueRef idx = LLVMConstInt(i32, (unsigned)reg, 0);
        LLVMValueRef ptr = LLVMBuildGEP2(b, i32, regs, &idx, 1, "");
        s->val[reg] = LLVMBuildLoad2(b, i32, ptr, "");
    }
    return s->val[reg];
}

static void shadow_set(RegShadow *s, int reg, LLVMValueRef val)
{
    s->val[reg] = val;
}

static void shadow_spill(LLVMBuilderRef b, LLVMTypeRef i32,
                          RegShadow *s, LLVMValueRef regs)
{
    int i;
    for (i = 0; i < 16; i++) {
        if (!s->val[i]) continue;
        LLVMValueRef idx = LLVMConstInt(i32, (unsigned)i, 0);
        LLVMValueRef ptr = LLVMBuildGEP2(b, i32, regs, &idx, 1, "");
        LLVMBuildStore(b, s->val[i], ptr);
    }
}

/* ------------------------------------------------------------------ */
/* Memory access helpers (VAX byte address -> M[addr>>2])              */
/* ------------------------------------------------------------------ */

/* Inline fast-path: direct host-memory load of width bytes at byte_addr,
   zero-extended to i32. Mirrors the original (pre-helper) IR. */
static LLVMValueRef emit_mem_load_inline(LLVMBuilderRef b, LLVMTypeRef i32,
                                          LLVMValueRef mem,
                                          LLVMValueRef byte_addr, int width)
{
    LLVMContextRef lctx = LLVMGetTypeContext(i32);
    LLVMTypeRef i8  = LLVMInt8TypeInContext(lctx);
    LLVMTypeRef i16 = LLVMInt16TypeInContext(lctx);
    if (width == 1) {
        LLVMValueRef ptr = LLVMBuildGEP2(b, i8, mem, &byte_addr, 1, "bp");
        LLVMValueRef val = LLVMBuildLoad2(b, i8, ptr, "bv");
        return LLVMBuildZExt(b, val, i32, "ze");
    } else if (width == 2) {
        LLVMValueRef ptr = LLVMBuildGEP2(b, i8, mem, &byte_addr, 1, "wp");
        LLVMValueRef val = LLVMBuildLoad2(b, i16, ptr, "wv");
        return LLVMBuildZExt(b, val, i32, "ze");
    } else {
        LLVMValueRef idx = LLVMBuildLShr(b, byte_addr, LLVMConstInt(i32, 2, 0), "wi");
        LLVMValueRef ptr = LLVMBuildGEP2(b, i32, mem, &idx, 1, "mp");
        return LLVMBuildLoad2(b, i32, ptr, "mv");
    }
}

/* Inline fast-path store. */
static void emit_mem_store_inline(LLVMBuilderRef b, LLVMTypeRef i32,
                                   LLVMValueRef mem, LLVMValueRef byte_addr,
                                   LLVMValueRef val, int width)
{
    LLVMContextRef lctx = LLVMGetTypeContext(i32);
    LLVMTypeRef i8  = LLVMInt8TypeInContext(lctx);
    LLVMTypeRef i16 = LLVMInt16TypeInContext(lctx);
    if (width == 1) {
        LLVMValueRef ptr = LLVMBuildGEP2(b, i8, mem, &byte_addr, 1, "bp");
        LLVMValueRef trv = LLVMBuildTrunc(b, val, i8, "bt");
        LLVMBuildStore(b, trv, ptr);
    } else if (width == 2) {
        LLVMValueRef ptr = LLVMBuildGEP2(b, i8, mem, &byte_addr, 1, "wp");
        LLVMValueRef trv = LLVMBuildTrunc(b, val, i16, "wt");
        LLVMBuildStore(b, trv, ptr);
    } else {
        LLVMValueRef idx = LLVMBuildLShr(b, byte_addr, LLVMConstInt(i32, 2, 0), "wi");
        LLVMValueRef ptr = LLVMBuildGEP2(b, i32, mem, &idx, 1, "mp");
        LLVMBuildStore(b, val, ptr);
    }
}

/* Build IR that loads *mapen_ptr at runtime and returns an i1: 1 if the
   guest MMU is enabled, 0 if not. mapen_ptr's address is baked into the
   IR as a constant; this works because cpu_state lives at a stable
   global address. Returns NULL if mapen_ptr was never registered (in
   which case the caller skips the runtime check and always inlines). */
static LLVMValueRef emit_mmu_on_check(LLVMBuilderRef b, LLVMTypeRef i32)
{
    if (!mapen_ptr) return NULL;
    LLVMContextRef lctx = LLVMGetTypeContext(i32);
    LLVMTypeRef    i64  = LLVMInt64TypeInContext(lctx);
    LLVMTypeRef    pi32 = LLVMPointerTypeInContext(lctx, 0);
    LLVMValueRef   addr = LLVMConstInt(i64, (uint64_t)(uintptr_t)mapen_ptr, 0);
    LLVMValueRef   p    = LLVMConstIntToPtr(addr, pi32);
    LLVMValueRef   v    = LLVMBuildLoad2(b, i32, p, "mapen");
    return LLVMBuildICmp(b, LLVMIntNE, v, LLVMConstInt(i32, 0, 0), "mmu_on");
}

/* Spill all live shadow values into the regs[] memory backing.
   Called before the slow-path helper invocation so that, on a fault
   exit, the register file in memory reflects the JIT's working state.
   (Note: any autoincrement side-effects already committed to shadow
   for this instruction are NOT unwound — see vax_jit.c helper notes.) */
static void emit_shadow_spill_for_fault(LLVMBuilderRef b, LLVMTypeRef i32,
                                         EmitCtx *ctx)
{
    if (ctx->shadow)
        shadow_spill(b, i32, ctx->shadow, ctx->regs);
}

/* Emit a memory load with inline fast-path + helper-fallback + fault exit.
   IR shape:
     %mmu_on = load *mapen != 0
     br i1 %mmu_on, %slow, %fast
   fast:                       ; MMU off: direct host-memory access
     %fv = inline load
     br %cont
   slow:
     <spill shadows>            ; for fault-exit visibility
     call helper(regs, psl, mem, va, width, fault_slot)
     %sv = ...
     %f  = load fault_slot
     br %f != 0, %fault_exit, %cont
   cont:
     %v = phi [%fv, fast_end], [%sv, slow_end_ok] */
static LLVMValueRef emit_mem_load(LLVMBuilderRef b, LLVMTypeRef i32,
                                   LLVMValueRef mem, LLVMValueRef byte_addr,
                                   int width, EmitCtx *ctx)
{
    LLVMValueRef mmu_on = emit_mmu_on_check(b, i32);
    if (!mmu_on)
        return emit_mem_load_inline(b, i32, mem, byte_addr, width);

    LLVMContextRef    lctx = LLVMGetTypeContext(i32);
    LLVMValueRef      fn   = LLVMGetBasicBlockParent(LLVMGetInsertBlock(b));
    LLVMBasicBlockRef fast = LLVMAppendBasicBlockInContext(lctx, fn, "mld_fast");
    LLVMBasicBlockRef slow = LLVMAppendBasicBlockInContext(lctx, fn, "mld_slow");
    LLVMBasicBlockRef cont = LLVMAppendBasicBlockInContext(lctx, fn, "mld_cont");
    LLVMBuildCondBr(b, mmu_on, slow, fast);

    LLVMPositionBuilderAtEnd(b, fast);
    LLVMValueRef fv = emit_mem_load_inline(b, i32, mem, byte_addr, width);
    LLVMBasicBlockRef fast_end = LLVMGetInsertBlock(b);
    LLVMBuildBr(b, cont);

    LLVMPositionBuilderAtEnd(b, slow);
    emit_shadow_spill_for_fault(b, i32, ctx);
    LLVMModuleRef mod    = LLVMGetGlobalParent(fn);
    LLVMTypeRef   ptr    = LLVMPointerTypeInContext(lctx, 0);
    LLVMTypeRef   parms[6] = { ptr, ptr, ptr, i32, i32, ptr };
    LLVMTypeRef   fn_ty  = LLVMFunctionType(i32, parms, 6, 0);
    LLVMValueRef  hfn    = LLVMGetNamedFunction(mod, "vax_jit_mem_load_helper");
    if (!hfn)
        hfn = LLVMAddFunction(mod, "vax_jit_mem_load_helper", fn_ty);
    LLVMValueRef args[6] = { ctx->regs, ctx->psl, mem, byte_addr,
                              LLVMConstInt(i32, (unsigned)width, 0),
                              ctx->fault_slot };
    LLVMValueRef sv = LLVMBuildCall2(b, fn_ty, hfn, args, 6, "ml");

    LLVMValueRef f = LLVMBuildLoad2(b, i32, ctx->fault_slot, "fst");
    LLVMValueRef faulted = LLVMBuildICmp(b, LLVMIntNE, f,
                                          LLVMConstInt(i32, 0, 0), "fl");
    LLVMBasicBlockRef slow_ok = LLVMAppendBasicBlockInContext(lctx, fn, "mld_ok");
    LLVMAddIncoming(ctx->fault_pc_phi,
                    (LLVMValueRef[]){ LLVMConstInt(i32, ctx->cur_pc, 0) },
                    (LLVMBasicBlockRef[]){ LLVMGetInsertBlock(b) },
                    1);
    LLVMBuildCondBr(b, faulted, ctx->fault_exit_bb, slow_ok);

    LLVMPositionBuilderAtEnd(b, slow_ok);
    LLVMBuildBr(b, cont);

    LLVMPositionBuilderAtEnd(b, cont);
    LLVMValueRef phi = LLVMBuildPhi(b, i32, "mlv");
    LLVMValueRef in_vals[2] = { fv, sv };
    LLVMBasicBlockRef in_bbs[2] = { fast_end, slow_ok };
    LLVMAddIncoming(phi, in_vals, in_bbs, 2);
    return phi;
}

/* Emit a memory store with inline fast-path + helper-fallback + fault exit. */
static void emit_mem_store(LLVMBuilderRef b, LLVMTypeRef i32,
                            LLVMValueRef mem, LLVMValueRef byte_addr,
                            LLVMValueRef val, int width, EmitCtx *ctx)
{
    LLVMValueRef mmu_on = emit_mmu_on_check(b, i32);
    if (!mmu_on) {
        emit_mem_store_inline(b, i32, mem, byte_addr, val, width);
        return;
    }

    LLVMContextRef    lctx = LLVMGetTypeContext(i32);
    LLVMValueRef      fn   = LLVMGetBasicBlockParent(LLVMGetInsertBlock(b));
    LLVMBasicBlockRef fast = LLVMAppendBasicBlockInContext(lctx, fn, "mst_fast");
    LLVMBasicBlockRef slow = LLVMAppendBasicBlockInContext(lctx, fn, "mst_slow");
    LLVMBasicBlockRef cont = LLVMAppendBasicBlockInContext(lctx, fn, "mst_cont");
    LLVMBuildCondBr(b, mmu_on, slow, fast);

    LLVMPositionBuilderAtEnd(b, fast);
    emit_mem_store_inline(b, i32, mem, byte_addr, val, width);
    LLVMBuildBr(b, cont);

    LLVMPositionBuilderAtEnd(b, slow);
    emit_shadow_spill_for_fault(b, i32, ctx);
    LLVMModuleRef mod    = LLVMGetGlobalParent(fn);
    LLVMTypeRef   ptr    = LLVMPointerTypeInContext(lctx, 0);
    LLVMTypeRef   parms[7] = { ptr, ptr, ptr, i32, i32, i32, ptr };
    LLVMTypeRef   fn_ty  = LLVMFunctionType(LLVMVoidTypeInContext(lctx),
                                            parms, 7, 0);
    LLVMValueRef  hfn    = LLVMGetNamedFunction(mod, "vax_jit_mem_store_helper");
    if (!hfn)
        hfn = LLVMAddFunction(mod, "vax_jit_mem_store_helper", fn_ty);
    LLVMValueRef args[7] = { ctx->regs, ctx->psl, mem, byte_addr, val,
                              LLVMConstInt(i32, (unsigned)width, 0),
                              ctx->fault_slot };
    LLVMBuildCall2(b, fn_ty, hfn, args, 7, "");

    LLVMValueRef f = LLVMBuildLoad2(b, i32, ctx->fault_slot, "fst");
    LLVMValueRef faulted = LLVMBuildICmp(b, LLVMIntNE, f,
                                          LLVMConstInt(i32, 0, 0), "fl");
    LLVMAddIncoming(ctx->fault_pc_phi,
                    (LLVMValueRef[]){ LLVMConstInt(i32, ctx->cur_pc, 0) },
                    (LLVMBasicBlockRef[]){ LLVMGetInsertBlock(b) },
                    1);
    LLVMBuildCondBr(b, faulted, ctx->fault_exit_bb, cont);

    LLVMPositionBuilderAtEnd(b, cont);
}

/* Compute effective byte address for a memory operand.
   For autodec/autoinc, also updates the register shadow.
   Returns NULL for non-memory operand kinds.                          */
static LLVMValueRef emit_operand_ea(LLVMBuilderRef b, LLVMTypeRef i32,
                                     VaxJITBlkOp *op, RegShadow *s,
                                     LLVMValueRef regs, LLVMValueRef mem,
                                     EmitCtx *ctx)
{
    LLVMValueRef base, disp, ea;
    switch (op->kind) {
    case JITBLK_REG_DEFERRED:
        return shadow_get(b, i32, s, op->reg, regs);
    case JITBLK_AUTODECREMENT:
        base = shadow_get(b, i32, s, op->reg, regs);
        ea   = LLVMBuildSub(b, base, LLVMConstInt(i32, op->width, 0), "ad");
        shadow_set(s, op->reg, ea);
        return ea;
    case JITBLK_AUTOINCREMENT:
        base = shadow_get(b, i32, s, op->reg, regs);
        shadow_set(s, op->reg,
                   LLVMBuildAdd(b, base, LLVMConstInt(i32, op->width, 0), "ai"));
        return base;
    case JITBLK_AUTOINC_DEF:
        base = shadow_get(b, i32, s, op->reg, regs);
        shadow_set(s, op->reg,
                   LLVMBuildAdd(b, base, LLVMConstInt(i32, 4, 0), "ai"));
        return emit_mem_load(b, i32, mem, base, 4, ctx);
    case JITBLK_ABSOLUTE:
        return LLVMConstInt(i32, (uint32_t)op->imm, 0);
    case JITBLK_ABS_DEFERRED: {
        LLVMValueRef addr = LLVMConstInt(i32, (uint32_t)op->imm, 0);
        return emit_mem_load(b, i32, mem, addr, 4, ctx);
    }
    case JITBLK_DISP:
        base = shadow_get(b, i32, s, op->reg, regs);
        disp = LLVMConstInt(i32, (uint32_t)op->imm, 0);
        return LLVMBuildAdd(b, base, disp, "ea");
    case JITBLK_DISP_DEFERRED:
        base = shadow_get(b, i32, s, op->reg, regs);
        disp = LLVMConstInt(i32, (uint32_t)op->imm, 0);
        ea   = LLVMBuildAdd(b, base, disp, "ea");
        return emit_mem_load(b, i32, mem, ea, 4, ctx);
    default:
        return NULL;
    }
}

/* Read the value of an operand given its pre-computed EA (NULL for non-memory). */
static LLVMValueRef emit_read_operand(LLVMBuilderRef b, LLVMTypeRef i32,
                                       VaxJITBlkOp *op, LLVMValueRef ea,
                                       RegShadow *s, LLVMValueRef regs,
                                       LLVMValueRef mem, EmitCtx *ctx)
{
    LLVMValueRef val;
    switch (op->kind) {
    case JITBLK_LITERAL:
    case JITBLK_IMMEDIATE:
        return LLVMConstInt(i32, (uint32_t)op->imm, 0);
    case JITBLK_REGISTER:
        val = shadow_get(b, i32, s, op->reg, regs);
        if (op->width < 4) {
            LLVMContextRef lctx = LLVMGetTypeContext(i32);
            LLVMTypeRef iN = (op->width == 1)
                             ? LLVMInt8TypeInContext(lctx)
                             : LLVMInt16TypeInContext(lctx);
            val = LLVMBuildTrunc(b, val, iN, "trunc");
            val = LLVMBuildZExt(b, val, i32, "ze");
        }
        return val;
    default:
        return emit_mem_load(b, i32, mem, ea, op->width, ctx);
    }
}

/* Write val to the operand's destination (shadow for register, memory via ea). */
static void emit_write_operand(LLVMBuilderRef b, LLVMTypeRef i32,
                                VaxJITBlkOp *op, LLVMValueRef ea,
                                LLVMValueRef val, RegShadow *s,
                                LLVMValueRef regs, LLVMValueRef mem,
                                EmitCtx *ctx)
{
    if (op->kind == JITBLK_REGISTER) {
        if (op->width < 4) {
            uint32_t wmask = (op->width == 1) ? 0xFFu : 0xFFFFu;
            LLVMValueRef old_val = shadow_get(b, i32, s, op->reg, regs);
            LLVMValueRef masked  = LLVMBuildAnd(b, old_val,
                                      LLVMConstInt(i32, ~wmask, 0), "msk");
            LLVMValueRef trimmed = LLVMBuildAnd(b, val,
                                      LLVMConstInt(i32, wmask, 0), "trm");
            shadow_set(s, op->reg, LLVMBuildOr(b, masked, trimmed, "mrg"));
        } else {
            shadow_set(s, op->reg, val);
        }
    } else {
        emit_mem_store(b, i32, mem, ea, val, op->width, ctx);
    }
}

/* Forward declaration (defined later in compile_block section) */
static void shadow_clear(RegShadow *s);

/* Emit IR for one VAX instruction into the current basic block. */
static void emit_insn(LLVMBuilderRef b, LLVMTypeRef i32,
                       VaxJITBlkInsn *insn, RegShadow *s,
                       LLVMValueRef regs, LLVMValueRef v_psl,
                       LLVMValueRef mem, EmitCtx *ctx)
{
    LLVMValueRef ea0, ea1, src, dst_old, result, cc;

    ctx->cur_pc = insn->insn_pc;

    switch (insn->opc) {

    case VAX_OPC_NOP:
        return;

    /* ADDL2 src, dst  :  dst = dst + src */
    case VAX_OPC_ADDL2:
        ea0     = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1     = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        src     = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        dst_old = emit_read_operand(b, i32, &insn->ops[1], ea1, s, regs, mem, ctx);
        result  = LLVMBuildAdd(b, dst_old, src, "r");
        cc      = build_cc_add(b, i32, src, dst_old, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &insn->ops[1], ea1, result, s, regs, mem, ctx);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* ADDL3 src1, src2, dst  :  dst = src2 + src1 */
    case VAX_OPC_ADDL3: {
        LLVMValueRef ea2, src2;
        ea0    = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1    = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        ea2    = emit_operand_ea(b, i32, &insn->ops[2], s, regs, mem, ctx);
        src    = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        src2   = emit_read_operand(b, i32, &insn->ops[1], ea1, s, regs, mem, ctx);
        result = LLVMBuildAdd(b, src2, src, "r");
        cc     = build_cc_add(b, i32, src, src2, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &insn->ops[2], ea2, result, s, regs, mem, ctx);
        build_psl_update(b, i32, v_psl, cc);
        return;
    }

    /* SUBL2 src, dst  :  dst = dst - src */
    case VAX_OPC_SUBL2:
        ea0     = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1     = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        src     = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        dst_old = emit_read_operand(b, i32, &insn->ops[1], ea1, s, regs, mem, ctx);
        result  = LLVMBuildSub(b, dst_old, src, "r");
        cc      = build_cc_sub(b, i32, src, dst_old, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &insn->ops[1], ea1, result, s, regs, mem, ctx);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* SUBL3 src1, src2, dst  :  dst = src2 - src1 */
    case VAX_OPC_SUBL3: {
        LLVMValueRef ea2, src2;
        ea0    = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1    = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        ea2    = emit_operand_ea(b, i32, &insn->ops[2], s, regs, mem, ctx);
        src    = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        src2   = emit_read_operand(b, i32, &insn->ops[1], ea1, s, regs, mem, ctx);
        result = LLVMBuildSub(b, src2, src, "r");
        cc     = build_cc_sub(b, i32, src, src2, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &insn->ops[2], ea2, result, s, regs, mem, ctx);
        build_psl_update(b, i32, v_psl, cc);
        return;
    }

    /* BISL2 src, dst  :  dst = dst | src */
    case VAX_OPC_BISL2:
        ea0     = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1     = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        src     = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        dst_old = emit_read_operand(b, i32, &insn->ops[1], ea1, s, regs, mem, ctx);
        result  = LLVMBuildOr(b, dst_old, src, "r");
        cc      = build_cc_logical(b, i32, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &insn->ops[1], ea1, result, s, regs, mem, ctx);
        build_psl_update_logical(b, i32, v_psl, cc);
        return;

    /* BISL3 src1, src2, dst  :  dst = src2 | src1 */
    case VAX_OPC_BISL3: {
        LLVMValueRef ea2, src2;
        ea0    = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1    = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        ea2    = emit_operand_ea(b, i32, &insn->ops[2], s, regs, mem, ctx);
        src    = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        src2   = emit_read_operand(b, i32, &insn->ops[1], ea1, s, regs, mem, ctx);
        result = LLVMBuildOr(b, src2, src, "r");
        cc     = build_cc_logical(b, i32, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &insn->ops[2], ea2, result, s, regs, mem, ctx);
        build_psl_update_logical(b, i32, v_psl, cc);
        return;
    }

    /* BICL2 src, dst  :  dst = dst & ~src */
    case VAX_OPC_BICL2:
        ea0     = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1     = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        src     = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        dst_old = emit_read_operand(b, i32, &insn->ops[1], ea1, s, regs, mem, ctx);
        result  = LLVMBuildAnd(b, dst_old, LLVMBuildNot(b, src, "ns"), "r");
        cc      = build_cc_logical(b, i32, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &insn->ops[1], ea1, result, s, regs, mem, ctx);
        build_psl_update_logical(b, i32, v_psl, cc);
        return;

    /* BICL3 src1, src2, dst  :  dst = src2 & ~src1 */
    case VAX_OPC_BICL3: {
        LLVMValueRef ea2, src2;
        ea0    = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1    = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        ea2    = emit_operand_ea(b, i32, &insn->ops[2], s, regs, mem, ctx);
        src    = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        src2   = emit_read_operand(b, i32, &insn->ops[1], ea1, s, regs, mem, ctx);
        result = LLVMBuildAnd(b, src2, LLVMBuildNot(b, src, "ns"), "r");
        cc     = build_cc_logical(b, i32, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &insn->ops[2], ea2, result, s, regs, mem, ctx);
        build_psl_update_logical(b, i32, v_psl, cc);
        return;
    }

    /* XORL2 src, dst  :  dst = dst ^ src */
    case VAX_OPC_XORL2:
        ea0     = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1     = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        src     = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        dst_old = emit_read_operand(b, i32, &insn->ops[1], ea1, s, regs, mem, ctx);
        result  = LLVMBuildXor(b, dst_old, src, "r");
        cc      = build_cc_logical(b, i32, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &insn->ops[1], ea1, result, s, regs, mem, ctx);
        build_psl_update_logical(b, i32, v_psl, cc);
        return;

    /* XORL3 src1, src2, dst  :  dst = src2 ^ src1 */
    case VAX_OPC_XORL3: {
        LLVMValueRef ea2, src2;
        ea0    = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1    = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        ea2    = emit_operand_ea(b, i32, &insn->ops[2], s, regs, mem, ctx);
        src    = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        src2   = emit_read_operand(b, i32, &insn->ops[1], ea1, s, regs, mem, ctx);
        result = LLVMBuildXor(b, src2, src, "r");
        cc     = build_cc_logical(b, i32, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &insn->ops[2], ea2, result, s, regs, mem, ctx);
        build_psl_update_logical(b, i32, v_psl, cc);
        return;
    }

    /* MOVL src, dst  :  dst = src */
    case VAX_OPC_MOVL:
        ea0    = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1    = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        src    = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        cc     = build_cc_logical(b, i32, src, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &insn->ops[1], ea1, src, s, regs, mem, ctx);
        build_psl_update_logical(b, i32, v_psl, cc);
        return;

    /* MCOML src, dst  :  dst = ~src */
    case VAX_OPC_MCOML:
        ea0    = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1    = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        src    = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        result = LLVMBuildNot(b, src, "r");
        cc     = build_cc_logical(b, i32, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &insn->ops[1], ea1, result, s, regs, mem, ctx);
        build_psl_update_logical(b, i32, v_psl, cc);
        return;

    /* CMPL src1, src2  :  src1 - src2, set CC, no store */
    case VAX_OPC_CMPL:
        ea0     = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1     = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        src     = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        dst_old = emit_read_operand(b, i32, &insn->ops[1], ea1, s, regs, mem, ctx);
        result  = LLVMBuildSub(b, src, dst_old, "r");
        cc      = build_cc_sub(b, i32, dst_old, src, result, 0xFFFFFFFFu, 0x80000000u);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* TSTL src  :  set CC_LOGICAL on src, no store */
    case VAX_OPC_TSTL:
        ea0 = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        src = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        cc  = build_cc_logical(b, i32, src, 0xFFFFFFFFu, 0x80000000u);
        build_psl_update_logical(b, i32, v_psl, cc);
        return;

    /* CLRL dst  :  dst = 0 */
    case VAX_OPC_CLRL:
        ea0    = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        result = LLVMConstInt(i32, 0, 0);
        cc     = build_cc_logical(b, i32, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &insn->ops[0], ea0, result, s, regs, mem, ctx);
        build_psl_update_logical(b, i32, v_psl, cc);
        return;

    /* INCL dst  :  dst = dst + 1 */
    case VAX_OPC_INCL:
        ea0     = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        dst_old = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        src     = LLVMConstInt(i32, 1, 0);
        result  = LLVMBuildAdd(b, dst_old, src, "r");
        cc      = build_cc_add(b, i32, src, dst_old, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &insn->ops[0], ea0, result, s, regs, mem, ctx);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* DECL dst  :  dst = dst - 1 */
    case VAX_OPC_DECL:
        ea0     = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        dst_old = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        src     = LLVMConstInt(i32, 1, 0);
        result  = LLVMBuildSub(b, dst_old, src, "r");
        cc      = build_cc_sub(b, i32, src, dst_old, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &insn->ops[0], ea0, result, s, regs, mem, ctx);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* MOVB src, dst  :  dst[7:0] = src[7:0], CC on byte */
    case VAX_OPC_MOVB:
        ea0 = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1 = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        src = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        cc  = build_cc_logical(b, i32, src, 0xFFu, 0x80u);
        emit_write_operand(b, i32, &insn->ops[1], ea1, src, s, regs, mem, ctx);
        build_psl_update_logical(b, i32, v_psl, cc);
        return;

    /* MOVW src, dst  :  dst[15:0] = src[15:0], CC on word */
    case VAX_OPC_MOVW:
        ea0 = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1 = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        src = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        cc  = build_cc_logical(b, i32, src, 0xFFFFu, 0x8000u);
        emit_write_operand(b, i32, &insn->ops[1], ea1, src, s, regs, mem, ctx);
        build_psl_update_logical(b, i32, v_psl, cc);
        return;

    /* MOVZBL src, dst  :  dst = zero_extend(src[7:0]), CC on longword */
    case VAX_OPC_MOVZBL:
        ea0 = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1 = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        src = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        cc  = build_cc_logical(b, i32, src, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &insn->ops[1], ea1, src, s, regs, mem, ctx);
        build_psl_update_logical(b, i32, v_psl, cc);
        return;

    /* MOVZWL src, dst  :  dst = zero_extend(src[15:0]), CC on longword */
    case VAX_OPC_MOVZWL:
        ea0 = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1 = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        src = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        cc  = build_cc_logical(b, i32, src, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &insn->ops[1], ea1, src, s, regs, mem, ctx);
        build_psl_update_logical(b, i32, v_psl, cc);
        return;

    /* PUSHL src  :  *(--SP) = src, CC on longword */
    case VAX_OPC_PUSHL: {
        LLVMValueRef sp, sp_new;
        ea0    = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        src    = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        sp     = shadow_get(b, i32, s, 14, regs);
        sp_new = LLVMBuildSub(b, sp, LLVMConstInt(i32, 4, 0), "sp_dec");
        shadow_set(s, 14, sp_new);
        emit_mem_store(b, i32, mem, sp_new, src, 4, ctx);
        cc = build_cc_logical(b, i32, src, 0xFFFFFFFFu, 0x80000000u);
        build_psl_update_logical(b, i32, v_psl, cc);
        return;
    }

    /* ASHL cnt, src, dst  :  dst = src << cnt (arithmetic shift) */
    case VAX_OPC_ASHL: {
        LLVMContextRef lctx = LLVMGetTypeContext(i32);
        LLVMTypeRef i8  = LLVMInt8TypeInContext(lctx);
        LLVMTypeRef i1  = LLVMInt1TypeInContext(lctx);
        ea0 = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1 = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        LLVMValueRef ea2 = emit_operand_ea(b, i32, &insn->ops[2], s, regs, mem, ctx);
        LLVMValueRef v_cnt = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        LLVMValueRef v_src = emit_read_operand(b, i32, &insn->ops[1], ea1, s, regs, mem, ctx);
        /* sign-extend cnt from byte to i32 */
        LLVMValueRef cnt8  = LLVMBuildTrunc(b, v_cnt, i8, "c8");
        LLVMValueRef cnt32 = LLVMBuildSExt(b, cnt8, i32, "c32");
        LLVMValueRef zero  = LLVMConstInt(i32, 0, 0);
        LLVMValueRef zero1 = LLVMConstInt(i1, 0, 0);
        LLVMValueRef c31   = LLVMConstInt(i32, 31, 0);
        LLVMValueRef c32   = LLVMConstInt(i32, 32, 0);
        /* determine shift direction */
        LLVMValueRef is_left  = LLVMBuildICmp(b, LLVMIntSGT, cnt32, zero, "il");
        LLVMValueRef is_right = LLVMBuildICmp(b, LLVMIntSLT, cnt32, zero, "ir");
        /* absolute value of cnt */
        LLVMValueRef neg_cnt = LLVMBuildNeg(b, cnt32, "nc");
        LLVMValueRef abs_cnt = LLVMBuildSelect(b, is_left, cnt32, neg_cnt, "ac");
        /* clamp to 31 for safe shifting (LLVM UB for shift >= bitwidth) */
        LLVMValueRef big     = LLVMBuildICmp(b, LLVMIntSGE, abs_cnt, c32, "big");
        LLVMValueRef clamped = LLVMBuildSelect(b, big, c31, abs_cnt, "cl");
        /* left shift: cnt >= 32 → 0 */
        LLVMValueRef shl_res  = LLVMBuildShl(b, v_src, clamped, "ls");
        LLVMValueRef left_res = LLVMBuildSelect(b, big, zero, shl_res, "lr");
        /* V for left shift: round-trip check (shr(shl(src,cnt),cnt) != src) */
        LLVMValueRef recover  = LLVMBuildAShr(b, shl_res, clamped, "rv");
        LLVMValueRef v_left   = LLVMBuildICmp(b, LLVMIntNE, recover, v_src, "vl");
        /* V for cnt >= 32 left: V = (src != 0) */
        LLVMValueRef v_big    = LLVMBuildICmp(b, LLVMIntNE, v_src, zero, "vb");
        LLVMValueRef v_left_f = LLVMBuildSelect(b, big, v_big, v_left, "vlf");
        /* right shift: cnt >= 32 → sign fill (all sign bits) */
        LLVMValueRef shr_res   = LLVMBuildAShr(b, v_src, clamped, "rs");
        LLVMValueRef shr_big   = LLVMBuildAShr(b, v_src, c31, "sb");
        LLVMValueRef right_res = LLVMBuildSelect(b, big, shr_big, shr_res, "rr");
        /* select direction */
        result = LLVMBuildSelect(b, is_left, left_res,
                     LLVMBuildSelect(b, is_right, right_res, v_src, "zr"), "res");
        /* v_flag: V=1 only for left shift with overflow (i1) */
        LLVMValueRef v_flag = LLVMBuildSelect(b, is_left, v_left_f, zero1, "vf");
        emit_write_operand(b, i32, &insn->ops[2], ea2, result, s, regs, mem, ctx);
        /* CC: N/Z from result, V from v_flag, C=0 always */
        LLVMValueRef cn_m = LLVMConstInt(i32, 0x08, 0);
        LLVMValueRef cz_m = LLVMConstInt(i32, 0x04, 0);
        LLVMValueRef cv_m = LLVMConstInt(i32, 0x02, 0);
        LLVMValueRef n_f  = LLVMBuildICmp(b, LLVMIntSLT, result, zero, "n");
        LLVMValueRef z_f  = LLVMBuildICmp(b, LLVMIntEQ,  result, zero, "z");
        LLVMValueRef pn   = LLVMBuildSelect(b, n_f,    cn_m, zero, "pn");
        LLVMValueRef pz   = LLVMBuildSelect(b, z_f,    cz_m, zero, "pz");
        LLVMValueRef pv   = LLVMBuildSelect(b, v_flag, cv_m, zero, "pv");
        cc = LLVMBuildOr(b, LLVMBuildOr(b, pn, pz, "cc1"), pv, "cc");
        build_psl_update(b, i32, v_psl, cc);
        return;
    }

    /* MOVQ src, dst  :  dst[63:0] = src[63:0], CC on quadword */
    case VAX_OPC_MOVQ: {
        LLVMValueRef v_lo, v_hi;
        LLVMValueRef zero = LLVMConstInt(i32, 0, 0);
        /* read source quadword */
        if (insn->ops[0].kind == JITBLK_REGISTER) {
            v_lo = shadow_get(b, i32, s, insn->ops[0].reg,             regs);
            v_hi = shadow_get(b, i32, s, (insn->ops[0].reg + 1) & 15, regs);
        } else {
            LLVMValueRef ea_lo = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
            LLVMValueRef ea_hi = LLVMBuildAdd(b, ea_lo, LLVMConstInt(i32, 4, 0), "eahi");
            v_lo = emit_mem_load(b, i32, mem, ea_lo, 4, ctx);
            v_hi = emit_mem_load(b, i32, mem, ea_hi, 4, ctx);
        }
        /* write destination quadword */
        if (insn->ops[1].kind == JITBLK_REGISTER) {
            shadow_set(s, insn->ops[1].reg,             v_lo);
            shadow_set(s, (insn->ops[1].reg + 1) & 15, v_hi);
        } else {
            LLVMValueRef ea_lo = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
            LLVMValueRef ea_hi = LLVMBuildAdd(b, ea_lo, LLVMConstInt(i32, 4, 0), "eadhi");
            emit_mem_store(b, i32, mem, ea_lo, v_lo, 4, ctx);
            emit_mem_store(b, i32, mem, ea_hi, v_hi, 4, ctx);
        }
        /* CC: N = bit31 of hi, Z = (lo==0 && hi==0), V=0, C=0 */
        LLVMValueRef n   = LLVMBuildICmp(b, LLVMIntSLT, v_hi, zero, "n");
        LLVMValueRef lz  = LLVMBuildICmp(b, LLVMIntEQ,  v_lo, zero, "lz");
        LLVMValueRef hz  = LLVMBuildICmp(b, LLVMIntEQ,  v_hi, zero, "hz");
        LLVMValueRef z   = LLVMBuildAnd(b, lz, hz, "z");
        LLVMValueRef cn_m = LLVMConstInt(i32, 0x08, 0);
        LLVMValueRef cz_m = LLVMConstInt(i32, 0x04, 0);
        LLVMValueRef pn  = LLVMBuildSelect(b, n, cn_m, zero, "pn");
        LLVMValueRef pz  = LLVMBuildSelect(b, z, cz_m, zero, "pz");
        cc = LLVMBuildOr(b, pn, pz, "cc");
        build_psl_update(b, i32, v_psl, cc);
        return;
    }

    /* MOVAB/MOVAL src, dst  :  dst = EA(src) */
    case VAX_OPC_MOVAB:
    case VAX_OPC_MOVAL: {
        LLVMValueRef ea = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        if (!ea) return;   /* register mode: architecturally unpredictable */
        ea1 = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        emit_write_operand(b, i32, &insn->ops[1], ea1, ea, s, regs, mem, ctx);
        cc = build_cc_logical(b, i32, ea, 0xFFFFFFFFu, 0x80000000u);
        build_psl_update_logical(b, i32, v_psl, cc);
        return;
    }

    /* PUSHAB/PUSHAL src  :  *(--SP) = EA(src) */
    case VAX_OPC_PUSHAB:
    case VAX_OPC_PUSHAL: {
        LLVMValueRef ea = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        if (!ea) return;   /* register mode: architecturally unpredictable */
        LLVMValueRef sp     = shadow_get(b, i32, s, 14, regs);
        LLVMValueRef sp_new = LLVMBuildSub(b, sp, LLVMConstInt(i32, 4, 0), "sp");
        shadow_set(s, 14, sp_new);
        emit_mem_store(b, i32, mem, sp_new, ea, 4, ctx);
        cc = build_cc_logical(b, i32, ea, 0xFFFFFFFFu, 0x80000000u);
        build_psl_update_logical(b, i32, v_psl, cc);
        return;
    }

    /* PUSHR mask.rw — push selected registers R14..R0 onto stack.
       mask is always a compile-time constant (scan rejects register mode). */
    case VAX_OPC_PUSHR: {
        LLVMValueRef mask_val = emit_read_operand(b, i32, &insn->ops[0], NULL,
                                                  s, regs, mem, ctx);
        uint32_t mask = (uint32_t)LLVMConstIntGetZExtValue(mask_val) & 0x7FFF;
        LLVMValueRef sp = shadow_get(b, i32, s, 14, regs);
        int ri;
        for (ri = 14; ri >= 0; ri--) {
            if (mask & (1u << ri)) {
                sp = LLVMBuildSub(b, sp, LLVMConstInt(i32, 4, 0), "sp");
                LLVMValueRef rv = shadow_get(b, i32, s, ri, regs);
                emit_mem_store(b, i32, mem, sp, rv, 4, ctx);
            }
        }
        shadow_set(s, 14, sp);  /* update SP in shadow */
        /* PUSHR does not set CC */
        return;
    }

    /* MOVPSL dst.wl — copy current PSL (including CC bits) to destination. */
    case VAX_OPC_MOVPSL: {
        ea0    = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        result = LLVMBuildLoad2(b, i32, v_psl, "psl");
        emit_write_operand(b, i32, &insn->ops[0], ea0, result, s, regs, mem, ctx);
        /* MOVPSL does not alter CC */
        return;
    }

    /* MOVZBW src.rb, dst.ww  :  dst[15:0] = zero_extend(src[7:0]), CC on word */
    case VAX_OPC_MOVZBW:
        ea0 = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1 = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        src = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        cc  = build_cc_logical(b, i32, src, 0xFFFFu, 0x8000u);
        emit_write_operand(b, i32, &insn->ops[1], ea1, src, s, regs, mem, ctx);
        build_psl_update_logical(b, i32, v_psl, cc);
        return;

    /* MOVF src.rf, dst.wf  :  dst = src (F-float is a longword), CC on float sign */
    case VAX_OPC_MOVF:
        ea0 = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1 = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        src = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        /* CC_IIZP_FP = CC_IIZP_W: N from bit 15 (float sign), Z from full longword */
        cc  = build_cc_logical(b, i32, src, 0xFFFFu, 0x8000u);
        emit_write_operand(b, i32, &insn->ops[1], ea1, src, s, regs, mem, ctx);
        build_psl_update_logical(b, i32, v_psl, cc);
        return;

    /* CLRQ dst.wq  :  dst = 0 (quadword), CC: N=0, Z=1, V=0, C preserved */
    case VAX_OPC_CLRQ: {
        LLVMValueRef zero = LLVMConstInt(i32, 0, 0);
        if (insn->ops[0].kind == JITBLK_REGISTER) {
            shadow_set(s, insn->ops[0].reg,             zero);
            shadow_set(s, (insn->ops[0].reg + 1) & 15, zero);
        } else {
            LLVMValueRef ea_lo = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
            LLVMValueRef ea_hi = LLVMBuildAdd(b, ea_lo, LLVMConstInt(i32, 4, 0), "eahi");
            emit_mem_store(b, i32, mem, ea_lo, zero, 4, ctx);
            emit_mem_store(b, i32, mem, ea_hi, zero, 4, ctx);
        }
        /* CC_ZZ1P: Z=1, C preserved */
        LLVMValueRef old_psl = LLVMBuildLoad2(b, i32, v_psl, "po");
        LLVMValueRef c_bit   = LLVMBuildAnd(b, old_psl, LLVMConstInt(i32, 1, 0), "cb");
        cc = LLVMBuildOr(b, LLVMConstInt(i32, 0x04, 0), c_bit, "cc");
        build_psl_update(b, i32, v_psl, cc);
        return;
    }

    /* ADWC src.rl, dst.ml  :  dst = dst + src + C */
    case VAX_OPC_ADWC: {
        ea0     = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1     = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        src     = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        dst_old = emit_read_operand(b, i32, &insn->ops[1], ea1, s, regs, mem, ctx);
        /* Extract carry-in from current PSL bit 0 */
        LLVMValueRef psl_val = LLVMBuildLoad2(b, i32, v_psl, "psl");
        LLVMValueRef carry_in = LLVMBuildAnd(b, psl_val, LLVMConstInt(i32, 1, 0), "cin");
        /* tmp = dst + src; result = tmp + carry_in */
        LLVMValueRef tmp = LLVMBuildAdd(b, dst_old, src, "tmp");
        result = LLVMBuildAdd(b, tmp, carry_in, "r");
        /* CC_ADD_L(r, op0, op1): standard add CC from src + dst_old */
        cc = build_cc_add(b, i32, src, dst_old, result, 0xFFFFFFFFu, 0x80000000u);
        /* Special case: if (r == op1) && op0 → set C.
           This handles the carry-through case where the main add carried
           but carry_in addition brought result back equal to dst_old. */
        LLVMValueRef zero = LLVMConstInt(i32, 0, 0);
        LLVMValueRef r_eq_dst = LLVMBuildICmp(b, LLVMIntEQ, result, dst_old, "req");
        LLVMValueRef src_nz   = LLVMBuildICmp(b, LLVMIntNE, src, zero, "snz");
        LLVMValueRef special  = LLVMBuildAnd(b, r_eq_dst, src_nz, "spc");
        LLVMValueRef extra_c  = LLVMBuildSelect(b, special, LLVMConstInt(i32, 1, 0), zero, "ec");
        cc = LLVMBuildOr(b, cc, extra_c, "cc2");
        emit_write_operand(b, i32, &insn->ops[1], ea1, result, s, regs, mem, ctx);
        build_psl_update(b, i32, v_psl, cc);
        return;
    }

    /* SBWC src.rl, dst.ml  :  dst = dst - src - C */
    case VAX_OPC_SBWC: {
        ea0     = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1     = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        src     = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        dst_old = emit_read_operand(b, i32, &insn->ops[1], ea1, s, regs, mem, ctx);
        LLVMValueRef psl_val = LLVMBuildLoad2(b, i32, v_psl, "psl");
        LLVMValueRef carry_in = LLVMBuildAnd(b, psl_val, LLVMConstInt(i32, 1, 0), "cin");
        LLVMValueRef tmp = LLVMBuildSub(b, dst_old, src, "tmp");
        result = LLVMBuildSub(b, tmp, carry_in, "r");
        cc = build_cc_sub(b, i32, src, dst_old, result, 0xFFFFFFFFu, 0x80000000u);
        /* Special case: if (op0 == op1) && r → set C */
        LLVMValueRef zero = LLVMConstInt(i32, 0, 0);
        LLVMValueRef eq   = LLVMBuildICmp(b, LLVMIntEQ, src, dst_old, "eq");
        LLVMValueRef r_nz = LLVMBuildICmp(b, LLVMIntNE, result, zero, "rnz");
        LLVMValueRef special = LLVMBuildAnd(b, eq, r_nz, "spc");
        LLVMValueRef extra_c = LLVMBuildSelect(b, special, LLVMConstInt(i32, 1, 0), zero, "ec");
        cc = LLVMBuildOr(b, cc, extra_c, "cc2");
        emit_write_operand(b, i32, &insn->ops[1], ea1, result, s, regs, mem, ctx);
        build_psl_update(b, i32, v_psl, cc);
        return;
    }

    /* ADAWI src.rw, dst.mw  :  dst = dst + src (word), set CC */
    case VAX_OPC_ADAWI: {
        ea0     = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1     = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        src     = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        dst_old = emit_read_operand(b, i32, &insn->ops[1], ea1, s, regs, mem, ctx);
        result  = LLVMBuildAnd(b, LLVMBuildAdd(b, dst_old, src, "sum"),
                               LLVMConstInt(i32, 0xFFFF, 0), "r");
        cc = build_cc_add(b, i32, src, dst_old, result, 0xFFFFu, 0x8000u);
        emit_write_operand(b, i32, &insn->ops[1], ea1, result, s, regs, mem, ctx);
        build_psl_update(b, i32, v_psl, cc);
        return;
    }

    /* POPR mask.rw — pop selected registers R0..R14 from stack. */
    case VAX_OPC_POPR: {
        LLVMValueRef mask_val = emit_read_operand(b, i32, &insn->ops[0], NULL,
                                                   s, regs, mem, ctx);
        uint32_t mask = (uint32_t)LLVMConstIntGetZExtValue(mask_val) & 0x7FFF;
        LLVMValueRef sp = shadow_get(b, i32, s, 14, regs);
        int ri;
        for (ri = 0; ri <= 13; ri++) {
            if (mask & (1u << ri)) {
                LLVMValueRef rv = emit_mem_load(b, i32, mem, sp, 4, ctx);
                shadow_set(s, ri, rv);
                sp = LLVMBuildAdd(b, sp, LLVMConstInt(i32, 4, 0), "sp");
            }
        }
        /* SP (R14): if bit 14 set, load SP from stack but don't increment */
        if (mask & (1u << 14)) {
            LLVMValueRef rv = emit_mem_load(b, i32, mem, sp, 4, ctx);
            shadow_set(s, 14, rv);
        } else {
            shadow_set(s, 14, sp);
        }
        /* POPR does not set CC */
        return;
    }

    /* ASHQ cnt.rb, src.rq, dst.wq  :  arithmetic shift quadword */
    case VAX_OPC_ASHQ: {
        LLVMContextRef lctx = LLVMGetTypeContext(i32);
        LLVMTypeRef i8   = LLVMInt8TypeInContext(lctx);
        LLVMTypeRef i64  = LLVMInt64TypeInContext(lctx);
        LLVMTypeRef i1   = LLVMInt1TypeInContext(lctx);
        LLVMValueRef zero32 = LLVMConstInt(i32, 0, 0);
        LLVMValueRef zero64 = LLVMConstInt(i64, 0, 0);

        ea0 = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        LLVMValueRef v_cnt = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);

        /* Read source quadword */
        LLVMValueRef v_lo, v_hi;
        if (insn->ops[1].kind == JITBLK_REGISTER) {
            v_lo = shadow_get(b, i32, s, insn->ops[1].reg,             regs);
            v_hi = shadow_get(b, i32, s, (insn->ops[1].reg + 1) & 15, regs);
        } else {
            LLVMValueRef ea_lo = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
            LLVMValueRef ea_hi = LLVMBuildAdd(b, ea_lo, LLVMConstInt(i32, 4, 0), "eahi");
            v_lo = emit_mem_load(b, i32, mem, ea_lo, 4, ctx);
            v_hi = emit_mem_load(b, i32, mem, ea_hi, 4, ctx);
        }

        /* Build 64-bit source */
        LLVMValueRef lo64 = LLVMBuildZExt(b, v_lo, i64, "lo64");
        LLVMValueRef hi64 = LLVMBuildShl(b, LLVMBuildZExt(b, v_hi, i64, "hi64e"),
                                          LLVMConstInt(i64, 32, 0), "hi64");
        LLVMValueRef src64 = LLVMBuildOr(b, lo64, hi64, "src64");

        /* Sign-extend cnt from byte */
        LLVMValueRef cnt8  = LLVMBuildTrunc(b, v_cnt, i8, "c8");
        LLVMValueRef cnt32 = LLVMBuildSExt(b, cnt8, i32, "c32");
        LLVMValueRef is_left  = LLVMBuildICmp(b, LLVMIntSGT, cnt32, zero32, "il");
        LLVMValueRef is_right = LLVMBuildICmp(b, LLVMIntSLT, cnt32, zero32, "ir");
        LLVMValueRef neg_cnt  = LLVMBuildNeg(b, cnt32, "nc");
        LLVMValueRef abs_cnt  = LLVMBuildSelect(b, is_left, cnt32, neg_cnt, "ac");
        LLVMValueRef abs_cnt64 = LLVMBuildZExt(b, abs_cnt, i64, "ac64");
        LLVMValueRef c63   = LLVMConstInt(i64, 63, 0);
        LLVMValueRef c64i  = LLVMConstInt(i32, 64, 0);
        LLVMValueRef big   = LLVMBuildICmp(b, LLVMIntSGE, abs_cnt, LLVMConstInt(i32, 64, 0), "big");
        LLVMValueRef clamped64 = LLVMBuildSelect(b, big, c63, abs_cnt64, "cl64");

        /* Left shift */
        LLVMValueRef shl_res  = LLVMBuildShl(b, src64, clamped64, "ls");
        LLVMValueRef left_res = LLVMBuildSelect(b, big, zero64, shl_res, "lr");
        /* V for left shift: round-trip check */
        LLVMValueRef recover  = LLVMBuildAShr(b, shl_res, clamped64, "rv");
        LLVMValueRef v_left   = LLVMBuildICmp(b, LLVMIntNE, recover, src64, "vl");
        LLVMValueRef v_big    = LLVMBuildICmp(b, LLVMIntNE, src64, zero64, "vb");
        LLVMValueRef v_left_f = LLVMBuildSelect(b, big, v_big, v_left, "vlf");

        /* Right shift */
        LLVMValueRef shr_res   = LLVMBuildAShr(b, src64, clamped64, "rs");
        LLVMValueRef shr_big   = LLVMBuildAShr(b, src64, c63, "sb");
        LLVMValueRef right_res = LLVMBuildSelect(b, big, shr_big, shr_res, "rr");

        /* Select direction */
        LLVMValueRef res64 = LLVMBuildSelect(b, is_left, left_res,
                                 LLVMBuildSelect(b, is_right, right_res, src64, "zr"), "res");
        LLVMValueRef v_flag = LLVMBuildSelect(b, is_left, v_left_f,
                                 LLVMConstInt(i1, 0, 0), "vf");

        /* Split result to lo/hi */
        LLVMValueRef r_lo = LLVMBuildTrunc(b, res64, i32, "rlo");
        LLVMValueRef r_hi = LLVMBuildTrunc(b, LLVMBuildLShr(b, res64,
                                LLVMConstInt(i64, 32, 0), "rsh"), i32, "rhi");

        /* Write destination quadword */
        if (insn->ops[2].kind == JITBLK_REGISTER) {
            shadow_set(s, insn->ops[2].reg,             r_lo);
            shadow_set(s, (insn->ops[2].reg + 1) & 15, r_hi);
        } else {
            LLVMValueRef ea_lo = emit_operand_ea(b, i32, &insn->ops[2], s, regs, mem, ctx);
            LLVMValueRef ea_hi = LLVMBuildAdd(b, ea_lo, LLVMConstInt(i32, 4, 0), "eadhi");
            emit_mem_store(b, i32, mem, ea_lo, r_lo, 4, ctx);
            emit_mem_store(b, i32, mem, ea_hi, r_hi, 4, ctx);
        }

        /* CC: N from hi bit31, Z from (lo|hi)==0, V from overflow, C=0 */
        LLVMValueRef n_f = LLVMBuildICmp(b, LLVMIntSLT, r_hi, zero32, "n");
        LLVMValueRef lz  = LLVMBuildICmp(b, LLVMIntEQ,  r_lo, zero32, "lz");
        LLVMValueRef hz  = LLVMBuildICmp(b, LLVMIntEQ,  r_hi, zero32, "hz");
        LLVMValueRef z_f = LLVMBuildAnd(b, lz, hz, "z");
        LLVMValueRef pn  = LLVMBuildSelect(b, n_f,    LLVMConstInt(i32, 0x08, 0), zero32, "pn");
        LLVMValueRef pz  = LLVMBuildSelect(b, z_f,    LLVMConstInt(i32, 0x04, 0), zero32, "pz");
        LLVMValueRef pv  = LLVMBuildSelect(b, v_flag,  LLVMConstInt(i32, 0x02, 0), zero32, "pv");
        cc = LLVMBuildOr(b, LLVMBuildOr(b, pn, pz, "cc1"), pv, "cc");
        build_psl_update(b, i32, v_psl, cc);
        (void)c64i;
        return;
    }

    /* EMUL mulr.rl, muld.rl, add.rl, prod.wq  :  prod = mulr * muld + sext(add) */
    case VAX_OPC_EMUL: {
        LLVMContextRef lctx = LLVMGetTypeContext(i32);
        LLVMTypeRef i64  = LLVMInt64TypeInContext(lctx);
        LLVMValueRef zero32 = LLVMConstInt(i32, 0, 0);

        ea0 = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1 = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        LLVMValueRef ea2 = emit_operand_ea(b, i32, &insn->ops[2], s, regs, mem, ctx);
        LLVMValueRef ea3 = emit_operand_ea(b, i32, &insn->ops[3], s, regs, mem, ctx);

        LLVMValueRef mulr = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);
        LLVMValueRef muld = emit_read_operand(b, i32, &insn->ops[1], ea1, s, regs, mem, ctx);
        LLVMValueRef add  = emit_read_operand(b, i32, &insn->ops[2], ea2, s, regs, mem, ctx);

        /* Extend to 64-bit signed, multiply, add */
        LLVMValueRef mulr64 = LLVMBuildSExt(b, mulr, i64, "mr64");
        LLVMValueRef muld64 = LLVMBuildSExt(b, muld, i64, "md64");
        LLVMValueRef prod64 = LLVMBuildMul(b, mulr64, muld64, "prod");
        LLVMValueRef add64  = LLVMBuildSExt(b, add, i64, "add64");
        LLVMValueRef res64  = LLVMBuildAdd(b, prod64, add64, "res64");

        /* Split to lo/hi */
        LLVMValueRef r_lo = LLVMBuildTrunc(b, res64, i32, "rlo");
        LLVMValueRef r_hi = LLVMBuildTrunc(b, LLVMBuildLShr(b, res64,
                                LLVMConstInt(i64, 32, 0), "rsh"), i32, "rhi");

        /* Write destination quadword */
        if (insn->ops[3].kind == JITBLK_REGISTER) {
            shadow_set(s, insn->ops[3].reg,             r_lo);
            shadow_set(s, (insn->ops[3].reg + 1) & 15, r_hi);
        } else {
            LLVMValueRef ea_lo = ea3;
            LLVMValueRef ea_hi = LLVMBuildAdd(b, ea_lo, LLVMConstInt(i32, 4, 0), "eadhi");
            emit_mem_store(b, i32, mem, ea_lo, r_lo, 4, ctx);
            emit_mem_store(b, i32, mem, ea_hi, r_hi, 4, ctx);
        }

        /* CC: N from hi bit31, Z from (lo|hi)==0, V=0, C=0 */
        LLVMValueRef n_f = LLVMBuildICmp(b, LLVMIntSLT, r_hi, zero32, "n");
        LLVMValueRef lz  = LLVMBuildICmp(b, LLVMIntEQ,  r_lo, zero32, "lz");
        LLVMValueRef hz  = LLVMBuildICmp(b, LLVMIntEQ,  r_hi, zero32, "hz");
        LLVMValueRef z_f = LLVMBuildAnd(b, lz, hz, "z");
        LLVMValueRef pn  = LLVMBuildSelect(b, n_f, LLVMConstInt(i32, 0x08, 0), zero32, "pn");
        LLVMValueRef pz  = LLVMBuildSelect(b, z_f, LLVMConstInt(i32, 0x04, 0), zero32, "pz");
        cc = LLVMBuildOr(b, pn, pz, "cc");
        build_psl_update(b, i32, v_psl, cc);
        return;
    }

    /* EDIV divr.rl, divd.rq, quo.wl, rem.wl  :  64-bit divide */
    case VAX_OPC_EDIV: {
        LLVMContextRef lctx = LLVMGetTypeContext(i32);
        LLVMTypeRef i64  = LLVMInt64TypeInContext(lctx);
        LLVMTypeRef i1   = LLVMInt1TypeInContext(lctx);
        LLVMValueRef zero32 = LLVMConstInt(i32, 0, 0);
        LLVMValueRef zero64 = LLVMConstInt(i64, 0, 0);

        ea0 = emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx);
        ea1 = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        LLVMValueRef ea2 = emit_operand_ea(b, i32, &insn->ops[2], s, regs, mem, ctx);
        LLVMValueRef ea3 = emit_operand_ea(b, i32, &insn->ops[3], s, regs, mem, ctx);

        LLVMValueRef divr = emit_read_operand(b, i32, &insn->ops[0], ea0, s, regs, mem, ctx);

        /* Read dividend quadword */
        LLVMValueRef dvd_lo, dvd_hi;
        if (insn->ops[1].kind == JITBLK_REGISTER) {
            dvd_lo = shadow_get(b, i32, s, insn->ops[1].reg,             regs);
            dvd_hi = shadow_get(b, i32, s, (insn->ops[1].reg + 1) & 15, regs);
        } else {
            LLVMValueRef ea_lo = ea1;
            LLVMValueRef ea_hi = LLVMBuildAdd(b, ea_lo, LLVMConstInt(i32, 4, 0), "eahi");
            dvd_lo = emit_mem_load(b, i32, mem, ea_lo, 4, ctx);
            dvd_hi = emit_mem_load(b, i32, mem, ea_hi, 4, ctx);
        }

        /* Build 64-bit dividend */
        LLVMValueRef lo64 = LLVMBuildZExt(b, dvd_lo, i64, "lo64");
        LLVMValueRef hi64 = LLVMBuildShl(b, LLVMBuildZExt(b, dvd_hi, i64, "hi64e"),
                                          LLVMConstInt(i64, 32, 0), "hi64");
        LLVMValueRef dvd64 = LLVMBuildOr(b, lo64, hi64, "dvd64");

        /* Check for divide by zero */
        LLVMValueRef is_zero = LLVMBuildICmp(b, LLVMIntEQ, divr, zero32, "dz");

        /* |divisor| and |dividend| */
        LLVMValueRef divr_neg = LLVMBuildICmp(b, LLVMIntSLT, divr, zero32, "dn");
        LLVMValueRef abs_divr = LLVMBuildSelect(b, divr_neg,
                                    LLVMBuildNeg(b, divr, "nd"), divr, "ad");
        LLVMValueRef abs_divr64 = LLVMBuildZExt(b, abs_divr, i64, "ad64");

        LLVMValueRef dvd_neg = LLVMBuildICmp(b, LLVMIntSLT, dvd64, zero64, "dvn");
        LLVMValueRef abs_dvd = LLVMBuildSelect(b, dvd_neg,
                                    LLVMBuildNeg(b, dvd64, "ndvd"), dvd64, "advd");

        /* Check if quotient would overflow: (|dvd| >> 32) >= |divr| */
        LLVMValueRef dvd_hi_u = LLVMBuildLShr(b, abs_dvd, LLVMConstInt(i64, 32, 0), "dvhi");
        LLVMValueRef overflow = LLVMBuildICmp(b, LLVMIntUGE, dvd_hi_u, abs_divr64, "ovf");
        LLVMValueRef bad = LLVMBuildOr(b, is_zero, overflow, "bad");

        /* Safe divisor for division (avoid UB, use 1 when bad) */
        LLVMValueRef safe_divr = LLVMBuildSelect(b, bad,
                                     LLVMConstInt(i64, 1, 0), abs_divr64, "sd");

        /* Unsigned division of absolute values */
        LLVMValueRef quo_u = LLVMBuildUDiv(b, abs_dvd, safe_divr, "qu");
        LLVMValueRef rem_u = LLVMBuildURem(b, abs_dvd, safe_divr, "ru");

        /* Apply signs */
        LLVMValueRef dvd_hi_sign = LLVMBuildICmp(b, LLVMIntSLT, dvd_hi,
                                       zero32, "dhs");
        LLVMValueRef sign_diff = LLVMBuildXor(b, divr_neg, dvd_hi_sign, "sdif");
        LLVMValueRef quo_s = LLVMBuildTrunc(b, quo_u, i32, "qs");
        LLVMValueRef quo_neg = LLVMBuildSelect(b, sign_diff,
                                   LLVMBuildNeg(b, quo_s, "qn"), quo_s, "qf");
        LLVMValueRef rem_s = LLVMBuildTrunc(b, rem_u, i32, "rs");
        LLVMValueRef rem_neg = LLVMBuildSelect(b, dvd_hi_sign,
                                   LLVMBuildNeg(b, rem_s, "rn"), rem_s, "rf");

        /* On bad: quo = dvd_lo, rem = 0 */
        LLVMValueRef quo_final = LLVMBuildSelect(b, bad, dvd_lo, quo_neg, "qfin");
        LLVMValueRef rem_final = LLVMBuildSelect(b, bad, zero32, rem_neg, "rfin");

        /* Write quotient and remainder */
        emit_write_operand(b, i32, &insn->ops[2], ea2, quo_final, s, regs, mem, ctx);
        emit_write_operand(b, i32, &insn->ops[3], ea3, rem_final, s, regs, mem, ctx);

        /* CC: N,Z from quotient, V if bad, C=0 */
        LLVMValueRef n_f = LLVMBuildICmp(b, LLVMIntSLT, quo_final, zero32, "n");
        LLVMValueRef z_f = LLVMBuildICmp(b, LLVMIntEQ,  quo_final, zero32, "z");
        LLVMValueRef pn  = LLVMBuildSelect(b, n_f, LLVMConstInt(i32, 0x08, 0), zero32, "pn");
        LLVMValueRef pz  = LLVMBuildSelect(b, z_f, LLVMConstInt(i32, 0x04, 0), zero32, "pz");
        LLVMValueRef pv  = LLVMBuildSelect(b, bad, LLVMConstInt(i32, 0x02, 0), zero32, "pv");
        cc = LLVMBuildOr(b, LLVMBuildOr(b, pn, pz, "cc1"), pv, "cc");
        build_psl_update(b, i32, v_psl, cc);
        (void)i1;
        return;
    }

    /* EXTZV pos.rl, size.rb, base.vb, dst.wl — C helper call */
    case VAX_OPC_EXTZV: {
        LLVMTypeRef ptr = LLVMPointerTypeInContext(LLVMGetTypeContext(i32), 0);
        /* Spill all shadows before calling C helper */
        shadow_spill(b, i32, s, regs);

        LLVMValueRef pos_val  = emit_read_operand(b, i32, &insn->ops[0], 
                                    emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx),
                                    s, regs, mem, ctx);
        LLVMValueRef size_val = emit_read_operand(b, i32, &insn->ops[1],
                                    emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx),
                                    s, regs, mem, ctx);

        /* base operand: register mode → pass (1, reg_num), memory → (0, address) */
        LLVMValueRef base_is_reg, base_val;
        if (insn->ops[2].kind == JITBLK_REGISTER) {
            base_is_reg = LLVMConstInt(i32, 1, 0);
            base_val    = LLVMConstInt(i32, (unsigned)insn->ops[2].reg, 0);
        } else {
            base_is_reg = LLVMConstInt(i32, 0, 0);
            base_val    = emit_read_operand(b, i32, &insn->ops[2],
                              emit_operand_ea(b, i32, &insn->ops[2], s, regs, mem, ctx),
                              s, regs, mem, ctx);
        }

        /* dst operand: register mode → (1, reg_num), memory → (0, address) */
        LLVMValueRef dst_is_reg, dst_val;
        if (insn->ops[3].kind == JITBLK_REGISTER) {
            dst_is_reg = LLVMConstInt(i32, 1, 0);
            dst_val    = LLVMConstInt(i32, (unsigned)insn->ops[3].reg, 0);
        } else {
            dst_is_reg = LLVMConstInt(i32, 0, 0);
            dst_val    = emit_operand_ea(b, i32, &insn->ops[3], s, regs, mem, ctx);
            if (!dst_val) dst_val = LLVMConstInt(i32, 0, 0);
        }

        /* Declare and call helper */
        LLVMModuleRef mod = LLVMGetGlobalParent(LLVMGetBasicBlockParent(
                                LLVMGetInsertBlock(b)));
        LLVMTypeRef parms[8] = { ptr, ptr, ptr, i32, i32, i32, i32, i32, };
        LLVMTypeRef fn_ty = LLVMFunctionType(LLVMVoidTypeInContext(
                                LLVMGetTypeContext(i32)), parms, 8, 0);
        LLVMValueRef ext_fn = LLVMAddFunction(mod, "vax_jit_extzv_helper", fn_ty);

        LLVMValueRef args[8] = { regs, v_psl, mem,
                                  pos_val, size_val, base_is_reg, base_val,
                                  dst_is_reg };
        /* Need 9th arg but helper takes 9 params, let me fix... */
        /* Actually the signature has 9 params. Let me redo. */
        LLVMTypeRef parms9[9] = { ptr, ptr, ptr, i32, i32, i32, i32, i32, i32 };
        fn_ty = LLVMFunctionType(LLVMVoidTypeInContext(
                    LLVMGetTypeContext(i32)), parms9, 9, 0);
        /* Remove the one we already added and re-add with correct type.
           Actually LLVMAddFunction will reuse the existing one if names match.
           Let's use a different approach: just build the correct one. */
        LLVMDeleteFunction(ext_fn);
        ext_fn = LLVMAddFunction(mod, "vax_jit_extzv_helper", fn_ty);
        LLVMValueRef args9[9] = { regs, v_psl, mem,
                                   pos_val, size_val, base_is_reg, base_val,
                                   dst_is_reg, dst_val };
        LLVMBuildCall2(b, fn_ty, ext_fn, args9, 9, "");

        /* Invalidate all shadows after helper call */
        shadow_clear(s);
        return;
    }

    /* INSV src.rl, pos.rl, size.rb, base.vb — C helper call */
    case VAX_OPC_INSV: {
        LLVMTypeRef ptr = LLVMPointerTypeInContext(LLVMGetTypeContext(i32), 0);
        shadow_spill(b, i32, s, regs);

        LLVMValueRef src_val = emit_read_operand(b, i32, &insn->ops[0],
                                   emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx),
                                   s, regs, mem, ctx);
        LLVMValueRef pos_val = emit_read_operand(b, i32, &insn->ops[1],
                                   emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx),
                                   s, regs, mem, ctx);
        LLVMValueRef size_val = emit_read_operand(b, i32, &insn->ops[2],
                                    emit_operand_ea(b, i32, &insn->ops[2], s, regs, mem, ctx),
                                    s, regs, mem, ctx);

        LLVMValueRef base_is_reg, base_val;
        if (insn->ops[3].kind == JITBLK_REGISTER) {
            base_is_reg = LLVMConstInt(i32, 1, 0);
            base_val    = LLVMConstInt(i32, (unsigned)insn->ops[3].reg, 0);
        } else {
            base_is_reg = LLVMConstInt(i32, 0, 0);
            base_val    = emit_read_operand(b, i32, &insn->ops[3],
                              emit_operand_ea(b, i32, &insn->ops[3], s, regs, mem, ctx),
                              s, regs, mem, ctx);
        }

        LLVMModuleRef mod = LLVMGetGlobalParent(LLVMGetBasicBlockParent(
                                LLVMGetInsertBlock(b)));
        LLVMTypeRef parms[8] = { ptr, ptr, ptr, i32, i32, i32, i32, i32 };
        LLVMTypeRef fn_ty = LLVMFunctionType(LLVMVoidTypeInContext(
                                LLVMGetTypeContext(i32)), parms, 8, 0);
        LLVMValueRef ext_fn = LLVMAddFunction(mod, "vax_jit_insv_helper", fn_ty);
        LLVMValueRef args[8] = { regs, v_psl, mem,
                                  src_val, pos_val, size_val,
                                  base_is_reg, base_val };
        LLVMBuildCall2(b, fn_ty, ext_fn, args, 8, "");
        shadow_clear(s);
        return;
    }

    /* MOVC3 len.rw, srcaddr.ab, dstaddr.ab — C helper call */
    case VAX_OPC_MOVC3: {
        LLVMTypeRef ptr = LLVMPointerTypeInContext(LLVMGetTypeContext(i32), 0);
        shadow_spill(b, i32, s, regs);

        LLVMValueRef len_val = emit_read_operand(b, i32, &insn->ops[0],
                                   emit_operand_ea(b, i32, &insn->ops[0], s, regs, mem, ctx),
                                   s, regs, mem, ctx);
        /* srcaddr and dstaddr are address operands (.ab) — we need the EA, not the value */
        LLVMValueRef src_ea = emit_operand_ea(b, i32, &insn->ops[1], s, regs, mem, ctx);
        LLVMValueRef dst_ea = emit_operand_ea(b, i32, &insn->ops[2], s, regs, mem, ctx);
        if (!src_ea) src_ea = emit_read_operand(b, i32, &insn->ops[1], NULL, s, regs, mem, ctx);
        if (!dst_ea) dst_ea = emit_read_operand(b, i32, &insn->ops[2], NULL, s, regs, mem, ctx);

        LLVMModuleRef mod = LLVMGetGlobalParent(LLVMGetBasicBlockParent(
                                LLVMGetInsertBlock(b)));
        LLVMTypeRef parms[6] = { ptr, ptr, ptr, i32, i32, i32 };
        LLVMTypeRef fn_ty = LLVMFunctionType(LLVMVoidTypeInContext(
                                LLVMGetTypeContext(i32)), parms, 6, 0);
        LLVMValueRef ext_fn = LLVMAddFunction(mod, "vax_jit_movc3_helper", fn_ty);
        LLVMValueRef args[6] = { regs, v_psl, mem, len_val, src_ea, dst_ea };
        LLVMBuildCall2(b, fn_ty, ext_fn, args, 6, "");
        shadow_clear(s);
        return;
    }

    default:
        return;
    }
}

/* ------------------------------------------------------------------ */
/* Branch condition: emit i1 from PSL for VAX conditional branches     */
/* PSL CC layout: bit3=N, bit2=Z, bit1=V, bit0=C                      */
/* ------------------------------------------------------------------ */

/* Extract one CC bit from PSL as i1 (reloads PSL each call). */
static LLVMValueRef emit_psl_bit(LLVMBuilderRef b, LLVMTypeRef i32,
                                  LLVMValueRef psl_val, int bit)
{
    LLVMContextRef lctx = LLVMGetTypeContext(i32);
    LLVMTypeRef i1  = LLVMInt1TypeInContext(lctx);
    LLVMValueRef sh = LLVMBuildLShr(b, psl_val,
                                     LLVMConstInt(i32, (unsigned)bit, 0), "ps");
    LLVMValueRef and = LLVMBuildAnd(b, sh, LLVMConstInt(i32, 1, 0), "pb");
    return LLVMBuildTrunc(b, and, i1, "bt");
}

/* Build an i1 condition value from the current PSL for the given VaxBCond. */
static LLVMValueRef emit_branch_cond(LLVMBuilderRef b, LLVMTypeRef i32,
                                      LLVMValueRef v_psl, int cond)
{
    LLVMValueRef psl_val = LLVMBuildLoad2(b, i32, v_psl, "psl");
    LLVMValueRef n = emit_psl_bit(b, i32, psl_val, 3);
    LLVMValueRef z = emit_psl_bit(b, i32, psl_val, 2);
    LLVMValueRef v = emit_psl_bit(b, i32, psl_val, 1);
    LLVMValueRef c = emit_psl_bit(b, i32, psl_val, 0);

    switch (cond) {
    case VAX_BCOND_NEQ:  return LLVMBuildNot(b, z, "");                         /* !Z */
    case VAX_BCOND_EQL:  return z;                                               /* Z  */
    case VAX_BCOND_GEQ:  return LLVMBuildICmp(b, LLVMIntEQ,  n, v, "");        /* N=V */
    case VAX_BCOND_LSS:  return LLVMBuildICmp(b, LLVMIntNE,  n, v, "");        /* N!=V */
    case VAX_BCOND_GTR: {                                                        /* !Z && N=V */
        LLVMValueRef nv_eq = LLVMBuildICmp(b, LLVMIntEQ, n, v, "nveq");
        return LLVMBuildAnd(b, LLVMBuildNot(b, z, "nz"), nv_eq, "");
    }
    case VAX_BCOND_LEQ: {                                                        /* Z || N!=V */
        LLVMValueRef nv_ne = LLVMBuildICmp(b, LLVMIntNE, n, v, "nvne");
        return LLVMBuildOr(b, z, nv_ne, "");
    }
    case VAX_BCOND_GTRU: {                                                       /* !C && !Z */
        return LLVMBuildAnd(b, LLVMBuildNot(b, c, "nc"),
                                LLVMBuildNot(b, z, "nz"), "");
    }
    case VAX_BCOND_LEQU: return LLVMBuildOr(b, c, z, "");                       /* C || Z */
    case VAX_BCOND_VC:   return LLVMBuildNot(b, v, "");                         /* !V */
    case VAX_BCOND_VS:   return v;                                               /* V  */
    case VAX_BCOND_GEQU: return LLVMBuildNot(b, c, "");                         /* !C */
    case VAX_BCOND_LSSU: return c;                                               /* C  */
    default:             return LLVMConstInt(LLVMInt1TypeInContext(LLVMGetTypeContext(i32)), 1, 0);
    }
}

/* ------------------------------------------------------------------ */
/* BB map: (guest_pc -> LLVMBasicBlockRef)                             */
/* ------------------------------------------------------------------ */

#define MAX_BB_ENTRIES (VAX_JIT_MAX_INSNS * 3)

typedef struct { int32_t pc; LLVMBasicBlockRef bb; } BBEntry;

static LLVMBasicBlockRef bb_map_lookup(BBEntry *map, int n, int32_t pc)
{
    int i;
    for (i = 0; i < n; i++)
        if (map[i].pc == pc) return map[i].bb;
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Shadow clear helper                                                  */
/* ------------------------------------------------------------------ */

static void shadow_clear(RegShadow *s)
{
    int i;
    for (i = 0; i < 16; i++) s->val[i] = NULL;
}

/* Write a constant PC value into the regs[15] slot directly (no shadow). */
static void emit_pc_store(LLVMBuilderRef b, LLVMTypeRef i32,
                           LLVMValueRef v_regs, int32_t pc_val)
{
    LLVMValueRef idx = LLVMConstInt(i32, 15, 0);
    LLVMValueRef ptr = LLVMBuildGEP2(b, i32, v_regs, &idx, 1, "");
    LLVMBuildStore(b, LLVMConstInt(i32, (uint32_t)pc_val, 0), ptr);
}

/* Decrement *sim_interval by delta; store back. */
static void emit_sim_interval_dec(LLVMBuilderRef b, LLVMTypeRef i32,
                                   LLVMValueRef v_siv, int delta)
{
    LLVMValueRef old  = LLVMBuildLoad2(b, i32, v_siv, "siv");
    LLVMValueRef nval = LLVMBuildSub(b, old, LLVMConstInt(i32, (unsigned)delta, 0), "sivn");
    LLVMBuildStore(b, nval, v_siv);
}

/* Compile instructions blk->insns[start_insn..end_insn) as a single native
   function.  entry_pc is the guest PC of insns[start_insn] and is used for
   the function symbol name.  Supports multi-BB regions, back-edges, and
   CALLS (which exits the segment; re-entry is a separate segment).
   Returns NULL on error.                                               */
static void *compile_block(VaxJITBlock *blk, int start_insn, int end_insn,
                            int32_t entry_pc, int32_t *sim_interval_ptr)
{
    static uint32_t bcnt = 0;
    char sym[32];
    int i;
    int32_t first_opc = (end_insn > start_insn) ? blk->insns[start_insn].opc : 0;
    snprintf(sym, sizeof(sym), "blk_%02x_%u", (uint32_t)(uint8_t)first_opc, bcnt++);
    (void)entry_pc; /* used for naming in future; suppress unused-variable warning */

    LLVMTypeRef i32  = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef ptr  = LLVMPointerTypeInContext(ctx, 0);
    /* Function: void(ptr regs, ptr psl, ptr mem, ptr sim_interval) */
    LLVMTypeRef ps[4] = { ptr, ptr, ptr, ptr };
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(sym, ctx);
    LLVMValueRef  fn  = LLVMAddFunction(mod, sym,
                            LLVMFunctionType(LLVMVoidTypeInContext(ctx),
                                             ps, 4, 0));
    LLVMBuilderRef b  = LLVMCreateBuilderInContext(ctx);

    LLVMValueRef v_regs = LLVMGetParam(fn, 0);
    LLVMValueRef v_psl  = LLVMGetParam(fn, 1);
    LLVMValueRef v_mem  = LLVMGetParam(fn, 2);
    LLVMValueRef v_siv  = LLVMGetParam(fn, 3);

    /* ---- Pre-pass: build BB map ---- */

    /* Collect insn_pcs for the range [start_insn, end_insn) */
    int32_t insn_pcs[VAX_JIT_MAX_INSNS];
    int n_range = end_insn - start_insn;
    for (i = 0; i < n_range; i++)
        insn_pcs[i] = blk->insns[start_insn + i].insn_pc;

    /* Determine which PCs need their own LLVM basic block */
    int32_t  bb_start_pcs[MAX_BB_ENTRIES];
    int      n_bb_starts = 0;

    /* First instruction always starts the entry BB */
    if (n_range > 0)
        bb_start_pcs[n_bb_starts++] = insn_pcs[0];

    for (i = 0; i < n_range; i++) {
        VaxJITBlkInsn *insn = &blk->insns[start_insn + i];
        if (insn->branch_target < 0) continue;
        /* CALLS branch_target is the return address — NOT an intra-region branch */
        if (insn->opc == 0xFB) continue;

        int32_t tgt = insn->branch_target;
        int is_uncond = (insn->branch_cond == VAX_BCOND_UNCOND);
        int j, found;

        /* Check if target is an intra-region instruction */
        int tgt_intra = 0;
        for (j = 0; j < n_range; j++)
            if (insn_pcs[j] == tgt) { tgt_intra = 1; break; }

        if (tgt_intra) {
            /* Target needs a BB */
            found = 0;
            for (j = 0; j < n_bb_starts; j++)
                if (bb_start_pcs[j] == tgt) { found = 1; break; }
            if (!found && n_bb_starts < MAX_BB_ENTRIES)
                bb_start_pcs[n_bb_starts++] = tgt;
        }

        /* Fall-through of a conditional branch needs a BB */
        if (!is_uncond && i + 1 < n_range) {
            int32_t ft = insn_pcs[i + 1];
            found = 0;
            for (j = 0; j < n_bb_starts; j++)
                if (bb_start_pcs[j] == ft) { found = 1; break; }
            if (!found && n_bb_starts < MAX_BB_ENTRIES)
                bb_start_pcs[n_bb_starts++] = ft;
        }
    }

    /* Always create a dedicated entry BB first.
       LLVM requires the entry block to have no predecessors. When the JIT
       re-enters a region at a loop-top PC (e.g. after sim_interval expiry),
       the first instruction's BB becomes a back-edge target and would violate
       this constraint if it were also the entry block. The entry BB just
       falls through unconditionally to the first instruction BB. */
    LLVMBasicBlockRef entry_bb =
        LLVMAppendBasicBlockInContext(ctx, fn, "entry");

    /* Create LLVM BBs for each BB-start PC */
    BBEntry bb_map[MAX_BB_ENTRIES];
    int n_bb_map = 0;
    {
        char bbname[32];
        for (i = 0; i < n_bb_starts; i++) {
            snprintf(bbname, sizeof(bbname), "bb_%x", (uint32_t)bb_start_pcs[i]);
            bb_map[n_bb_map].pc = bb_start_pcs[i];
            bb_map[n_bb_map].bb = LLVMAppendBasicBlockInContext(ctx, fn, bbname);
            n_bb_map++;
        }
    }

    /* Create exit BBs for each unique intra-region exit target (branch targets
       NOT in insn_pcs — exits the region).  CALLS branch_target is the return
       address for cache re-entry, not an intra-region branch — skip it here. */
    int32_t  exit_pcs[MAX_BB_ENTRIES];
    BBEntry  exit_map[MAX_BB_ENTRIES];
    int      n_exits = 0;
    for (i = 0; i < n_range; i++) {
        VaxJITBlkInsn *insn = &blk->insns[start_insn + i];
        if (insn->branch_target < 0) continue;
        if (insn->opc == 0xFB) continue;  /* CALLS exits inline, not via exit BB */
        int32_t tgt = insn->branch_target;
        int j, tgt_intra = 0;
        for (j = 0; j < n_range; j++)
            if (insn_pcs[j] == tgt) { tgt_intra = 1; break; }
        if (tgt_intra) continue;

        /* Exit target: find or create exit BB */
        int found = 0;
        for (j = 0; j < n_exits; j++)
            if (exit_pcs[j] == tgt) { found = 1; break; }
        if (!found && n_exits < MAX_BB_ENTRIES) {
            char bbname[32];
            snprintf(bbname, sizeof(bbname), "exit_%x", (uint32_t)tgt);
            exit_pcs[n_exits]    = tgt;
            exit_map[n_exits].pc = tgt;
            exit_map[n_exits].bb = LLVMAppendBasicBlockInContext(ctx, fn, bbname);
            n_exits++;
        }
    }

    /* If has_back_edge: create sim_expired BB */
    LLVMBasicBlockRef sim_expired_bb = NULL;
    if (blk->has_back_edge) {
        sim_expired_bb = LLVMAppendBasicBlockInContext(ctx, fn, "sim_expired");
    }

    /* Final fallthrough BB (for the fall-off-end case) */
    LLVMBasicBlockRef fallthrough_bb =
        LLVMAppendBasicBlockInContext(ctx, fn, "fallthrough");

    /* Fault-exit BB: shared sink for every mem-op slow-path that signals a
       translation/access fault. Each contributing slow path AddIncomings its
       cur_pc constant to fault_pc_phi; the BB stores that PC into regs[15]
       and returns. The interpreter resumes at that PC and naturally re-faults
       through the normal SIMH path (setting fault_PC/fault_p1/recq unwind). */
    LLVMBasicBlockRef fault_exit_bb =
        LLVMAppendBasicBlockInContext(ctx, fn, "fault_exit");

    /* ---- Main emission ---- */

    /* Populate entry_bb: alloca fault_slot, then unconditional branch
       to the first instruction BB. */
    RegShadow shadow;
    shadow_clear(&shadow);
    EmitCtx emit_ctx = { 0 };
    emit_ctx.regs   = v_regs;
    emit_ctx.psl    = v_psl;
    emit_ctx.shadow = &shadow;
    emit_ctx.fault_exit_bb = fault_exit_bb;
    EmitCtx *ectx = &emit_ctx;
    {
        LLVMBasicBlockRef first_bb = (n_bb_map > 0) ? bb_map[0].bb : fallthrough_bb;
        LLVMPositionBuilderAtEnd(b, entry_bb);
        emit_ctx.fault_slot = LLVMBuildAlloca(b, i32, "fault_slot");
        /* Use a constant-false branch to fault_exit_bb so it always has a
           predecessor edge (required for a well-formed phi); the optimizer
           folds this away when no real fault path adds an incoming. */
        LLVMValueRef false_c = LLVMConstInt(LLVMInt1TypeInContext(ctx), 0, 0);
        LLVMBuildCondBr(b, false_c, fault_exit_bb, first_bb);

        /* Build the phi up front; seed it from the dummy entry edge so the
           phi is well-formed even if no mem op contributes. */
        LLVMPositionBuilderAtEnd(b, fault_exit_bb);
        emit_ctx.fault_pc_phi = LLVMBuildPhi(b, i32, "fault_pc");
        LLVMAddIncoming(emit_ctx.fault_pc_phi,
                        (LLVMValueRef[]){ LLVMConstInt(i32, 0, 0) },
                        (LLVMBasicBlockRef[]){ entry_bb },
                        1);

        LLVMPositionBuilderAtEnd(b, first_bb);
    }

    int builder_terminated = 0;

    for (i = 0; i < n_range; i++) {
        VaxJITBlkInsn *insn = &blk->insns[start_insn + i];

        /* Check if this instruction's PC starts a new BB (after the first) */
        if (i > 0) {
            LLVMBasicBlockRef target_bb = bb_map_lookup(bb_map, n_bb_map, insn->insn_pc);
            if (target_bb != NULL) {
                /* Terminate current BB if needed */
                if (!builder_terminated) {
                    shadow_spill(b, i32, &shadow, v_regs);
                    LLVMBuildBr(b, target_bb);
                }
                /* Switch to new BB */
                LLVMPositionBuilderAtEnd(b, target_bb);
                shadow_clear(&shadow);
                builder_terminated = 0;
            }
        }

        if (builder_terminated) {
            /* Dead code after unconditional branch: skip emission */
            /* but keep looping so we can detect the next BB start */
            continue;
        }

        /* CALLS/RET must be handled before the branch_target check:
           CALLS stores the return address in branch_target (>= 0) but is NOT
           a branch — it exits the segment and the callee runs separately. */
        if (insn->opc == 0xFB /* CALLS */ || insn->opc == 0x04 /* RET */) {
            /* CALLS: spill return address into R15, call helper, then EXIT.
               The helper sets PC = callee+2.  The callee runs via the interpreter
               or its own JIT block.  When the callee RETs, PC = return address,
               and a cache hit re-enters this compiled segment.
               RET: call ret helper (tears down frame, sets PC = saved return addr),
               then exit. */

            if (insn->opc == 0xFB) {
                /* R15 = return address (insn->branch_target); helper reads it for frame */
                shadow_set(&shadow, 15,
                           LLVMConstInt(i32, (uint32_t)insn->branch_target, 0));
            }
            shadow_spill(b, i32, &shadow, v_regs);

            if (insn->opc == 0xFB) {
                /* vax_jit_calls_helper(ptr regs, ptr psl, ptr mem, i32 argc, i32 callee) */
                LLVMTypeRef  parms5[5] = { ptr, ptr, ptr, i32, i32 };
                LLVMTypeRef  fn5_ty    = LLVMFunctionType(
                                             LLVMVoidTypeInContext(ctx), parms5, 5, 0);
                LLVMValueRef ext_fn    = LLVMAddFunction(mod,
                                             "vax_jit_calls_helper", fn5_ty);

                VaxJITBlkOp *op0 = &insn->ops[0];
                LLVMValueRef argc_val;
                if (op0->kind == JITBLK_LITERAL || op0->kind == JITBLK_IMMEDIATE) {
                    argc_val = LLVMConstInt(i32, (uint32_t)op0->imm, 0);
                } else {
                    LLVMValueRef idx = LLVMConstInt(i32, (unsigned)op0->reg, 0);
                    LLVMValueRef rp  = LLVMBuildGEP2(b, i32, v_regs, &idx, 1, "");
                    argc_val = LLVMBuildLoad2(b, i32, rp, "argc");
                }

                LLVMValueRef callee_val =
                    LLVMConstInt(i32, (uint32_t)insn->ops[1].imm, 0);

                LLVMValueRef args5[5] = { v_regs, v_psl, v_mem, argc_val, callee_val };
                LLVMBuildCall2(b, fn5_ty, ext_fn, args5, 5, "");
                /* Helper set PC = callee+2; exit so the callee runs on its own. */
            } else {
                /* vax_jit_ret_helper(ptr regs, ptr psl, ptr mem) */
                LLVMTypeRef  parms3[3] = { ptr, ptr, ptr };
                LLVMTypeRef  fn3_ty    = LLVMFunctionType(
                                             LLVMVoidTypeInContext(ctx), parms3, 3, 0);
                LLVMValueRef ext_fn    = LLVMAddFunction(mod,
                                             "vax_jit_ret_helper", fn3_ty);

                LLVMValueRef args3[3] = { v_regs, v_psl, v_mem };
                LLVMBuildCall2(b, fn3_ty, ext_fn, args3, 3, "");
            }

            LLVMBuildRetVoid(b);
            builder_terminated = 1;
        } else if (insn->branch_target >= 0) {
            /* Branch instruction: spill, then emit the branch */
            shadow_spill(b, i32, &shadow, v_regs);
            shadow_clear(&shadow);

            int32_t tgt = insn->branch_target;
            int is_uncond = (insn->branch_cond == VAX_BCOND_UNCOND);

            /* Is the target intra-region? */
            int j, tgt_intra = 0;
            for (j = 0; j < n_range; j++)
                if (insn_pcs[j] == tgt) { tgt_intra = 1; break; }

            LLVMBasicBlockRef tgt_bb = tgt_intra
                ? bb_map_lookup(bb_map, n_bb_map, tgt)
                : NULL;

            /* Exit BB for this target (if exit) */
            LLVMBasicBlockRef ex_bb = NULL;
            if (!tgt_intra) {
                for (j = 0; j < n_exits; j++)
                    if (exit_pcs[j] == tgt) { ex_bb = exit_map[j].bb; break; }
            }

            if (is_uncond) {
                if (tgt_intra && tgt_bb) {
                    if (tgt < insn->insn_pc) {
                        /* Unconditional backward intra-region (loop back-edge):
                           write loop-top PC for sim_expired exit, check sim_interval. */
                        emit_pc_store(b, i32, v_regs, tgt);
                        LLVMValueRef old_iv  = LLVMBuildLoad2(b, i32, v_siv, "siv");
                        LLVMValueRef new_iv  = LLVMBuildSub(b, old_iv,
                                                  LLVMConstInt(i32, (unsigned)n_range, 0),
                                                  "sivn");
                        LLVMBuildStore(b, new_iv, v_siv);
                        LLVMValueRef expired = LLVMBuildICmp(b, LLVMIntSLE, new_iv,
                                                  LLVMConstInt(i32, 0, 0), "exp");
                        LLVMBuildCondBr(b, expired, sim_expired_bb, tgt_bb);
                    } else {
                        /* Unconditional forward intra-region jump (not a loop):
                           just branch to the target BB. */
                        LLVMBuildBr(b, tgt_bb);
                    }
                } else {
                    /* Unconditional exit */
                    LLVMBuildBr(b, ex_bb ? ex_bb : fallthrough_bb);
                }
                builder_terminated = 1;
            } else {
                /* Conditional branch */
                LLVMValueRef cond_val = emit_branch_cond(b, i32, v_psl,
                                                          insn->branch_cond);
                /* Fall-through BB */
                LLVMBasicBlockRef ft_bb = (i + 1 < n_range)
                    ? bb_map_lookup(bb_map, n_bb_map, insn_pcs[i + 1])
                    : fallthrough_bb;
                if (!ft_bb) ft_bb = fallthrough_bb;

                if (tgt_intra && tgt_bb) {
                    if (tgt < insn->insn_pc) {
                        /* Conditional backward intra-region (loop back-edge):
                           on the taken path, decrement sim_interval and check
                           for expiry before looping — same guard as uncond back-edge. */
                        LLVMBasicBlockRef iv_check_bb =
                            LLVMAppendBasicBlockInContext(ctx, fn, "cond_back_iv");
                        LLVMBuildCondBr(b, cond_val, iv_check_bb, ft_bb);

                        LLVMPositionBuilderAtEnd(b, iv_check_bb);
                        emit_pc_store(b, i32, v_regs, tgt);
                        LLVMValueRef old_iv = LLVMBuildLoad2(b, i32, v_siv, "siv");
                        LLVMValueRef new_iv = LLVMBuildSub(b, old_iv,
                                                LLVMConstInt(i32, (unsigned)n_range, 0),
                                                "sivn");
                        LLVMBuildStore(b, new_iv, v_siv);
                        LLVMValueRef expired = LLVMBuildICmp(b, LLVMIntSLE, new_iv,
                                                LLVMConstInt(i32, 0, 0), "exp");
                        LLVMBuildCondBr(b, expired, sim_expired_bb, tgt_bb);
                    } else {
                        /* Conditional forward intra-region: true→target, false→fall-through */
                        LLVMBuildCondBr(b, cond_val, tgt_bb, ft_bb);
                    }
                } else {
                    /* Conditional exit: true→exit, false→fall-through */
                    LLVMBuildCondBr(b, cond_val,
                                    ex_bb ? ex_bb : fallthrough_bb,
                                    ft_bb);
                }
                builder_terminated = 1;
            }
        } else {
            /* Normal non-branch instruction */
            emit_insn(b, i32, insn, &shadow, v_regs, v_psl, v_mem, ectx);
            builder_terminated = 0;
        }
    }

    /* Final fallthrough exit (if builder is still active) */
    if (!builder_terminated) {
        shadow_set(&shadow, 15,
                   LLVMConstInt(i32, (uint32_t)blk->fallthrough_pc, 0));
        shadow_spill(b, i32, &shadow, v_regs);
        LLVMBuildBr(b, fallthrough_bb);
    }

    /* ---- Populate fallthrough BB ---- */
    LLVMPositionBuilderAtEnd(b, fallthrough_bb);
    emit_pc_store(b, i32, v_regs, blk->fallthrough_pc);
    LLVMBuildRetVoid(b);

    /* ---- Populate exit BBs ---- */
    for (i = 0; i < n_exits; i++) {
        LLVMPositionBuilderAtEnd(b, exit_map[i].bb);
        emit_pc_store(b, i32, v_regs, exit_pcs[i]);
        LLVMBuildRetVoid(b);
    }

    /* ---- Populate sim_expired BB ---- */
    if (sim_expired_bb) {
        LLVMPositionBuilderAtEnd(b, sim_expired_bb);
        /* PC already spilled before the back-edge branch; just ret. */
        LLVMBuildRetVoid(b);
    }

    /* ---- Populate fault_exit BB ---- */
    /* Phi was pre-seeded with a never-taken edge from entry_bb (see entry
       setup above); each mem-op slow-path AddIncoming'd its cur_pc. Store
       the resolved fault PC into regs[15] and return to the interpreter. */
    LLVMPositionBuilderAtEnd(b, fault_exit_bb);
    {
        LLVMValueRef idx = LLVMConstInt(i32, 15, 0);
        LLVMValueRef ptr = LLVMBuildGEP2(b, i32, v_regs, &idx, 1, "");
        LLVMBuildStore(b, emit_ctx.fault_pc_phi, ptr);
    }
    LLVMBuildRetVoid(b);

    LLVMDisposeBuilder(b);

    /* Verify IR before handing to optimizer — catches malformed BBs early */
    if (LLVMVerifyFunction(fn, LLVMPrintMessageAction)) {
        fprintf(stderr, "vax_jit: IR verification failed for %s\n", sym);
        LLVMDumpValue(fn);
        return NULL;
    }

    LLVMErrorRef err = jit_add_module(mod);
    if (err) {
        char *m = LLVMGetErrorMessage(err);
        fprintf(stderr, "vax_jit: compile_block opc=%02x: %s\n",
                (uint8_t)first_opc, m);
        LLVMDisposeErrorMessage(m);
        return NULL;
    }
    return jit_lookup(sym);
}

/* ------------------------------------------------------------------ */
/* Public block execution API                                           */
/* ------------------------------------------------------------------ */

void *vax_jit_llvm_compile_block(VaxJITBlock *blk, int32_t *sim_interval)
{
    return compile_block(blk, 0, blk->n_insns, blk->region_start_pc, sim_interval);
}

/* Compile all segments in a scanned region.
   Segments are divided by CALLS instructions: the main segment (0..calls_idx)
   is entries[0], and each post-CALLS continuation segment is an additional entry.
   Returns n_entries == 0 if the main segment fails to compile.                  */
VaxJITCompileResult vax_jit_llvm_compile_block_multi(VaxJITBlock *blk,
                                                      int32_t     *sim_interval)
{
    VaxJITCompileResult result;
    result.n_entries = 0;

    /* Find CALLS boundaries */
    int calls_idx[VAX_JIT_MAX_SEGMENTS];
    int n_calls = 0;
    for (int i = 0; i < blk->n_insns && n_calls < VAX_JIT_MAX_SEGMENTS - 1; i++) {
        if (blk->insns[i].opc == 0xFB)
            calls_idx[n_calls++] = i;
    }

    /* Build segment boundaries: [seg_start[k], seg_end[k]) */
    int seg_start[VAX_JIT_MAX_SEGMENTS];
    int seg_end[VAX_JIT_MAX_SEGMENTS];
    int32_t seg_entry_pc[VAX_JIT_MAX_SEGMENTS];
    int n_segs = n_calls + 1;

    seg_start[0]    = 0;
    seg_end[0]      = (n_calls > 0) ? calls_idx[0] + 1 : blk->n_insns;
    seg_entry_pc[0] = blk->region_start_pc;

    for (int k = 0; k < n_calls; k++) {
        seg_start[k + 1]    = calls_idx[k] + 1;
        seg_end[k + 1]      = (k + 1 < n_calls) ? calls_idx[k + 1] + 1 : blk->n_insns;
        seg_entry_pc[k + 1] = blk->insns[calls_idx[k]].branch_target; /* return address */
    }

    /* Compile each segment */
    for (int k = 0; k < n_segs; k++) {
        if (seg_start[k] >= seg_end[k]) continue;  /* empty (shouldn't happen) */
        void *fn = compile_block(blk, seg_start[k], seg_end[k],
                                 seg_entry_pc[k], sim_interval);
        if (!fn) {
            if (k == 0) return result;  /* main segment failed — give up */
            continue;                   /* re-entry segment failed — skip it */
        }
        result.entries[result.n_entries].entry_pc = seg_entry_pc[k];
        result.entries[result.n_entries].fn       = fn;
        result.entries[result.n_entries].n_insns  = seg_end[k] - seg_start[k];
        result.n_entries++;
    }

    return result;
}

void vax_jit_llvm_run_block(void *fn, int32_t *regs, int32_t *psl,
                             int32_t *mem, int32_t *sim_interval)
{
    typedef void (*BlockFn)(int32_t *, int32_t *, int32_t *, int32_t *);
    ((BlockFn)fn)(regs, psl, mem, sim_interval);
}

int vax_jit_llvm_exec_block(VaxJITBlock *blk, int32_t *regs,
                             int32_t *psl, int32_t *mem,
                             int32_t *sim_interval)
{
    typedef void (*BlockFn)(int32_t *, int32_t *, int32_t *, int32_t *);
    BlockFn fn = (BlockFn)compile_block(blk, 0, blk->n_insns,
                                        blk->region_start_pc, sim_interval);
    if (!fn) return 0;
    fn(regs, psl, mem, sim_interval);
    return 1;
}

/* ------------------------------------------------------------------ */
/* Init / destroy                                                       */
/* ------------------------------------------------------------------ */

int vax_jit_llvm_init(void)
{
    LLVMInitializeNativeTarget();
    LLVMInitializeNativeAsmPrinter();
    LLVMInstallFatalErrorHandler(llvm_fatal);

    ctx = LLVMContextCreate();
    if (!ctx) {
        fprintf(stderr, "vax_jit: failed to create LLVM context\n");
        return -1;
    }

    LLVMErrorRef err = LLVMOrcCreateLLJIT(&jit, NULL);
    if (err) {
        char *msg = LLVMGetErrorMessage(err);
        fprintf(stderr, "vax_jit: ORC init failed: %s\n", msg);
        LLVMDisposeErrorMessage(msg);
        LLVMContextDispose(ctx);
        ctx = NULL;
        return -1;
    }

    if (!compile_nop())                              return 0;
    if (!compile_intl2(VAX_INTL_ADD,     "vax_addl")) return 0;
    if (!compile_intl2(VAX_INTL_SUB,     "vax_subl")) return 0;
    if (!compile_intl2(VAX_INTL_OR,      "vax_orl"))  return 0;
    if (!compile_intl2(VAX_INTL_AND_NOT, "vax_bicl")) return 0;
    if (!compile_intl2(VAX_INTL_XOR,     "vax_xorl")) return 0;
    if (!compile_intl2(VAX_INTL_MOV,     "vax_movl")) return 0;
    if (!compile_intl2(VAX_INTL_COM,     "vax_coml")) return 0;
    if (!compile_cmpl())                              return 0;
    if (!compile_tstl())                              return 0;

    fprintf(stdout, "VAX JIT enabled (LLVM ORC v2)\n");
    return 0;
}

void vax_jit_llvm_destroy(void)
{
    if (jit) { LLVMOrcDisposeLLJIT(jit); jit = NULL; }
    if (ctx) { LLVMContextDispose(ctx);  ctx = NULL; }
}

/* ------------------------------------------------------------------ */
/* CALLS/RET helper registration                                        */
/* ------------------------------------------------------------------ */

void vax_jit_llvm_register_call_helpers(void *calls_fn, void *ret_fn)
{
    LLVMOrcExecutionSessionRef es       = LLVMOrcLLJITGetExecutionSession(jit);
    LLVMOrcJITDylibRef         main_lib = LLVMOrcLLJITGetMainJITDylib(jit);

    LLVMJITSymbolFlags flags;
    flags.GenericFlags = (uint8_t)(LLVMJITSymbolGenericFlagsExported |
                                   LLVMJITSymbolGenericFlagsCallable);
    flags.TargetFlags  = 0;

    LLVMOrcCSymbolMapPair syms[2];
    syms[0].Name = LLVMOrcExecutionSessionIntern(es, "vax_jit_calls_helper");
    syms[0].Sym.Address = (LLVMOrcExecutorAddress)(uintptr_t)calls_fn;
    syms[0].Sym.Flags   = flags;

    syms[1].Name = LLVMOrcExecutionSessionIntern(es, "vax_jit_ret_helper");
    syms[1].Sym.Address = (LLVMOrcExecutorAddress)(uintptr_t)ret_fn;
    syms[1].Sym.Flags   = flags;

    LLVMOrcMaterializationUnitRef mu = LLVMOrcAbsoluteSymbols(syms, 2);
    LLVMErrorRef err = LLVMOrcJITDylibDefine(main_lib, mu);
    if (err) {
        char *msg = LLVMGetErrorMessage(err);
        fprintf(stderr, "vax_jit: failed to register call helpers: %s\n", msg);
        LLVMDisposeErrorMessage(msg);
    }
}

/* ------------------------------------------------------------------ */
/* EXTZV/INSV/MOVC3 helper registration                                */
/* ------------------------------------------------------------------ */

void vax_jit_llvm_register_misc_helpers(void *extzv_fn, void *insv_fn, void *movc3_fn)
{
    LLVMOrcExecutionSessionRef es       = LLVMOrcLLJITGetExecutionSession(jit);
    LLVMOrcJITDylibRef         main_lib = LLVMOrcLLJITGetMainJITDylib(jit);

    LLVMJITSymbolFlags flags;
    flags.GenericFlags = (uint8_t)(LLVMJITSymbolGenericFlagsExported |
                                   LLVMJITSymbolGenericFlagsCallable);
    flags.TargetFlags  = 0;

    LLVMOrcCSymbolMapPair syms[3];
    syms[0].Name = LLVMOrcExecutionSessionIntern(es, "vax_jit_extzv_helper");
    syms[0].Sym.Address = (LLVMOrcExecutorAddress)(uintptr_t)extzv_fn;
    syms[0].Sym.Flags   = flags;

    syms[1].Name = LLVMOrcExecutionSessionIntern(es, "vax_jit_insv_helper");
    syms[1].Sym.Address = (LLVMOrcExecutorAddress)(uintptr_t)insv_fn;
    syms[1].Sym.Flags   = flags;

    syms[2].Name = LLVMOrcExecutionSessionIntern(es, "vax_jit_movc3_helper");
    syms[2].Sym.Address = (LLVMOrcExecutorAddress)(uintptr_t)movc3_fn;
    syms[2].Sym.Flags   = flags;

    LLVMOrcMaterializationUnitRef mu = LLVMOrcAbsoluteSymbols(syms, 3);
    LLVMErrorRef err = LLVMOrcJITDylibDefine(main_lib, mu);
    if (err) {
        char *msg = LLVMGetErrorMessage(err);
        fprintf(stderr, "vax_jit: failed to register misc helpers: %s\n", msg);
        LLVMDisposeErrorMessage(msg);
    }
}

void vax_jit_llvm_register_mem_helpers(void *load_fn, void *store_fn)
{
    LLVMOrcExecutionSessionRef es       = LLVMOrcLLJITGetExecutionSession(jit);
    LLVMOrcJITDylibRef         main_lib = LLVMOrcLLJITGetMainJITDylib(jit);

    LLVMJITSymbolFlags flags;
    flags.GenericFlags = (uint8_t)(LLVMJITSymbolGenericFlagsExported |
                                   LLVMJITSymbolGenericFlagsCallable);
    flags.TargetFlags  = 0;

    LLVMOrcCSymbolMapPair syms[2];
    syms[0].Name = LLVMOrcExecutionSessionIntern(es, "vax_jit_mem_load_helper");
    syms[0].Sym.Address = (LLVMOrcExecutorAddress)(uintptr_t)load_fn;
    syms[0].Sym.Flags   = flags;

    syms[1].Name = LLVMOrcExecutionSessionIntern(es, "vax_jit_mem_store_helper");
    syms[1].Sym.Address = (LLVMOrcExecutorAddress)(uintptr_t)store_fn;
    syms[1].Sym.Flags   = flags;

    LLVMOrcMaterializationUnitRef mu = LLVMOrcAbsoluteSymbols(syms, 2);
    LLVMErrorRef err = LLVMOrcJITDylibDefine(main_lib, mu);
    if (err) {
        char *msg = LLVMGetErrorMessage(err);
        fprintf(stderr, "vax_jit: failed to register mem helpers: %s\n", msg);
        LLVMDisposeErrorMessage(msg);
    }
}

void vax_jit_llvm_register_mmu_state(int32_t *p)
{
    mapen_ptr = p;
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

int vax_jit_llvm_nop(void)
{
    if (!fn_nop) return 0;
    fn_nop();
    return 1;
}

/* op: VaxIntOpL enum value
   src_is_const/src_val: literal (is_const=1) or register index (is_const=0)
   dst_reg: register index to read dst_old from and write result to */
int vax_jit_llvm_intl2(int32_t *regs, int32_t *psl, int op,
                        int src_is_const, int32_t src_val, int dst_reg)
{
    if (!jit || op < 0 || op >= VAX_INTL_NOPS || !fn_intl2[op]) return 0;
    fn_intl2[op](regs, psl,
                 (int32_t)src_is_const, src_val, (int32_t)dst_reg);
    return 1;
}

/* CMPL src1, src2 — computes src1 - src2, sets CC_SUB, no store */
int vax_jit_llvm_cmpl(int32_t *regs, int32_t *psl,
                      int src1_is_const, int32_t src1_val,
                      int src2_is_const, int32_t src2_val)
{
    if (!fn_cmpl) return 0;
    fn_cmpl(regs, psl,
            (int32_t)src1_is_const, src1_val,
            (int32_t)src2_is_const, src2_val);
    return 1;
}

/* TSTL src — sets CC_LOGICAL based on src, no store */
int vax_jit_llvm_tstl(int32_t *regs, int32_t *psl,
                      int src_is_const, int32_t src_val)
{
    if (!fn_tstl) return 0;
    fn_tstl(regs, psl, (int32_t)src_is_const, src_val);
    return 1;
}

/* Enable/disable IR dump — call before init so handlers are dumped at
   compile time (they are compiled once during vax_jit_llvm_init).     */
void vax_jit_llvm_set_ir_dump(int enable)
{
    ir_dump = enable;
}
