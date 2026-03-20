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

/* PSL = (PSL & ~0xF) | cc_bits */
static void build_psl_update(LLVMBuilderRef b, LLVMTypeRef i32,
                              LLVMValueRef v_psl, LLVMValueRef cc)
{
    LLVMValueRef old  = LLVMBuildLoad2(b, i32, v_psl, "po");
    LLVMValueRef mask = LLVMConstInt(i32, 0xFFFFFFF0u, 0);
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
    build_psl_update(b, i32, LLVMGetParam(fn, 1),
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
#define VAX_OPC_SUBL2 0xC2
#define VAX_OPC_BISL2 0xC8
#define VAX_OPC_BICL2 0xCA
#define VAX_OPC_XORL2 0xCC
#define VAX_OPC_MOVL  0xD0
#define VAX_OPC_CMPL  0xD1
#define VAX_OPC_MCOML 0xD2
#define VAX_OPC_CLRL  0xD4
#define VAX_OPC_TSTL  0xD5
#define VAX_OPC_INCL  0xD6
#define VAX_OPC_DECL  0xD7
#define VAX_OPC_MOVB   0x90
#define VAX_OPC_MOVZBL 0x9A
#define VAX_OPC_MOVW   0xB0
#define VAX_OPC_PUSHL  0xDD

/* ------------------------------------------------------------------ */
/* Register shadow: tracks loaded/modified register values             */
/* NULL = not yet loaded; non-NULL = loaded (and possibly modified).   */
/* ------------------------------------------------------------------ */

typedef struct {
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

static LLVMValueRef emit_mem_load(LLVMBuilderRef b, LLVMTypeRef i32,
                                   LLVMValueRef mem, LLVMValueRef byte_addr,
                                   int width)
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

static void emit_mem_store(LLVMBuilderRef b, LLVMTypeRef i32,
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

/* Compute effective byte address for a memory operand.
   For autodec/autoinc, also updates the register shadow.
   Returns NULL for non-memory operand kinds.                          */
static LLVMValueRef emit_operand_ea(LLVMBuilderRef b, LLVMTypeRef i32,
                                     VaxJITBlkOp *op, RegShadow *s,
                                     LLVMValueRef regs, LLVMValueRef mem)
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
        return emit_mem_load(b, i32, mem, base, 4);
    case JITBLK_ABSOLUTE:
        return LLVMConstInt(i32, (uint32_t)op->imm, 0);
    case JITBLK_ABS_DEFERRED: {
        LLVMValueRef addr = LLVMConstInt(i32, (uint32_t)op->imm, 0);
        return emit_mem_load(b, i32, mem, addr, 4);
    }
    case JITBLK_DISP:
        base = shadow_get(b, i32, s, op->reg, regs);
        disp = LLVMConstInt(i32, (uint32_t)op->imm, 0);
        return LLVMBuildAdd(b, base, disp, "ea");
    case JITBLK_DISP_DEFERRED:
        base = shadow_get(b, i32, s, op->reg, regs);
        disp = LLVMConstInt(i32, (uint32_t)op->imm, 0);
        ea   = LLVMBuildAdd(b, base, disp, "ea");
        return emit_mem_load(b, i32, mem, ea, 4);
    default:
        return NULL;
    }
}

/* Read the value of an operand given its pre-computed EA (NULL for non-memory). */
static LLVMValueRef emit_read_operand(LLVMBuilderRef b, LLVMTypeRef i32,
                                       VaxJITBlkOp *op, LLVMValueRef ea,
                                       RegShadow *s, LLVMValueRef regs,
                                       LLVMValueRef mem)
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
        return emit_mem_load(b, i32, mem, ea, op->width);
    }
}

/* Write val to the operand's destination (shadow for register, memory via ea). */
static void emit_write_operand(LLVMBuilderRef b, LLVMTypeRef i32,
                                VaxJITBlkOp *op, LLVMValueRef ea,
                                LLVMValueRef val, RegShadow *s,
                                LLVMValueRef regs, LLVMValueRef mem)
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
        emit_mem_store(b, i32, mem, ea, val, op->width);
    }
}

/* Emit IR for one VAX instruction into the current basic block. */
static void emit_insn(LLVMBuilderRef b, LLVMTypeRef i32,
                       VaxJITBlock *blk, RegShadow *s,
                       LLVMValueRef regs, LLVMValueRef v_psl,
                       LLVMValueRef mem)
{
    LLVMValueRef ea0, ea1, src, dst_old, result, cc;

    switch (blk->opc) {

    case VAX_OPC_NOP:
        return;

    /* ADDL2 src, dst  :  dst = dst + src */
    case VAX_OPC_ADDL2:
        ea0     = emit_operand_ea(b, i32, &blk->ops[0], s, regs, mem);
        ea1     = emit_operand_ea(b, i32, &blk->ops[1], s, regs, mem);
        src     = emit_read_operand(b, i32, &blk->ops[0], ea0, s, regs, mem);
        dst_old = emit_read_operand(b, i32, &blk->ops[1], ea1, s, regs, mem);
        result  = LLVMBuildAdd(b, dst_old, src, "r");
        cc      = build_cc_add(b, i32, src, dst_old, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &blk->ops[1], ea1, result, s, regs, mem);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* SUBL2 src, dst  :  dst = dst - src */
    case VAX_OPC_SUBL2:
        ea0     = emit_operand_ea(b, i32, &blk->ops[0], s, regs, mem);
        ea1     = emit_operand_ea(b, i32, &blk->ops[1], s, regs, mem);
        src     = emit_read_operand(b, i32, &blk->ops[0], ea0, s, regs, mem);
        dst_old = emit_read_operand(b, i32, &blk->ops[1], ea1, s, regs, mem);
        result  = LLVMBuildSub(b, dst_old, src, "r");
        cc      = build_cc_sub(b, i32, src, dst_old, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &blk->ops[1], ea1, result, s, regs, mem);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* BISL2 src, dst  :  dst = dst | src */
    case VAX_OPC_BISL2:
        ea0     = emit_operand_ea(b, i32, &blk->ops[0], s, regs, mem);
        ea1     = emit_operand_ea(b, i32, &blk->ops[1], s, regs, mem);
        src     = emit_read_operand(b, i32, &blk->ops[0], ea0, s, regs, mem);
        dst_old = emit_read_operand(b, i32, &blk->ops[1], ea1, s, regs, mem);
        result  = LLVMBuildOr(b, dst_old, src, "r");
        cc      = build_cc_logical(b, i32, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &blk->ops[1], ea1, result, s, regs, mem);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* BICL2 src, dst  :  dst = dst & ~src */
    case VAX_OPC_BICL2:
        ea0     = emit_operand_ea(b, i32, &blk->ops[0], s, regs, mem);
        ea1     = emit_operand_ea(b, i32, &blk->ops[1], s, regs, mem);
        src     = emit_read_operand(b, i32, &blk->ops[0], ea0, s, regs, mem);
        dst_old = emit_read_operand(b, i32, &blk->ops[1], ea1, s, regs, mem);
        result  = LLVMBuildAnd(b, dst_old, LLVMBuildNot(b, src, "ns"), "r");
        cc      = build_cc_logical(b, i32, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &blk->ops[1], ea1, result, s, regs, mem);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* XORL2 src, dst  :  dst = dst ^ src */
    case VAX_OPC_XORL2:
        ea0     = emit_operand_ea(b, i32, &blk->ops[0], s, regs, mem);
        ea1     = emit_operand_ea(b, i32, &blk->ops[1], s, regs, mem);
        src     = emit_read_operand(b, i32, &blk->ops[0], ea0, s, regs, mem);
        dst_old = emit_read_operand(b, i32, &blk->ops[1], ea1, s, regs, mem);
        result  = LLVMBuildXor(b, dst_old, src, "r");
        cc      = build_cc_logical(b, i32, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &blk->ops[1], ea1, result, s, regs, mem);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* MOVL src, dst  :  dst = src */
    case VAX_OPC_MOVL:
        ea0    = emit_operand_ea(b, i32, &blk->ops[0], s, regs, mem);
        ea1    = emit_operand_ea(b, i32, &blk->ops[1], s, regs, mem);
        src    = emit_read_operand(b, i32, &blk->ops[0], ea0, s, regs, mem);
        cc     = build_cc_logical(b, i32, src, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &blk->ops[1], ea1, src, s, regs, mem);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* MCOML src, dst  :  dst = ~src */
    case VAX_OPC_MCOML:
        ea0    = emit_operand_ea(b, i32, &blk->ops[0], s, regs, mem);
        ea1    = emit_operand_ea(b, i32, &blk->ops[1], s, regs, mem);
        src    = emit_read_operand(b, i32, &blk->ops[0], ea0, s, regs, mem);
        result = LLVMBuildNot(b, src, "r");
        cc     = build_cc_logical(b, i32, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &blk->ops[1], ea1, result, s, regs, mem);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* CMPL src1, src2  :  src1 - src2, set CC, no store */
    case VAX_OPC_CMPL:
        ea0     = emit_operand_ea(b, i32, &blk->ops[0], s, regs, mem);
        ea1     = emit_operand_ea(b, i32, &blk->ops[1], s, regs, mem);
        src     = emit_read_operand(b, i32, &blk->ops[0], ea0, s, regs, mem);
        dst_old = emit_read_operand(b, i32, &blk->ops[1], ea1, s, regs, mem);
        result  = LLVMBuildSub(b, src, dst_old, "r");
        cc      = build_cc_sub(b, i32, dst_old, src, result, 0xFFFFFFFFu, 0x80000000u);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* TSTL src  :  set CC_LOGICAL on src, no store */
    case VAX_OPC_TSTL:
        ea0 = emit_operand_ea(b, i32, &blk->ops[0], s, regs, mem);
        src = emit_read_operand(b, i32, &blk->ops[0], ea0, s, regs, mem);
        cc  = build_cc_logical(b, i32, src, 0xFFFFFFFFu, 0x80000000u);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* CLRL dst  :  dst = 0 */
    case VAX_OPC_CLRL:
        ea0    = emit_operand_ea(b, i32, &blk->ops[0], s, regs, mem);
        result = LLVMConstInt(i32, 0, 0);
        cc     = build_cc_logical(b, i32, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &blk->ops[0], ea0, result, s, regs, mem);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* INCL dst  :  dst = dst + 1 */
    case VAX_OPC_INCL:
        ea0     = emit_operand_ea(b, i32, &blk->ops[0], s, regs, mem);
        dst_old = emit_read_operand(b, i32, &blk->ops[0], ea0, s, regs, mem);
        src     = LLVMConstInt(i32, 1, 0);
        result  = LLVMBuildAdd(b, dst_old, src, "r");
        cc      = build_cc_add(b, i32, src, dst_old, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &blk->ops[0], ea0, result, s, regs, mem);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* DECL dst  :  dst = dst - 1 */
    case VAX_OPC_DECL:
        ea0     = emit_operand_ea(b, i32, &blk->ops[0], s, regs, mem);
        dst_old = emit_read_operand(b, i32, &blk->ops[0], ea0, s, regs, mem);
        src     = LLVMConstInt(i32, 1, 0);
        result  = LLVMBuildSub(b, dst_old, src, "r");
        cc      = build_cc_sub(b, i32, src, dst_old, result, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &blk->ops[0], ea0, result, s, regs, mem);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* MOVB src, dst  :  dst[7:0] = src[7:0], CC on byte */
    case VAX_OPC_MOVB:
        ea0 = emit_operand_ea(b, i32, &blk->ops[0], s, regs, mem);
        ea1 = emit_operand_ea(b, i32, &blk->ops[1], s, regs, mem);
        src = emit_read_operand(b, i32, &blk->ops[0], ea0, s, regs, mem);
        cc  = build_cc_logical(b, i32, src, 0xFFu, 0x80u);
        emit_write_operand(b, i32, &blk->ops[1], ea1, src, s, regs, mem);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* MOVW src, dst  :  dst[15:0] = src[15:0], CC on word */
    case VAX_OPC_MOVW:
        ea0 = emit_operand_ea(b, i32, &blk->ops[0], s, regs, mem);
        ea1 = emit_operand_ea(b, i32, &blk->ops[1], s, regs, mem);
        src = emit_read_operand(b, i32, &blk->ops[0], ea0, s, regs, mem);
        cc  = build_cc_logical(b, i32, src, 0xFFFFu, 0x8000u);
        emit_write_operand(b, i32, &blk->ops[1], ea1, src, s, regs, mem);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* MOVZBL src, dst  :  dst = zero_extend(src[7:0]), CC on longword */
    case VAX_OPC_MOVZBL:
        ea0 = emit_operand_ea(b, i32, &blk->ops[0], s, regs, mem);
        ea1 = emit_operand_ea(b, i32, &blk->ops[1], s, regs, mem);
        src = emit_read_operand(b, i32, &blk->ops[0], ea0, s, regs, mem);
        cc  = build_cc_logical(b, i32, src, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &blk->ops[1], ea1, src, s, regs, mem);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* MOVZWL src, dst  :  dst = zero_extend(src[15:0]), CC on longword */
    case VAX_OPC_MOVZWL:
        ea0 = emit_operand_ea(b, i32, &blk->ops[0], s, regs, mem);
        ea1 = emit_operand_ea(b, i32, &blk->ops[1], s, regs, mem);
        src = emit_read_operand(b, i32, &blk->ops[0], ea0, s, regs, mem);
        cc  = build_cc_logical(b, i32, src, 0xFFFFFFFFu, 0x80000000u);
        emit_write_operand(b, i32, &blk->ops[1], ea1, src, s, regs, mem);
        build_psl_update(b, i32, v_psl, cc);
        return;

    /* PUSHL src  :  *(--SP) = src, CC on longword */
    case VAX_OPC_PUSHL: {
        LLVMValueRef sp, sp_new;
        ea0    = emit_operand_ea(b, i32, &blk->ops[0], s, regs, mem);
        src    = emit_read_operand(b, i32, &blk->ops[0], ea0, s, regs, mem);
        sp     = shadow_get(b, i32, s, 14, regs);
        sp_new = LLVMBuildSub(b, sp, LLVMConstInt(i32, 4, 0), "sp_dec");
        shadow_set(s, 14, sp_new);
        emit_mem_store(b, i32, mem, sp_new, src, 4);
        cc = build_cc_logical(b, i32, src, 0xFFFFFFFFu, 0x80000000u);
        build_psl_update(b, i32, v_psl, cc);
        return;
    }

    default:
        return;
    }
}

/* Compile a single block to a native function and return its pointer.
   Returns NULL on error.                                               */
static void *compile_block(VaxJITBlock *blk)
{
    static uint32_t bcnt = 0;
    char sym[32];
    snprintf(sym, sizeof(sym), "blk_%02x_%u", (uint32_t)(uint8_t)blk->opc, bcnt++);

    LLVMTypeRef i32  = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef ptr  = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef ps[3] = { ptr, ptr, ptr };
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext(sym, ctx);
    LLVMValueRef  fn  = LLVMAddFunction(mod, sym,
                            LLVMFunctionType(LLVMVoidTypeInContext(ctx),
                                             ps, 3, 0));
    LLVMBuilderRef b  = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(b,
        LLVMAppendBasicBlockInContext(ctx, fn, "entry"));

    LLVMValueRef v_regs = LLVMGetParam(fn, 0);
    LLVMValueRef v_psl  = LLVMGetParam(fn, 1);
    LLVMValueRef v_mem  = LLVMGetParam(fn, 2);

    RegShadow shadow;
    int i;
    for (i = 0; i < 16; i++) shadow.val[i] = NULL;

    emit_insn(b, i32, blk, &shadow, v_regs, v_psl, v_mem);

    shadow_spill(b, i32, &shadow, v_regs);
    LLVMBuildRetVoid(b);
    LLVMDisposeBuilder(b);

    LLVMErrorRef err = jit_add_module(mod);
    if (err) {
        char *m = LLVMGetErrorMessage(err);
        fprintf(stderr, "vax_jit: compile_block opc=%02x: %s\n",
                (uint8_t)blk->opc, m);
        LLVMDisposeErrorMessage(m);
        return NULL;
    }
    return jit_lookup(sym);
}

/* ------------------------------------------------------------------ */
/* Public block execution API                                           */
/* ------------------------------------------------------------------ */

int vax_jit_llvm_exec_block(VaxJITBlock *blk, int32_t *regs,
                             int32_t *psl, int32_t *mem)
{
    typedef void (*BlockFn)(int32_t *, int32_t *, int32_t *);
    BlockFn fn = (BlockFn)compile_block(blk);
    if (!fn) return 0;
    fn(regs, psl, mem);
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
