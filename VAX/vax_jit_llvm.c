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
                                  LLVMValueRef result)
{
    LLVMValueRef z    = LLVMConstInt(i32, 0, 0);
    LLVMValueRef sign = LLVMConstInt(i32, 0x80000000u, 0);
    LLVMValueRef cn   = LLVMConstInt(i32, 0x08, 0);
    LLVMValueRef cz   = LLVMConstInt(i32, 0x04, 0);
    LLVMValueRef cv   = LLVMConstInt(i32, 0x02, 0);
    LLVMValueRef cc   = LLVMConstInt(i32, 0x01, 0);

    LLVMValueRef nb = LLVMBuildSelect(b,
        LLVMBuildICmp(b, LLVMIntSLT, result, z, "nc"), cn, z, "n");
    LLVMValueRef zb = LLVMBuildSelect(b,
        LLVMBuildICmp(b, LLVMIntEQ,  result, z, "zc"), cz, z, "z");

    /* V: (~src ^ dst_old) & (src ^ result) has bit 31 set */
    LLVMValueRef t1   = LLVMBuildXor(b, LLVMBuildNot(b, src, "ns"), dst_old, "t1");
    LLVMValueRef t2   = LLVMBuildXor(b, src, result, "t2");
    LLVMValueRef vm   = LLVMBuildAnd(b, LLVMBuildAnd(b, t1, t2, "t3"), sign, "vm");
    LLVMValueRef vb   = LLVMBuildSelect(b,
        LLVMBuildICmp(b, LLVMIntNE, vm, z, "vc"), cv, z, "v");

    /* C: (uint32)result < (uint32)dst_old */
    LLVMValueRef cb   = LLVMBuildSelect(b,
        LLVMBuildICmp(b, LLVMIntULT, result, dst_old, "cc_c"), cc, z, "c");

    return LLVMBuildOr(b,
               LLVMBuildOr(b, LLVMBuildOr(b, nb, zb, "nz"), vb, "nzv"),
               cb, "cc");
}

/* CC bits for subtraction (result = dst - src): N,Z,V,C */
static LLVMValueRef build_cc_sub(LLVMBuilderRef b, LLVMTypeRef i32,
                                  LLVMValueRef src, LLVMValueRef dst_old,
                                  LLVMValueRef result)
{
    LLVMValueRef z    = LLVMConstInt(i32, 0, 0);
    LLVMValueRef sign = LLVMConstInt(i32, 0x80000000u, 0);
    LLVMValueRef cn   = LLVMConstInt(i32, 0x08, 0);
    LLVMValueRef cz   = LLVMConstInt(i32, 0x04, 0);
    LLVMValueRef cv   = LLVMConstInt(i32, 0x02, 0);
    LLVMValueRef cc   = LLVMConstInt(i32, 0x01, 0);

    LLVMValueRef nb = LLVMBuildSelect(b,
        LLVMBuildICmp(b, LLVMIntSLT, result, z, "nc"), cn, z, "n");
    LLVMValueRef zb = LLVMBuildSelect(b,
        LLVMBuildICmp(b, LLVMIntEQ,  result, z, "zc"), cz, z, "z");

    /* V: (dst ^ src) & (dst ^ result) has bit 31 set */
    LLVMValueRef t1   = LLVMBuildAnd(b,
                            LLVMBuildXor(b, dst_old, src,    "t1"),
                            LLVMBuildXor(b, dst_old, result, "t2"), "t3");
    LLVMValueRef vm   = LLVMBuildAnd(b, t1, sign, "vm");
    LLVMValueRef vb   = LLVMBuildSelect(b,
        LLVMBuildICmp(b, LLVMIntNE, vm, z, "vc"), cv, z, "v");

    /* C: borrow = (uint32)src > (uint32)dst_old */
    LLVMValueRef cb   = LLVMBuildSelect(b,
        LLVMBuildICmp(b, LLVMIntUGT, src, dst_old, "cc_c"), cc, z, "c");

    return LLVMBuildOr(b,
               LLVMBuildOr(b, LLVMBuildOr(b, nb, zb, "nz"), vb, "nzv"),
               cb, "cc");
}

/* CC bits for logical ops: N,Z set; V=0,C=0 */
static LLVMValueRef build_cc_logical(LLVMBuilderRef b, LLVMTypeRef i32,
                                      LLVMValueRef result)
{
    LLVMValueRef z  = LLVMConstInt(i32, 0, 0);
    LLVMValueRef cn = LLVMConstInt(i32, 0x08, 0);
    LLVMValueRef cz = LLVMConstInt(i32, 0x04, 0);
    LLVMValueRef nb = LLVMBuildSelect(b,
        LLVMBuildICmp(b, LLVMIntSLT, result, z, "nc"), cn, z, "n");
    LLVMValueRef zb = LLVMBuildSelect(b,
        LLVMBuildICmp(b, LLVMIntEQ,  result, z, "zc"), cz, z, "z");
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
        cc     = build_cc_add(b, i32, src, dst_old, result);
        break;
    case VAX_INTL_SUB:
        result = LLVMBuildSub(b, dst_old, src, "r");
        cc     = build_cc_sub(b, i32, src, dst_old, result);
        break;
    case VAX_INTL_OR:
        result = LLVMBuildOr(b, dst_old, src, "r");
        cc     = build_cc_logical(b, i32, result);
        break;
    case VAX_INTL_AND_NOT: {
        LLVMValueRef ns = LLVMBuildNot(b, src, "ns");
        result = LLVMBuildAnd(b, dst_old, ns, "r");
        cc     = build_cc_logical(b, i32, result);
        break;
    }
    case VAX_INTL_XOR:
        result = LLVMBuildXor(b, dst_old, src, "r");
        cc     = build_cc_logical(b, i32, result);
        break;
    case VAX_INTL_MOV:
        result = src;
        cc     = build_cc_logical(b, i32, result);
        break;
    case VAX_INTL_COM:
        result = LLVMBuildNot(b, src, "r");
        cc     = build_cc_logical(b, i32, result);
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
    LLVMValueRef cc     = build_cc_sub(b, i32, src2, src1, result);
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
                     build_cc_logical(b, i32, src));
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
