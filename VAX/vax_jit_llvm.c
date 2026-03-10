/* vax_jit_llvm.c: LLVM ORC JIT engine for the VAX simulator
 *
 * This file is intentionally isolated from all SIMH and VAX headers.
 * It uses only standard C types so that LLVM's macro-heavy headers
 * never mix with SIMH's own short-name macros (BB, FP, SP, PC, etc.)
 * which would cause collisions.
 *
 * The public API uses only int and void* — see vax_jit.h for the
 * SIMH-facing wrapper that maps VAXCPUState to these primitives.
 *
 * Build requires: -I/usr/include/llvm-c-18
 *                 -I/usr/lib/llvm-18/include   (for llvm/Config/)
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
/* LLVM / ORC state                                                     */
/* ------------------------------------------------------------------ */

static LLVMOrcLLJITRef jit = NULL;
static LLVMContextRef  ctx = NULL;

static void llvm_fatal(const char *reason)
{
    fprintf(stderr, "LLVM fatal error: %s\n", reason);
    exit(1);
}

/* ------------------------------------------------------------------ */
/* Init / destroy                                                       */
/* ------------------------------------------------------------------ */

int vax_jit_llvm_init(void)
{
    LLVMErrorRef err;

    LLVMInitializeNativeTarget();
    LLVMInitializeNativeAsmPrinter();
    LLVMInstallFatalErrorHandler(llvm_fatal);

    ctx = LLVMContextCreate();
    if (!ctx) {
        fprintf(stderr, "vax_jit: failed to create LLVM context\n");
        return -1;
    }

    err = LLVMOrcCreateLLJIT(&jit, NULL);
    if (err) {
        char *msg = LLVMGetErrorMessage(err);
        fprintf(stderr, "vax_jit: ORC init failed: %s\n", msg);
        LLVMDisposeErrorMessage(msg);
        LLVMContextDispose(ctx);
        ctx = NULL;
        return -1;
    }

    fprintf(stdout, "VAX JIT enabled (LLVM ORC v2)\n");
    return 0;
}

void vax_jit_llvm_destroy(void)
{
    if (jit) { LLVMOrcDisposeLLJIT(jit); jit = NULL; }
    if (ctx) { LLVMContextDispose(ctx);  ctx = NULL; }
}

/* ------------------------------------------------------------------ */
/* NOP: emit void @vax_nop() { ret void }, compile, execute            */
/* ------------------------------------------------------------------ */

static int jit_nop(void)
{
    LLVMModuleRef              mod;
    LLVMBuilderRef             builder;
    LLVMTypeRef                fn_type;
    LLVMValueRef               fn;
    LLVMBasicBlockRef          entry_bb;
    LLVMOrcThreadSafeModuleRef tsm;
    LLVMOrcJITDylibRef         dylib;
    LLVMErrorRef               err;
    LLVMOrcExecutorAddress     addr = 0;
    void                       (*fn_ptr)(void);

    mod     = LLVMModuleCreateWithNameInContext("vax_nop_mod", ctx);
    fn_type = LLVMFunctionType(LLVMVoidTypeInContext(ctx), NULL, 0, 0);
    fn      = LLVMAddFunction(mod, "vax_nop", fn_type);
    entry_bb = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    builder = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(builder, entry_bb);
    LLVMBuildRetVoid(builder);
    LLVMDisposeBuilder(builder);

    tsm  = LLVMOrcCreateNewThreadSafeModule(
               mod, LLVMOrcCreateNewThreadSafeContext());
    dylib = LLVMOrcLLJITGetMainJITDylib(jit);
    err   = LLVMOrcLLJITAddLLVMIRModule(jit, dylib, tsm);
    if (err) {
        char *msg = LLVMGetErrorMessage(err);
        fprintf(stderr, "vax_jit NOP: add module failed: %s\n", msg);
        LLVMDisposeErrorMessage(msg);
        return 0;
    }

    err = LLVMOrcLLJITLookup(jit, &addr, "vax_nop");
    if (err) {
        char *msg = LLVMGetErrorMessage(err);
        fprintf(stderr, "vax_jit NOP: lookup failed: %s\n", msg);
        LLVMDisposeErrorMessage(msg);
        return 0;
    }

    fn_ptr = (void (*)(void))(uintptr_t)addr;
    fn_ptr();
    fprintf(stdout, "vax_jit: executed NOP via JIT\n");
    return 1;
}

/* ------------------------------------------------------------------ */
/* ADDL2: R[dst] = R[dst] + R[src],  update PSL condition codes        */
/*                                                                      */
/* IR:  void @vax_addl2(ptr %regs, ptr %psl, i32 %src_idx, i32 %dst_idx) */
/* CC (matches interpreter's CC_ADD_L macro):                           */
/*   N = result[31]                                                     */
/*   Z = result == 0                                                    */
/*   V = (~src ^ dst) & (src ^ result) has bit 31 set                  */
/*   C = (uint32)result < (uint32)dst_old                               */
/* ------------------------------------------------------------------ */

static int jit_addl2(int32_t *regs, int32_t *psl, int src_idx, int dst_idx)
{
    LLVMModuleRef              mod;
    LLVMBuilderRef             builder;
    LLVMTypeRef                param_types[4];
    LLVMTypeRef                fn_type;
    LLVMValueRef               fn;
    LLVMBasicBlockRef          entry_bb;
    LLVMOrcThreadSafeModuleRef tsm;
    LLVMOrcJITDylibRef         dylib;
    LLVMErrorRef               err;
    LLVMOrcExecutorAddress     addr = 0;
    void (*fn_ptr)(int32_t *, int32_t *, int32_t, int32_t);

    LLVMTypeRef i32  = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef ptr  = LLVMPointerTypeInContext(ctx, 0);  /* opaque ptr */

    /* Build: void @vax_addl2(ptr %regs, ptr %psl, i32 %src_idx, i32 %dst_idx) */
    param_types[0] = ptr;
    param_types[1] = ptr;
    param_types[2] = i32;
    param_types[3] = i32;
    mod     = LLVMModuleCreateWithNameInContext("vax_addl2_mod", ctx);
    fn_type = LLVMFunctionType(LLVMVoidTypeInContext(ctx), param_types, 4, 0);
    fn      = LLVMAddFunction(mod, "vax_addl2", fn_type);
    entry_bb = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    builder  = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(builder, entry_bb);

    LLVMValueRef v_regs    = LLVMGetParam(fn, 0);
    LLVMValueRef v_psl     = LLVMGetParam(fn, 1);
    LLVMValueRef v_src_idx = LLVMGetParam(fn, 2);
    LLVMValueRef v_dst_idx = LLVMGetParam(fn, 3);

    /* src = regs[src_idx] */
    LLVMValueRef src_ptr = LLVMBuildGEP2(builder, i32, v_regs, &v_src_idx, 1, "src_ptr");
    LLVMValueRef src     = LLVMBuildLoad2(builder, i32, src_ptr, "src");

    /* dst_old = regs[dst_idx] */
    LLVMValueRef dst_ptr = LLVMBuildGEP2(builder, i32, v_regs, &v_dst_idx, 1, "dst_ptr");
    LLVMValueRef dst_old = LLVMBuildLoad2(builder, i32, dst_ptr, "dst_old");

    /* result = src + dst_old */
    LLVMValueRef result = LLVMBuildAdd(builder, src, dst_old, "result");
    LLVMBuildStore(builder, result, dst_ptr);

    /* --- condition codes --- */
    LLVMValueRef zero    = LLVMConstInt(i32, 0, 0);
    LLVMValueRef lsign   = LLVMConstInt(i32, 0x80000000u, 0);
    LLVMValueRef cc_n_v  = LLVMConstInt(i32, 0x08, 0);   /* CC_N */
    LLVMValueRef cc_z_v  = LLVMConstInt(i32, 0x04, 0);   /* CC_Z */
    LLVMValueRef cc_ov_v = LLVMConstInt(i32, 0x02, 0);   /* CC_V */
    LLVMValueRef cc_c_v  = LLVMConstInt(i32, 0x01, 0);   /* CC_C */

    /* N: result[31] set */
    LLVMValueRef n_cond = LLVMBuildICmp(builder, LLVMIntSLT, result, zero, "n_cond");
    LLVMValueRef n_bit  = LLVMBuildSelect(builder, n_cond, cc_n_v, zero, "n_bit");

    /* Z: result == 0 */
    LLVMValueRef z_cond = LLVMBuildICmp(builder, LLVMIntEQ, result, zero, "z_cond");
    LLVMValueRef z_bit  = LLVMBuildSelect(builder, z_cond, cc_z_v, zero, "z_bit");

    /* V: (~src ^ dst_old) & (src ^ result) has bit 31 set */
    LLVMValueRef not_src = LLVMBuildNot(builder, src, "not_src");
    LLVMValueRef t1      = LLVMBuildXor(builder, not_src, dst_old, "t1");
    LLVMValueRef t2      = LLVMBuildXor(builder, src, result, "t2");
    LLVMValueRef t3      = LLVMBuildAnd(builder, t1, t2, "t3");
    LLVMValueRef v_msb   = LLVMBuildAnd(builder, t3, lsign, "v_msb");
    LLVMValueRef v_cond  = LLVMBuildICmp(builder, LLVMIntNE, v_msb, zero, "v_cond");
    LLVMValueRef v_bit   = LLVMBuildSelect(builder, v_cond, cc_ov_v, zero, "v_bit");

    /* C: (uint32)result < (uint32)dst_old */
    LLVMValueRef c_cond = LLVMBuildICmp(builder, LLVMIntULT, result, dst_old, "c_cond");
    LLVMValueRef c_bit  = LLVMBuildSelect(builder, c_cond, cc_c_v, zero, "c_bit");

    /* cc = N | Z | V | C */
    LLVMValueRef cc_nz  = LLVMBuildOr(builder, n_bit, z_bit, "cc_nz");
    LLVMValueRef cc_nzv = LLVMBuildOr(builder, cc_nz, v_bit, "cc_nzv");
    LLVMValueRef cc     = LLVMBuildOr(builder, cc_nzv, c_bit, "cc");

    /* PSL = (PSL & ~0xF) | cc */
    LLVMValueRef psl_old     = LLVMBuildLoad2(builder, i32, v_psl, "psl_old");
    LLVMValueRef cc_mask_inv = LLVMConstInt(i32, 0xFFFFFFF0u, 0);
    LLVMValueRef psl_no_cc   = LLVMBuildAnd(builder, psl_old, cc_mask_inv, "psl_no_cc");
    LLVMValueRef psl_new     = LLVMBuildOr(builder, psl_no_cc, cc, "psl_new");
    LLVMBuildStore(builder, psl_new, v_psl);

    LLVMBuildRetVoid(builder);
    LLVMDisposeBuilder(builder);

    tsm   = LLVMOrcCreateNewThreadSafeModule(
                mod, LLVMOrcCreateNewThreadSafeContext());
    dylib = LLVMOrcLLJITGetMainJITDylib(jit);
    err   = LLVMOrcLLJITAddLLVMIRModule(jit, dylib, tsm);
    if (err) {
        char *msg = LLVMGetErrorMessage(err);
        fprintf(stderr, "vax_jit ADDL2: add module: %s\n", msg);
        LLVMDisposeErrorMessage(msg);
        return 0;
    }
    err = LLVMOrcLLJITLookup(jit, &addr, "vax_addl2");
    if (err) {
        char *msg = LLVMGetErrorMessage(err);
        fprintf(stderr, "vax_jit ADDL2: lookup: %s\n", msg);
        LLVMDisposeErrorMessage(msg);
        return 0;
    }

    fn_ptr = (void (*)(int32_t *, int32_t *, int32_t, int32_t))(uintptr_t)addr;
    fn_ptr(regs, psl, (int32_t)src_idx, (int32_t)dst_idx);
    fprintf(stdout, "vax_jit: executed ADDL2 R%d, R%d via JIT\n", src_idx, dst_idx);
    return 1;
}

int vax_jit_llvm_addl2(int32_t *regs, int32_t *psl, int src, int dst)
{
    if (!jit) return 0;
    return jit_addl2(regs, psl, src, dst);
}

int vax_jit_llvm_nop(void)
{
    if (!jit) return 0;
    return jit_nop();
}
