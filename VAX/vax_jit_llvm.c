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
    return 1;
}

/* ------------------------------------------------------------------ */
/* Dispatch — opcode is a plain int, no SIMH types                     */
/* ------------------------------------------------------------------ */

#define VAX_OPC_NOP 0x01

int vax_jit_llvm_execute(int opc)
{
    if (!jit)
        return 0;
    switch (opc) {
    case VAX_OPC_NOP:
        return jit_nop();
    default:
        return 0;
    }
}
