// Microbench: a DEFINED in-cubin device function (helper_mul3) called INDIRECTLY
// through a c[0x4] (.nv.constant4) function-pointer slot — the FlashAttnFwdSm90
// dispatch idiom. Exercises ResolveIndirectCall's defined-symbol const4 path.
//
// This .cu is the BASE used to produce the checked-in artifacts; the shipped
// kernel_sm90.sass + kernel.cubin are HAND-DERIVED from it (nvcc 12.2/sm_90 emits
// the in-cubin call via CALL.REL, not the c[0x4] CALL.ABS.NOINC idiom we need,
// and does not route a defined function through bank-4). Construction:
//   1. nvcc -arch=sm_90a -cubin -> base cubin: helper_mul3 defined at SASS 0x260,
//      reached by `CALL.REL.NOINC 0x260`.
//   2. SASS edit: replaced that call with the flash idiom
//        MOV R16,0x0; LDC.64 R16,c[0x4][R16]; LEPC R20,0x240;
//        CALL.ABS.NOINC R16; BPT.TRAP 0x1
//   3. cubin ELF surgery: appended a .rela.nv.constant4 entry mapping slot 0x0 ->
//      the DEFINED STT_FUNC symbol $const4_devcall$helper_mul3 (value=0x260).
// Oracle (host_cpu.cpp): y[i] == x[i] * 3.
#include <cstdio>
extern "C" __device__ __noinline__ float helper_mul3(float x) { return x * 3.0f; }
extern "C" __global__ void const4_devcall(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (x[i] < -1e30f) printf("neg %d\n", i);   // runtime-guarded -> table kept
        y[i] = helper_mul3(x[i]);
    }
}
