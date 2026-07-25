// trap_stub: BPT.TRAP lowering + harness stub-filter microbench.
//
// Two kernels:
//  - kern_trap_stub: whole body = printf(error) + __trap(). This is exactly
//    the shape of the CUTLASS arch-conditional guard (CUTE_INVALID_CONTROL_PATH)
//    that wgmma kernels collapse to when the cubin is built for sm_90 instead
//    of sm_90a (CUTLASS examples 54/61/63/67). SASS: CALL.ABS.NOINC (vprintf)
//    then BPT.TRAP 0x1. The lifter faithfully emits llvm.trap (ud2 on x86), so
//    an execution harness must FILTER this kernel out of the drivable set;
//    launching it is a guaranteed SIGILL on GPU and CPU alike.
//  - kern_trap_cold: a kernel with a COLD __trap() path plus real work. The
//    harness must KEEP this one (its normal exit is not trap-preceded), and it
//    must execute correctly on x86 when the trap path is not taken.
#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void kern_trap_stub() {
    printf("ERROR : Arch conditional MMA instruction used without targeting "
           "appropriate compute capability. Aborting.\n");
    __trap();
}

extern "C" __global__ void kern_trap_cold(int* out, const int* in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    if (in[tid] == 0x7eadbeef) __trap();  // never taken for driver inputs
    out[tid] = in[tid] + 1;
}

int main() {
    constexpr int N = 256;
    int h_in[N], h_out[N];
    for (int i = 0; i < N; i++) { h_in[i] = i; h_out[i] = 0; }
    int *d_in, *d_out;
    cudaMalloc(&d_in, N * 4); cudaMalloc(&d_out, N * 4);
    cudaMemcpy(d_in, h_in, N * 4, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * 4);
    kern_trap_cold<<<1, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * 4, cudaMemcpyDeviceToHost);
    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, 4, N, f); fclose(f);
    bool ok = (h_out[0] == 1 && h_out[255] == 256);
    printf(ok ? "TEST PASSED\n" : "TEST FAILED (got %d, %d)\n", h_out[0], h_out[255]);
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
