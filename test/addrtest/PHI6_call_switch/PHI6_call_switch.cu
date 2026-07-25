__device__ __noinline__ int fn_a(int x) { return x * 3 + 7; }
__device__ __noinline__ int fn_b(int x) { return x * x - 5; }
__device__ __noinline__ int fn_c(int x) { return (x >> 1) + 100; }
__device__ __noinline__ int fn_d(int x) { return ~x & 0xFF; }

extern "C" __global__ void PHI6_call_switch(const int* in, int* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int val = in[tid];
    int r;
    switch (val % 4) {
        case 0: r = fn_a(val); break;
        case 1: r = fn_b(val); break;
        case 2: r = fn_c(val); break;
        default: r = fn_d(val); break;
    }
    // Post-convergence: PHI merges 4 paths
    if (r > 100) r = fn_a(r);
    else r = fn_b(r);
    out[tid] = r;
}
