// Microbench: threads with global_id >= n return BEFORE __syncthreads
// (damage-cuda's exact shape: blockDim 256, n=32 -> 224 early exits).
// Exercises the x86 wrapper's barrier with partial participation.
extern "C" __global__ void early_exit_barrier(const int* in, int* out, int n) {
    extern __shared__ int cache[];
    int lid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + lid;
    if (gid >= n) return;
    cache[lid] = in[gid] != -1 ? 1 : 0;
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (lid < i) cache[lid] += cache[lid + i];
        __syncthreads();
    }
    if (lid == 0) out[blockIdx.x] = cache[0];
}
