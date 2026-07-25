__device__ int d_zero;  // not initialized — CUDA zero-initializes

extern "C" __global__ void CB4g_device_zeroinit(int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = d_zero + i;
}
