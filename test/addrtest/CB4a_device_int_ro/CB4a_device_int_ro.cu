__device__ int d_val = 42;

extern "C" __global__ void CB4a_device_int_ro(int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = d_val + i;
}
