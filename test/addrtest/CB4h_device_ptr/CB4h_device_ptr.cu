__device__ float* d_ptr;

extern "C" __global__ void CB4h_device_ptr(float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = d_ptr[i];
}
