__device__ __noinline__ float helper_scale(float x, float s) { return x * s; }

extern "C" __global__ void MF_init(float* buf, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) buf[i] = (float)(i + 1);
}

extern "C" __global__ void MF_scale(float* buf, float s, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) buf[i] = helper_scale(buf[i], s);
}

extern "C" __global__ void MF_add(const float* a, const float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) c[i] = a[i] + b[i];
}
