extern "C" __global__ void SH12_two_barriers(const float* a, const float* b, float* out, int N) {
    __shared__ float s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) s[threadIdx.x] = a[i];
    __syncthreads();
    float x = 0.0f;
    if (i < N) x = s[threadIdx.x];
    if (i < N) s[threadIdx.x] = x + b[i];
    __syncthreads();
    if (i < N) out[i] = s[threadIdx.x];
}
