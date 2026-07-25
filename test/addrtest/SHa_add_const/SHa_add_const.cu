extern "C" __global__ void SHa_add_const(const float* in, float* out, int N, float K) {
    __shared__ float s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) s[threadIdx.x] = in[i];
    __syncthreads();
    if (i < N) out[i] = s[threadIdx.x] + K;
}
