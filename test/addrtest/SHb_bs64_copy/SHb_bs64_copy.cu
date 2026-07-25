extern "C" __global__ void SHb_bs64_copy(const float* in, float* out, int N) {
    __shared__ float s[64];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) s[threadIdx.x] = in[i];
    __syncthreads();
    if (i < N) out[i] = s[threadIdx.x];
}
