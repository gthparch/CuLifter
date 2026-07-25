extern "C" __global__ void SHa_stencil3_shared(const float* in, float* out, int N) {
    __shared__ float s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) s[threadIdx.x] = in[i];
    __syncthreads();
    if (i >= 1 && i < N-1 && threadIdx.x >= 1 && threadIdx.x < blockDim.x-1)
        out[i] = s[threadIdx.x-1] + s[threadIdx.x] + s[threadIdx.x+1];
}
