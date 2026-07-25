extern "C" __global__ void SHa_offset_plus4(const float* in, float* out, int N) {
    __shared__ float s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) s[threadIdx.x] = in[i];
    __syncthreads();
    if (i < N && threadIdx.x + 4 < blockDim.x) {
        int gi = blockIdx.x * blockDim.x + threadIdx.x + 4;
        if (gi < N) out[i] = s[threadIdx.x + 4];
    }
}
