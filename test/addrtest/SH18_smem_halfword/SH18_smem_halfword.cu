extern "C" __global__ void SH18_smem_halfword(const short* in, short* out, int N) {
    __shared__ short s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) s[threadIdx.x] = in[i];
    __syncthreads();
    if (i < N) out[i] = s[threadIdx.x];
}
