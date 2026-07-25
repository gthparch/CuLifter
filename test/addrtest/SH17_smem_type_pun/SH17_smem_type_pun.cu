extern "C" __global__ void SH17_smem_type_pun(const float* in, unsigned int* out, int N) {
    extern __shared__ char smem_raw[];
    float* sf = (float*)smem_raw;
    unsigned int* si = (unsigned int*)smem_raw;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) sf[threadIdx.x] = in[i];
    __syncthreads();
    if (i < N) out[i] = si[threadIdx.x];
}
