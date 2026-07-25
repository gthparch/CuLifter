extern "C" __global__ void SH16_smem_bank_stride(const float* in, float* out, int N) {
    __shared__ float s[32 * 33];
    int i = blockIdx.x * 32 + threadIdx.x;
    if (i < N) s[threadIdx.x * 33] = in[i];
    __syncthreads();
    if (i < N) out[i] = s[threadIdx.x * 33];
}
