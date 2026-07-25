extern "C" __global__ void SHt_u8_copy(const unsigned char* in, unsigned char* out, int N) {
    __shared__ unsigned char s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) s[threadIdx.x] = in[i];
    __syncthreads();
    if (i < N) out[i] = s[threadIdx.x];
}
