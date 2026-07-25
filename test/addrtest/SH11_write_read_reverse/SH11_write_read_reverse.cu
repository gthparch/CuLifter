extern "C" __global__ void SH11_write_read_reverse(const float* in, float* out, int N) {
    __shared__ float s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) s[threadIdx.x] = in[i];
    __syncthreads();
    if (i < N) {
        int rev = blockDim.x - 1 - threadIdx.x;
        int gi = blockIdx.x * blockDim.x + rev;
        if (gi < N) out[i] = s[rev];
    }
}
