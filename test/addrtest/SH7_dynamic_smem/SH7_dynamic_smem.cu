extern "C" __global__ void SH7_dynamic_smem(const float* in, float* out, int N) {
    extern __shared__ float s[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        s[threadIdx.x] = in[i];
        __syncthreads();
        out[i] = s[threadIdx.x];
    }
}
