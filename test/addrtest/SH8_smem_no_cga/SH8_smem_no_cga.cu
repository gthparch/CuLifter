extern "C" __global__ void SH8_smem_no_cga(const float* in, float* out, int N) {
    __shared__ float s[256];
    int i = threadIdx.x;
    if (i < N) {
        s[i] = in[i];
        __syncthreads();
        out[i] = s[i];
    }
}
