extern "C" __global__ void SHt_f64_copy(const double* in, double* out, int N) {
    __shared__ double s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) s[threadIdx.x] = in[i];
    __syncthreads();
    if (i < N) out[i] = s[threadIdx.x];
}
