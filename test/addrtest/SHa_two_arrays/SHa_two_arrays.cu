extern "C" __global__ void SHa_two_arrays(const float* a, const float* b, float* out, int N) {
    __shared__ float sa[256], sb[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { sa[threadIdx.x] = a[i]; sb[threadIdx.x] = b[i]; }
    __syncthreads();
    if (i < N) out[i] = sa[threadIdx.x] + sb[threadIdx.x];
}
