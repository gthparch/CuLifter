extern "C" __global__ void SHt_i32_copy(const int* in, int* out, int N) {
    __shared__ int s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) s[threadIdx.x] = in[i];
    __syncthreads();
    if (i < N) out[i] = s[threadIdx.x];
}
