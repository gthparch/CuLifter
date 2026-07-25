extern "C" __global__ void SHa_dynamic_int(const int* in, int* out, int N) {
    extern __shared__ int sd[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) sd[threadIdx.x] = in[i];
    __syncthreads();
    if (i < N) out[i] = sd[threadIdx.x];
}
