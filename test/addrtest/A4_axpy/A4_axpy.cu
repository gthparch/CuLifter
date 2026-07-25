extern "C" __global__ void A4_axpy(const float* in, float* out, int N, float a, float b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = a * in[i] + b;
}
