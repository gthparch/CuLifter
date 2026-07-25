extern "C" __global__ void BF_small64_copy(const float* in, float* out, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<N) out[i] = in[i] * 2.0f + 1.0f;
}
