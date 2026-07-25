extern "C" __global__ void MU_sin_cos(const float* in, float* out_sin, float* out_cos, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { out_sin[i] = sinf(in[i]); out_cos[i] = cosf(in[i]); }
}
