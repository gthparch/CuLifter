extern "C" __global__ void N6_predicated_neighbor(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float left  = (i > 0)     ? in[i - 1] : 0.0f;
        float right = (i < N - 1) ? in[i + 1] : 0.0f;
        out[i] = left + in[i] + right;
    }
}
