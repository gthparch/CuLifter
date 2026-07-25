extern "C" __global__ void P2b_sel_sub_f32(const float* A, const float* B,
    float* out, int N, int selector, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= offset && i < N) {
        const float* src = (selector == 0) ? A : B;
        out[i] = src[i - offset];
    }
}
