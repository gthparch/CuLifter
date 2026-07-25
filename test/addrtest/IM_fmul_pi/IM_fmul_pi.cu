extern "C" __global__ void IM_fmul_pi(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = in[i];
        float c = 3.141592653589793f;
        out[i] = x * c;
    }
}
