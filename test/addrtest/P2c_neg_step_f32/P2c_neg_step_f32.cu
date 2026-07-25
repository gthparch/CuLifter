extern "C" __global__ void P2c_neg_step_f32(const float* in, float* out, int N, int step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int j = i;
        j -= step;
        if (j >= 0 && j < N)
            out[i] = in[j];
    }
}
