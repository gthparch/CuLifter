extern "C" __global__ void P2c_neg_step_u32(const unsigned int* in, unsigned int* out, int N, int step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int j = i;
        j -= step;
        if (j >= 0 && j < N)
            out[i] = in[j];
    }
}
