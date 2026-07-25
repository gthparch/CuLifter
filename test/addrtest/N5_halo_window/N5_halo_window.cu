extern "C" __global__ void N5_halo_window(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 4 && i < N - 4) {
        float sum = 0.0f;
        sum += in[i-4] + in[i-3] + in[i-2] + in[i-1];
        sum += in[i];
        sum += in[i+1] + in[i+2] + in[i+3] + in[i+4];
        out[i] = sum;
    }
}
