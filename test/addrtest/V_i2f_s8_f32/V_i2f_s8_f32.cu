extern "C" __global__ void V_i2f_s8_f32(const signed char* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = (float)((float)in[i]);
}
