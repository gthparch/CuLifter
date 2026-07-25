extern "C" __global__ void V_i2f_s16_f32(const short* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = (float)((float)in[i]);
}
