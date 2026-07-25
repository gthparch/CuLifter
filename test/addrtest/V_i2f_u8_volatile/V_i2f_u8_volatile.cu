extern "C" __global__ void V_i2f_u8_volatile(const unsigned char* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        volatile unsigned char v = in[i];
        out[i] = (float)v;
    }
}
