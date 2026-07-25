extern "C" __global__ void V_f2i_f32_s16(const float* in, short* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = (short)((short)roundf(in[i]));
}
