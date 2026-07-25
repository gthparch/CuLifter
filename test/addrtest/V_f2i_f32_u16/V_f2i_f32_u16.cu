extern "C" __global__ void V_f2i_f32_u16(const float* in, unsigned short* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = (unsigned short)((unsigned short)roundf(in[i]));
}
