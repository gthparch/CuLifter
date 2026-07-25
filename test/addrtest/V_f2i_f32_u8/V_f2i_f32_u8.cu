extern "C" __global__ void V_f2i_f32_u8(const float* in, unsigned char* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = (unsigned char)((unsigned char)roundf(in[i]));
}
