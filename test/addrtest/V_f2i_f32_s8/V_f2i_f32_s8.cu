extern "C" __global__ void V_f2i_f32_s8(const float* in, signed char* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = (signed char)((signed char)roundf(in[i]));
}
