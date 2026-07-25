extern "C" __global__ void V_u8_to_f32_manual(const int* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = (float)(in[i] & 0xFF);
}
