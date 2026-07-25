extern "C" __global__ void V_store_u8(const int* in, unsigned char* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = (unsigned char)(in[i] & 0xFF);
}
