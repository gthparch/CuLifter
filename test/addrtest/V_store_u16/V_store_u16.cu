extern "C" __global__ void V_store_u16(const int* in, unsigned short* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = (unsigned short)(in[i] & 0xFFFF);
}
