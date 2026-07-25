extern "C" __global__ void V_prmt_byte0(const int* in, int* out, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<N) out[i] = __byte_perm(in[i], 0, 0x4440);
}
