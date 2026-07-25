extern "C" __global__ void V_prmt_byteswap(const unsigned int* in, unsigned int* out, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<N) out[i] = __byte_perm(in[i], 0, 0x0123);
}
