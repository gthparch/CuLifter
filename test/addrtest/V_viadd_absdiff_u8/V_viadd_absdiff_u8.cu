extern "C" __global__ void V_viadd_absdiff_u8(const unsigned char* a, const unsigned char* b,
    unsigned char* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int d = (int)a[i] - (int)b[i];
        out[i] = (unsigned char)(d < 0 ? -d : d);
    }
}
