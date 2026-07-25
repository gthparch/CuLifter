extern "C" __global__ void B28e_viadd_sat(const unsigned char* a,
                                   const unsigned char* b,
                                   unsigned char* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int sum = (unsigned int)a[i] + (unsigned int)b[i];
        out[i] = (unsigned char)(sum < 255u ? sum : 255u);
    }
}
