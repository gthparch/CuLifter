extern "C" __global__ void V_viadd_clamp_u8(const unsigned char* a, const unsigned char* b,
    unsigned char* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int sum = (int)a[i] + (int)b[i];
        out[i] = (unsigned char)(sum > 255 ? 255 : sum);
    }
}
