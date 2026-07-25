extern "C" __global__ void V_prmt_signext_half(const int* in, int* out_lo, int* out_hi, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        short lo = (short)(in[i] & 0xFFFF);
        short hi = (short)((in[i] >> 16) & 0xFFFF);
        out_lo[i] = (int)lo;  // sign-extended
        out_hi[i] = (int)hi;  // sign-extended
    }
}
