extern "C" __global__ void V_prmt_signext_byte(const int* in, int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Extract low byte, sign-extend to 32 bits
        int b = (signed char)(in[i] & 0xFF);
        out[i] = b;
    }
}
