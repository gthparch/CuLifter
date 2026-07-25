extern "C" __global__ void IM_float_as_int(const float* in, int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int bits = __float_as_int(in[i]);
        out[i] = bits >> 23;  // extract exponent
    }
}
