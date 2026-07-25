extern "C" __global__ void IM_int_as_float(const int* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // __int_as_float reinterprets bits, not converts
        float f = __int_as_float(in[i]);
        out[i] = f * 2.0f;
    }
}
