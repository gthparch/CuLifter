extern "C" __global__ void IM_mixed_float_int(const float* fin, const int* iin,
    float* fout, int* iout, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        fout[i] = fin[i] * 3.14159265f;     // float constant
        iout[i] = iin[i] / 7;               // integer magic constant
    }
}
