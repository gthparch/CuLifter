extern "C" __global__ void addr_X4_base_large_off_float(const float* in1, const float* in2,
                                    float* out, int N, int selector,
                                    unsigned long long base_offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* base = (selector == 0) ? in1 : in2;
        // base_offset is 0 in practice but forces 64-bit address math
        unsigned long long addr = (unsigned long long)base + base_offset +
                                  (unsigned long long)i * sizeof(float);
        out[i] = *(const float*)addr;
    }
}
