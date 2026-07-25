extern "C" __global__ void V_lea_hi_basic(const unsigned int* a, const unsigned int* b,
    unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // (a << 2) high word + b = LEA.HI semantics
        unsigned long long wide = ((unsigned long long)a[i] << 2);
        out[i] = (unsigned int)(wide >> 32) + b[i];
    }
}
