extern "C" __global__ void P2b_sel_sub_u32(const unsigned int* A, const unsigned int* B,
    unsigned int* out, int N, int selector, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= offset && i < N) {
        const unsigned int* src = (selector == 0) ? A : B;
        out[i] = src[i - offset];
    }
}
