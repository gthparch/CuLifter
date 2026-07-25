extern "C" __global__ void H6_base_large_param(const float* in, float* out,
    unsigned long long byte_offset, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* p = (const float*)((const char*)in + byte_offset);
        out[i] = p[i];
    }
}
