// __noinline__ prevents inlining, forcing a CALL instruction in SASS.
// Without it, nvcc inlines small functions and CALL disappears.
__device__ __noinline__ float helper_square(float x) {
    return x * x + 1.0f;
}

__device__ __noinline__ float helper_cube(float x) {
    return x * x * x;
}

extern "C" __global__ void test_call(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = helper_square(in[i]) + helper_cube(in[i]);
}
