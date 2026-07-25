// Microbench: register-indirect CALL to DEVICE functions defined in-cubin
// (function-pointer dispatch through c[0x4] slots) — the FlashAttention
// dispatch pattern; exercises ResolveIndirectCall's defined-symbol path.
__device__ __noinline__ float fa(float x) { return x * 2.0f; }
__device__ __noinline__ float fb(float x) { return x + 10.0f; }
extern "C" __global__ void indirect_call(const float* x, float* y, int sel, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float (*f)(float) = sel ? fa : fb;
        y[i] = f(x[i]);
    }
}
