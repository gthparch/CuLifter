// Many uniform (non-thread-varying) branches force the compiler to use
// higher uniform predicate registers @UP3, @UP4, etc.
extern "C" __global__ void test_upred(float* out, const float* in,
                                       int n, int mode, int flag1,
                                       int flag2, int flag3, int flag4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    if (mode == 1) v *= 2.0f;
    if (mode == 2) v *= 3.0f;
    if (flag1) v += 1.0f;
    if (flag2) v -= 0.5f;
    if (flag3) v *= v;
    if (flag4) v = 1.0f / v;
    out[i] = v;
}
