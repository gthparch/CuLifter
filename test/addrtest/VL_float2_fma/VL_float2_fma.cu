extern "C" __global__ void VL_float2_fma(const float2* a, const float2* b, const float2* c,
    float2* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float2 va = a[i], vb = b[i], vc = c[i];
        out[i] = make_float2(va.x * vb.x + vc.x, va.y * vb.y + vc.y);
    }
}
