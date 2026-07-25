extern "C" __global__ void VL_float4_dot(const float4* a, const float4* b, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float4 va = a[i], vb = b[i];
        out[i] = va.x*vb.x + va.y*vb.y + va.z*vb.z + va.w*vb.w;
    }
}
