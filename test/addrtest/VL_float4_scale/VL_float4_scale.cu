extern "C" __global__ void VL_float4_scale(const float4* in, float4* out, float alpha, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float4 v = in[i];
        v.x *= alpha; v.y *= alpha; v.z *= alpha; v.w *= alpha;
        out[i] = v;
    }
}
