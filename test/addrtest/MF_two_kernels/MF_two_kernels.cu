extern "C" __global__ void MF_kernel_simple(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] * 2.0f;
}

extern "C" __global__ void MF_kernel_complex(const float4* in, float4* out, float alpha, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float4 v = in[i];
        v.x = v.x * alpha + 1.0f;
        v.y = v.y * alpha + 2.0f;
        v.z = v.z * alpha + 3.0f;
        v.w = v.w * alpha + 4.0f;
        out[i] = v;
    }
}
