extern "C" __global__ void VL_saxpy_vec4(const float4* x, float4* y, float alpha, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float4 vx = x[i], vy = y[i];
        vy.x += alpha * vx.x; vy.y += alpha * vx.y;
        vy.z += alpha * vx.z; vy.w += alpha * vx.w;
        y[i] = vy;
    }
}
