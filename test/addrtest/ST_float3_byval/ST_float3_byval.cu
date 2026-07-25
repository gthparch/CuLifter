extern "C" __global__ void ST_float3_byval(float* out, float3 offset, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = offset.x + offset.y + offset.z + (float)i;
}
