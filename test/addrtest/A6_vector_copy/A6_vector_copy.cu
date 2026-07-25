extern "C" __global__ void A6_vector_copy(const float4* in, float4* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = in[i];
}
