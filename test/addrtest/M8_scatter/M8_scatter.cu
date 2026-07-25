extern "C" __global__ void M8_scatter(const float* data, const int* indices,
    float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[indices[i]] = data[i];
}
