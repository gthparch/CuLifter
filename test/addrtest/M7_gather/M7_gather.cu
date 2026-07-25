extern "C" __global__ void M7_gather(const float* data, const int* indices,
    float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = data[indices[i]];
}
