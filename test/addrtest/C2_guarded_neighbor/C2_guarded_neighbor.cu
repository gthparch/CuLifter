extern "C" __global__ void C2_guarded_neighbor(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < N - 1)
        out[i] = in[i-1] + in[i] + in[i+1];
    else if (i < N)
        out[i] = in[i];
}
