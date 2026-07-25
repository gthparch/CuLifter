extern "C" __global__ void A5_two_input(const float* __restrict__ in1,
    const float* __restrict__ in2, float* __restrict__ out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = in1[i] + in2[i];
}
