extern "C" __global__ void addr_M1_select_two_float(const float* in1, const float* in2,
                                    float* out, int N, int selector) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* src = (selector == 0) ? in1 : in2;
        out[i] = src[i];
    }
}
