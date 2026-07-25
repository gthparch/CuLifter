extern "C" __global__ void V_shr_arith(const int* in, int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] >> 31;
}
