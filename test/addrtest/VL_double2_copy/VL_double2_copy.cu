extern "C" __global__ void VL_double2_copy(const double2* in, double2* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i];
}
