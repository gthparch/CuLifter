// Any kernel with parameters generates LDCU on SM100.
// LDCU loads kernel arguments from constant memory into uniform registers.
extern "C" __global__ void test_ldcu(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;
}
