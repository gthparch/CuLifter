// Clamp pattern: fminf(fmaxf(v, lo), hi) generates FMNMX3 on SM100.
// On SM90, this compiles to two separate FMNMX instructions.
extern "C" __global__ void test_fmnmx3(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in[i];
        // Try multiple clamp patterns to maximize FMNMX3 generation
        float clamped = fminf(fmaxf(v, 0.0f), 1.0f);    // clamp to [0,1]
        float relu6   = fminf(fmaxf(v, 0.0f), 6.0f);    // ReLU6
        out[i] = clamped + relu6;
    }
}
