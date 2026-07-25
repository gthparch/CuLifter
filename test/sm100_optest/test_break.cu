// Early loop exit with break generates BREAK instruction.
extern "C" __global__ void test_break(int* out, const int* in, int n, int threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int sum = 0;
    for (int j = 0; j < 100; j++) {
        sum += in[(i + j) % n];
        if (sum > threshold) break;
    }
    out[i] = sum;
}
