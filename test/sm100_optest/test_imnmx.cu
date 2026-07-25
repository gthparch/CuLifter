// Integer min/max generates IMNMX on SM100.
extern "C" __global__ void test_imnmx(int* out, const int* a, const int* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int mn = min(a[i], b[i]);
        int mx = max(a[i], b[i]);
        out[i] = mx - mn;  // range
    }
}
