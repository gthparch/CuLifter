#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

__global__ void test_upred(float* out, const float* in,
                           int n, int mode, int flag1,
                           int flag2, int flag3, int flag4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    if (mode == 1) v *= 2.0f;
    if (mode == 2) v *= 3.0f;
    if (flag1) v += 1.0f;
    if (flag2) v -= 0.5f;
    if (flag3) v *= v;
    if (flag4) v = 1.0f / v;
    out[i] = v;
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    int mode = 1, flag1 = 1, flag2 = 0, flag3 = 1, flag4 = 0;
    bool dump = (argc > 1 && strcmp(argv[1], "--dump") == 0);

    float *h_in = new float[N], *h_out = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1) * 0.1f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(float));

    test_upred<<<4, 256>>>(d_out, d_in, N, mode, flag1, flag2, flag3, flag4);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (dump) {
        FILE* f = fopen("test_upred_ref_output.bin", "wb");
        fwrite(h_out, sizeof(float), N, f);
        fclose(f);
        printf("Dumped %d floats to test_upred_ref_output.bin\n", N);
    }

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float v = h_in[i];
        if (mode == 1) v *= 2.0f;
        if (mode == 2) v *= 3.0f;
        if (flag1) v += 1.0f;
        if (flag2) v -= 0.5f;
        if (flag3) v *= v;
        if (flag4) v = 1.0f / v;
        if (fabsf(h_out[i] - v) > 1e-4f) {
            if (errors < 5) fprintf(stderr, "FAIL at %d: got %f expected %f\n", i, h_out[i], v);
            errors++;
        }
    }
    printf("%s\n", errors == 0 ? "TEST PASSED" : "FAIL");

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return errors ? 1 : 0;
}
