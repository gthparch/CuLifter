#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

__device__ __noinline__ float helper_square(float x) {
    return x * x + 1.0f;
}

__device__ __noinline__ float helper_cube(float x) {
    return x * x * x;
}

__global__ void test_call(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = helper_square(in[i]) + helper_cube(in[i]);
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    bool dump = (argc > 1 && strcmp(argv[1], "--dump") == 0);

    float *h_in = new float[N], *h_out = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1) * 0.01f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(float));

    test_call<<<4, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (dump) {
        FILE* f = fopen("test_call_ref_output.bin", "wb");
        fwrite(h_out, sizeof(float), N, f);
        fclose(f);
        printf("Dumped %d floats to test_call_ref_output.bin\n", N);
    }

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        float expected = (x * x + 1.0f) + (x * x * x);
        if (fabsf(h_out[i] - expected) > 1e-4f) {
            if (errors < 5) fprintf(stderr, "FAIL at %d: got %f expected %f\n", i, h_out[i], expected);
            errors++;
        }
    }
    printf("%s\n", errors == 0 ? "TEST PASSED" : "FAIL");

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return errors ? 1 : 0;
}
