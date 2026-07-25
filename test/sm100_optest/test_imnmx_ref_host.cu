#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

__global__ void test_imnmx(int* out, const int* a, const int* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int mn = min(a[i], b[i]);
        int mx = max(a[i], b[i]);
        out[i] = mx - mn;
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    bool dump = (argc > 1 && strcmp(argv[1], "--dump") == 0);

    int *h_a = new int[N], *h_b = new int[N], *h_out = new int[N];
    for (int i = 0; i < N; i++) { h_a[i] = i * 3 - 500; h_b[i] = i * 2 - 300; }

    int *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(int));

    test_imnmx<<<4, 256>>>(d_out, d_a, d_b, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    if (dump) {
        FILE* f = fopen("test_imnmx_ref_output.bin", "wb");
        fwrite(h_out, sizeof(int), N, f);
        fclose(f);
        printf("Dumped %d ints to test_imnmx_ref_output.bin\n", N);
    }

    int errors = 0;
    for (int i = 0; i < N; i++) {
        int expected = abs(h_a[i] - h_b[i]);
        if (h_out[i] != expected) {
            if (errors < 5) fprintf(stderr, "FAIL at %d: got %d expected %d\n", i, h_out[i], expected);
            errors++;
        }
    }
    printf("%s\n", errors == 0 ? "TEST PASSED" : "FAIL");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    delete[] h_a; delete[] h_b; delete[] h_out;
    return errors ? 1 : 0;
}
