#include <cstdio>
#include <cstdlib>
#include <cstring>

__global__ void test_break(int* out, const int* in, int n, int threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int sum = 0;
    for (int j = 0; j < 100; j++) {
        sum += in[(i + j) % n];
        if (sum > threshold) break;
    }
    out[i] = sum;
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    int threshold = 500;
    bool dump = (argc > 1 && strcmp(argv[1], "--dump") == 0);

    int *h_in = new int[N], *h_out = new int[N];
    for (int i = 0; i < N; i++) h_in[i] = (i % 10) + 1;

    int *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(int));

    test_break<<<4, 256>>>(d_out, d_in, N, threshold);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    if (dump) {
        FILE* f = fopen("test_break_ref_output.bin", "wb");
        fwrite(h_out, sizeof(int), N, f);
        fclose(f);
        printf("Dumped %d ints to test_break_ref_output.bin\n", N);
    }

    // CPU verification
    int *cpu_out = new int[N];
    for (int i = 0; i < N; i++) {
        int sum = 0;
        for (int j = 0; j < 100; j++) {
            sum += h_in[(i + j) % N];
            if (sum > threshold) break;
        }
        cpu_out[i] = sum;
    }

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != cpu_out[i]) {
            if (errors < 5) fprintf(stderr, "FAIL at %d: got %d expected %d\n", i, h_out[i], cpu_out[i]);
            errors++;
        }
    }
    printf("%s\n", errors == 0 ? "TEST PASSED" : "FAIL");

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out; delete[] cpu_out;
    return errors ? 1 : 0;
}
