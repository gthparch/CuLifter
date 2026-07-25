#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

extern "C" __global__
void ISA1c_atomicMax_global(int* out, const int* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        atomicMax(&out[0], in[tid]);
    }
}

int main(int argc, char** argv) {
    const int N = 256;
    int h_in[256], h_out[1];
    h_out[0] = -2147483647;
    int expected = -2147483647;
    for (int i = 0; i < N; i++) {
        h_in[i] = (i * 17 + 3) % 1000 - 500;
        if (h_in[i] > expected) expected = h_in[i];
    }

    int *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    int init = -2147483647;
    cudaMemcpy(d_out, &init, sizeof(int), cudaMemcpyHostToDevice);

    ISA1c_atomicMax_global<<<1, 256>>>(d_out, d_in, N);
    cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(int), 1, f);
        fclose(f);
    }

    if (h_out[0] == expected) {
        printf("TEST PASSED\n");
    } else {
        printf("TEST FAILED: got %d, expected %d\n", h_out[0], expected);
        return 1;
    }
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
