#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

extern "C" __global__
void ISA1b_atomicCAS_global(int* out, const int* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        atomicCAS(&out[0], 0, in[tid]);
    }
}

int main(int argc, char** argv) {
    const int N = 32;
    int h_in[32], h_out[1] = {0};
    for (int i = 0; i < N; i++) h_in[i] = 100 + i;

    int *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(int));

    ISA1b_atomicCAS_global<<<1, 32>>>(d_out, d_in, N);
    cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(int), 1, f);
        fclose(f);
    }

    // First thread (tid=0) should win CAS: out[0] = in[0] = 100
    if (h_out[0] == 100) {
        printf("TEST PASSED\n");
    } else {
        printf("TEST FAILED: got %d, expected 100\n", h_out[0]);
        return 1;
    }
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
