#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

extern "C" __global__
void ISA3a_match_any(int* out, const int* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int val = in[tid];
        unsigned mask = __match_any_sync(0xFFFFFFFF, val);
        out[tid] = __popc(mask);
    }
}

int main(int argc, char** argv) {
    const int N = 32;
    int h_in[32], h_out[32];
    // Groups: 0-7 have value 10, 8-15 have value 20, 16-31 have value 30
    for (int i = 0; i < 8;  i++) h_in[i] = 10;
    for (int i = 8; i < 16; i++) h_in[i] = 20;
    for (int i = 16; i < 32; i++) h_in[i] = 30;

    int *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    ISA3a_match_any<<<1, 32>>>(d_out, d_in, N);
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(int), N, f);
        fclose(f);
    }

    int pass = 1;
    for (int i = 0; i < 8; i++)  if (h_out[i] != 8)  { pass = 0; break; }
    for (int i = 8; i < 16; i++) if (h_out[i] != 8)  { pass = 0; break; }
    for (int i = 16; i < 32; i++) if (h_out[i] != 16) { pass = 0; break; }

    if (pass) {
        printf("TEST PASSED\n");
    } else {
        printf("TEST FAILED\n");
        for (int i = 0; i < N; i++) printf("  [%d] = %d\n", i, h_out[i]);
        return 1;
    }
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
