#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

extern "C" __global__
void ISA4a_reduce_add(int* out, const int* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int val = in[tid];
        int sum = __reduce_add_sync(0xFFFFFFFF, val);
        if (threadIdx.x == 0) {
            out[0] = sum;
        }
    }
}

int main(int argc, char** argv) {
    const int N = 32;
    int h_in[32], h_out[1] = {0};
    int expected = 0;
    for (int i = 0; i < N; i++) { h_in[i] = i + 1; expected += h_in[i]; }

    int *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(int));

    ISA4a_reduce_add<<<1, 32>>>(d_out, d_in, N);
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
