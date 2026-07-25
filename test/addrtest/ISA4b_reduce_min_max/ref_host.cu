#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

extern "C" __global__
void ISA4b_reduce_min_max(int* out_min, int* out_max, const int* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int val = in[tid];
        int mn = __reduce_min_sync(0xFFFFFFFF, val);
        int mx = __reduce_max_sync(0xFFFFFFFF, val);
        if (threadIdx.x == 0) {
            out_min[0] = mn;
            out_max[0] = mx;
        }
    }
}

int main(int argc, char** argv) {
    const int N = 32;
    int h_in[32];
    int h_out[2] = {0, 0};  // [min, max]
    for (int i = 0; i < N; i++) h_in[i] = (i * 7 + 13) % 100 - 50;
    int expected_min = h_in[0], expected_max = h_in[0];
    for (int i = 1; i < N; i++) {
        if (h_in[i] < expected_min) expected_min = h_in[i];
        if (h_in[i] > expected_max) expected_max = h_in[i];
    }

    int *d_in, *d_out_min, *d_out_max;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out_min, sizeof(int));
    cudaMalloc(&d_out_max, sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out_min, 0, sizeof(int));
    cudaMemset(d_out_max, 0, sizeof(int));

    ISA4b_reduce_min_max<<<1, 32>>>(d_out_min, d_out_max, d_in, N);
    cudaMemcpy(&h_out[0], d_out_min, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_out[1], d_out_max, sizeof(int), cudaMemcpyDeviceToHost);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(int), 2, f);
        fclose(f);
    }

    if (h_out[0] == expected_min && h_out[1] == expected_max) {
        printf("TEST PASSED\n");
    } else {
        printf("TEST FAILED: min=%d (exp %d), max=%d (exp %d)\n",
               h_out[0], expected_min, h_out[1], expected_max);
        return 1;
    }
    cudaFree(d_in); cudaFree(d_out_min); cudaFree(d_out_max);
    return 0;
}
