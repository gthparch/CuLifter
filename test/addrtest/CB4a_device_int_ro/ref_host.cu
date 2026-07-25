#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

__device__ int d_val = 42;

extern "C" __global__ void CB4a_device_int_ro(int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = d_val + i;
}

int main(int argc, char** argv) {
    constexpr int N = 256;
    int h_out[N]; memset(h_out, 0, sizeof(h_out));
    int *d_out; cudaMalloc(&d_out, N*sizeof(int));
    cudaMemset(d_out, 0, N*sizeof(int));
    CB4a_device_int_ro<<<(N+255)/256, 256>>>(d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(int), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(int), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (h_out[i] != 42 + i) { printf("FAIL at %d: %d vs %d\n", i, h_out[i], 42+i); ok=false; break; }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_out); return ok ? 0 : 1;
}
