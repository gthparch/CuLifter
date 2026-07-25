#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void S_copy_u32_1d(const unsigned int* in, unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = in[i];
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    unsigned int h_in[N], h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) h_in[i] = (uint32_t)((i%7)+1)*100;
    unsigned int *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(uint32_t)); cudaMalloc(&d_out, N*sizeof(uint32_t));
    cudaMemcpy(d_in, h_in, N*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(uint32_t));
    S_copy_u32_1d<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(uint32_t), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) if (h_out[i] != h_in[i]) { printf("FAIL at %d\n", i); ok=false; break; }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
