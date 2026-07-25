#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void S_sten3_u8_1d(const unsigned char* in, unsigned char* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1)
        out[i] = (unsigned char)(in[i - 1] + in[i] + in[i + 1]);
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    unsigned char h_in[N], h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) h_in[i] = (uint8_t)((i%7)+1);
    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(uint8_t)); cudaMalloc(&d_out, N*sizeof(uint8_t));
    cudaMemcpy(d_in, h_in, N*sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(uint8_t));
    S_sten3_u8_1d<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(uint8_t), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 1; i < N-1; i++) {
        if (abs((int)h_out[i]-(int)(uint8_t)(h_in[i-1] + h_in[i] + h_in[i+1])) > 0) { printf("FAIL at %d\n", i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
