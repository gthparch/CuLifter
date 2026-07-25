#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void S_sten5_u16_1d(const unsigned short* in, unsigned short* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 && i < N - 2)
        out[i] = (unsigned short)(in[i - 2] + in[i - 1] + in[i] + in[i + 1] + in[i + 2]);
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    unsigned short h_in[N], h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) h_in[i] = (uint16_t)((i%7)+1)*10;
    unsigned short *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(uint16_t)); cudaMalloc(&d_out, N*sizeof(uint16_t));
    cudaMemcpy(d_in, h_in, N*sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(uint16_t));
    S_sten5_u16_1d<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(uint16_t), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(uint16_t), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 2; i < N-2; i++) {
        if (abs((int)h_out[i]-(int)(uint16_t)(h_in[i-2]+h_in[i-1]+h_in[i]+h_in[i+1]+h_in[i+2])) > 0) { printf("FAIL at %d\n", i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
