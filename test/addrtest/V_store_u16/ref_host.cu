#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
extern "C" __global__ void V_store_u16(const int* in, unsigned short* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = (unsigned short)(in[i] & 0xFFFF);
}
int main(int argc, char** argv) {
    constexpr int N = 1024;
    int h_in[N]; unsigned short h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = i*7+3;
    memset(h_out, 0, sizeof(h_out));
    int *d_in; unsigned short *d_out;
    cudaMalloc(&d_in, N*4); cudaMalloc(&d_out, N*2);
    cudaMemcpy(d_in, h_in, N*4, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*2);
    V_store_u16<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*2, cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) { FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, 2, N, f); fclose(f); }
    bool ok = true;
    for (int i = 0; i < N; i++) if (h_out[i] != (unsigned short)(h_in[i]&0xFFFF)) { printf("FAIL at %d\n", i); ok = false; break; }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
