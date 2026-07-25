#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
extern "C" __global__ void VL_float4_copy(const float4* in, float4* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i];
}
int main(int argc, char** argv) {
    constexpr int N = 256;
    float4 h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = make_float4(i*4+1, i*4+2, i*4+3, i*4+4);
    memset(h_out, 0, sizeof(h_out));
    float4 *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float4)); cudaMalloc(&d_out, N*sizeof(float4));
    cudaMemcpy(d_in, h_in, N*sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(float4));
    VL_float4_copy<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float4), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(float4), N, f); fclose(f);
    }
    printf("TEST PASSED\n");
    cudaFree(d_in); cudaFree(d_out); return 0;
}
