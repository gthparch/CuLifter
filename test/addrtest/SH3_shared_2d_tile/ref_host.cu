#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void SH3_shared_2d_tile(const float* in, float* out, int W, int H) {
    __shared__ float tile[16][16];
    int col = blockIdx.x * 16 + threadIdx.x;
    int row = blockIdx.y * 16 + threadIdx.y;
    if (col < W && row < H)
        tile[threadIdx.y][threadIdx.x] = in[row * W + col];
    __syncthreads();
    if (col < W && row < H)
        out[row * W + col] = tile[threadIdx.y][threadIdx.x];
}

int main(int argc, char** argv) {
    constexpr int W = 32, H = 32;
    float h_in[1024], h_out[1024];
    for (int i = 0; i < 1024; i++) h_in[i] = (float)(i*3+7);
    memset(h_out, 0, sizeof(h_out));
    float *d_in, *d_out;
    cudaMalloc(&d_in, 1024*sizeof(float)); cudaMalloc(&d_out, 1024*sizeof(float));
    cudaMemcpy(d_in, h_in, 1024*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, 1024*sizeof(float));
    dim3 b(16,16), g((W+15)/16,(H+15)/16);
    SH3_shared_2d_tile<<<g, b>>>(d_in, d_out, W, H);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 1024*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(float), 1024, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < 1024; i++)
        if (h_out[i] != h_in[i]) { printf("FAIL at %d\n",i); ok=false; break; }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
