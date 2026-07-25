#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void S_copy_u32_2d(const unsigned int* in, unsigned int* out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row < H) {
        int c = row * W + col;
        out[c] = in[c];
    }
}

int main(int argc, char** argv) {
    constexpr int W = 32, H = 32;
    unsigned int h_in[1024], h_out[1024];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < 1024; i++) h_in[i] = (uint32_t)((i%7)+1)*100;
    unsigned int *d_in, *d_out;
    cudaMalloc(&d_in, 1024*sizeof(uint32_t)); cudaMalloc(&d_out, 1024*sizeof(uint32_t));
    cudaMemcpy(d_in, h_in, 1024*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, 1024*sizeof(uint32_t));
    dim3 b(16,16), g((W+15)/16,(H+15)/16);
    S_copy_u32_2d<<<g, b>>>(d_in, d_out, W, H);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 1024*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(uint32_t), 1024, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < 1024; i++) if (h_out[i] != h_in[i]) { printf("FAIL at %d\n", i); ok=false; break; }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
