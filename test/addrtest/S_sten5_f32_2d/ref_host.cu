#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void S_sten5_f32_2d(const float* in, float* out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row >= 2 && row < H - 2) {
        int c = row * W + col;
        out[c] = (float)(in[c - 2*W] + in[c - W] + in[c] + in[c + W] + in[c + 2*W]);
    }
}

int main(int argc, char** argv) {
    constexpr int W = 32, H = 32;
    float h_in[1024], h_out[1024];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < 1024; i++) h_in[i] = (float)((i%7)+1)*0.1f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, 1024*sizeof(float)); cudaMalloc(&d_out, 1024*sizeof(float));
    cudaMemcpy(d_in, h_in, 1024*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, 1024*sizeof(float));
    dim3 b(16,16), g((W+15)/16,(H+15)/16);
    S_sten5_f32_2d<<<g, b>>>(d_in, d_out, W, H);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 1024*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(float), 1024, f); fclose(f);
    }
    bool ok = true;
    for (int row = 2; row < H-2; row++) for (int col = 0; col < W; col++) {
        int c = row*W+col;
        if (fabsf(h_out[c]-(float)(h_in[c-2*W]+h_in[c-W]+h_in[c]+h_in[c+W]+h_in[c+2*W])) > 1e-5f) { printf("FAIL at row=%d col=%d\n", row, col); ok=false; goto done; }
    } done:
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
