#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void D2_pitch32(const float* in, float* out, int pitch, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row >= 1 && row < H - 1) {
        int c = row * pitch + col;
        out[c] = in[c - pitch] + in[c] + in[c + pitch];
    }
}

int main(int argc, char** argv) {
    constexpr int W = 32, H = 32;
    constexpr int pitch = 32;
    float h_in[1024], h_out[1024];
    memset(h_out, 0, sizeof(h_out));
    float *d_in, *d_out;
    cudaMalloc(&d_in, 1024*sizeof(float)); cudaMalloc(&d_out, 1024*sizeof(float));
    for (int i = 0; i < 1024; i++) h_in[i] = (float)(i%7+1)*0.1f;
    cudaMemcpy(d_in, h_in, 1024*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, 1024*sizeof(float));
    dim3 block(16,16); dim3 grid((32+15)/16,(32+15)/16);
    D2_pitch32<<<grid, block>>>(d_in, d_out, pitch, 32, 32);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 1024*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }
    bool ok = true;
    for (int row = 1; row < H-1; row++) for (int col = 0; col < W; col++) {
        int c = row*pitch+col;
        float expected = h_in[c-pitch]+h_in[c]+h_in[c+pitch];
        if (fabsf(h_out[c]-expected) > 1e-5f) {
            printf("FAIL at row=%d col=%d\n", row, col); ok=false; goto done;
        }
    } done:
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
