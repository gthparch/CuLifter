#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void N3_stencil_2d_cross(const float* in, float* out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= 1 && col < W-1 && row >= 1 && row < H-1) {
        int c = row*W + col;
        out[c] = in[c-W] + in[c+W] + in[c-1] + in[c+1] + in[c];
    }
}

int main(int argc, char** argv) {
    constexpr int W = 32, H = 32, total = W*H;
    float h_in[total], h_out[total];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < total; i++) h_in[i] = (float)(i%7+1)*0.1f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, total*sizeof(float)); cudaMalloc(&d_out, total*sizeof(float));
    cudaMemcpy(d_in, h_in, total*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, total*sizeof(float));
    dim3 block(16,16); dim3 grid((W+15)/16,(H+15)/16);
    N3_stencil_2d_cross<<<grid, block>>>(d_in, d_out, W, H);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, total*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }
    bool ok = true;
    for (int r = 1; r < H-1; r++) for (int c = 1; c < W-1; c++) {
        int i = r*W+c;
        float e = h_in[i-W]+h_in[i+W]+h_in[i-1]+h_in[i+1]+h_in[i];
        if (fabsf(h_out[i]-e) > 1e-5f) { printf("FAIL at %d\n",i); ok=false; goto done; }
    } done:
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
