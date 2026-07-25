#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void D2_base_selected(const float* in1, const float* in2,
    float* out, int W, int H, int selector) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row >= 1 && row < H - 1) {
        const float* in = (selector == 0) ? in1 : in2;
        int c = row * W + col;
        out[c] = in[c - W] + in[c] + in[c + W];
    }
}

int main(int argc, char** argv) {
    constexpr int W = 32, H = 32;
    float h_in1[1024], h_in2[1024], h_out[1024];
    memset(h_out, 0, sizeof(h_out));
    float *d_in1, *d_in2, *d_out;
    cudaMalloc(&d_in1, 1024*sizeof(float)); cudaMalloc(&d_in2, 1024*sizeof(float));
    cudaMalloc(&d_out, 1024*sizeof(float));
    for (int i = 0; i < 1024; i++) { h_in1[i]=(float)(i%7+1)*0.1f; h_in2[i]=(float)(i%5+1)*0.2f; }
    cudaMemcpy(d_in1, h_in1, 1024*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, h_in2, 1024*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, 1024*sizeof(float));
    dim3 block(16,16); dim3 grid((W+15)/16,(H+15)/16);
    D2_base_selected<<<grid, block>>>(d_in1, d_in2, d_out, W, H, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 1024*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }
    bool ok = true;
    for (int row = 1; row < H-1; row++) for (int col = 0; col < W; col++) {
        int c = row*W+col;
        float expected = h_in1[c-W]+h_in1[c]+h_in1[c+W];
        if (fabsf(h_out[c]-expected) > 1e-5f) {
            printf("FAIL\n"); ok=false; goto done;
        }
    } done:
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in1); cudaFree(d_in2); cudaFree(d_out);
    return ok ? 0 : 1;
}
