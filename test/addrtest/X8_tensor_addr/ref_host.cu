#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void X8_tensor_addr(const float* in, float* out,
    size_t C, size_t H_stride, size_t W_stride, int B, int Ht, int Wt, int Ch) {
    int b = blockIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B && h < Ht && w < Wt) {
        for (int c = 0; c < Ch; c++) {
            size_t idx = (size_t)b * H_stride + (size_t)h * W_stride + (size_t)w * C + (size_t)c;
            size_t oidx = ((size_t)b * Ht * Wt + (size_t)h * Wt + (size_t)w) * Ch + c;
            out[oidx] = in[idx];
        }
    }
}

int main(int argc, char** argv) {
    constexpr int B = 2, Ht = 8, Wt = 8, Ch = 4;
    constexpr int total = B * Ht * Wt * Ch;
    float h_in[total], h_out[total];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < total; i++) h_in[i] = (float)(i % 7 + 1) * 0.1f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    cudaMemcpy(d_in, h_in, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, total * sizeof(float));
    size_t C = Ch, H_stride = (size_t)Ht * Wt * Ch, W_stride = (size_t)Wt * Ch;
    dim3 block(8, 8); dim3 grid((Wt+7)/8, (Ht+7)/8, B);
    X8_tensor_addr<<<grid, block>>>(d_in, d_out, C, H_stride, W_stride, B, Ht, Wt, Ch);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < total; i++) {
        if (fabsf(h_out[i] - h_in[i]) > 1e-5f) {
            printf("FAIL at %d: got %f expected %f\n", i, h_out[i], h_in[i]);
            ok = false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
