#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void addr_X1_stencil_64pitch_float(const float* in, float* out,
                                    size_t pitch, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= 1 && col < W - 1 && row >= 1 && row < H - 1) {
        size_t c = (size_t)row * pitch + (size_t)col;
        size_t n = (size_t)(row - 1) * pitch + (size_t)col;
        size_t s = (size_t)(row + 1) * pitch + (size_t)col;
        size_t w = (size_t)row * pitch + (size_t)(col - 1);
        size_t e = (size_t)row * pitch + (size_t)(col + 1);
        out[c] = in[n] + in[s] + in[w] + in[e] + in[c];
    }
}

int main(int argc, char** argv) {
    constexpr int W = 32, H = 32;
    constexpr size_t pitch = W;
    constexpr int total = W * H;
    float *h_in = new float[total];
    float *h_out = new float[total];
    memset(h_out, 0, total * sizeof(float));
    for (int i = 0; i < total; i++) h_in[i] = (float)(i % 7 + 1) * 0.1f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    cudaMemcpy(d_in, h_in, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, total * sizeof(float));

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    addr_X1_stencil_64pitch_float<<<grid, block>>>(d_in, d_out, pitch, W, H);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(float), total, f);
        fclose(f);
    }

    bool ok = true;
    for (int row = 1; row < H - 1; row++) {
        for (int col = 1; col < W - 1; col++) {
            int c = row * W + col;
            int n = (row-1) * W + col;
            int s = (row+1) * W + col;
            int w = row * W + (col-1);
            int e = row * W + (col+1);
            float expected = h_in[n] + h_in[s] + h_in[w] + h_in[e] + h_in[c];
            if (fabsf(h_out[c] - expected) > 1e-5f) {
                printf("FAIL at row=%d col=%d: got %f expected %f\n",
                       row, col, h_out[c], expected);
                ok = false;
                goto done;
            }
        }
    }
    done:
    printf(ok ? "TEST PASSED\n" : "FAIL\n");

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
