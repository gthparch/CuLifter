#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void X1m_double(const double* in, double* out,
                                   size_t pitch, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= 1 && col < W - 1 && row >= 1 && row < H - 1) {
        double v_n = in[(size_t)(row - 1) * pitch + (size_t)col];
        double v_w = in[(size_t)row * pitch + (size_t)(col - 1)];
        double v_s = in[(size_t)(row + 1) * pitch + (size_t)col];
        double v_e = in[(size_t)row * pitch + (size_t)(col + 1)];
        double v_c = in[(size_t)row * pitch + (size_t)col];
        size_t oc = (size_t)row * pitch + (size_t)col;
        out[oc] = v_n + v_w + v_s + v_e + v_c;
    }
}

int main(int argc, char** argv) {
    constexpr int W = 32, H = 32, total = W * H;
    double h_in[1024], h_out[1024];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < total; i++) h_in[i] = (double)((i % 7 + 1) * 0.1);
    double *d_in, *d_out;
    cudaMalloc(&d_in, total * sizeof(double)); cudaMalloc(&d_out, total * sizeof(double));
    cudaMemcpy(d_in, h_in, total * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, total * sizeof(double));
    dim3 block(16, 16); dim3 grid((W+15)/16, (H+15)/16);
    X1m_double<<<grid, block>>>(d_in, d_out, (size_t)W, W, H);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, total * sizeof(double), cudaMemcpyDeviceToHost);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }

    bool ok = true;
    for (int row = 1; row < H - 1; row++) {
        for (int col = 1; col < W - 1; col++) {
            int c = row * W + col;
            double expected = h_in[(row-1)*W + col] + h_in[row*W + (col-1)] + h_in[(row+1)*W + col] + h_in[row*W + (col+1)] + h_in[row*W + col];
            if (fabs(h_out[c] - expected) > 1e-10) {
                printf("FAIL at row=%d col=%d\n", row, col);
                ok = false; goto done;
            }
        }
    }
    done:
    printf(ok ? "TEST PASSED\n" : "FAIL\n");

    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
