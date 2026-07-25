#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void addr_H4_pitch_2d_sizet(const float* in, float* out,
                                    size_t pitch, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row < H) {
        size_t idx = (size_t)row * pitch + (size_t)col;
        out[idx] = in[idx];
    }
}

int main(int argc, char** argv) {
    constexpr int W = 32, H = 32;
    constexpr size_t pitch = W;  // elements per row
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
    addr_H4_pitch_2d_sizet<<<grid, block>>>(d_in, d_out, pitch, W, H);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(float), total, f);
        fclose(f);
    }

    bool ok = true;
    for (int i = 0; i < total; i++) {
        if (h_out[i] != h_in[i]) {
            printf("FAIL at %d: got %f expected %f\n",
                   i, h_out[i], h_in[i]);
            ok = false;
            break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
