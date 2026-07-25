#include <cstdio>
#include <cuda_runtime.h>

#define TILE 16

extern "C" __global__ void kern_shared_2d_tile(const float* __restrict__ in,
                                                float* __restrict__ out,
                                                int rows, int cols) {
    __shared__ float tile[TILE][TILE + 1]; // +1 to avoid bank conflicts
    int bx = blockIdx.x * TILE;
    int by = blockIdx.y * TILE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load tile from global to shared (row-major)
    int in_row = by + ty;
    int in_col = bx + tx;
    if (in_row < rows && in_col < cols)
        tile[ty][tx] = in[in_row * cols + in_col];
    else
        tile[ty][tx] = 0.0f;
    __syncthreads();

    // Store tile transposed from shared to global
    int out_row = bx + ty;
    int out_col = by + tx;
    if (out_row < cols && out_col < rows)
        out[out_row * rows + out_col] = tile[tx][ty];
}

int main() {
    constexpr int R = 32, C = 32;
    float h_in[R * C], h_out[C * R];
    for (int i = 0; i < R * C; i++) h_in[i] = (float)(i + 1);
    float *d_in, *d_out;
    cudaMalloc(&d_in, R * C * sizeof(float));
    cudaMalloc(&d_out, C * R * sizeof(float));
    cudaMemcpy(d_in, h_in, R * C * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(TILE, TILE);
    dim3 grid((C + TILE - 1) / TILE, (R + TILE - 1) / TILE);
    kern_shared_2d_tile<<<grid, block>>>(d_in, d_out, R, C);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, C * R * sizeof(float), cudaMemcpyDeviceToHost);
    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(float), C * R, f); fclose(f);
    // Verify: h_out[j*R+i] == h_in[i*C+j]
    bool ok = true;
    for (int i = 0; i < R && ok; i++)
        for (int j = 0; j < C && ok; j++)
            if (fabsf(h_out[j * R + i] - h_in[i * C + j]) > 1e-5f) ok = false;
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
