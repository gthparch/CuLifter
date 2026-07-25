// SHFL.DOWN warp reduction: sum 32 consecutive floats via warp shuffle
// Thread 0 of each warp accumulates the warp-wide sum.
// Output: N/32 floats, each the sum of a 32-element group.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_shfl_reduce_sum(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N) ? in[tid] : 0.0f;

    // Classic warp reduction via SHFL.DOWN
    unsigned mask = 0xFFFFFFFF;
    val += __shfl_down_sync(mask, val, 16);
    val += __shfl_down_sync(mask, val, 8);
    val += __shfl_down_sync(mask, val, 4);
    val += __shfl_down_sync(mask, val, 2);
    val += __shfl_down_sync(mask, val, 1);

    // Lane 0 of each warp writes the result
    int laneId = threadIdx.x & 31;
    int warpId = tid / 32;
    if (laneId == 0 && warpId < (N / 32)) {
        out[warpId] = val;
    }
}

int main() {
    constexpr int N = 256;         // 8 warps
    constexpr int NWARPS = N / 32; // 8 outputs

    float h_in[N];
    float h_out[NWARPS];
    memset(h_out, 0, sizeof(h_out));

    // Input: 1.0, 2.0, 3.0, ..., N
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1);

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, NWARPS * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, NWARPS * sizeof(float));

    kern_shfl_reduce_sum<<<1, N>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, NWARPS * sizeof(float), cudaMemcpyDeviceToHost);

    // Save reference
    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(float), NWARPS, f);
    fclose(f);

    // Verify: warp k sums elements [32*k+1 .. 32*k+32]
    bool ok = true;
    for (int w = 0; w < NWARPS; w++) {
        float expected = 0.0f;
        for (int i = 0; i < 32; i++) expected += (float)(w * 32 + i + 1);
        if (fabsf(h_out[w] - expected) > 0.5f) {
            printf("FAIL warp %d: got %f expected %f\n", w, h_out[w], expected);
            ok = false;
        }
    }
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");

    cudaFree(d_in);
    cudaFree(d_out);
    return ok ? 0 : 1;
}
