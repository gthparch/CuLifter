// SHFL.IDX broadcast: lane 0 broadcasts its value to all lanes in the warp.
// Each thread writes what it received via shfl. Lane 0 should write its own value.
// Output: N floats (every thread writes back what it got from the broadcast).

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_shfl_broadcast(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N) ? in[tid] : 0.0f;

    // Broadcast from lane 0 of each warp
    unsigned mask = 0xFFFFFFFF;
    float broadcast_val = __shfl_sync(mask, val, 0);

    if (tid < N) {
        out[tid] = broadcast_val;
    }
}

int main() {
    constexpr int N = 256;

    float h_in[N];
    float h_out[N];
    memset(h_out, 0, sizeof(h_out));

    // Input: 1.0, 2.0, ..., N
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1);

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(float));

    kern_shfl_broadcast<<<1, N>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Save reference
    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(float), N, f);
    fclose(f);

    // Verify: each group of 32 threads gets lane 0's value
    bool ok = true;
    for (int w = 0; w < N / 32; w++) {
        float expected = (float)(w * 32 + 1);  // lane 0 of warp w
        for (int lane = 0; lane < 32; lane++) {
            int idx = w * 32 + lane;
            if (fabsf(h_out[idx] - expected) > 1e-4f) {
                printf("FAIL [%d]: got %f expected %f\n", idx, h_out[idx], expected);
                ok = false;
                break;
            }
        }
        if (!ok) break;
    }
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");

    cudaFree(d_in);
    cudaFree(d_out);
    return ok ? 0 : 1;
}
