#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void kern_shared_reduce(const float* __restrict__ in,
                                               float* __restrict__ out,
                                               int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (gid < N) ? in[gid] : 0.0f;
    __syncthreads();

    // Tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        out[blockIdx.x] = sdata[0];
}

int main() {
    constexpr int N = 1024;
    constexpr int BLOCK = 256;
    constexpr int GRID = (N + BLOCK - 1) / BLOCK;
    float h_in[N], h_out[GRID];
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, GRID * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    kern_shared_reduce<<<GRID, BLOCK>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, GRID * sizeof(float), cudaMemcpyDeviceToHost);
    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(float), GRID, f); fclose(f);
    // Each block sums 256 ones = 256.0
    bool ok = true;
    for (int i = 0; i < GRID; i++)
        if (fabsf(h_out[i] - 256.0f) > 1e-3f) { ok = false; break; }
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
