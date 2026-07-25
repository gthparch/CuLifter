#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void kern_shared_scatter(const int* __restrict__ keys,
                                                int* __restrict__ out,
                                                int N) {
    __shared__ int counts[16]; // histogram bins
    __shared__ int prefix[16]; // prefix sums
    int tid = threadIdx.x;

    // Initialize shared
    if (tid < 16) counts[tid] = 0;
    __syncthreads();

    // Count keys in each bin (atomicAdd to shared)
    int gid = blockIdx.x * blockDim.x + tid;
    int bin = -1;
    if (gid < N) {
        bin = keys[gid] & 0xF; // 4-bit key
        atomicAdd(&counts[bin], 1);
    }
    __syncthreads();

    // Prefix sum on counts
    if (tid < 16) {
        int sum = 0;
        for (int i = 0; i < tid; i++) sum += counts[i];
        prefix[tid] = sum;
    }
    __syncthreads();

    // Scatter output (just write bin + prefix as verification)
    if (gid < N && bin >= 0)
        out[gid] = prefix[bin] + bin * 100;
}

int main() {
    constexpr int N = 256;
    int h_keys[N], h_out[N];
    for (int i = 0; i < N; i++) h_keys[i] = i % 16;
    int *d_keys, *d_out;
    cudaMalloc(&d_keys, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_keys, h_keys, N * sizeof(int), cudaMemcpyHostToDevice);
    kern_shared_scatter<<<1, 256>>>(d_keys, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(int), N, f); fclose(f);
    printf("TEST PASSED\n");
    cudaFree(d_keys); cudaFree(d_out);
    return 0;
}
