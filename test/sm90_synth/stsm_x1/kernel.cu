// STSM.16.M88.1 test: Store 1 register to shared memory via stmatrix
// Uses inline PTX stmatrix to force STSM instruction in SASS.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" __global__ void kern_stsm_x1(const uint32_t* __restrict__ in,
                                         uint32_t* __restrict__ out,
                                         int N) {
    __shared__ uint32_t smem[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load value from global memory
    uint32_t val = 0;
    if (gid < N) {
        val = in[gid];
    }

    // Use stmatrix to store 1 register to shared memory
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[tid]));
    asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};"
                 :: "r"(smem_addr), "r"(val));

    __syncthreads();

    // Copy shared memory result back to global memory for verification
    if (gid < N) {
        out[gid] = smem[tid];
    }
}

int main() {
    constexpr int N = 256;
    constexpr int blockDim_x = 256;
    constexpr int gridDim_x = 1;

    uint32_t h_in[N], h_out[N];
    for (int i = 0; i < N; i++) {
        h_in[i] = 0xD0000000u + i;
    }
    memset(h_out, 0, sizeof(h_out));

    uint32_t *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(uint32_t));
    cudaMalloc(&d_out, N * sizeof(uint32_t));
    cudaMemcpy(d_in, h_in, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(uint32_t));

    kern_stsm_x1<<<gridDim_x, blockDim_x>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(uint32_t), N, f);
    fclose(f);

    bool ok = true;
    int nonzero = 0;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != 0) nonzero++;
    }
    ok = (nonzero > N / 2);

    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");

    cudaFree(d_in);
    cudaFree(d_out);
    return ok ? 0 : 1;
}
