// LDSM.16.M88.4 test: Load 4 matrix fragments (4 registers) from shared memory
// Uses inline PTX ldmatrix.x4 to force LDSM.16.M88.4 in SASS.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" __global__ void kern_ldsm_x4(const uint32_t* __restrict__ in,
                                         uint32_t* __restrict__ out,
                                         int N) {
    __shared__ uint32_t smem[1024];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load input data into shared memory (4 elements per thread)
    if (gid < N) {
        smem[tid * 4]     = in[gid * 4];
        smem[tid * 4 + 1] = in[gid * 4 + 1];
        smem[tid * 4 + 2] = in[gid * 4 + 2];
        smem[tid * 4 + 3] = in[gid * 4 + 3];
    }
    __syncthreads();

    // Use ldmatrix.x4 to load 4 registers from shared memory
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[tid * 4]));
    uint32_t frag0, frag1, frag2, frag3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(frag0), "=r"(frag1), "=r"(frag2), "=r"(frag3)
                 : "r"(smem_addr));

    // Store results to global memory
    if (gid < N) {
        out[gid * 4]     = frag0;
        out[gid * 4 + 1] = frag1;
        out[gid * 4 + 2] = frag2;
        out[gid * 4 + 3] = frag3;
    }
}

int main() {
    constexpr int N = 128;
    constexpr int ELEMS = N * 4;
    constexpr int blockDim_x = 128;
    constexpr int gridDim_x = 1;

    uint32_t h_in[ELEMS], h_out[ELEMS];
    for (int i = 0; i < ELEMS; i++) {
        h_in[i] = 0xC0000000u + i;
    }
    memset(h_out, 0, sizeof(h_out));

    uint32_t *d_in, *d_out;
    cudaMalloc(&d_in, ELEMS * sizeof(uint32_t));
    cudaMalloc(&d_out, ELEMS * sizeof(uint32_t));
    cudaMemcpy(d_in, h_in, ELEMS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, ELEMS * sizeof(uint32_t));

    kern_ldsm_x4<<<gridDim_x, blockDim_x>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, ELEMS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(uint32_t), ELEMS, f);
    fclose(f);

    bool ok = true;
    int nonzero = 0;
    for (int i = 0; i < ELEMS; i++) {
        if (h_out[i] != 0) nonzero++;
    }
    ok = (nonzero > ELEMS / 2);

    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");

    cudaFree(d_in);
    cudaFree(d_out);
    return ok ? 0 : 1;
}
