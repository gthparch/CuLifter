// STSM.16.M88.2 test: stmatrix.x2 stores 2 matrices — only lanes 0-15 supply
// row addresses; the addresses of lanes 16-31 are IGNORED by hardware
// (PTX ISA: "the values of addresses of the remaining threads are ignored").
// Every lane still passes its own address; words 64..127 of each warp's
// 128-word window are a canary that must stay zero — a per-lane-contiguous
// lowering writes through lanes 16-31's addresses and clobbers it.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void kern_stsm_x2(const uint32_t* __restrict__ in,
                                        uint32_t* __restrict__ out,
                                        int N) {
    __shared__ uint32_t smem[1024];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid * 4]     = 0u;
    smem[tid * 4 + 1] = 0u;
    smem[tid * 4 + 2] = 0u;
    smem[tid * 4 + 3] = 0u;
    __syncthreads();

    uint32_t f0 = in[gid * 2];
    uint32_t f1 = in[gid * 2 + 1];

    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[tid * 4]));
    asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};"
                 :: "r"(smem_addr), "r"(f0), "r"(f1));
    __syncthreads();

    // Full readback: rows window (lanes 0-15) + canary (lanes 16-31)
    if (gid < N) {
        out[gid * 4]     = smem[tid * 4];
        out[gid * 4 + 1] = smem[tid * 4 + 1];
        out[gid * 4 + 2] = smem[tid * 4 + 2];
        out[gid * 4 + 3] = smem[tid * 4 + 3];
    }
}

int main() {
    constexpr int N = 128;
    constexpr int IN_ELEMS = N * 2;
    constexpr int OUT_ELEMS = N * 4;
    constexpr int blockDim_x = 128;
    constexpr int gridDim_x = 1;

    uint32_t h_in[IN_ELEMS], h_out[OUT_ELEMS];
    for (int i = 0; i < IN_ELEMS; i++) {
        h_in[i] = 0xD0000000u + i;
    }
    memset(h_out, 0, sizeof(h_out));

    uint32_t *d_in, *d_out;
    cudaMalloc(&d_in, IN_ELEMS * sizeof(uint32_t));
    cudaMalloc(&d_out, OUT_ELEMS * sizeof(uint32_t));
    cudaMemcpy(d_in, h_in, IN_ELEMS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, OUT_ELEMS * sizeof(uint32_t));

    kern_stsm_x2<<<gridDim_x, blockDim_x>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, OUT_ELEMS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(uint32_t), OUT_ELEMS, f);
    fclose(f);

    // Layout: word (8m+r)*4+w of warp window = frag m of lane 4r+w (m in 0..1);
    // words 64..127 of each warp window = 0 (ignored addresses, canary).
    int n_fail = 0;
    for (int wq = 0; wq < blockDim_x / 32; wq++) {
        for (int m = 0; m < 2; m++)
            for (int r = 0; r < 8; r++)
                for (int w = 0; w < 4; w++) {
                    uint32_t expect = h_in[(32 * wq + 4 * r + w) * 2 + m];
                    uint32_t got = h_out[(32 * wq + 8 * m + r) * 4 + w];
                    if (got != expect) n_fail++;
                }
        for (int k = 64; k < 128; k++)
            if (h_out[128 * wq + k] != 0) n_fail++;
    }

    printf(n_fail == 0 ? "TEST PASSED\n" : "TEST FAILED (%d mismatches)\n", n_fail);

    cudaFree(d_in);
    cudaFree(d_out);
    return n_fail == 0 ? 0 : 1;
}
