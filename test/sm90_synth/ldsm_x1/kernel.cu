// LDSM.16.M88.1 test: ldmatrix.x1 loads one 8x8 b16 matrix per warp.
// Lanes 0-7 supply the row addresses (16-byte aligned, one row each);
// fragment of lane L = word (L&3) of the row addressed by lane (L>>2).
// The original fixture addressed rows at &smem[tid] (4-byte aligned —
// ldmatrix rows must be 16-byte aligned) and shipped an all-zero dead
// capture; rebuilt with aligned rows + a self-checking GPU main().

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void kern_ldsm_x1(const uint32_t* __restrict__ in,
                                        uint32_t* __restrict__ out,
                                        int N) {
    __shared__ uint32_t smem[128];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int warp = tid / 32, lane = tid % 32;

    // Fill this warp's 8-row (32-word) region: smem[i] = in[i]
    if (gid < N) {
        smem[tid] = in[gid];
    }
    __syncthreads();

    // Lane L addresses row (L&7) of its warp's region (lanes 8-31's
    // addresses are ignored by HW but stay valid and 16B-aligned).
    uint32_t smem_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem[(8 * warp + (lane & 7)) * 4]));
    uint32_t frag;
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
                 : "=r"(frag)
                 : "r"(smem_addr));

    if (gid < N) {
        out[gid] = frag;
    }
}

int main() {
    constexpr int N = 128;
    constexpr int blockDim_x = 128;
    constexpr int gridDim_x = 1;

    uint32_t h_in[N], h_out[N];
    for (int i = 0; i < N; i++) {
        h_in[i] = 0xA0000000u + i;
    }
    memset(h_out, 0, sizeof(h_out));

    uint32_t *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(uint32_t));
    cudaMalloc(&d_out, N * sizeof(uint32_t));
    cudaMemcpy(d_in, h_in, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(uint32_t));

    kern_ldsm_x1<<<gridDim_x, blockDim_x>>>(d_in, d_out, N);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("TEST FAILED (launch: %s)\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_out, d_out, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(uint32_t), N, f);
    fclose(f);

    // ldmatrix layout: out[32w+L] = in[(8w + (L>>2))*4 + (L&3)]
    int n_fail = 0;
    for (int w = 0; w < blockDim_x / 32; w++)
        for (int L = 0; L < 32; L++)
            if (h_out[32 * w + L] != h_in[(8 * w + (L >> 2)) * 4 + (L & 3)])
                n_fail++;

    printf(n_fail == 0 ? "TEST PASSED\n" : "TEST FAILED (%d mismatches)\n", n_fail);

    cudaFree(d_in);
    cudaFree(d_out);
    return n_fail == 0 ? 0 : 1;
}
