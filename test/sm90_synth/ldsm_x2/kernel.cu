// LDSM.16.M88.2 test: ldmatrix.x2 loads two 8x8 b16 matrices per warp.
// Lanes 0-15 supply the row addresses (16-byte aligned; lanes 8m..8m+7
// address the rows of matrix m); fragment m of lane L = word (L&3) of the
// row addressed by lane 8m+(L>>2). The original fixture addressed rows at
// &smem[tid*2] (8-byte aligned — ldmatrix rows must be 16-byte aligned)
// and shipped an all-zero dead capture; rebuilt with aligned rows + a
// self-checking GPU main().

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void kern_ldsm_x2(const uint32_t* __restrict__ in,
                                        uint32_t* __restrict__ out,
                                        int N) {
    __shared__ uint32_t smem[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int warp = tid / 32, lane = tid % 32;

    // Fill this warp's 16-row (64-word) region: smem[i] = in[i]
    if (gid < N) {
        smem[tid * 2]     = in[gid * 2];
        smem[tid * 2 + 1] = in[gid * 2 + 1];
    }
    __syncthreads();

    // Lane L addresses row (L&15) of its warp's region (lanes 16-31's
    // addresses are ignored by HW but stay valid and 16B-aligned).
    uint32_t smem_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem[(16 * warp + (lane & 15)) * 4]));
    uint32_t frag0, frag1;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                 : "=r"(frag0), "=r"(frag1)
                 : "r"(smem_addr));

    if (gid < N) {
        out[gid * 2]     = frag0;
        out[gid * 2 + 1] = frag1;
    }
}

int main() {
    constexpr int N = 128;
    constexpr int ELEMS = N * 2;
    constexpr int blockDim_x = 128;
    constexpr int gridDim_x = 1;

    uint32_t h_in[ELEMS], h_out[ELEMS];
    for (int i = 0; i < ELEMS; i++) {
        h_in[i] = 0xB0000000u + i;
    }
    memset(h_out, 0, sizeof(h_out));

    uint32_t *d_in, *d_out;
    cudaMalloc(&d_in, ELEMS * sizeof(uint32_t));
    cudaMalloc(&d_out, ELEMS * sizeof(uint32_t));
    cudaMemcpy(d_in, h_in, ELEMS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, ELEMS * sizeof(uint32_t));

    kern_ldsm_x2<<<gridDim_x, blockDim_x>>>(d_in, d_out, N);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("TEST FAILED (launch: %s)\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_out, d_out, ELEMS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(uint32_t), ELEMS, f);
    fclose(f);

    // ldmatrix layout: out[(32w+L)*2+m] = in[(16w + 8m + (L>>2))*4 + (L&3)]
    int n_fail = 0;
    for (int w = 0; w < blockDim_x / 32; w++)
        for (int L = 0; L < 32; L++)
            for (int m = 0; m < 2; m++)
                if (h_out[(32 * w + L) * 2 + m] !=
                    h_in[(16 * w + 8 * m + (L >> 2)) * 4 + (L & 3)])
                    n_fail++;

    printf(n_fail == 0 ? "TEST PASSED\n" : "TEST FAILED (%d mismatches)\n", n_fail);

    cudaFree(d_in);
    cudaFree(d_out);
    return n_fail == 0 ? 0 : 1;
}
