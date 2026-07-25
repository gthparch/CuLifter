// STSM.16.MT88.4 test: stmatrix.x4.trans stores 4 TRANSPOSED 8x8 b16 matrices.
// Memory receives F^T of the standard fragment matrix F:
//   T[r][c] = F[c][r] = halfword (r&1) of reg m of lane 4c+(r>>1)
//   => word w of the row addressed by lane 8m+r
//      = sel(r&1, frag_m[lane 8w+(r>>1)]) | sel(r&1, frag_m[lane 8w+4+(r>>1)]) << 16
// where sel(0,x)=lo16(x), sel(1,x)=hi16(x).
// Cross-check: ldmatrix.x4.trans of the stored rows round-trips the fragments.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void kern_stsm_mt88(const uint32_t* __restrict__ in,
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

    uint32_t f0 = in[gid * 4];
    uint32_t f1 = in[gid * 4 + 1];
    uint32_t f2 = in[gid * 4 + 2];
    uint32_t f3 = in[gid * 4 + 3];

    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[tid * 4]));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};"
                 :: "r"(smem_addr), "r"(f0), "r"(f1), "r"(f2), "r"(f3));
    __syncthreads();

    if (gid < N) {
        out[gid * 4]     = smem[tid * 4];
        out[gid * 4 + 1] = smem[tid * 4 + 1];
        out[gid * 4 + 2] = smem[tid * 4 + 2];
        out[gid * 4 + 3] = smem[tid * 4 + 3];
    }

    uint32_t g0, g1, g2, g3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(g0), "=r"(g1), "=r"(g2), "=r"(g3)
                 : "r"(smem_addr));
    if (gid < N) {
        out[(N + gid) * 4]     = g0;
        out[(N + gid) * 4 + 1] = g1;
        out[(N + gid) * 4 + 2] = g2;
        out[(N + gid) * 4 + 3] = g3;
    }
}

int main() {
    constexpr int N = 128;
    constexpr int ELEMS = N * 4;
    constexpr int blockDim_x = 128;
    constexpr int gridDim_x = 1;

    uint32_t h_in[ELEMS], h_out[2 * ELEMS];
    for (int i = 0; i < ELEMS; i++) {
        h_in[i] = (0xA000u + 2 * i) | ((0xB001u + 2 * i) << 16);  // distinct halfwords
    }
    memset(h_out, 0, sizeof(h_out));

    uint32_t *d_in, *d_out;
    cudaMalloc(&d_in, ELEMS * sizeof(uint32_t));
    cudaMalloc(&d_out, 2 * ELEMS * sizeof(uint32_t));
    cudaMemcpy(d_in, h_in, ELEMS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, 2 * ELEMS * sizeof(uint32_t));

    kern_stsm_mt88<<<gridDim_x, blockDim_x>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, 2 * ELEMS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(uint32_t), 2 * ELEMS, f);
    fclose(f);

    // Verify against the transposed st.matrix layout + round-trip identity.
    auto frag = [&](int wq, int lane, int m) { return h_in[(32 * wq + lane) * 4 + m]; };
    int n_fail = 0;
    for (int wq = 0; wq < blockDim_x / 32; wq++) {
        for (int m = 0; m < 4; m++)
            for (int r = 0; r < 8; r++)
                for (int w = 0; w < 4; w++) {
                    uint32_t v0 = frag(wq, 8 * w + (r >> 1), m);
                    uint32_t v1 = frag(wq, 8 * w + 4 + (r >> 1), m);
                    uint16_t lo = (uint16_t)((r & 1) ? (v0 >> 16) : (v0 & 0xffff));
                    uint16_t hi = (uint16_t)((r & 1) ? (v1 >> 16) : (v1 & 0xffff));
                    uint32_t expect = (uint32_t)lo | ((uint32_t)hi << 16);
                    uint32_t got = h_out[(32 * wq + 8 * m + r) * 4 + w];
                    if (got != expect) n_fail++;
                }
    }
    for (int i = 0; i < ELEMS; i++)
        if (h_out[ELEMS + i] != h_in[i]) n_fail++;

    printf(n_fail == 0 ? "TEST PASSED\n" : "TEST FAILED (%d mismatches)\n", n_fail);

    cudaFree(d_in);
    cudaFree(d_out);
    return n_fail == 0 ? 0 : 1;
}
