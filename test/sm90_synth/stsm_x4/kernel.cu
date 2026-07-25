// STSM.16.M88.4 test: stmatrix.x4 stores 4 8x8 b16 matrices to shared memory
// with the warp-collective fragment redistribution (inverse of ldmatrix):
//   word w of the 16B row addressed by lane 8m+r  =  reg m of lane 4r+w
// Linear smem readback pins the layout; an ldmatrix.x4 of the stored rows
// must round-trip the original fragments (identity).

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void kern_stsm_x4(const uint32_t* __restrict__ in,
                                        uint32_t* __restrict__ out,
                                        int N) {
    __shared__ uint32_t smem[1024];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Zero the readback window (smem is not zero-initialized on HW)
    smem[tid * 4]     = 0u;
    smem[tid * 4 + 1] = 0u;
    smem[tid * 4 + 2] = 0u;
    smem[tid * 4 + 3] = 0u;
    __syncthreads();

    // Each lane's 4 source fragments (distinct words pin the distribution)
    uint32_t f0 = in[gid * 4];
    uint32_t f1 = in[gid * 4 + 1];
    uint32_t f2 = in[gid * 4 + 2];
    uint32_t f3 = in[gid * 4 + 3];

    // Each lane addresses its own 16-byte row: lane L of warp w -> smem[tid*4]
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[tid * 4]));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
                 :: "r"(smem_addr), "r"(f0), "r"(f1), "r"(f2), "r"(f3));
    __syncthreads();

    // 1) Linear readback of what landed in shared memory
    if (gid < N) {
        out[gid * 4]     = smem[tid * 4];
        out[gid * 4 + 1] = smem[tid * 4 + 1];
        out[gid * 4 + 2] = smem[tid * 4 + 2];
        out[gid * 4 + 3] = smem[tid * 4 + 3];
    }

    // 2) Round-trip: ldmatrix.x4 of the stored rows must return the fragments
    uint32_t g0, g1, g2, g3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
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
        h_in[i] = 0xC0000000u + i;
    }
    memset(h_out, 0, sizeof(h_out));

    uint32_t *d_in, *d_out;
    cudaMalloc(&d_in, ELEMS * sizeof(uint32_t));
    cudaMalloc(&d_out, 2 * ELEMS * sizeof(uint32_t));
    cudaMemcpy(d_in, h_in, ELEMS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, 2 * ELEMS * sizeof(uint32_t));

    kern_stsm_x4<<<gridDim_x, blockDim_x>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, 2 * ELEMS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(uint32_t), 2 * ELEMS, f);
    fclose(f);

    // Verify on-GPU result against the PTX ISA st.matrix layout:
    //   smem word (8m+r)*4+w of warp W = frag m of lane 4r+w of warp W
    // and the ldmatrix round-trip identity.
    int n_fail = 0;
    for (int wq = 0; wq < blockDim_x / 32; wq++) {
        for (int m = 0; m < 4; m++)
            for (int r = 0; r < 8; r++)
                for (int w = 0; w < 4; w++) {
                    uint32_t expect = h_in[(32 * wq + 4 * r + w) * 4 + m];
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
