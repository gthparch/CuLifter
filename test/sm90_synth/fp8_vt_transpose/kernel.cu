// fp8_vt_transpose — minimal isolation of the FA3 e4m3 in-kernel V-transpose.
//
// FA3 fp8 transposes V (row-major [seqlen, hdim], 1 byte/elem e4m3) into a
// K-major Vt tile using exactly this per-fragment sequence (from the e4m3
// kernel.sass around 0xa0b0):
//
//   LDSM.16.MT88.4 R4, [src]        ; transposed load, 16-bit transpose unit
//   PRMT R8, R4, 0x6420, R5         ; even-byte gather  {R5,R4}[0,2,4,6]
//   PRMT R9, R4, 0x7531, R5         ; odd-byte  gather  {R5,R4}[1,3,5,7]
//   PRMT R10,R6, 0x6420, R7
//   PRMT R11,R6, 0x7531, R7
//   STSM.16.M88.4 [dst], R8         ; non-transposed store
//
// LDSM.16.MT88 transposes 16-bit units, but fp8 packs TWO e4m3 bytes per
// 16-bit unit (two adjacent hdim values). The 16-bit transpose keeps each
// (hdim 2j, hdim 2j+1) pair together; the PRMT then de-interleaves the even
// and odd bytes to complete the genuine 8-bit (e4m3) transpose. This kernel
// reproduces ONE full x4 fragment-transpose on a known fp8 matrix so we can
// compare the resulting smem byte layout against a CPU transpose.
//
// Tile geometry (matches the x4 ldmatrix exactly, NO lane aliasing): the
// source V is 16 seqlen rows x 32 hdim bytes (= 16 b16 cols), so the four
// 8x8 b16 matrices of x4 are fully populated and all 32 warp lanes address
// distinct b16 rows: lane L -> seqlen row (L & 0xf), bcol-half (L >> 4).
// After the transpose, Vt in smem must satisfy Vt[h][s] == V[s][h].

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#define SEQ 16
#define HD  32       // hdim bytes (e4m3); = 16 b16 cols for the full x4

extern "C" __global__ void kern_fp8_vt(const uint8_t* __restrict__ vin,
                                       uint8_t* __restrict__ vout) {
    __shared__ uint8_t vsrc[SEQ * HD];
    __shared__ uint8_t vdst[HD * SEQ];

    int tid = threadIdx.x;            // one warp = 32 lanes
    // EVERY lane fills ALL of vsrc (redundant but race-immune: identical values,
    // no cross-lane dependency, so even with the DCE'd __syncwarp the ldmatrix
    // always sees a fully-populated vsrc — isolates the transpose from any fill
    // ordering artifact).
    for (int i = 0; i < SEQ * HD; i++) vsrc[i] = vin[i];
    for (int i = 0; i < HD * SEQ; i++) vdst[i] = 0;
    __syncwarp();

    // ldmatrix.x4.trans.b16: lane L addresses one 16-byte b16 row. The source
    // tile is 16 seqlen rows x 32 hdim bytes = 16 b16 rows x 16 b16 cols. The
    // x4 covers the 2x2 grid of 8x8 b16 matrices: lane L -> seqlen row (L & 0xf)
    // of bcol-half (L >> 4) (bytes [(L>>4)*16 .. +15] of that seqlen row).
    uint32_t base = (uint32_t)__cvta_generic_to_shared(vsrc);
    uint32_t addr = base + (uint32_t)((tid & 0xf) * HD + ((tid >> 4) & 1) * 16);
    uint32_t r4, r5, r6, r7;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7) : "r"(addr));

    // fp8 byte de-interleave (completes the 8-bit transpose). PTX prmt.b32 d,a,b,c
    // uses c as the selector; the REAL FA3 kernel emits `PRMT Rd, Ra, 0x6420, Rb`
    // = prmt(a=Ra, b=Rb, sel=0x6420) (even-byte de-interleave of {Rb,Ra}). So the
    // selector must be the IMMEDIATE and Rb the b-operand: prmt(a=r4, b=r5, c=imm).
    uint32_t r8, r9, r10, r11;
    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(r8)  : "r"(r4), "r"(r5), "n"(0x6420));
    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(r9)  : "r"(r4), "r"(r5), "n"(0x7531));
    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(r10) : "r"(r6), "r"(r7), "n"(0x6420));
    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(r11) : "r"(r6), "r"(r7), "n"(0x7531));

    // stmatrix.x4 (non-transposed) into Vt smem. Dest b16 rows are SEQ=16 bytes
    // wide. Lane (8m+r) addresses dst hdim row (8m+r) — i.e. lane L addresses
    // hdim row L (0..31). Each dst row holds SEQ=16 seqlen bytes.
    uint32_t daddr = (uint32_t)__cvta_generic_to_shared(vdst) + (uint32_t)(tid * SEQ);
    asm volatile(
        "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};"
        :: "r"(daddr), "r"(r8), "r"(r9), "r"(r10), "r"(r11));
    __syncwarp();

    for (int i = tid; i < HD * SEQ; i += 32) vout[i] = vdst[i];
}

int main() {
    uint8_t hV[SEQ * HD], hVt[HD * SEQ];
    for (int s = 0; s < SEQ; s++)
        for (int h = 0; h < HD; h++)
            hV[s * HD + h] = (uint8_t)(s * HD + h);   // distinct mod 256? 16*32=512 -> NOT distinct
    memset(hVt, 0, sizeof(hVt));

    uint8_t *dV, *dVt;
    cudaMalloc(&dV, sizeof(hV));
    cudaMalloc(&dVt, sizeof(hVt));
    cudaMemcpy(dV, hV, sizeof(hV), cudaMemcpyHostToDevice);
    cudaMemset(dVt, 0, sizeof(hVt));

    kern_fp8_vt<<<1, 32>>>(dV, dVt);
    cudaDeviceSynchronize();
    cudaMemcpy(hVt, dVt, sizeof(hVt), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(hVt, 1, sizeof(hVt), f);
    fclose(f);

    printf("=== Vt smem (row-major [HD=%d][SEQ=%d]) after device transpose ===\n", HD, SEQ);
    int nmatch = 0, ntot = 0, nzero = 0;
    for (int h = 0; h < HD; h++) {
        printf("h=%2d: ", h);
        for (int s = 0; s < SEQ; s++) {
            uint8_t got = hVt[h * SEQ + s];
            uint8_t exp = (uint8_t)(s * HD + h);   // V[s][h]
            printf("%3d ", got);
            ntot++;
            if (got == exp) nmatch++;
            if (got == 0) nzero++;
        }
        printf("\n");
    }
    printf("transpose match %d/%d  (zeros=%d)\n", nmatch, ntot, nzero);
    printf(nmatch == ntot ? "TEST PASSED\n" : "TEST FAILED\n");
    cudaFree(dV); cudaFree(dVt);
    return nmatch == ntot ? 0 : 1;
}
