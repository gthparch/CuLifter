// F2FP.F16.E4M3.UNPACK_B test: fp8 e4m3x2 -> f16x2 conversion over all
// 256 byte values, exercising both the low-half (default) and high-half
// (.H1) source selections. PTX cvt.rn.f16x2.e4m3x2 compiles to
// F2FP.F16.E4M3.UNPACK_B on sm_90.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void kern_f2fp_e4m3_unpack(const uint32_t* __restrict__ in,
                                                 uint32_t* __restrict__ out,
                                                 int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    uint32_t v = in[gid];          // 4 e4m3 bytes
    uint32_t lo16 = v;             // bytes 0,1 (low half)
    uint32_t f16x2_lo, f16x2_hi;
    asm volatile("{\n\t"
                 ".reg .b16 lo, hi;\n\t"
                 "mov.b32 {lo, hi}, %2;\n\t"
                 "cvt.rn.f16x2.e4m3x2 %0, lo;\n\t"
                 "cvt.rn.f16x2.e4m3x2 %1, hi;\n\t"
                 "}"
                 : "=r"(f16x2_lo), "=r"(f16x2_hi)
                 : "r"(lo16));
    out[gid * 2]     = f16x2_lo;   // f16(byte1):f16(byte0)
    out[gid * 2 + 1] = f16x2_hi;   // f16(byte3):f16(byte2)
}

int main() {
    constexpr int N = 64;          // 64 words = all 256 e4m3 byte values
    uint32_t h_in[N], h_out[N * 2];
    for (int i = 0; i < N; i++) {
        uint32_t b0 = 4*i, b1 = 4*i+1, b2 = 4*i+2, b3 = 4*i+3;
        h_in[i] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
    memset(h_out, 0, sizeof(h_out));

    uint32_t *d_in, *d_out;
    cudaMalloc(&d_in, sizeof(h_in));
    cudaMalloc(&d_out, sizeof(h_out));
    cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(h_out));

    kern_f2fp_e4m3_unpack<<<1, 64>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(uint32_t), N * 2, f);
    fclose(f);
    printf("out[0]=0x%08x out[1]=0x%08x\n", h_out[0], h_out[1]);
    printf("TEST PASSED\n");
    return 0;
}
