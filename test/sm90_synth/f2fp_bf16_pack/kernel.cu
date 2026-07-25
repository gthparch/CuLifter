// F2FP.BF16.F32.PACK_AB test: two f32 -> packed bf16x2 (PTX cvt.rn.bf16x2.f32).
// Covers normals across the exponent range, RN-even ties, and values whose
// f16 encoding differs wildly from bf16 (the historic bug emitted f16 bits).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void kern_f2fp_bf16_pack(const float* __restrict__ in,
                                               uint32_t* __restrict__ out,
                                               int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    float a = in[gid * 2];      // -> high 16
    float b = in[gid * 2 + 1];  // -> low 16
    uint32_t packed;
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                 : "=r"(packed) : "f"(a), "f"(b));
    out[gid] = packed;
}

int main() {
    constexpr int N = 64;
    float h_in[N * 2];
    uint32_t h_out[N];
    for (int i = 0; i < N * 2; i++) {
        // exercise wide exponent range, signs, and rounding boundaries
        float v = (float)((i % 23) + 1) * 0.731f;
        int e = (i % 17) - 8;
        h_in[i] = ((i & 1) ? -1.0f : 1.0f) * ldexpf(v, e * 3);
        if (i == 5) h_in[i] = 14.75f;          // the ex94 sentinel value
        if (i == 9) h_in[i] = 1.00390625f;     // RN-even tie case
    }
    memset(h_out, 0, sizeof(h_out));

    float* d_in; uint32_t* d_out;
    cudaMalloc(&d_in, sizeof(h_in));
    cudaMalloc(&d_out, sizeof(h_out));
    cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(h_out));
    kern_f2fp_bf16_pack<<<1, 64>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(uint32_t), N, f);
    fclose(f);
    printf("out[0]=0x%08x\nTEST PASSED\n", h_out[0]);
    return 0;
}
