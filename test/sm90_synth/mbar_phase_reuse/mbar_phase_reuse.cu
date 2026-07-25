// mbar_phase_reuse — TWO mbarriers REUSED across a loop (FA3-mainloop pattern).
//
// Each iteration arrives+waits two mbarriers, so each barrier's phase parity
// flips and is REUSED every iteration — the property FlashAttention-3's KV-tile
// mainloop depends on (and that the single-cycle mbar_prod_cons did not test).
// Per iteration: every lane writes its shared slot; barrier W orders all 32
// writes before the cross-lane neighbor read; barrier R orders all 32 reads
// before the next iteration overwrites the slots (the write-after-read guard).
// The two barriers make this correct on x86's independent-thread model WITHOUT
// relying on SIMT warp lockstep (a single-barrier warp-synchronous loop would be
// a WAR race here — that is a separate warp-lockstep concern, not mbarrier).
//
// out[lane] = sum over i in [0,K) of ( base[(lane+1)&31] + i ).
#include <cstdint>
#include <cuda.h>

#define K 8

extern "C" __global__ void mbar_phase_reuse(const float* __restrict__ in,
                                            float* out, int N) {
    __shared__ float smem[32];
    __shared__ __align__(8) uint64_t bW;
    __shared__ __align__(8) uint64_t bR;

    int lane = threadIdx.x;            // 0..31, single warp
    if (lane >= 32) return;

    if (lane == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "l"(&bW), "r"(32));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "l"(&bR), "r"(32));
    }
    __syncthreads();

    float base = in[lane];
    float acc = 0.0f;

    for (int i = 0; i < K; i++) {
        uint32_t ph = (uint32_t)(i & 1);
        smem[lane] = base + (float)i;
        // Barrier W: all writes complete before any neighbor read.
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "l"(&bW));
        asm volatile(
            "{\n\t.reg .pred P;\n\tLW:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra LW;\n\t}\n\t" :: "l"(&bW), "r"(ph));
        float v = smem[(lane + 1) & 31];
        // Barrier R: all reads complete before the next iteration overwrites.
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "l"(&bR));
        asm volatile(
            "{\n\t.reg .pred P;\n\tLR:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra LR;\n\t}\n\t" :: "l"(&bR), "r"(ph));
        acc += v;
    }

    out[lane] = acc;
}

int main() {
    constexpr int N = 32;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) { h_in[i] = (float)(i + 1); h_out[i] = -1.0f; }

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    mbar_phase_reuse<<<1, 32>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(float), N, f);
    fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp = 0.0f;
        for (int k = 0; k < K; k++) exp += h_in[(i + 1) & 31] + (float)k;
        if (fabsf(h_out[i] - exp) > 1e-4f) {
            printf("MISMATCH at %d: got %f expected %f\n", i, h_out[i], exp);
            ok = false; break;
        }
    }
    printf(ok ? "GPU REF PASSED\n" : "GPU REF FAILED\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
