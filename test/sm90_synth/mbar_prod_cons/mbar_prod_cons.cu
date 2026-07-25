// mbar_prod_cons — CROSS-WARP producer/consumer mbarrier microbench.
//
// The minimal pattern the NOP/always-ready mbarrier model gets WRONG: the
// CONSUMER warp (warp 0, tids 0..31) reads data produced by the PRODUCER warp
// (warp 1, tids 32..63), gated by an mbarrier. Because warp 0 has the lower
// tids, the wrapper runs it first — so without a REAL barrier the consumer reads
// shared memory before the producer has written it (=> 0, wrong). A correct
// mbarrier makes the consumer block on the producer's 32 arrivals.
//
// Single block, 64 threads (2 warps), one barrier cycle. out[i] = 2*in[i].
#include <cstdint>
#include <cuda.h>

extern "C" __global__ void mbar_prod_cons(const float* __restrict__ in,
                                          float* out, int N) {
    __shared__ float smem[32];
    __shared__ __align__(8) uint64_t mbar;

    int tid  = threadIdx.x;            // 0..63
    int warp = tid >> 5;               // 0 = consumer, 1 = producer
    int lane = tid & 31;

    // Consumer warp inits the barrier to the PRODUCER warp's arrival count (32).
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                     :: "l"(&mbar), "r"(32));
    }
    __syncthreads();

    if (warp == 1) {
        // Producer: write the consumer's shared slot, then arrive.
        smem[lane] = in[lane] * 2.0f;
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "l"(&mbar));
    } else {
        // Consumer: block until the producer's tile is ready, then read it.
        uint32_t phase = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred P;\n\t"
            "L_w:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra L_w;\n\t"
            "}\n\t"
            :: "l"(&mbar), "r"(phase));
        out[lane] = smem[lane];
    }
}

int main() {
    constexpr int N = 32;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) { h_in[i] = (float)(i + 1); h_out[i] = -1.0f; }

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(float));

    mbar_prod_cons<<<1, 64>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(float), N, f);
    fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp = 2.0f * (float)(i + 1);
        if (fabsf(h_out[i] - exp) > 1e-5f) {
            printf("MISMATCH at %d: got %f expected %f\n", i, h_out[i], exp);
            ok = false; break;
        }
    }
    printf(ok ? "GPU REF PASSED\n" : "GPU REF FAILED\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
