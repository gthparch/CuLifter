#include <cstdio>
#include <cstdint>
#include <cuda.h>

// SYNCS variant test: mbarrier with invalidation
// Uses mbarrier.init + arrive + try_wait + inval to exercise SYNCS variants
extern "C" __global__ void stas_redas(const float* __restrict__ in, float* out, int N) {
    __shared__ float smem[256];
    __shared__ __align__(8) uint64_t mbar;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread 0 init barrier
    if (threadIdx.x == 0) {
        asm volatile(
            "mbarrier.init.shared::cta.b64 [%0], %1;"
            :: "l"(&mbar), "r"(blockDim.x)
        );
    }
    __syncthreads();

    // Copy data to shared
    if (tid < N) {
        smem[threadIdx.x] = in[tid];
    }

    // All threads arrive
    asm volatile(
        "mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "l"(&mbar)
    );

    // Wait on barrier
    {
        uint32_t phase = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred P;\n\t"
            "L_W:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra L_W;\n\t"
            "}\n\t"
            :: "l"(&mbar), "r"(phase)
        );
    }

    // Invalidate barrier (thread 0 only) - may generate different SYNCS variants
    if (threadIdx.x == 0) {
        asm volatile(
            "mbarrier.inval.shared::cta.b64 [%0];"
            :: "l"(&mbar)
        );
    }
    __syncthreads();

    // Read from shared, scale by 5.0f, write to global
    if (tid < N)
        out[tid] = smem[threadIdx.x] * 5.0f;
}

int main() {
    constexpr int N = 256;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1);

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    stas_redas<<<1, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(float), N, f);
    fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = 5.0f * (float)(i + 1);
        if (fabsf(h_out[i] - expected) > 1e-5f) {
            printf("MISMATCH at %d: got %f expected %f\n", i, h_out[i], expected);
            ok = false;
            break;
        }
    }
    if (ok) printf("GPU REF PASSED\n");

    cudaFree(d_in);
    cudaFree(d_out);
    return ok ? 0 : 1;
}
