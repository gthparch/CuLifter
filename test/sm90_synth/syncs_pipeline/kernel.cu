#include <cstdio>
#include <cstdint>
#include <cuda.h>

// SYNCS multi-stage pipeline using mbarrier
// Two barriers for two pipeline stages: generates multiple SYNCS variants
extern "C" __global__ void syncs_pipeline(const float* __restrict__ in, float* out, int N) {
    __shared__ float smem_a[256];
    __shared__ float smem_b[256];
    __shared__ __align__(8) uint64_t mbar_a;
    __shared__ __align__(8) uint64_t mbar_b;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread 0 initializes both barriers
    if (threadIdx.x == 0) {
        asm volatile(
            "mbarrier.init.shared::cta.b64 [%0], %1;"
            :: "l"(&mbar_a), "r"(blockDim.x)
        );
        asm volatile(
            "mbarrier.init.shared::cta.b64 [%0], %1;"
            :: "l"(&mbar_b), "r"(blockDim.x)
        );
    }
    __syncthreads();

    // Phase A: copy first batch
    if (tid < N) smem_a[threadIdx.x] = in[tid];
    asm volatile(
        "mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "l"(&mbar_a)
    );

    // Phase B: copy second batch (same data, different smem)
    if (tid < N) smem_b[threadIdx.x] = in[tid];
    asm volatile(
        "mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "l"(&mbar_b)
    );

    // Wait for phase A
    {
        uint32_t phase = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred P;\n\t"
            "L_A:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra L_A;\n\t"
            "}\n\t"
            :: "l"(&mbar_a), "r"(phase)
        );
    }

    float va = 0.0f;
    if (tid < N) va = smem_a[threadIdx.x];

    // Wait for phase B
    {
        uint32_t phase = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred P;\n\t"
            "L_B:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra L_B;\n\t"
            "}\n\t"
            :: "l"(&mbar_b), "r"(phase)
        );
    }

    float vb = 0.0f;
    if (tid < N) vb = smem_b[threadIdx.x];

    // Output: sum of both phases plus 10.0f
    if (tid < N)
        out[tid] = va + vb + 10.0f;
}

int main() {
    constexpr int N = 256;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1);

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    syncs_pipeline<<<1, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(float), N, f);
    fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = 2.0f * (float)(i + 1) + 10.0f;
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
