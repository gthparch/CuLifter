#include <cstdio>
#include <cstdint>
#include <cuda.h>

// SYNCS.ARRIVE pattern using mbarrier inline PTX
// mbarrier init + arrive + try_wait generates SYNCS SASS instructions on SM90
extern "C" __global__ void syncs_arrive(const float* __restrict__ in, float* out, int N) {
    __shared__ float smem[256];
    // mbarrier object in shared memory (64-bit aligned, needs 8 bytes)
    __shared__ __align__(8) uint64_t mbar;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread 0 initializes the mbarrier
    if (threadIdx.x == 0) {
        asm volatile(
            "mbarrier.init.shared::cta.b64 [%0], %1;"
            :: "l"(&mbar), "r"(blockDim.x)
        );
    }
    __syncthreads();

    // Each thread does a synchronous copy
    if (tid < N) {
        smem[threadIdx.x] = in[tid];
    }

    // Signal arrival on the mbarrier
    asm volatile(
        "mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "l"(&mbar)
    );

    // Wait on the mbarrier (phase 0)
    {
        uint32_t phase = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred P;\n\t"
            "WAIT_LOOP:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra WAIT_LOOP;\n\t"
            "}\n\t"
            :: "l"(&mbar), "r"(phase)
        );
    }

    // Compute: multiply by 3.0f
    if (tid < N)
        out[tid] = smem[threadIdx.x] * 3.0f;
}

int main() {
    constexpr int N = 256;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1);

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    syncs_arrive<<<1, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(float), N, f);
    fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = 3.0f * (float)(i + 1);
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
