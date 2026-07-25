#include <cuda/pipeline>
#include <cstdio>
#include <cstdint>

// LDGSTS + barrier: two-phase async copy with explicit barrier wait
extern "C" __global__ void ldgsts_barrier(const float* __restrict__ in, float* out, int N) {
    __shared__ float smem_a[256];
    __shared__ float smem_b[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Phase 1: copy first half
    pipe.producer_acquire();
    if (tid < N)
        cuda::memcpy_async(&smem_a[threadIdx.x], &in[tid], sizeof(float), pipe);
    pipe.producer_commit();

    // Phase 2: copy second half (offset by N)
    pipe.producer_acquire();
    if (tid < N)
        cuda::memcpy_async(&smem_b[threadIdx.x], &in[tid], sizeof(float), pipe);
    pipe.producer_commit();

    // Wait for phase 1
    cuda::pipeline_consumer_wait_prior<1>(pipe);
    __syncthreads();

    float va = 0.0f;
    if (tid < N) va = smem_a[threadIdx.x];

    // Wait for phase 2
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    __syncthreads();

    float vb = 0.0f;
    if (tid < N) vb = smem_b[threadIdx.x];

    if (tid < N)
        out[tid] = va + vb;  // in[tid] + in[tid] = 2*in[tid]
}

int main() {
    constexpr int N = 256;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1);

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    ldgsts_barrier<<<1, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(float), N, f);
    fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = 2.0f * (float)(i + 1);
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
