#include <cuda/pipeline>
#include <cstdio>
#include <cstdint>

// LDGSTS in a loop: async copy + accumulate across 4 iterations
extern "C" __global__ void ldgsts_loop(const float* __restrict__ in, float* out, int N) {
    __shared__ float smem[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;

    // 4 iterations, each does async copy + accumulate
    for (int iter = 0; iter < 4; iter++) {
        int src_idx = iter * N + tid;
        cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
        pipe.producer_acquire();
        if (tid < N)
            cuda::memcpy_async(&smem[threadIdx.x], &in[src_idx], sizeof(float), pipe);
        pipe.producer_commit();
        pipe.consumer_wait();
        __syncthreads();
        if (tid < N)
            acc += smem[threadIdx.x];
        __syncthreads();
    }

    if (tid < N)
        out[tid] = acc;
}

int main() {
    constexpr int N = 256;
    constexpr int TOTAL = N * 4;
    float h_in[TOTAL], h_out[N];
    for (int i = 0; i < TOTAL; i++) h_in[i] = (float)((i % N) + 1);

    float *d_in, *d_out;
    cudaMalloc(&d_in, TOTAL * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, TOTAL * sizeof(float), cudaMemcpyHostToDevice);

    ldgsts_loop<<<1, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(float), N, f);
    fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        // Each iteration adds (i+1), 4 iterations => 4*(i+1)
        float expected = 4.0f * (float)(i + 1);
        if (fabsf(h_out[i] - expected) > 1e-4f) {
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
