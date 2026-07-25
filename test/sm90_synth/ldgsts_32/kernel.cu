#include <cuda/pipeline>
#include <cstdio>
#include <cstdint>

// LDGSTS 32-bit: async copy 4 bytes from global to shared per thread
extern "C" __global__ void ldgsts_32(const float* __restrict__ in, float* out, int N) {
    __shared__ float smem[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    pipe.producer_acquire();
    if (tid < N)
        cuda::memcpy_async(&smem[threadIdx.x], &in[tid], sizeof(float), pipe);
    pipe.producer_commit();
    pipe.consumer_wait();
    __syncthreads();
    if (tid < N)
        out[tid] = smem[threadIdx.x] + 1.0f;
}

int main() {
    constexpr int N = 256;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1);

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    ldgsts_32<<<1, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(float), N, f);
    fclose(f);

    // Verify
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = (float)(i + 1) + 1.0f;
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
