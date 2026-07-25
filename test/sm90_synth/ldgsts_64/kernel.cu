#include <cuda/pipeline>
#include <cstdio>
#include <cstdint>

// LDGSTS 64-bit: async copy 8 bytes from global to shared per thread
extern "C" __global__ void ldgsts_64(const double* __restrict__ in, double* out, int N) {
    __shared__ double smem[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    pipe.producer_acquire();
    if (tid < N)
        cuda::memcpy_async(&smem[threadIdx.x], &in[tid], sizeof(double), pipe);
    pipe.producer_commit();
    pipe.consumer_wait();
    __syncthreads();
    if (tid < N)
        out[tid] = smem[threadIdx.x] + 1.0;
}

int main() {
    constexpr int N = 256;
    double h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (double)(i + 1);

    double *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(double));
    cudaMalloc(&d_out, N * sizeof(double));
    cudaMemcpy(d_in, h_in, N * sizeof(double), cudaMemcpyHostToDevice);

    ldgsts_64<<<1, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(double), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(double), N, f);
    fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        double expected = (double)(i + 1) + 1.0;
        if (fabs(h_out[i] - expected) > 1e-10) {
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
