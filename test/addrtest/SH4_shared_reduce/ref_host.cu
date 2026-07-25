#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void SH4_shared_reduce(const float* in, float* out, int N) {
    __shared__ float smem[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    smem[threadIdx.x] = (i < N) ? in[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[blockIdx.x] = smem[0];
}

int main(int argc, char** argv) {
    constexpr int N = 1024, BS = 256;
    constexpr int NBLK = (N + BS - 1) / BS;
    float h_in[N], h_out[NBLK];
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;
    memset(h_out, 0, sizeof(h_out));
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float)); cudaMalloc(&d_out, NBLK*sizeof(float));
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, NBLK*sizeof(float));
    SH4_shared_reduce<<<NBLK, BS>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, NBLK*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(float), NBLK, f); fclose(f);
    }
    bool ok = true;
    for (int b = 0; b < NBLK; b++) {
        int count = (b < NBLK-1) ? BS : (N - b*BS);
        if (fabsf(h_out[b]-(float)count) > 0.5f) {
            printf("FAIL at block %d: got %f exp %d\n", b, h_out[b], count); ok=false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
