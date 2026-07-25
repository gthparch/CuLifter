#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void SH14_reduce_sum_16(const float* in, float* out, int N) {
    __shared__ float s[16];
    int i = blockIdx.x * 16 + threadIdx.x;
    s[threadIdx.x] = (i < N) ? in[i] : 0.0f;
    __syncthreads();
    for (int stride = 16/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}

int main(int argc, char** argv) {
    constexpr int N = 1024, BS = 16, NBLK = (N+BS-1)/BS;
    float h_in[N], h_out[NBLK];
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;
    memset(h_out, 0, sizeof(h_out));
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float)); cudaMalloc(&d_out, NBLK*sizeof(float));
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, NBLK*sizeof(float));
    SH14_reduce_sum_16<<<NBLK, BS>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, NBLK*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(float), NBLK, f); fclose(f);
    }
    bool ok = true;
    for (int b = 0; b < NBLK; b++) {
        int count = (b < NBLK-1) ? BS : (N-b*BS);
        if (fabsf(h_out[b]-(float)count) > 0.5f) { printf("FAIL blk %d\n",b); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
