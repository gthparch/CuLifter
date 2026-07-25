#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void SH10_write_read_neighbor(const float* in, float* out, int N) {
    __shared__ float s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) s[threadIdx.x] = in[i];
    __syncthreads();
    if (i < N) {
        int nb = threadIdx.x ^ 1;
        int gi = blockIdx.x * blockDim.x + nb;
        if (gi < N) out[i] = s[nb];
    }
}


int main(int argc, char** argv) {
    constexpr int N = 1024, BS = 256;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i*3+7);
    memset(h_out, 0, sizeof(h_out));
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float)); cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(float));
    SH10_write_read_neighbor<<<(N+BS-1)/BS, BS>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(float), N, f); fclose(f);
    }
    bool ok = true;
    for (int blk = 0; blk < (N+BS-1)/BS; blk++)
        for (int t = 0; t < BS; t++) {
            int i = blk*BS+t; if (i>=N) continue;
            int nb = t^1; int gi = blk*BS+nb;
            if (gi < N && h_out[i] != h_in[gi]) { printf("FAIL at %d\n",i); ok=false; goto done; }
        }
    done:
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
