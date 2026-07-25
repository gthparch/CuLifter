#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
extern "C" __global__ void WS4_prefix_sum(const float* in, float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N) ? in[tid] : 0.0f;
    for (int d = 1; d < 32; d <<= 1) {
        float n = __shfl_up_sync(0xffffffff, val, d);
        if (threadIdx.x % 32 >= d) val += n;
    }
    if (tid < N) out[tid] = val;
}

int main(int argc, char** argv) {
    constexpr int N = 256;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;
    memset(h_out, 0, sizeof(h_out));
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*4); cudaMalloc(&d_out, N*4);
    cudaMemcpy(d_in, h_in, N*4, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*4);
    WS4_prefix_sum<<<1, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*4, cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, 4, N, f); fclose(f);
    }
    bool ok = true;
    for (int w = 0; w < N/32; w++)
        for (int i = 0; i < 32; i++) {
            float exp = (float)(i + 1);
            if (fabsf(h_out[w*32+i] - exp) > 0.5f) {
                printf("FAIL at %d: %f vs %f\n", w*32+i, h_out[w*32+i], exp); ok=false; goto done;
            }
        }
    done:
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out); return ok ? 0 : 1;
}
