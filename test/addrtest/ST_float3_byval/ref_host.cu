#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
extern "C" __global__ void ST_float3_byval(float* out, float3 offset, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = offset.x + offset.y + offset.z + (float)i;
}

int main(int argc, char** argv) {
    constexpr int N = 256;
    float h_out[N]; memset(h_out, 0, sizeof(h_out));
    float *d_out; cudaMalloc(&d_out, N*4);
    cudaMemset(d_out, 0, N*4);
    float3 offset = make_float3(1.0f, 2.0f, 3.0f);
    ST_float3_byval<<<(N+255)/256, 256>>>(d_out, offset, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*4, cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, 4, N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (fabsf(h_out[i] - (6.0f + (float)i)) > 1e-4f) { printf("FAIL %d\n", i); ok=false; break; }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_out); return ok ? 0 : 1;
}
