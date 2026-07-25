#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
struct Params { float scale; int offset; float bias; };
extern "C" __global__ void ST_custom_struct(const float* in, float* out, Params p, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i + p.offset] * p.scale + p.bias;
}
int main(int argc, char** argv) {
    constexpr int N = 256;
    float h_in[N+10], h_out[N];
    for (int i = 0; i < N+10; i++) h_in[i] = (float)(i+1);
    memset(h_out, 0, sizeof(h_out));
    float *d_in, *d_out;
    cudaMalloc(&d_in, (N+10)*4); cudaMalloc(&d_out, N*4);
    cudaMemcpy(d_in, h_in, (N+10)*4, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*4);
    Params p = {2.0f, 5, 10.0f};
    ST_custom_struct<<<(N+255)/256, 256>>>(d_in, d_out, p, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*4, cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, 4, N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp = h_in[i+5] * 2.0f + 10.0f;
        if (fabsf(h_out[i]-exp)>1e-3f) { printf("FAIL %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out); return ok ? 0 : 1;
}
