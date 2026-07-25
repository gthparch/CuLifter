#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void P2_ptr_metadata(const float* in, float* out,
    int N, int stride, int offset, int scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int src_idx = i * stride + offset;
        if (src_idx >= 0 && src_idx < N)
            out[i] = in[src_idx] * (float)scale;
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    float h_in[N], h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i%7+1)*0.1f;
    int stride = 1, offset = 0, scale = 2;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float)); cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(float));
    P2_ptr_metadata<<<(N+255)/256, 256>>>(d_in, d_out, N, stride, offset, scale);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        int src = i*stride+offset;
        if (src >= 0 && src < N) {
            float e = h_in[src] * (float)scale;
            if (fabsf(h_out[i]-e) > 1e-5f) { printf("FAIL at %d\n",i); ok=false; break; }
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
