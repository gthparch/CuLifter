#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void H3_idx64_multi(const float* in, float* out,
    size_t a, size_t b, size_t c, int N) {
    size_t i = (size_t)threadIdx.x;
    size_t idx = a * i + b * (size_t)blockIdx.x + c;
    int j = (int)(idx % (size_t)N);
    if (j < N) out[j] = in[j];
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    float h_in[N], h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i%7+1)*0.1f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float)); cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(float));
    H3_idx64_multi<<<1, 256>>>(d_in, d_out, 131072ULL, 65536ULL, 0ULL, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < 256; i++) {
        size_t idx = 131072ULL*i + 0ULL;
        int j = (int)(idx % 1024ULL);
        if (h_out[j] != h_in[j]) { printf("FAIL at %d\n",j); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
