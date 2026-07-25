#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void X5_gather_u64(const float* data,
    const unsigned long long* indices, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = data[(int)(indices[i] % (unsigned long long)N)];
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    float h_data[N]; unsigned long long h_idx[N]; float h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) { h_data[i]=(float)(i%7+1)*0.1f; h_idx[i]=(unsigned long long)((i*7)%N); }
    float *d_data, *d_out; unsigned long long *d_idx;
    cudaMalloc(&d_data, N*sizeof(float)); cudaMalloc(&d_idx, N*sizeof(unsigned long long));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_data, h_data, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, h_idx, N*sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(float));
    X5_gather_u64<<<(N+255)/256, 256>>>(d_data, d_idx, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        int j = (int)(h_idx[i] % 1024ULL);
        if (h_out[i] != h_data[j]) { printf("FAIL at %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_data); cudaFree(d_idx); cudaFree(d_out);
    return ok ? 0 : 1;
}
