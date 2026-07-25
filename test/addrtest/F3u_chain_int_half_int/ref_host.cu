#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void F3u_chain_int_half_int(const int* a, const int* b, const int* c,
    int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        half ha = __int2half_rn(a[i]);
        half hb = __int2half_rn(b[i]);
        half hc = __int2half_rn(c[i]);
        half hr = __hfma(ha, hb, hc);
        out[i] = __half2int_rn(hr);
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    int h_a[N], h_b[N], h_c[N], h_out[N];
    for (int i = 0; i < N; i++) { h_a[i]=(i%5)+1; h_b[i]=(i%3)+1; h_c[i]=(i%7); }
    memset(h_out, 0, sizeof(h_out));
    int *d_a, *d_b, *d_c, *d_out;
    cudaMalloc(&d_a, N*sizeof(int)); cudaMalloc(&d_b, N*sizeof(int));
    cudaMalloc(&d_c, N*sizeof(int)); cudaMalloc(&d_out, N*sizeof(int));
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(int));
    F3u_chain_int_half_int<<<(N+255)/256, 256>>>(d_a, d_b, d_c, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(int), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(int), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float ha = __half2float(__int2half_rn(h_a[i]));
        float hb = __half2float(__int2half_rn(h_b[i]));
        float hc = __half2float(__int2half_rn(h_c[i]));
        int exp = (int)roundf(__half2float(__float2half(ha*hb+hc)));
        if (h_out[i] != exp) { printf("FAIL at %d: got %d exp %d\n",i,h_out[i],exp); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_out);
    return ok ? 0 : 1;
}
