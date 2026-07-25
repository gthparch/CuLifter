#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void F3t_f2i_s16(const half* in, short* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = __half2short_rn(in[i]);
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    half h_in[N]; short h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = __float2half((float)((i%41)-20));
    memset(h_out, 0, sizeof(h_out));
    half *d_in; short *d_out;
    cudaMalloc(&d_in, N*sizeof(half)); cudaMalloc(&d_out, N*sizeof(short));
    cudaMemcpy(d_in, h_in, N*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(short));
    F3t_f2i_s16<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(short), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(short), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float fv = __half2float(h_in[i]);
        short exp = (short)roundf(fv);
        if (h_out[i] != exp) { printf("FAIL at %d: got %d exp %d\n",i,h_out[i],exp); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
