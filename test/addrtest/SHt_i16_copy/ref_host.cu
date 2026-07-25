#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void SHt_i16_copy(const short* in, short* out, int N) {
    __shared__ short s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) s[threadIdx.x] = in[i];
    __syncthreads();
    if (i < N) out[i] = s[threadIdx.x];
}

int main(int argc, char** argv) {
    constexpr int N = 1024, BS = 256;
    short h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (short)((i%41)-20);
    memset(h_out, 0, sizeof(h_out));
    short *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(short)); cudaMalloc(&d_out, N*sizeof(short));
    cudaMemcpy(d_in, h_in, N*sizeof(short), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(short));
    SHt_i16_copy<<<(N+BS-1)/BS, BS>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(short), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(short), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (h_out[i]!=h_in[i]) { printf("FAIL at %d\n",i); ok=false; break; }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
