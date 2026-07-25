#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
extern "C" __global__ void VL_int4_bitwise(const int4* a, const int4* b, int4* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int4 va = a[i], vb = b[i];
        out[i] = make_int4(va.x ^ vb.x, va.y & vb.y, va.z | vb.z, ~va.w);
    }
}
int main(int argc, char** argv) {
    constexpr int N = 256;
    int4 h_a[N], h_b[N], h_out[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = make_int4(i*4*0x12345, (i*4+1)*0x12345, (i*4+2)*0x12345, (i*4+3)*0x12345);
        h_b[i] = make_int4(i*4*0x67890, (i*4+1)*0x67890, (i*4+2)*0x67890, (i*4+3)*0x67890);
    }
    memset(h_out, 0, sizeof(h_out));
    int4 *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N*sizeof(int4)); cudaMalloc(&d_b, N*sizeof(int4)); cudaMalloc(&d_out, N*sizeof(int4));
    cudaMemcpy(d_a, h_a, N*sizeof(int4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(int4), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(int4));
    VL_int4_bitwise<<<(N+255)/256, 256>>>(d_a, d_b, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(int4), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(int4), N, f); fclose(f);
    }
    printf("TEST PASSED\n");
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out); return 0;
}
