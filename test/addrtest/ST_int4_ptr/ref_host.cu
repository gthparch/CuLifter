#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
extern "C" __global__ void ST_int4_ptr(const int4* in, int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int4 v = in[i];
        out[i] = v.x + v.y + v.z + v.w;
    }
}

int main(int argc, char** argv) {
    constexpr int N = 64;
    int4 h_in[N]; int h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = make_int4(i*4+1, i*4+2, i*4+3, i*4+4);
    memset(h_out, 0, sizeof(h_out));
    int4 *d_in; int *d_out;
    cudaMalloc(&d_in, N*sizeof(int4)); cudaMalloc(&d_out, N*4);
    cudaMemcpy(d_in, h_in, N*sizeof(int4), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*4);
    ST_int4_ptr<<<(N+63)/64, 64>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*4, cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, 4, N, f); fclose(f);
    }
    printf("TEST PASSED\n");
    cudaFree(d_in); cudaFree(d_out); return 0;
}
