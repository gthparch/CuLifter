#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

__device__ int d_arr[256];

extern "C" __global__ void CB4c_device_array(const int* in, int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d_arr[i] = in[i] * 3;
        out[i] = d_arr[i];
    }
}

int main(int argc, char** argv) {
    constexpr int N = 256;
    int h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = i + 1;
    memset(h_out, 0, sizeof(h_out));
    int *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(int)); cudaMalloc(&d_out, N*sizeof(int));
    cudaMemcpy(d_in, h_in, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(int));
    CB4c_device_array<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(int), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(int), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (h_out[i] != h_in[i]*3) { printf("FAIL at %d\n", i); ok=false; break; }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out); return ok ? 0 : 1;
}
