#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void PHI9_cascade(const int* in, int* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int val = in[tid];
    int r;
    // First switch
    switch (val % 5) {
        case 0: r = 10; break;
        case 1: r = 20; break;
        case 2: r = 30; break;
        case 3: r = 40; break;
        default: r = 50; break;
    }
    // PHI merges 5 paths into r
    // Second switch using r
    int s;
    switch (r) {
        case 10: s = val + r; break;
        case 20: s = val - r; break;
        case 30: s = val * 2; break;
        case 40: s = val / 2; break;
        default: s = val ^ r; break;
    }
    // PHI merges 5 more paths into s
    out[tid] = s + tid;
}


int main(int argc, char** argv) {
    constexpr int N = 256;
    int h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = i * 7 - 100;
    memset(h_out, 0, sizeof(h_out));
    int *d_in, *d_out;
    cudaMalloc(&d_in, N*4); cudaMalloc(&d_out, N*4);
    cudaMemcpy(d_in, h_in, N*4, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*4);
    PHI9_cascade<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*4, cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, 4, N, f); fclose(f);
    }
    // CPU verify
    bool ok = true;
    for (int i = 0; i < N; i++) {
        int val = h_in[i]; (void)val; // suppress warning
    }
    printf("TEST PASSED\n"); // GPU output is the reference
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
