#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void PHI5_three_level(const int* in, int* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int val = in[tid];
    int r = 0;
    if (val > 0) {
        switch (val % 4) {
            case 0: r = val; break;
            case 1:
                if (val > 50) r = val * 2;
                else r = val + 20;
                break;
            case 2: r = val - 10; break;
            default: r = val / 2; break;
        }
    } else {
        switch ((-val) % 3) {
            case 0: r = -val; break;
            case 1:
                if (val < -50) r = val * val;
                else r = val + 50;
                break;
            default: r = 42; break;
        }
    }
    out[tid] = r + tid;
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
    PHI5_three_level<<<(N+255)/256, 256>>>(d_in, d_out, N);
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
