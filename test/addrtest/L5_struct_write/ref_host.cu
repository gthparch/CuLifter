#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

struct Vec3 { float x, y, z; };

extern "C" __global__ void L5_struct_write(const float* in, Vec3* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i].x = in[i] * 1.0f;
        out[i].y = in[i] * 2.0f;
        out[i].z = in[i] * 3.0f;
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    float *h_in = new float[N];
    Vec3 h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i % 7 + 1) * 0.1f;
    float *d_in; Vec3 *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(Vec3));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(Vec3));
    L5_struct_write<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(Vec3), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_out[i].x - h_in[i]*1.0f) > 1e-5f ||
            fabsf(h_out[i].y - h_in[i]*2.0f) > 1e-5f ||
            fabsf(h_out[i].z - h_in[i]*3.0f) > 1e-5f) {
            printf("FAIL at %d\n", i); ok = false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out); delete[] h_in;
    return ok ? 0 : 1;
}
