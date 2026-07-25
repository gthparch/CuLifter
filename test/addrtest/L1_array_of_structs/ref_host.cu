#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

struct MyStruct { float x; int y; float z; };

extern "C" __global__ void L1_array_of_structs(const MyStruct* arr, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = arr[i].x + (float)arr[i].y + arr[i].z;
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    MyStruct *h_arr = new MyStruct[N];
    float h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) { h_arr[i] = {(float)(i%7+1)*0.1f, i%5+1, (float)(i%3+1)*0.2f}; }
    MyStruct *d_arr; float *d_out;
    cudaMalloc(&d_arr, N * sizeof(MyStruct));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_arr, h_arr, N * sizeof(MyStruct), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(float));
    L1_array_of_structs<<<(N+255)/256, 256>>>(d_arr, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = h_arr[i].x + (float)h_arr[i].y + h_arr[i].z;
        if (fabsf(h_out[i] - expected) > 1e-5f) {
            printf("FAIL at %d\n", i); ok = false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_arr); cudaFree(d_out); delete[] h_arr;
    return ok ? 0 : 1;
}
