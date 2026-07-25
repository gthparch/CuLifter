#include <stdio.h>
#include <cuda_runtime.h>

__global__ void loop1(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int sum = 0;
    #pragma unroll 1
    for (int i = idx; i < n; i += 1)
        sum += i;

    C[idx] = sum + A[idx] + B[idx];
}

int main() {
    int n = 1 << 20;
    size_t sz = n * sizeof(float);
    float *h_A = (float*)malloc(sz), *h_B = (float*)malloc(sz), *h_C = (float*)malloc(sz);
    for (int i = 0; i < n; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sz);
    cudaMalloc(&d_B, sz);
    cudaMalloc(&d_C, sz);
    cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    loop1<<<blocks, threads>>>(d_A, d_B, d_C, n);
    cudaMemcpy(h_C, d_C, sz, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++)
        printf("%f ", h_C[i]);
    printf("\n");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
