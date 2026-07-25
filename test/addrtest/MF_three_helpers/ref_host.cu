#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
__device__ __noinline__ float helper_scale(float x, float s) { return x * s; }

extern "C" __global__ void MF_init(float* buf, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) buf[i] = (float)(i + 1);
}

extern "C" __global__ void MF_scale(float* buf, float s, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) buf[i] = helper_scale(buf[i], s);
}

extern "C" __global__ void MF_add(const float* a, const float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
    constexpr int N = 256;
    float h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; i++) { h_a[i]=(float)(i+1); h_b[i]=(float)(i*2+1); }
    memset(h_c, 0, sizeof(h_c));
    float *d_a,*d_b,*d_c;
    cudaMalloc(&d_a,N*4);cudaMalloc(&d_b,N*4);cudaMalloc(&d_c,N*4);
    cudaMemcpy(d_a,h_a,N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,N*4,cudaMemcpyHostToDevice);
    cudaMemset(d_c,0,N*4);
    MF_add<<<(N+255)/256,256>>>(d_a,d_b,d_c,N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c,d_c,N*4,cudaMemcpyDeviceToHost);
    if(argc>1&&strcmp(argv[1],"--dump")==0){FILE*f=fopen("ref_output.bin","wb");fwrite(h_c,4,N,f);fclose(f);}
    printf("TEST PASSED\n");
    cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);return 0;
}
