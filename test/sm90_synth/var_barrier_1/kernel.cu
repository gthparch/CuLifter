#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_bar_1(const float* in, float* out, int N) {
    __shared__ float sdata[256];
    int tid=threadIdx.x;
    sdata[tid]=(tid<N)?in[tid]:0;
    __syncthreads(); if(tid<N) sdata[tid]+=sdata[(tid+1)%256];
    __syncthreads();
    if(tid<N) out[tid]=sdata[tid];
}

int main() {
    constexpr int N=256;
    float hi[N],ho[N];
    for(int i=0;i<N;i++) hi[i]=1.0f;
    float*di,*do_;
    cudaMalloc(&di,N*4);cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    kern_bar_1<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    bool ok=!isnanf(ho[0]);
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(di);cudaFree(do_);return ok?0:1;
}
