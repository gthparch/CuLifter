#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_local_256(const float* in, float* out, int N) {
    float buf[64];
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=N) return;
    for(int j=0;j<32;j++) buf[j]=in[(tid+j)%N];
    for(int j=0;j<32;j++) buf[j]*=2.0f;
    for(int j=32;j<64;j++) buf[j]=buf[j%32]+0.1f*(float)j;
    float sum=0; for(int j=0;j<64;j++) sum+=buf[j];
    out[tid]=sum;
}

int main() {
    constexpr int N=64;
    float hi[N],ho[N];
    for(int i=0;i<N;i++) hi[i]=(float)(i+1)*0.01f;
    float*di,*do_;
    cudaMalloc(&di,N*4);cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    kern_local_256<<<1,64>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    bool ok=!isnanf(ho[0])&&ho[0]!=0;
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(di);cudaFree(do_);return ok?0:1;
}
