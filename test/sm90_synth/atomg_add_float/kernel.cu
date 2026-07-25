#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_atomg_add(float* hist, const int* data, int N) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=N) return;
    atomicAdd(&hist[data[tid]&0xF], 1.0f);
}

int main() {
    constexpr int N=256, BINS=16;
    int hd[N]; float hh[BINS]={};
    for(int i=0;i<N;i++) hd[i]=i%BINS;
    int*dd; float*dh;
    cudaMalloc(&dd,N*4);cudaMalloc(&dh,BINS*4);
    cudaMemcpy(dd,hd,N*4,cudaMemcpyHostToDevice);
    cudaMemset(dh,0,BINS*4);
    kern_atomg_add<<<1,256>>>(dh,dd,N);
    cudaDeviceSynchronize();
    cudaMemcpy(hh,dh,BINS*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(hh,4,BINS,f);fclose(f);
    bool ok=fabsf(hh[0]-16.0f)<1e-3f;
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(dd);cudaFree(dh); return ok?0:1;
}
