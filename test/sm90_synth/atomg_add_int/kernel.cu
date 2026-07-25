#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void kern_atomg_add_int(int* hist, const int* data, int N) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=N) return;
    atomicAdd(&hist[data[tid]&0x7], 1);
}

int main() {
    constexpr int N=256, BINS=8;
    int hd[N],hh[BINS]={};
    for(int i=0;i<N;i++) hd[i]=i%BINS;
    int*dd,*dh;
    cudaMalloc(&dd,N*4);cudaMalloc(&dh,BINS*4);
    cudaMemcpy(dd,hd,N*4,cudaMemcpyHostToDevice);
    cudaMemset(dh,0,BINS*4);
    kern_atomg_add_int<<<1,256>>>(dh,dd,N);
    cudaDeviceSynchronize();
    cudaMemcpy(hh,dh,BINS*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(hh,4,BINS,f);fclose(f);
    bool ok=(hh[0]==32);
    printf(ok?"TEST PASSED\n":"TEST FAILED (got %d)\n",hh[0]);
    cudaFree(dd);cudaFree(dh); return ok?0:1;
}
