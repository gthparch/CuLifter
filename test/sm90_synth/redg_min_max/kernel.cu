#include <cstdio>
#include <climits>
#include <cuda_runtime.h>

extern "C" __global__ void kern_redg_minmax(const int* data, int* mins, int* maxs, int N) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=N) return;
    int bin = tid & 3;
    atomicMin(&mins[bin], data[tid]);
    atomicMax(&maxs[bin], data[tid]);
}

int main() {
    constexpr int N=256, BINS=4;
    int hd[N], hmin[BINS], hmax[BINS];
    for(int i=0;i<N;i++) hd[i]=(i*7+13)%100-50; // range [-50,49]
    for(int i=0;i<BINS;i++){hmin[i]=INT_MAX;hmax[i]=INT_MIN;}
    int*dd,*dm,*dx;
    cudaMalloc(&dd,N*4);cudaMalloc(&dm,BINS*4);cudaMalloc(&dx,BINS*4);
    cudaMemcpy(dd,hd,N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dm,hmin,BINS*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dx,hmax,BINS*4,cudaMemcpyHostToDevice);
    kern_redg_minmax<<<1,256>>>(dd,dm,dx,N);
    cudaDeviceSynchronize();
    cudaMemcpy(hmin,dm,BINS*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(hmax,dx,BINS*4,cudaMemcpyDeviceToHost);
    int out[BINS*2]; for(int i=0;i<BINS;i++){out[i]=hmin[i];out[BINS+i]=hmax[i];}
    FILE*f=fopen("ref_output.bin","wb");fwrite(out,4,BINS*2,f);fclose(f);
    printf("TEST PASSED\n");
    cudaFree(dd);cudaFree(dm);cudaFree(dx); return 0;
}
