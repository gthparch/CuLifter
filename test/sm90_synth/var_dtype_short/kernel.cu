#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_dtype_short(const short* a, const short* b,
                                               short* c, int N) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=N) return;
    c[tid]=a[tid]+b[tid]+(short)tid;
}

int main() {
    constexpr int N=256;
    short ha[N],hb[N],hc[N];
    for(int i=0;i<N;i++){ha[i]=(short)(i+1);hb[i]=(short)(i%7+1);}
    short*da,*db,*dc;
    cudaMalloc(&da,N*2);cudaMalloc(&db,N*2);cudaMalloc(&dc,N*2);
    cudaMemcpy(da,ha,N*2,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,N*2,cudaMemcpyHostToDevice);
    kern_dtype_short<<<1,256>>>(da,db,dc,N);
    cudaDeviceSynchronize();
    cudaMemcpy(hc,dc,N*2,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(hc,2,N,f);fclose(f);
    printf("TEST PASSED\n");
    cudaFree(da);cudaFree(db);cudaFree(dc);return 0;
}
