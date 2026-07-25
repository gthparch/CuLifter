#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_dtype_float(const float* a, const float* b,
                                               float* c, int N) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=N) return;
    c[tid]=a[tid]+b[tid]+(float)tid;
}

int main() {
    constexpr int N=256;
    float ha[N],hb[N],hc[N];
    for(int i=0;i<N;i++){ha[i]=(float)(i+1);hb[i]=(float)(i%7+1);}
    float*da,*db,*dc;
    cudaMalloc(&da,N*4);cudaMalloc(&db,N*4);cudaMalloc(&dc,N*4);
    cudaMemcpy(da,ha,N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,N*4,cudaMemcpyHostToDevice);
    kern_dtype_float<<<1,256>>>(da,db,dc,N);
    cudaDeviceSynchronize();
    cudaMemcpy(hc,dc,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(hc,4,N,f);fclose(f);
    printf("TEST PASSED\n");
    cudaFree(da);cudaFree(db);cudaFree(dc);return 0;
}
