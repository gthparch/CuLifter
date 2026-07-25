#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_dtype_double(const double* a, const double* b,
                                               double* c, int N) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=N) return;
    c[tid]=a[tid]+b[tid]+(double)tid;
}

int main() {
    constexpr int N=256;
    double ha[N],hb[N],hc[N];
    for(int i=0;i<N;i++){ha[i]=(double)(i+1);hb[i]=(double)(i%7+1);}
    double*da,*db,*dc;
    cudaMalloc(&da,N*8);cudaMalloc(&db,N*8);cudaMalloc(&dc,N*8);
    cudaMemcpy(da,ha,N*8,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,N*8,cudaMemcpyHostToDevice);
    kern_dtype_double<<<1,256>>>(da,db,dc,N);
    cudaDeviceSynchronize();
    cudaMemcpy(hc,dc,N*8,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(hc,8,N,f);fclose(f);
    printf("TEST PASSED\n");
    cudaFree(da);cudaFree(db);cudaFree(dc);return 0;
}
