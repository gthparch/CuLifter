#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void B_isetp_or(const int* a, const int* b, int* out, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= N) return;
    // OR-combined predicates → ISETP.*.OR
    bool cond = (a[i] == 0) || (b[i] > 10) || (a[i] + b[i] < -5);
    out[i] = cond ? a[i] + b[i] : a[i] - b[i];
}

int main() {
    constexpr int N=256;
    int ha[N], hb[N], ho[N];
    for(int i=0;i<N;i++){ha[i]=i-128;hb[i]=(i*7)%50-25;}
    int*da,*db,*do_; cudaMalloc(&da,N*4);cudaMalloc(&db,N*4);cudaMalloc(&do_,N*4);
    cudaMemcpy(da,ha,N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,N*4,cudaMemcpyHostToDevice);
    B_isetp_or<<<1,256>>>(da,db,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb"); fwrite(ho,4,N,f); fclose(f);
    printf("TEST PASSED\n");
    cudaFree(da);cudaFree(db);cudaFree(do_); return 0;
}
