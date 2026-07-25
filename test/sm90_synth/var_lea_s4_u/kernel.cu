#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void kern_lea_s4_u(const int* in, int* out, int N) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=N) return;
    int idx=tid;
    int safe=idx%N;
    out[tid]=in[safe*4%N]+idx;
}

int main() {
    constexpr int N=256;
    int hi[N],ho[N];
    for(int i=0;i<N;i++) hi[i]=i*3;
    int*di,*do_;
    cudaMalloc(&di,N*4);cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    kern_lea_s4_u<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    printf("TEST PASSED\n");
    cudaFree(di);cudaFree(do_);return 0;
}
