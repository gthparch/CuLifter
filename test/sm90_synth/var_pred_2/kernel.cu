#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void kern_pred_2(const int* in, int* out, int N) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=N) return;
    bool p0=(in[tid+0*N]>0);
    bool p1=(in[tid+1*N]>0);
    int result=0;
    if(p0) result+=in[tid+0*N];
    if(p1) result+=in[tid+1*N];
    out[tid]=result;
}

int main() {
    constexpr int N=256;
    int hi[N*2],ho[N];
    for(int i=0;i<N*2;i++) hi[i]=(i%31)-15;
    int*di,*do_;
    cudaMalloc(&di,N*2*4);cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*2*4,cudaMemcpyHostToDevice);
    kern_pred_2<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    printf("TEST PASSED\n");
    cudaFree(di);cudaFree(do_);return 0;
}
