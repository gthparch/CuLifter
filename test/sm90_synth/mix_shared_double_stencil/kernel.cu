#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_shared_double_stencil(const double* in, double* out, int N) {
    __shared__ double sdata[258]; // N+2 for halo
    int tid=threadIdx.x, gid=blockIdx.x*blockDim.x+tid;
    sdata[tid+1]=(gid<N)?in[gid]:0.0;
    if(tid==0) sdata[0]=(gid>0)?in[gid-1]:0.0;
    if(tid==blockDim.x-1||gid==N-1) sdata[tid+2]=(gid+1<N)?in[gid+1]:0.0;
    __syncthreads();
    if(gid<N) out[gid]=0.25*sdata[tid]+0.5*sdata[tid+1]+0.25*sdata[tid+2];
}

int main() {
    constexpr int N=256;
    double hi[N],ho[N];
    for(int i=0;i<N;i++) hi[i]=(double)(i+1);
    double*di,*do_;
    cudaMalloc(&di,N*8);cudaMalloc(&do_,N*8);
    cudaMemcpy(di,hi,N*8,cudaMemcpyHostToDevice);
    kern_shared_double_stencil<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*8,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,8,N,f);fclose(f);
    // out[1]=0.25*1+0.5*2+0.25*3=0.25+1.0+0.75=2.0
    bool ok=fabs(ho[1]-2.0)<1e-10;
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(di);cudaFree(do_); return ok?0:1;
}
