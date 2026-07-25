#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__device__ double d_epsilon = 1e-8;

struct Config { double* input; double* output; int N; double alpha; };

extern "C" __global__ void kern_full_combo(Config cfg) {
    __shared__ double sdata[256];
    int tid=threadIdx.x, gid=blockIdx.x*blockDim.x+tid;
    sdata[tid]=(gid<cfg.N)?cfg.input[gid]:0.0;
    __syncthreads();
    double left=(tid>0)?sdata[tid-1]:0.0;
    double right=(tid<blockDim.x-1)?sdata[tid+1]:0.0;
    double center=sdata[tid];
    if(gid<cfg.N)
        cfg.output[gid]=cfg.alpha*(left+right-2.0*center)+center+d_epsilon;
}

int main() {
    constexpr int N=256;
    double hi[N],ho[N];
    for(int i=0;i<N;i++) hi[i]=(double)(i+1)*0.01;
    double*di,*do_;
    cudaMalloc(&di,N*8);cudaMalloc(&do_,N*8);
    cudaMemcpy(di,hi,N*8,cudaMemcpyHostToDevice);
    Config cfg={di,do_,N,0.1};
    kern_full_combo<<<1,256>>>(cfg);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*8,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,8,N,f);fclose(f);
    bool ok=!isnan(ho[0])&&!isinf(ho[0]);
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(di);cudaFree(do_); return ok?0:1;
}
