#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void kern_shared_int_reduce(const int* in, int* out, int N) {
    __shared__ int sdata[256];
    int tid=threadIdx.x, gid=blockIdx.x*blockDim.x+tid;
    sdata[tid]=(gid<N)?in[gid]:0;
    __syncthreads();
    for(int s=128;s>0;s>>=1){if(tid<s)sdata[tid]+=sdata[tid+s];__syncthreads();}
    if(tid==0) out[blockIdx.x]=sdata[0];
}

int main() {
    constexpr int N=256;
    int hi[N],ho[1];
    for(int i=0;i<N;i++) hi[i]=i+1;
    int*di,*do_;
    cudaMalloc(&di,N*4);cudaMalloc(&do_,4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    kern_shared_int_reduce<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,1,f);fclose(f);
    int exp=256*257/2;
    bool ok=(ho[0]==exp);
    printf(ok?"TEST PASSED\n":"TEST FAILED (got %d exp %d)\n",ho[0],exp);
    cudaFree(di);cudaFree(do_); return ok?0:1;
}
