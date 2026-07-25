#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__device__ float d_scale = 2.0f;  // goes to c[0x4]

extern "C" __global__ void kern_c4_shared_reduce(const float* in, float* out, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x*blockDim.x+tid;
    sdata[tid] = (gid<N) ? in[gid] * d_scale : 0.0f;  // c[4] read + shared store
    __syncthreads();
    for (int s=128;s>0;s>>=1) {
        if (tid<s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    if (tid==0) out[blockIdx.x] = sdata[0];
}

int main() {
    constexpr int N=256;
    float hi[N],ho[1];
    for(int i=0;i<N;i++) hi[i]=1.0f;
    float*di,*do_;
    cudaMalloc(&di,N*4);cudaMalloc(&do_,4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    kern_c4_shared_reduce<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,1,f);fclose(f);
    bool ok=fabsf(ho[0]-512.0f)<1.0f; // 256*1.0*2.0=512
    printf(ok?"TEST PASSED\n":"TEST FAILED (got %f)\n",ho[0]);
    cudaFree(di);cudaFree(do_); return ok?0:1;
}
