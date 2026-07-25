#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_shared_copy(const float* in, float* out, int N) {
    __shared__ float buf[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x*blockDim.x+tid;
    if (gid<N) buf[tid]=in[gid];
    __syncthreads();
    if (gid<N) out[gid]=buf[tid]*2.0f;
}

int main() {
    constexpr int N=256;
    float hi[N],ho[N];
    for(int i=0;i<N;i++) hi[i]=(float)(i+1);
    float*di,*do_;
    cudaMalloc(&di,N*4);cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    kern_shared_copy<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    bool ok=fabsf(ho[0]-2.0f)<1e-5f;
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(di);cudaFree(do_); return ok?0:1;
}
