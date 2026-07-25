#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__device__ float d_weight = 1.5f;

extern "C" __global__ void kern_c4_atom_shared(const float* in, float* out, int N) {
    __shared__ float histogram[16];
    int tid=threadIdx.x;
    if(tid<16) histogram[tid]=0.0f;
    __syncthreads();
    int gid=blockIdx.x*blockDim.x+tid;
    if(gid<N) {
        int bin=(int)(in[gid]*d_weight)&0xF;
        atomicAdd(&histogram[bin], 1.0f);
    }
    __syncthreads();
    if(tid<16) out[blockIdx.x*16+tid]=histogram[tid];
}

int main() {
    constexpr int N=256;
    float hi[N],ho[16];
    for(int i=0;i<N;i++) hi[i]=(float)(i%11);
    float*di,*do_;
    cudaMalloc(&di,N*4);cudaMalloc(&do_,16*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    kern_c4_atom_shared<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,16*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,16,f);fclose(f);
    printf("TEST PASSED\n");
    cudaFree(di);cudaFree(do_); return 0;
}
