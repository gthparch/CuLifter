#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__device__ float d_scale = 3.14159f;  // non-zero init → ELF .nv.constant4 has data

extern "C" __global__ void kern_c4_scalar_init(const float* in, float* out, int N) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=N) return;
    out[tid] = in[tid] * d_scale;
}

int main() {
    constexpr int N=256;
    float hi[N],ho[N];
    for(int i=0;i<N;i++) hi[i]=(float)(i+1);
    float*di,*do_;
    cudaMalloc(&di,N*4);cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    kern_c4_scalar_init<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    bool ok=fabsf(ho[0]-3.14159f)<0.01f;
    printf(ok?"TEST PASSED\n":"TEST FAILED (got %f, expected %f)\n",ho[0],3.14159f);
    cudaFree(di);cudaFree(do_); return ok?0:1;
}
