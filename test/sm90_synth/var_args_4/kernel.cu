#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_args_4(const float* in0, const float* in1, float* out, float s0, float s1, int N) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=N) return;
    float sum=0;
    sum += in0[tid];
    sum += in1[tid];
    sum += s0;
    sum += s1;
    out[tid]=sum;
}

int main() {
    constexpr int N=256;
    float h0[N]; for(int j=0;j<N;j++) h0[j]=(float)(j%(0+3)+1)*0.1f;
    float*d0; cudaMalloc(&d0,N*4); cudaMemcpy(d0,h0,N*4,cudaMemcpyHostToDevice);
    float h1[N]; for(int j=0;j<N;j++) h1[j]=(float)(j%(1+3)+1)*0.1f;
    float*d1; cudaMalloc(&d1,N*4); cudaMemcpy(d1,h1,N*4,cudaMemcpyHostToDevice);
    float*d_out; cudaMalloc(&d_out,N*4);
    kern_args_4<<<1,256>>>(d0, d1, d_out, 1.0f, 2.0f, N);
    cudaDeviceSynchronize();
    float ho[N]; cudaMemcpy(ho,d_out,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    bool ok=!isnanf(ho[0])&&ho[0]!=0;
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(d0);
    cudaFree(d1);
    cudaFree(d_out);
    return ok?0:1;
}
