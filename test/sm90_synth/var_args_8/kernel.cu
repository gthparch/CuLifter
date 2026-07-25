#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_args_8(const float* in0, const float* in1, const float* in2, const float* in3, float* out, float s0, float s1, float s2, float s3, int N) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=N) return;
    float sum=0;
    sum += in0[tid];
    sum += in1[tid];
    sum += in2[tid];
    sum += in3[tid];
    sum += s0;
    sum += s1;
    sum += s2;
    sum += s3;
    out[tid]=sum;
}

int main() {
    constexpr int N=256;
    float h0[N]; for(int j=0;j<N;j++) h0[j]=(float)(j%(0+3)+1)*0.1f;
    float*d0; cudaMalloc(&d0,N*4); cudaMemcpy(d0,h0,N*4,cudaMemcpyHostToDevice);
    float h1[N]; for(int j=0;j<N;j++) h1[j]=(float)(j%(1+3)+1)*0.1f;
    float*d1; cudaMalloc(&d1,N*4); cudaMemcpy(d1,h1,N*4,cudaMemcpyHostToDevice);
    float h2[N]; for(int j=0;j<N;j++) h2[j]=(float)(j%(2+3)+1)*0.1f;
    float*d2; cudaMalloc(&d2,N*4); cudaMemcpy(d2,h2,N*4,cudaMemcpyHostToDevice);
    float h3[N]; for(int j=0;j<N;j++) h3[j]=(float)(j%(3+3)+1)*0.1f;
    float*d3; cudaMalloc(&d3,N*4); cudaMemcpy(d3,h3,N*4,cudaMemcpyHostToDevice);
    float*d_out; cudaMalloc(&d_out,N*4);
    kern_args_8<<<1,256>>>(d0, d1, d2, d3, d_out, 1.0f, 2.0f, 3.0f, 4.0f, N);
    cudaDeviceSynchronize();
    float ho[N]; cudaMemcpy(ho,d_out,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    bool ok=!isnanf(ho[0])&&ho[0]!=0;
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(d0);
    cudaFree(d1);
    cudaFree(d2);
    cudaFree(d3);
    cudaFree(d_out);
    return ok?0:1;
}
