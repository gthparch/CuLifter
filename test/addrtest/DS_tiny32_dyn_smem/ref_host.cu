#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
extern "C" __global__ void DS_tiny32_dyn_smem(const float* in, float* out, int N) {
    extern __shared__ float s[];
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N) s[threadIdx.x]=in[i];
    __syncthreads();
    if(i<N) out[i]=s[threadIdx.x]*2.0f;
}

int main(int argc, char** argv) {
    constexpr int N=128,BS=32;
    float *h_in=(float*)malloc(N*4),*h_out=(float*)malloc(N*4);
    for(int i=0;i<N;i++) h_in[i]=(float)(i+1);
    memset(h_out,0,N*4);
    float *d_in,*d_out;
    cudaMalloc(&d_in,N*4);cudaMalloc(&d_out,N*4);
    cudaMemcpy(d_in,h_in,N*4,cudaMemcpyHostToDevice);
    cudaMemset(d_out,0,N*4);
    DS_tiny32_dyn_smem<<<(N+BS-1)/BS,BS,32*sizeof(float)>>>(d_in,d_out,N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out,d_out,N*4,cudaMemcpyDeviceToHost);
    if(argc>1&&strcmp(argv[1],"--dump")==0){FILE*f=fopen("ref_output.bin","wb");fwrite(h_out,4,N,f);fclose(f);}
    bool ok=true;
    for(int i=0;i<N;i++){float exp=h_in[i]*2.0f;if(fabsf(h_out[i]-exp)>1e-5f){printf("FAIL %d\n",i);ok=false;break;}}
    printf(ok?"TEST PASSED\n":"FAIL\n");
    free(h_in);free(h_out);cudaFree(d_in);cudaFree(d_out);return ok?0:1;
}
