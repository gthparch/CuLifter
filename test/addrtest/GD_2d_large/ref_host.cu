#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
extern "C" __global__ void GD_2d_large(const float* in, float* out, int W, int H) {
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    if(col<W&&row<H) { int c=row*W+col; out[c]=in[c]*2.0f+1.0f; }
}

int main(int argc, char** argv) {
    constexpr int W=128,H=128,T=16384;
    float *h_in=(float*)malloc(T*4),*h_out=(float*)malloc(T*4);
    for(int i=0;i<T;i++) h_in[i]=(float)((i*7+3)%1000)*0.01f;
    memset(h_out,0,T*4);
    float *d_in,*d_out;
    cudaMalloc(&d_in,T*4);cudaMalloc(&d_out,T*4);
    cudaMemcpy(d_in,h_in,T*4,cudaMemcpyHostToDevice);
    cudaMemset(d_out,0,T*4);
    dim3 b(16,16),g((W+15)/16,(H+15)/16);
    GD_2d_large<<<g,b>>>(d_in,d_out,W,H);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out,d_out,T*4,cudaMemcpyDeviceToHost);
    if(argc>1&&strcmp(argv[1],"--dump")==0){FILE*f=fopen("ref_output.bin","wb");fwrite(h_out,4,T,f);fclose(f);}
    bool ok=true;
    for(int i=0;i<T;i++){float exp=h_in[i]*2.0f+1.0f;if(fabsf(h_out[i]-exp)>1e-4f){printf("FAIL %d\n",i);ok=false;break;}}
    printf(ok?"TEST PASSED\n":"FAIL\n");
    free(h_in);free(h_out);cudaFree(d_in);cudaFree(d_out);return ok?0:1;
}
