#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
extern "C" __global__ void GD_3d_cube(const float* in, float* out, int X, int Y, int Z) {
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    int z=blockIdx.z*blockDim.z+threadIdx.z;
    if(x<X&&y<Y&&z<Z) { int c=(z*Y+y)*X+x; out[c]=in[c]+1.0f; }
}

int main(int argc, char** argv) {
    constexpr int X=16,Y=16,Z=16,T=4096;
    float *h_in=(float*)malloc(T*4),*h_out=(float*)malloc(T*4);
    for(int i=0;i<T;i++) h_in[i]=(float)(i%100);
    memset(h_out,0,T*4);
    float *d_in,*d_out;
    cudaMalloc(&d_in,T*4);cudaMalloc(&d_out,T*4);
    cudaMemcpy(d_in,h_in,T*4,cudaMemcpyHostToDevice);
    cudaMemset(d_out,0,T*4);
    dim3 b(8,8,4),g((X+7)/8,(Y+7)/8,(Z+3)/4);
    GD_3d_cube<<<g,b>>>(d_in,d_out,X,Y,Z);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out,d_out,T*4,cudaMemcpyDeviceToHost);
    if(argc>1&&strcmp(argv[1],"--dump")==0){FILE*f=fopen("ref_output.bin","wb");fwrite(h_out,4,T,f);fclose(f);}
    bool ok=true;
    for(int i=0;i<T;i++){float exp=h_in[i]+1.0f;if(fabsf(h_out[i]-exp)>1e-5f){printf("FAIL %d\n",i);ok=false;break;}}
    printf(ok?"TEST PASSED\n":"FAIL\n");
    free(h_in);free(h_out);cudaFree(d_in);cudaFree(d_out);return ok?0:1;
}
