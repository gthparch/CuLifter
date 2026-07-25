#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__device__ int d_offset = 0;  // c[4] read for index offset

extern "C" __global__ void kern_c4_shared_transpose(const float* in, float* out,
                                                     int W, int H) {
    __shared__ float tile[16][17];
    int tx=threadIdx.x, ty=threadIdx.y;
    int gx=blockIdx.x*16+tx, gy=blockIdx.y*16+ty;
    if(gx<W&&gy<H) tile[ty][tx]=in[(gy+d_offset)*W+gx]; else tile[ty][tx]=0;
    __syncthreads();
    int ox=blockIdx.y*16+tx, oy=blockIdx.x*16+ty;
    if(ox<H&&oy<W) out[oy*H+ox]=tile[tx][ty];
}

int main() {
    constexpr int W=32,H=32,N=W*H;
    float hi[N],ho[N];
    for(int i=0;i<N;i++) hi[i]=(float)(i+1);
    float*di,*do_;
    cudaMalloc(&di,N*4);cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4,cudaMemcpyHostToDevice);
    dim3 b(16,16),g(2,2);
    kern_c4_shared_transpose<<<g,b>>>(di,do_,W,H);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    bool ok=true;
    for(int i=0;i<H&&ok;i++)for(int j=0;j<W&&ok;j++)
        if(fabsf(ho[j*H+i]-hi[i*W+j])>1e-5f) ok=false;
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(di);cudaFree(do_); return ok?0:1;
}
