#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 32

__global__ void kern_tensorT_min(const double* in, double* out, int R, int C, int stride) {
    __shared__ double tile[TILE*TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x*TILE+tx, gy = blockIdx.y*TILE+ty;
    if (gx<C && gy<R) tile[ty*TILE+tx] = in[gy*stride+gx]; else tile[ty*TILE+tx]=0;
    __syncthreads();
    int ox = blockIdx.y*TILE+tx, oy = blockIdx.x*TILE+ty;
    if (ox<R && oy<C) out[oy*R+ox] = tile[tx*TILE+ty];
}

int main() {
    constexpr int R=64,C=64,S=C;
    double *hi=new double[R*C], *ho=new double[C*R];
    for(int i=0;i<R*C;i++) hi[i]=(double)i;
    double*di,*do_; cudaMalloc(&di,R*C*8); cudaMalloc(&do_,C*R*8);
    cudaMemcpy(di,hi,R*C*8,cudaMemcpyHostToDevice);
    dim3 b(TILE,TILE),g((C+TILE-1)/TILE,(R+TILE-1)/TILE);
    kern_tensorT_min<<<g,b>>>(di,do_,R,C,S);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,C*R*8,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,8,C*R,f);fclose(f);
    bool ok=true;
    for(int i=0;i<R&&ok;i++)for(int j=0;j<C&&ok;j++)
        if(fabs(ho[j*R+i]-hi[i*C+j])>1e-10) ok=false;
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    delete[]hi; delete[]ho; cudaFree(di); cudaFree(do_); return ok?0:1;
}
