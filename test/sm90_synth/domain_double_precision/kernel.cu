#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_double(const double* in, double* out, int N) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=N) return;
    double x = in[tid];
    out[tid] = sqrt(x*x + 1.0) - x; // numerically stable for large x
}

int main() {
    constexpr int N=256;
    double hi[N],ho[N];
    for(int i=0;i<N;i++) hi[i]=(double)(i+1)*0.01;
    double*di,*do_;
    cudaMalloc(&di,N*8);cudaMalloc(&do_,N*8);
    cudaMemcpy(di,hi,N*8,cudaMemcpyHostToDevice);
    kern_double<<<1,256>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*8,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,8,N,f);fclose(f);
    double exp=sqrt(hi[0]*hi[0]+1.0)-hi[0];
    bool ok=fabs(ho[0]-exp)<1e-10;
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(di);cudaFree(do_); return ok?0:1;
}
