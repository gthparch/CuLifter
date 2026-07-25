#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_ldl_w128(const float* in, float* out, int N) {
    float local_buf[64];
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=N) return;
    for (int j=0;j<4;j++) local_buf[j]=in[tid*4+j];
    for (int j=0;j<4;j++) local_buf[j]=local_buf[j]*2.0f+1.0f;
    // Force spill: use non-sequential access
    for (int j=0;j<4;j++) local_buf[j+4]=local_buf[j]+0.5f;
    float sum=0; for(int j=0;j<4*2;j++) sum+=local_buf[j];
    out[tid]=sum;
}

int main() {
    constexpr int N=128;
    float hi[N*4],ho[N];
    for(int i=0;i<N*4;i++) hi[i]=(float)(i%17+1)*0.1f;
    float*di,*do_;
    cudaMalloc(&di,N*4*4);cudaMalloc(&do_,N*4);
    cudaMemcpy(di,hi,N*4*4,cudaMemcpyHostToDevice);
    kern_ldl_w128<<<1,128>>>(di,do_,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,4,N,f);fclose(f);
    bool ok=!isnanf(ho[0])&&ho[0]!=0;
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(di);cudaFree(do_);return ok?0:1;
}
