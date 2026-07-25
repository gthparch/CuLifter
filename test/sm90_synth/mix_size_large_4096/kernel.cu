#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_large(const float* a, const float* b, float* c, float alpha, int N) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=N) return;
    c[tid]=a[tid]*alpha+b[tid];
}

int main() {
    constexpr int N=4096;
    float*ha=new float[N],*hb=new float[N],*hc=new float[N];
    for(int i=0;i<N;i++){ha[i]=(float)(i%31+1)*0.1f;hb[i]=(float)(i%17+1)*0.2f;}
    float*da,*db,*dc;
    cudaMalloc(&da,N*4);cudaMalloc(&db,N*4);cudaMalloc(&dc,N*4);
    cudaMemcpy(da,ha,N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,N*4,cudaMemcpyHostToDevice);
    kern_large<<<(N+255)/256,256>>>(da,db,dc,2.5f,N);
    cudaDeviceSynchronize();
    cudaMemcpy(hc,dc,N*4,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(hc,4,N,f);fclose(f);
    float exp=ha[0]*2.5f+hb[0];
    bool ok=fabsf(hc[0]-exp)<1e-3f;
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(da);cudaFree(db);cudaFree(dc);
    delete[]ha;delete[]hb;delete[]hc; return ok?0:1;
}
