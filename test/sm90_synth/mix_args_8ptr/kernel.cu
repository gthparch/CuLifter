#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_8ptr(
    const float* a, const float* b, const float* c, const float* d,
    float* e, float* f_, float* g, float* h, int N
) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=N) return;
    e[tid]=a[tid]+b[tid];
    f_[tid]=c[tid]-d[tid];
    g[tid]=a[tid]*c[tid];
    h[tid]=b[tid]+d[tid]+1.0f;
}

int main() {
    constexpr int N=256;
    float ha[N],hb[N],hc[N],hd[N],he[N],hf[N],hg[N],hh[N];
    for(int i=0;i<N;i++){ha[i]=(float)(i+1);hb[i]=(float)(i%7+1);
        hc[i]=(float)(i%13+1)*0.1f;hd[i]=(float)(i%5+1)*0.5f;}
    float*da,*db,*dc,*dd,*de,*df,*dg,*dh;
    cudaMalloc(&da,N*4);cudaMalloc(&db,N*4);cudaMalloc(&dc,N*4);cudaMalloc(&dd,N*4);
    cudaMalloc(&de,N*4);cudaMalloc(&df,N*4);cudaMalloc(&dg,N*4);cudaMalloc(&dh,N*4);
    cudaMemcpy(da,ha,N*4,cudaMemcpyHostToDevice);cudaMemcpy(db,hb,N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dc,hc,N*4,cudaMemcpyHostToDevice);cudaMemcpy(dd,hd,N*4,cudaMemcpyHostToDevice);
    kern_8ptr<<<1,256>>>(da,db,dc,dd,de,df,dg,dh,N);
    cudaDeviceSynchronize();
    cudaMemcpy(he,de,N*4,cudaMemcpyDeviceToHost);cudaMemcpy(hf,df,N*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(hg,dg,N*4,cudaMemcpyDeviceToHost);cudaMemcpy(hh,dh,N*4,cudaMemcpyDeviceToHost);
    float out[N*4];memcpy(out,he,N*4);memcpy(out+N,hf,N*4);
    memcpy(out+2*N,hg,N*4);memcpy(out+3*N,hh,N*4);
    FILE*f=fopen("ref_output.bin","wb");fwrite(out,4,N*4,f);fclose(f);
    bool ok=fabsf(he[0]-(ha[0]+hb[0]))<1e-3f;
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(da);cudaFree(db);cudaFree(dc);cudaFree(dd);
    cudaFree(de);cudaFree(df);cudaFree(dg);cudaFree(dh); return ok?0:1;
}
