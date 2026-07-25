#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_arg_mixed(
    const float* a, const float* b, const float* c,
    float* d, float* e,
    int N, float alpha, float beta
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    d[tid] = a[tid] * alpha + b[tid] * beta + c[tid];
    e[tid] = d[tid] * 0.5f;
}

int main() {
    constexpr int N = 256;
    float ha[N],hb[N],hc[N],hd[N],he[N];
    for(int i=0;i<N;i++){ha[i]=(float)(i+1);hb[i]=(float)(i%7+1);hc[i]=0.5f;}
    float *da,*db,*dc,*dd,*de;
    cudaMalloc(&da,N*4);cudaMalloc(&db,N*4);cudaMalloc(&dc,N*4);
    cudaMalloc(&dd,N*4);cudaMalloc(&de,N*4);
    cudaMemcpy(da,ha,N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dc,hc,N*4,cudaMemcpyHostToDevice);
    kern_arg_mixed<<<1,256>>>(da,db,dc,dd,de,N,2.0f,3.0f);
    cudaDeviceSynchronize();
    cudaMemcpy(hd,dd,N*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(he,de,N*4,cudaMemcpyDeviceToHost);
    float out[N*2]; memcpy(out,hd,N*4); memcpy(out+N,he,N*4);
    FILE*f=fopen("ref_output.bin","wb");fwrite(out,4,N*2,f);fclose(f);
    float exp_d = ha[0]*2.0f+hb[0]*3.0f+hc[0];
    bool ok = fabsf(hd[0]-exp_d)<1e-3f && fabsf(he[0]-exp_d*0.5f)<1e-3f;
    printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    cudaFree(da);cudaFree(db);cudaFree(dc);cudaFree(dd);cudaFree(de);
    return ok?0:1;
}
