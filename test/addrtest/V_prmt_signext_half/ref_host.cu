#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
extern "C" __global__ void V_prmt_signext_half(const int* in, int* out_lo, int* out_hi, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        short lo = (short)(in[i] & 0xFFFF);
        short hi = (short)((in[i] >> 16) & 0xFFFF);
        out_lo[i] = (int)lo;
        out_hi[i] = (int)hi;
    }
}
int main(int argc, char** argv) {
    constexpr int N=1024;
    int h_in[N],h_lo[N],h_hi[N];
    for(int i=0;i<N;i++) h_in[i]=((i*7-500)&0xFFFF)|((((i*13+300)&0xFFFF))<<16);
    memset(h_lo,0,sizeof(h_lo)); memset(h_hi,0,sizeof(h_hi));
    int *d_in,*d_lo,*d_hi;
    cudaMalloc(&d_in,N*4);cudaMalloc(&d_lo,N*4);cudaMalloc(&d_hi,N*4);
    cudaMemcpy(d_in,h_in,N*4,cudaMemcpyHostToDevice);
    cudaMemset(d_lo,0,N*4);cudaMemset(d_hi,0,N*4);
    V_prmt_signext_half<<<(N+255)/256,256>>>(d_in,d_lo,d_hi,N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_lo,d_lo,N*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hi,d_hi,N*4,cudaMemcpyDeviceToHost);
    if(argc>1&&strcmp(argv[1],"--dump")==0){FILE*f=fopen("ref_output.bin","wb");fwrite(h_lo,4,N,f);fwrite(h_hi,4,N,f);fclose(f);}
    bool ok=true;
    for(int i=0;i<N;i++){
        int el=(int)(short)(h_in[i]&0xFFFF);int eh=(int)(short)((h_in[i]>>16)&0xFFFF);
        if(h_lo[i]!=el||h_hi[i]!=eh){printf("FAIL %d\n",i);ok=false;break;}
    }
    printf(ok?"TEST PASSED\n":"FAIL\n");
    cudaFree(d_in);cudaFree(d_lo);cudaFree(d_hi);
    return ok?0:1;
}
