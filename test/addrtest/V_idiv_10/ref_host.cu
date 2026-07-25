#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
extern "C" __global__ void V_idiv_10(const int* in, int* out_q, int* out_r, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out_q[i] = in[i] / 10;
        out_r[i] = in[i] % 10;
    }
}
int main(int argc, char** argv) {
    constexpr int N=1024;
    int h_in[N],h_q[N],h_r[N];
    for(int i=0;i<N;i++) h_in[i]=i*17+3;
    memset(h_q,0,sizeof(h_q));memset(h_r,0,sizeof(h_r));
    int *d_in,*d_q,*d_r;
    cudaMalloc(&d_in,N*4);cudaMalloc(&d_q,N*4);cudaMalloc(&d_r,N*4);
    cudaMemcpy(d_in,h_in,N*4,cudaMemcpyHostToDevice);
    cudaMemset(d_q,0,N*4);cudaMemset(d_r,0,N*4);
    V_idiv_10<<<(N+255)/256,256>>>(d_in,d_q,d_r,N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_q,d_q,N*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_r,d_r,N*4,cudaMemcpyDeviceToHost);
    if(argc>1&&strcmp(argv[1],"--dump")==0){FILE*f=fopen("ref_output.bin","wb");fwrite(h_q,4,N,f);fwrite(h_r,4,N,f);fclose(f);}
    bool ok=true;
    for(int i=0;i<N;i++){if(h_q[i]!=h_in[i]/10||h_r[i]!=h_in[i]%10){printf("FAIL %d\n",i);ok=false;break;}}
    printf(ok?"TEST PASSED\n":"FAIL\n");
    cudaFree(d_in);cudaFree(d_q);cudaFree(d_r);
    return ok?0:1;
}
