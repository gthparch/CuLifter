#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
extern "C" __global__ void IM_float_as_int(const float* in, int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int bits = __float_as_int(in[i]);
        out[i] = bits >> 23;
    }
}
int main(int argc, char** argv) {
    constexpr int N = 256;
    float h_in[N]; int h_out[N];
    for(int i=0;i<N;i++) h_in[i]=(float)(i+1);
    memset(h_out,0,sizeof(h_out));
    float *d_in; int *d_out;
    cudaMalloc(&d_in,N*4);cudaMalloc(&d_out,N*4);
    cudaMemcpy(d_in,h_in,N*4,cudaMemcpyHostToDevice);
    cudaMemset(d_out,0,N*4);
    IM_float_as_int<<<(N+255)/256,256>>>(d_in,d_out,N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out,d_out,N*4,cudaMemcpyDeviceToHost);
    if(argc>1&&strcmp(argv[1],"--dump")==0){FILE*f=fopen("ref_output.bin","wb");fwrite(h_out,4,N,f);fclose(f);}
    bool ok=true;
    for(int i=0;i<N;i++){int bits;memcpy(&bits,&h_in[i],4);int exp=bits>>23;
    if(h_out[i]!=exp){printf("FAIL %d\n",i);ok=false;break;}}
    printf(ok?"TEST PASSED\n":"FAIL\n");
    cudaFree(d_in);cudaFree(d_out);return ok?0:1;
}
