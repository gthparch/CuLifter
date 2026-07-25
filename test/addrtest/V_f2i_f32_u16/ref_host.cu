#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
extern "C" __global__ void V_f2i_f32_u16(const float* in, unsigned short* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = (unsigned short)((unsigned short)roundf(in[i]));
}
int main(int argc, char** argv) {
    constexpr int N=1024;
    float h_in[N]; unsigned short h_out[N];
    for(int i=0;i<N;i++) h_in[i]=(float)(i%60000);
    memset(h_out,0,sizeof(h_out));
    float *d_in; unsigned short *d_out;
    cudaMalloc(&d_in,N*sizeof(float));cudaMalloc(&d_out,N*sizeof(unsigned short));
    cudaMemcpy(d_in,h_in,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(d_out,0,N*sizeof(unsigned short));
    V_f2i_f32_u16<<<(N+255)/256,256>>>(d_in,d_out,N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out,d_out,N*sizeof(unsigned short),cudaMemcpyDeviceToHost);
    if(argc>1&&strcmp(argv[1],"--dump")==0){FILE*f=fopen("ref_output.bin","wb");fwrite(h_out,sizeof(unsigned short),N,f);fclose(f);}
    bool ok=true;
    for(int i=0;i<N;i++){
        unsigned short exp=(unsigned short)((unsigned short)roundf(h_in[i]));
        if(h_out[i]!=exp){printf("FAIL %d\n",i);ok=false;break;}
    }
    printf(ok?"TEST PASSED\n":"FAIL\n");
    cudaFree(d_in);cudaFree(d_out);
    return ok?0:1;
}
