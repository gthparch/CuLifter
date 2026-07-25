#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>
extern "C" __global__ void V_viadd_absdiff_u8(const unsigned char* a, const unsigned char* b,
    unsigned char* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int d = (int)a[i] - (int)b[i];
        out[i] = (unsigned char)(d < 0 ? -d : d);
    }
}
int main(int argc, char** argv) {
    constexpr int N=1024;
    unsigned char h_a[N],h_b[N],h_out[N];
    for(int i=0;i<N;i++){h_a[i]=(unsigned char)(i%256);h_b[i]=(unsigned char)((i*3+50)%256);}
    memset(h_out,0,sizeof(h_out));
    unsigned char *d_a,*d_b,*d_out;
    cudaMalloc(&d_a,N);cudaMalloc(&d_b,N);cudaMalloc(&d_out,N);
    cudaMemcpy(d_a,h_a,N,cudaMemcpyHostToDevice);cudaMemcpy(d_b,h_b,N,cudaMemcpyHostToDevice);
    cudaMemset(d_out,0,N);
    V_viadd_absdiff_u8<<<(N+255)/256,256>>>(d_a,d_b,d_out,N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out,d_out,N,cudaMemcpyDeviceToHost);
    if(argc>1&&strcmp(argv[1],"--dump")==0){FILE*f=fopen("ref_output.bin","wb");fwrite(h_out,1,N,f);fclose(f);}
    bool ok=true;
    for(int i=0;i<N;i++){unsigned char exp=(unsigned char)abs((int)h_a[i]-(int)h_b[i]);if(h_out[i]!=exp){printf("FAIL %d\n",i);ok=false;break;}}
    printf(ok?"TEST PASSED\n":"FAIL\n");
    cudaFree(d_a);cudaFree(d_b);cudaFree(d_out);
    return ok?0:1;
}
