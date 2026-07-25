#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
extern "C" __global__ void V_lop3_xnor_or(const unsigned* a, const unsigned* b,
    const unsigned* c, unsigned* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = (unsigned)((~(a[i] ^ b[i])) | c[i]);
}
int main(int argc, char** argv) {
    constexpr int N=1024;
    unsigned h_a[N],h_b[N],h_c[N],h_out[N];
    for(int i=0;i<N;i++){h_a[i]=(unsigned)(i*0x12345);h_b[i]=(unsigned)(i*0x67890);h_c[i]=(unsigned)(i*0xABCDE);}
    memset(h_out,0,sizeof(h_out));
    unsigned *d_a,*d_b,*d_c,*d_out;
    cudaMalloc(&d_a,N*4);cudaMalloc(&d_b,N*4);cudaMalloc(&d_c,N*4);cudaMalloc(&d_out,N*4);
    cudaMemcpy(d_a,h_a,N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,h_c,N*4,cudaMemcpyHostToDevice);
    cudaMemset(d_out,0,N*4);
    V_lop3_xnor_or<<<(N+255)/256,256>>>(d_a,d_b,d_c,d_out,N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out,d_out,N*4,cudaMemcpyDeviceToHost);
    if(argc>1&&strcmp(argv[1],"--dump")==0){FILE*f=fopen("ref_output.bin","wb");fwrite(h_out,4,N,f);fclose(f);}
    bool ok=true;
    for(int i=0;i<N;i++){unsigned exp=(unsigned)((~(h_a[i] ^ h_b[i])) | h_c[i]);if(h_out[i]!=exp){printf("FAIL %d\n",i);ok=false;break;}}
    printf(ok?"TEST PASSED\n":"FAIL\n");
    cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);cudaFree(d_out);
    return ok?0:1;
}
