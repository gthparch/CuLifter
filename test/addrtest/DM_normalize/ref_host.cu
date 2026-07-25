#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
extern "C" __global__ void DM_normalize(const float* vx, const float* vy, const float* vz,
    float* ox, float* oy, float* oz, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<N) {
        float len = sqrtf(vx[i]*vx[i]+vy[i]*vy[i]+vz[i]*vz[i]+1e-8f);
        float inv = 1.0f/len;
        ox[i]=vx[i]*inv; oy[i]=vy[i]*inv; oz[i]=vz[i]*inv;
    }
}

int main(int argc, char** argv) {
    constexpr int N=256;
    float hx[N],hy[N],hz[N],ox[N],oy[N],oz[N];
    for(int i=0;i<N;i++){hx[i]=(float)((i*3+1)%17)*0.3f;hy[i]=(float)((i*5+2)%13)*0.4f;hz[i]=(float)((i*7+3)%11)*0.5f;}
    memset(ox,0,sizeof(ox));memset(oy,0,sizeof(oy));memset(oz,0,sizeof(oz));
    float *dx,*dy,*dz,*dox,*doy,*doz;
    cudaMalloc(&dx,N*4);cudaMalloc(&dy,N*4);cudaMalloc(&dz,N*4);
    cudaMalloc(&dox,N*4);cudaMalloc(&doy,N*4);cudaMalloc(&doz,N*4);
    cudaMemcpy(dx,hx,N*4,cudaMemcpyHostToDevice);cudaMemcpy(dy,hy,N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dz,hz,N*4,cudaMemcpyHostToDevice);
    cudaMemset(dox,0,N*4);cudaMemset(doy,0,N*4);cudaMemset(doz,0,N*4);
    DM_normalize<<<(N+255)/256,256>>>(dx,dy,dz,dox,doy,doz,N);
    cudaDeviceSynchronize();
    cudaMemcpy(ox,dox,N*4,cudaMemcpyDeviceToHost);cudaMemcpy(oy,doy,N*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(oz,doz,N*4,cudaMemcpyDeviceToHost);
    if(argc>1&&strcmp(argv[1],"--dump")==0){FILE*f=fopen("ref_output.bin","wb");fwrite(ox,4,N,f);fwrite(oy,4,N,f);fwrite(oz,4,N,f);fclose(f);}
    bool ok=true;
    for(int i=0;i<N;i++){float l=sqrtf(ox[i]*ox[i]+oy[i]*oy[i]+oz[i]*oz[i]);if(fabsf(l-1.0f)>0.01f){printf("FAIL %d: len=%f\n",i,l);ok=false;break;}}
    printf(ok?"TEST PASSED\n":"FAIL\n");
    cudaFree(dx);cudaFree(dy);cudaFree(dz);cudaFree(dox);cudaFree(doy);cudaFree(doz);
    return ok?0:1;
}
