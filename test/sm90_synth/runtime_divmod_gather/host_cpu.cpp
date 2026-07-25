#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "kernel_wrapper.h"
extern "C" void kern_runtime_divmod_gather();
int main(){
    const int total=200, D=18, SRC=4096;
    std::vector<float> src(SRC); for(int i=0;i<SRC;i++)src[i]=(float)(i+1)*0.25f;
    std::vector<float> out(total,-999.f);
    int block=64,grid=(total+block-1)/block;
    launchKernel(kern_runtime_divmod_gather,grid,block,
        ArgAt{0x210},src.data(),ArgAt{0x218},out.data(),
        ArgAt{0x220},total,ArgAt{0x224},D,ArgAt{0x228},SRC);
    int mism=0,first=-1;
    for(int t=0;t<total;t++){
        int qo=t/D,rem=t%D,idx=(qo*D+rem)%SRC;
        float e=src[idx]+(float)qo*1000.0f+(float)rem;
        if(memcmp(&out[t],&e,4)){if(first<0)first=t;mism++;}
    }
    printf("runtime_divmod_gather: mism=%d/%d first=%d\n",mism,total,first);
    if(first>=0){int qo=first/D,rem=first%D;printf("first t=%d got=%g exp=%g (q=%d r=%d)\n",first,out[first],src[(qo*D+rem)%SRC]+qo*1000.0f+rem,qo,rem);}
    if(mism==0){printf("REF VERIFIED\nTEST PASSED\n");return 0;}
    printf("FAILED\n");return 1;
}
