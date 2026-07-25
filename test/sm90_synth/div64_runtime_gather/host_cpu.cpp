// x86 driver + self-contained oracle for the IntDivRecognize 64-bit-divide gap
// (reduced from 42_ampere_tensorop_group_conv im2col npq-decompose).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "kernel_wrapper.h"
extern "C" void kern_div64_runtime_gather();
int main(){
    const int total=200, D=18, SRC=4096;   // D non-pow2 -> magic divide
    std::vector<float> src(SRC), go(total,-9.f);
    for(int i=0;i<SRC;i++) src[i]=(float)(i+1)*0.25f;
    std::vector<int> q(total,-1), r(total,-1);
    int block=64,grid=(total+block-1)/block;
    launchKernel(kern_div64_runtime_gather,grid,block,
        ArgAt{0x210},src.data(),ArgAt{0x218},go.data(),
        ArgAt{0x220},q.data(),ArgAt{0x228},r.data(),
        ArgAt{0x230},total,ArgAt{0x234},D,ArgAt{0x238},SRC);
    int qbad=0,rbad=0,gbad=0,first=-1;
    for(int t=0;t<total;t++){
        int eq=t/D, er=t%D; float eg=src[t%SRC];
        if(q[t]!=eq){if(first<0)first=t;qbad++;}
        if(r[t]!=er){if(first<0)first=t;rbad++;}
        if(memcmp(&go[t],&eg,4)){if(first<0)first=t;gbad++;}
    }
    printf("=== div64_runtime_gather (IntDivRecognize 64-bit-divide gap; 42 conv repro) ===\n");
    printf("total=%d D=%d (non-pow2) SRC=%d\n",total,D,SRC);
    printf("quotient bad=%d/%d  remainder bad=%d/%d  gather bad=%d/%d  first_bad=%d\n",
           qbad,total,rbad,total,gbad,total,first);
    if(first>=0){
        printf("first t=%d: got q=%d r=%d  expected q=%d r=%d\n",first,q[first],r[first],first/D,first%D);
        printf("q got:");for(int i=0;i<24;i++)printf(" %d",q[i]);printf("\n");
        printf("q exp:");for(int i=0;i<24;i++)printf(" %d",i/D);printf("\n");
    }
    if(qbad==0&&rbad==0&&gbad==0){printf("REF VERIFIED (64-bit div/mod exact)\nTEST PASSED\n");return 0;}
    printf("FAILED\n");return 1;
}
