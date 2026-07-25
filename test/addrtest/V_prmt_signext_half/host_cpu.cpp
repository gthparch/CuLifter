#include <cstdint>
#include <iostream>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void V_prmt_signext_half();
int main() {
    constexpr int N=1024;
    int *h_in=new int[N],*h_lo=new int[N],*h_hi=new int[N];
    memset(h_lo,0,N*4); memset(h_hi,0,N*4);
    for(int i=0;i<N;i++) h_in[i]=((i*7-500)&0xFFFF)|((((i*13+300)&0xFFFF))<<16);
    launchKernel(V_prmt_signext_half,(N+255)/256,256,h_in,h_lo,h_hi,N);
    bool ok=true;
    for(int i=0;i<N;i++) {
        int exp_lo=(int)(short)(h_in[i]&0xFFFF);
        int exp_hi=(int)(short)((h_in[i]>>16)&0xFFFF);
        if(h_lo[i]!=exp_lo||h_hi[i]!=exp_hi){std::cout<<"FAIL "<<i<<std::endl;ok=false;break;}
    }
    if(ok) std::cout<<"TEST PASSED"<<std::endl;
    delete[] h_in;delete[] h_lo;delete[] h_hi;
    return ok?0:1;
}
