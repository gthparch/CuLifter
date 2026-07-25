#include <cstdint>
#include <iostream>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void V_prmt_byteswap();
int main() {
    constexpr int N=1024;
    uint32_t *h_in=new uint32_t[N],*h_out=new uint32_t[N];
    memset(h_out,0,N*4);
    for(int i=0;i<N;i++) h_in[i]=0x01020304+i;
    launchKernel(V_prmt_byteswap,(N+255)/256,256,h_in,h_out,N);
    bool ok=true;
    for(int i=0;i<N;i++){
        uint32_t x=h_in[i];
        uint32_t exp=((x>>24)&0xFF)|((x>>8)&0xFF00)|((x<<8)&0xFF0000)|((x<<24)&0xFF000000);
        if(h_out[i]!=exp){std::cout<<"FAIL "<<i<<std::endl;ok=false;break;}
    }
    if(ok) std::cout<<"TEST PASSED"<<std::endl;
    delete[] h_in;delete[] h_out;
    return ok?0:1;
}
