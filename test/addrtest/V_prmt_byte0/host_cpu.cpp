#include <cstdint>
#include <iostream>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void V_prmt_byte0();
int main() {
    constexpr int N=1024;
    int *h_in=new int[N],*h_out=new int[N];
    memset(h_out,0,N*4);
    for(int i=0;i<N;i++) h_in[i]=i*0x01010101;
    launchKernel(V_prmt_byte0,(N+255)/256,256,h_in,h_out,N);
    bool ok=true;
    for(int i=0;i<N;i++) if(h_out[i]!=(h_in[i]&0xFF)) {std::cout<<"FAIL "<<i<<std::endl;ok=false;break;}
    if(ok) std::cout<<"TEST PASSED"<<std::endl;
    delete[] h_in;delete[] h_out;
    return ok?0:1;
}
