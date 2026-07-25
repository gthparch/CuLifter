#include <cstdint>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include "kernel_wrapper.h"
extern "C" void V_viadd_absdiff_u8();
int main() {
    constexpr int N=1024;
    uint8_t *h_a=new uint8_t[N],*h_b=new uint8_t[N],*h_out=new uint8_t[N];
    memset(h_out,0,N);
    for(int i=0;i<N;i++){h_a[i]=(uint8_t)(i%256);h_b[i]=(uint8_t)((i*3+50)%256);}
    launchKernel(V_viadd_absdiff_u8,(N+255)/256,256,h_a,h_b,h_out,N);
    bool ok=true;
    for(int i=0;i<N;i++){
        uint8_t exp=(uint8_t)abs((int)h_a[i]-(int)h_b[i]);
        if(h_out[i]!=exp){std::cout<<"FAIL "<<i<<std::endl;ok=false;break;}
    }
    if(ok) std::cout<<"TEST PASSED"<<std::endl;
    delete[] h_a;delete[] h_b;delete[] h_out;
    return ok?0:1;
}
