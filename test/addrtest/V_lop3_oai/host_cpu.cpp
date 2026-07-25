#include <cstdint>
#include <iostream>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void V_lop3_oai();
int main() {
    constexpr int N=1024;
    uint32_t *h_a=new uint32_t[N],*h_b=new uint32_t[N],*h_c=new uint32_t[N],*h_out=new uint32_t[N];
    memset(h_out,0,N*4);
    for(int i=0;i<N;i++){h_a[i]=(uint32_t)(i*0x12345);h_b[i]=(uint32_t)(i*0x67890);h_c[i]=(uint32_t)(i*0xABCDE);}
    launchKernel(V_lop3_oai,(N+255)/256,256,h_a,h_b,h_c,h_out,N);
    bool ok=true;
    for(int i=0;i<N;i++){
        uint32_t exp=(uint32_t)((h_a[i] | h_b[i]) & ~h_c[i]);
        if(h_out[i]!=exp){std::cout<<"FAIL "<<i<<std::endl;ok=false;break;}
    }
    if(ok) std::cout<<"TEST PASSED"<<std::endl;
    delete[] h_a;delete[] h_b;delete[] h_c;delete[] h_out;
    return ok?0:1;
}
