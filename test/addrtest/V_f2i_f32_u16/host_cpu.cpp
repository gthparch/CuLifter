#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void V_f2i_f32_u16();
int main() {
    constexpr int N=1024;
    float *h_in=new float[N]; unsigned short *h_out=new unsigned short[N];
    memset(h_out,0,N*sizeof(unsigned short));
    for(int i=0;i<N;i++) h_in[i]=(float)(i%60000);
    launchKernel(V_f2i_f32_u16,(N+255)/256,256,(float*)h_in,h_out,N);
    bool ok=true;
    for(int i=0;i<N;i++){
        unsigned short exp=(unsigned short)((unsigned short)roundf(h_in[i]));
        if(h_out[i]!=exp){std::cout<<"FAIL "<<i<<std::endl;ok=false;break;}
    }
    if(ok) std::cout<<"TEST PASSED"<<std::endl;
    delete[] h_in;delete[] h_out;
    return ok?0:1;
}
