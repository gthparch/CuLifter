#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void V_i2f_u8_f32();
int main() {
    constexpr int N=1024;
    unsigned char *h_in=new unsigned char[N]; float *h_out=new float[N];
    memset(h_out,0,N*sizeof(float));
    for(int i=0;i<N;i++) h_in[i]=(unsigned char)(i%256);
    launchKernel(V_i2f_u8_f32,(N+255)/256,256,(unsigned char*)h_in,h_out,N);
    bool ok=true;
    for(int i=0;i<N;i++){
        float exp=(float)((float)h_in[i]);
        if(fabsf(h_out[i]-exp)>0.5f){std::cout<<"FAIL "<<i<<std::endl;ok=false;break;}
    }
    if(ok) std::cout<<"TEST PASSED"<<std::endl;
    delete[] h_in;delete[] h_out;
    return ok?0:1;
}
