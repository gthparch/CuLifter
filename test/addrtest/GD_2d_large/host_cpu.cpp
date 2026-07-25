#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void GD_2d_large();
int main() {
    constexpr int W=128,H=128,T=16384;
    float *h_in=new float[T],*h_out=new float[T];
    memset(h_out,0,T*4);
    for(int i=0;i<T;i++) h_in[i]=(float)((i*7+3)%1000)*0.01f;
    launchKernel2D(GD_2d_large,Dim3{(W+15)/16,(H+15)/16},Dim3{16,16},h_in,h_out,W,H);
    bool ok=true;
    for(int i=0;i<T;i++){float exp=h_in[i]*2.0f+1.0f;if(fabsf(h_out[i]-exp)>1e-4f){std::cout<<"FAIL "<<i<<std::endl;ok=false;break;}}
    if(ok) std::cout<<"TEST PASSED"<<std::endl;
    delete[]h_in;delete[]h_out;return ok?0:1;
}
