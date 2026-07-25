#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void DS_tiny32_dyn_smem();
int main() {
    constexpr int N=128,BS=32;
    float *h_in=new float[N],*h_out=new float[N];
    memset(h_out,0,N*4);
    for(int i=0;i<N;i++) h_in[i]=(float)(i+1);
    launchKernel(DS_tiny32_dyn_smem,(N+BS-1)/BS,BS,h_in,h_out,N);
    bool ok=true;
    for(int i=0;i<N;i++){float exp=h_in[i]*2.0f;if(fabsf(h_out[i]-exp)>1e-5f){std::cout<<"FAIL "<<i<<std::endl;ok=false;break;}}
    if(ok) std::cout<<"TEST PASSED"<<std::endl;
    delete[]h_in;delete[]h_out;return ok?0:1;
}
