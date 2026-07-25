#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void GD_3d_cube();
int main() {
    constexpr int X=16,Y=16,Z=16,T=4096;
    float *h_in=new float[T],*h_out=new float[T];
    memset(h_out,0,T*4);
    for(int i=0;i<T;i++) h_in[i]=(float)(i%100);
    launchKernel3D(GD_3d_cube,Dim3{(X+7)/8,(Y+7)/8,(Z+3)/4},Dim3{8,8,4},h_in,h_out,X,Y,Z);
    bool ok=true;
    for(int i=0;i<T;i++){float exp=h_in[i]+1.0f;if(fabsf(h_out[i]-exp)>1e-5f){std::cout<<"FAIL "<<i<<std::endl;ok=false;break;}}
    if(ok) std::cout<<"TEST PASSED"<<std::endl;
    delete[]h_in;delete[]h_out;return ok?0:1;
}
