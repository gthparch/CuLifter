#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include "kernel_wrapper.h"
extern "C" void DM_idw_safe();
int main() {
    constexpr int N=64;
    float *h_x=new float[N],*h_y=new float[N],*h_v=new float[N],*h_out=new float[N];
    memset(h_out,0,N*4);
    for(int i=0;i<N;i++){h_x[i]=(float)(i%8)*1.5f;h_y[i]=(float)(i/8)*1.5f;h_v[i]=(float)(i*7%100);}
    launchKernel(DM_idw_safe,(N+63)/64,64,h_x,h_y,h_v,h_out,N);
    FILE*f=fopen("ref_output.bin","rb");
    if(!f){std::cerr<<"No ref"<<std::endl;return 1;}
    float ref[N];fread(ref,4,N,f);fclose(f);
    bool ok=true;
    for(int i=0;i<N;i++) if(fabsf(h_out[i]-ref[i])>fabsf(ref[i])*0.01f+1e-4f){std::cout<<"FAIL "<<i<<std::endl;ok=false;break;}
    if(ok) std::cout<<"TEST PASSED"<<std::endl;
    delete[]h_x;delete[]h_y;delete[]h_v;delete[]h_out;
    return ok?0:1;
}
