#include <cstdio>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void kern_args_4();
int main(){
    constexpr int N=256;
    float h0[N],h1[N],out[N]; memset(out,0,sizeof(out));
    for(int j=0;j<N;j++){h0[j]=(float)(j%(0+3)+1)*0.1f;h1[j]=(float)(j%(1+3)+1)*0.1f;}
    launchKernel(kern_args_4,1,256,h0,h1,out,1.0f,2.0f,N);
    FILE*f=fopen("ref_output.bin","rb");
    if(!f){bool ok=!std::isnan(out[0])&&out[0]!=0;printf(ok?"TEST PASSED\n":"FAIL\n");return ok?0:1;}
    float ref[N];fread(ref,4,N,f);fclose(f);
    bool ok=true;
    for(int i=0;i<N;i++){if(std::fabs(out[i]-ref[i])>1e-3f){printf("FAIL at %d: got %f ref %f\n",i,out[i],ref[i]);ok=false;break;}}
    if(ok) printf("TEST PASSED\n");return ok?0:1;}
