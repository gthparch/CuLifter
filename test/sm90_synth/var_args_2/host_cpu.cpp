#include <cstdio>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void kern_args_2();
int main(){
    constexpr int N=256;
    float in0[N],out[N]; memset(out,0,sizeof(out));
    for(int i=0;i<N;i++) in0[i]=(float)(i%3+1)*0.1f;
    float s0=1.0f;
    launchKernel(kern_args_2,1,256,in0,out,s0,N);
    FILE*f=fopen("ref_output.bin","rb");
    if(!f){bool ok=!std::isnan(out[0])&&out[0]!=0;printf(ok?"TEST PASSED\n":"FAIL\n");return ok?0:1;}
    float ref[N];fread(ref,4,N,f);fclose(f);
    bool ok=true;
    for(int i=0;i<N;i++){if(std::fabs(out[i]-ref[i])>1e-4f){printf("FAIL at %d: got %f ref %f\n",i,out[i],ref[i]);ok=false;break;}}
    if(ok) printf("TEST PASSED\n");
    return ok?0:1;
}
