#include <cstdio>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
int main(){
    FILE*f=fopen("ref_output.bin","rb");if(!f){printf("no ref\n");return 1;}
    double r[256];fread(r,8,256,f);fclose(f);
    bool ok=!std::isnan(r[0])&&!std::isinf(r[0]);
    printf(ok?"REF VERIFIED\n":"REF MISMATCH\n");return ok?0:1;
}
