#include <cstdio>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
int main(){
    FILE*f=fopen("ref_output.bin","rb");if(!f){printf("no ref\n");return 1;}
    float r[256];fread(r,4,256,f);fclose(f);
    bool ok=fabsf(r[0]-1.5f)<1e-5f;
    printf(ok?"REF VERIFIED\n":"REF MISMATCH\n");return ok?0:1;
}
