#include <cstdio>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
int main(){
    FILE*f=fopen("ref_output.bin","rb");if(!f){printf("no ref\n");return 1;}
    float r[16];fread(r,4,16,f);fclose(f);
    bool ok=fabsf(r[0]-16.0f)<1e-3f;
    printf(ok?"REF VERIFIED\n":"REF MISMATCH\n");return ok?0:1;
}
