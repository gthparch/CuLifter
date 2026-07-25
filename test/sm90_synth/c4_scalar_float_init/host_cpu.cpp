#include <cstdio>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
int main(){
    FILE*f=fopen("ref_output.bin","rb");if(!f){printf("no ref\n");return 1;}
    float r[256];fread(r,4,256,f);fclose(f);
    bool ok=fabsf(r[0]-3.14159f)<0.01f;
    printf(ok?"REF VERIFIED\n":"REF MISMATCH (got %f)\n",r[0]);return ok?0:1;
}
