#include <cstdio>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
int main(){FILE*f=fopen("ref_output.bin","rb");if(!f){printf("no ref\n");return 1;}
float r;fread(&r,4,1,f);fclose(f);bool ok=fabsf(r-512.0f)<1.0f;
printf(ok?"REF VERIFIED\n":"REF MISMATCH\n");return ok?0:1;}
