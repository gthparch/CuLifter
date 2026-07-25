#include <cstdio>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
int main(){FILE*f=fopen("ref_output.bin","rb");if(!f){printf("no ref\n");return 1;}
double r[256];fread(r,8,256,f);fclose(f);bool ok=fabs(r[1]-2.0)<1e-10;
printf(ok?"REF VERIFIED\n":"REF MISMATCH\n");return ok?0:1;}
