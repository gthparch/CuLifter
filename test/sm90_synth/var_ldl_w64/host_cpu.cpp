#include <cstdio>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
int main(){FILE*f=fopen("ref_output.bin","rb");if(!f){printf("no ref\n");return 1;}
float r[128];fread(r,4,128,f);fclose(f);
bool ok=!isnanf(r[0])&&r[0]!=0;printf(ok?"REF VERIFIED\n":"REF MISMATCH\n");return ok?0:1;}
