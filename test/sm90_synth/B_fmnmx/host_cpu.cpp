#include <cstdio>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
int main(){FILE*f=fopen("ref_output.bin","rb");if(!f){printf("no ref\n");return 1;}
float r[384];fread(r,4,384,f);fclose(f);
bool ok=(r[0]==-6.4f&&r[2]==0.0f);printf(ok?"REF VERIFIED\n":"REF MISMATCH\n");return ok?0:1;}
