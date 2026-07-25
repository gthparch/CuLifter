#include <cstdio>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
int main() { FILE*f=fopen("ref_output.bin","rb"); if(!f){printf("no ref\n");return 1;}
float r[1024]; fread(r,4,1024,f); fclose(f);
bool ok=(r[0]==2.0f && r[1]==4.0f && r[2]==6.0f && r[3]==8.0f);
printf(ok?"REF VERIFIED\n":"REF MISMATCH\n"); return ok?0:1; }
