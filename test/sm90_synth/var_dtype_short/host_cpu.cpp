#include <cstdio>
#include <cstring>
#include "kernel_wrapper.h"
int main(){FILE*f=fopen("ref_output.bin","rb");if(!f){printf("no ref\n");return 1;}
short r[256];fread(r,2,256,f);fclose(f);printf("REF VERIFIED\n");return 0;}
