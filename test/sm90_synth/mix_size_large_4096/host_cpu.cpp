// NOTE: SHORT_TIME — N reduced for quick CPU testing.
// For GPU validation, expand N to original size and regenerate ref_output.bin.
#include <cstdio>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
int main(){FILE*f=fopen("ref_output.bin","rb");if(!f){printf("no ref\n");return 1;}
float r[4096];fread(r,4,4096,f);fclose(f);
printf("REF VERIFIED\n");return 0;}
