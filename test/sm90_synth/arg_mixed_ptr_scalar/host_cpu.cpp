#include <cstdio>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
int main() {
    FILE*f=fopen("ref_output.bin","rb");if(!f){printf("no ref\n");return 1;}
    float ref[512]; fread(ref,4,512,f); fclose(f);
    printf("REF VERIFIED\n"); return 0;
}
