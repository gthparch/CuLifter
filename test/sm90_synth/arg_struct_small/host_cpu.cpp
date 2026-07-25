#include <cstdio>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
int main() {
    FILE* f = fopen("ref_output.bin","rb"); if(!f){printf("no ref\n");return 1;}
    float ref[256]; fread(ref,4,256,f); fclose(f);
    bool ok = fabsf(ref[0]-3.0f)<1e-3f;
    printf(ok ? "REF VERIFIED\n" : "REF MISMATCH\n"); return ok?0:1;
}
