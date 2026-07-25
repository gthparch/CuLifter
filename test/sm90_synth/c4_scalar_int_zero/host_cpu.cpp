#include <cstdio>
#include <cstring>
#include "kernel_wrapper.h"
int main(){
    FILE*f=fopen("ref_output.bin","rb");if(!f){printf("no ref\n");return 1;}
    int r[256];fread(r,4,256,f);fclose(f);
    bool ok=(r[0]==0&&r[255]==255);
    printf(ok?"REF VERIFIED\n":"REF MISMATCH\n");return ok?0:1;
}
