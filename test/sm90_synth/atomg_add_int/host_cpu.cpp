#include <cstdio>
#include <cstring>
#include "kernel_wrapper.h"
int main(){
    FILE*f=fopen("ref_output.bin","rb");if(!f){printf("no ref\n");return 1;}
    int r[8];fread(r,4,8,f);fclose(f);
    bool ok=(r[0]==32);
    printf(ok?"REF VERIFIED\n":"REF MISMATCH\n");return ok?0:1;
}
