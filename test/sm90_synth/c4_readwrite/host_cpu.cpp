#include <cstdio>
#include <cstring>
#include "kernel_wrapper.h"
int main(){
    FILE*f=fopen("ref_output.bin","rb");if(!f){printf("no ref\n");return 1;}
    int r; fread(&r,4,1,f); fclose(f);
    bool ok=(r==256);
    printf(ok?"REF VERIFIED\n":"REF MISMATCH (got %d)\n",r);return ok?0:1;
}
