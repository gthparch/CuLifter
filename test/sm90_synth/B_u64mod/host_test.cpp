#include <cstdint>
#include <cstdio>
#include "kernel_wrapper.h"
extern "C" void kern_u64mod();
int main() {
    constexpr int N = 16;
    uint32_t A[N] = {2,4,1,8,2,5,6,125,61,2,3,100,65535,123456789,7,2147483647u};
    uint32_t B[N] = {2,4,1,8,16,5,6,1,  61,1,3,100,65535,987654321,7,2};
    uint32_t M[N] = {7,7,7,7,7, 7,7,7,  7, 3,5,97, 65537,1000003,11,2147483647u};
    uint32_t* out = new uint32_t[1024]();
    uint32_t* a = new uint32_t[1024](); uint32_t* b = new uint32_t[1024](); uint32_t* n = new uint32_t[1024]();
    for (int i=0;i<N;i++){a[i]=A[i];b[i]=B[i];n[i]=M[i];}
    launchKernel(kern_u64mod, 1, N, ArgAt{0x210}, out, ArgAt{0x218}, a, ArgAt{0x220}, b, ArgAt{0x228}, n, ArgAt{0x230}, N);
    int bad=0;
    for (int i=0;i<N;i++){
        uint32_t exp = (uint32_t)(((uint64_t)A[i]*B[i]) % M[i]);
        const char* mk = out[i]==exp?"ok":"MISMATCH";
        if (out[i]!=exp) bad++;
        printf("%2d: %u*%u %% %u = lifted %u, native %u  %s\n", i, A[i],B[i],M[i], out[i], exp, mk);
    }
    printf("bad=%d\n", bad);
    return bad?1:0;
}
