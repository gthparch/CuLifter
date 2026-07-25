#include <cstdint>
#include <cstdio>
#include "kernel_wrapper.h"
extern "C" void kern_montprod();
extern "C" void kern_mr32_eff();

// ---- CPU reference (same code as sprp32.h) ----
static inline uint32_t ref_mont_prod32(const uint32_t a, const uint32_t b,
                                       const uint32_t n, const uint32_t npi)
{
  const uint64_t t = (uint64_t)a*b;
  const uint32_t m = (uint32_t)((uint32_t)t*npi);
  const uint32_t u = (t + (uint64_t)m*n) >> 32;
  if (u < (t >> 32))
    return (uint32_t)(u-n);
  return u >= n ? (uint32_t)(u-n) : u;
}
static inline uint32_t ref_modular_inverse32(const uint32_t a)
{
  static const unsigned char mask[128] = {255,85,51,73,199,93,59,17,15,229,195,89,215,237,203,33,31,117,83,105,231,125,91,49,47,5,227,121,247,13,235,65,63,149,115,137,7,157,123,81,79,37,3,153,23,45,11,97,95,181,147,169,39,189,155,113,111,69,35,185,55,77,43,129,127,213,179,201,71,221,187,145,143,101,67,217,87,109,75,161,159,245,211,233,103,253,219,177,175,133,99,249,119,141,107,193,191,21,243,9,135,29,251,209,207,165,131,25,151,173,139,225,223,53,19,41,167,61,27,241,239,197,163,57,183,205,171,1};
  uint32_t ret = mask[(a >> 1) & 127];
  ret *= 2 + a * ret;
  ret *= 2 + a * ret;
  return ret;
}

int main() {
    constexpr int N = 16;
    // n odd; a,b < n.  Last cases push t + m*n past 2^64 (overflow-fix path).
    uint32_t A[N] = {1, 2, 1,          5,     1234567,    1,          61,         12345,
                     0xFFFFFFF0u, 0x80000000u, 3, 99999, 0xFFFFFFF9u, 2, 0x7FFFFFFEu, 0xFFFFFFF9u};
    uint32_t B[N] = {1, 2, 2,          6,     7654321,    0xFFFFFFF9u-1, 61,      54321,
                     0xFFFFFFF0u, 0x80000001u, 3, 88888, 0xFFFFFFF9u, 2, 0x7FFFFFFDu, 1};
    uint32_t M[N] = {3, 3, 0xFFFFFFFBu, 7,    0xFFFFFFFBu, 0xFFFFFFF9u, 67,      65537,
                     0xFFFFFFFBu, 0xFFFFFFFBu, 5, 100003, 0xFFFFFFFBu, 3, 0x7FFFFFFFu, 0xFFFFFFFBu};
    uint32_t* out_p = new uint32_t[1024]();
    uint32_t* out_i = new uint32_t[1024]();
    uint32_t* a = new uint32_t[1024](); uint32_t* b = new uint32_t[1024](); uint32_t* n = new uint32_t[1024]();
    uint32_t* npi = new uint32_t[1024]();
    for (int i=0;i<N;i++){a[i]=A[i];b[i]=B[i];n[i]=M[i];npi[i]=ref_modular_inverse32(M[i]);}
    launchKernel(kern_montprod, 1, N, ArgAt{0x210}, out_p, ArgAt{0x218}, out_i,
                 ArgAt{0x220}, a, ArgAt{0x228}, b, ArgAt{0x230}, n,
                 ArgAt{0x238}, npi, ArgAt{0x240}, N);
    int bad=0;
    for (int i=0;i<N;i++){
        uint32_t enpi = ref_modular_inverse32(M[i]);
        uint32_t eprod = ref_mont_prod32(A[i], B[i], M[i], enpi);
        const char* mki = out_i[i]==enpi?"ok":"NPI_MISMATCH";
        const char* mkp = out_p[i]==eprod?"ok":"PROD_MISMATCH";
        if (out_i[i]!=enpi || out_p[i]!=eprod) bad++;
        printf("%2d: n=%10u npi lifted %10u native %10u %s | mont(%u,%u) lifted %10u native %10u %s\n",
               i, M[i], out_i[i], enpi, mki, A[i], B[i], out_p[i], eprod, mkp);
    }
    printf("bad=%d\n", bad);

    // ---- full mr32_eff: count SPRP passes over n = 3,5,...,65 with bases {2,7,61} ----
    constexpr int MRN = 32;
    uint32_t* bs = new uint32_t[1024]();
    uint32_t* nn = new uint32_t[1024]();
    int* vv = new int[1024]();
    bs[0]=2; bs[1]=7; bs[2]=61;
    for (int i=0;i<MRN;i++) nn[i] = 2u*i + 3u;
    launchKernel(kern_mr32_eff, 1, 256, ArgAt{0x210}, bs, ArgAt{0x218}, nn,
                 ArgAt{0x220}, vv, ArgAt{0x228}, MRN);
    int primes = 0;
    for (int i=0;i<MRN;i++){
        unsigned x = nn[i]; bool p = x >= 2;
        for (unsigned dd=2; dd*dd<=x; dd++) if (x%dd==0){p=false;break;}
        if (p) primes++;
    }
    int expect = 3*primes;
    printf("mr32_eff: val lifted %d, native-expected %d (%d primes among 3..%u)  %s\n",
           vv[0], expect, primes, 2u*(MRN-1)+3u, vv[0]==expect?"ok":"MR_MISMATCH");
    if (vv[0]!=expect) bad++;
    printf("overall bad=%d\n", bad);
    return bad?1:0;
}
