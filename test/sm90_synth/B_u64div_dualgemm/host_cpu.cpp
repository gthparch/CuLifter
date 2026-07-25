#include <climits>
#include <cstdint>
#include <cstdio>
#include "kernel_wrapper.h"
extern "C" void kern_s64div();

int main() {
    constexpr int N = 24;
    // 64-bit dividends/divisors spanning: all four sign combos, |d|=1, large
    // 64-bit magnitudes, divisor > |dividend|, INT64 edges, and divisors that
    // exercise a non-trivial reciprocal (so the Newton-Raphson chain runs).
    long long A[N] = {
        1000000000007LL,  1000000000007LL, -1000000000007LL, -1000000000007LL,
        9223372036854775807LL, -9223372036854775807LL, 0LL, 123456789012345LL,
        -123456789012345LL, 98765432101234LL, 5LL, -5LL,
        7000000000000LL, 7000000000000LL, 4294967296LL, 4294967295LL,
        1LL << 40, -(1LL << 40), 123456789012345678LL, -123456789012345678LL,
        1000003LL, 999999999989LL, 9223372036854775806LL, -8000000000000000000LL };
    long long B[N] = {
        1000003LL,        -1000003LL,       1000003LL,        -1000003LL,
        3LL, 3LL, 7LL, 1000000007LL,
        1000000007LL, -65537LL, 1000000000000LL, -1000000000000LL,
        -3LL, 3LL, 65536LL, -65535LL,
        1LL << 20, 1LL << 20, 1000000000039LL, -1000000000039LL,
        1000003LL, 999999999989LL, 1LL, 7LL };
    long long* q = new long long[1024]();
    long long* r = new long long[1024]();
    long long* a = new long long[1024]();
    long long* b = new long long[1024]();
    for (int i = 0; i < N; i++) { a[i] = A[i]; b[i] = B[i]; }
    launchKernel(kern_s64div, 1, N, ArgAt{0x210}, q, ArgAt{0x218}, r,
                 ArgAt{0x220}, a, ArgAt{0x228}, b, ArgAt{0x230}, N);
    int bad = 0;
    for (int i = 0; i < N; i++) {
        long long eq = A[i] / B[i], er = A[i] % B[i];
        bool ok = (q[i] == eq) && (r[i] == er);
        if (!ok) bad++;
        printf("%2d: %21lld / %21lld = lifted (%lld, %lld), native (%lld, %lld)  %s\n",
               i, A[i], B[i], q[i], r[i], eq, er, ok ? "ok" : "MISMATCH");
    }
    printf("bad=%d\n", bad);
    if (bad == 0) printf("TEST PASSED\n");
    return bad ? 1 : 0;
}
