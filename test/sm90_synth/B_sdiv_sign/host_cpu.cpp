#include <climits>
#include <cstdio>
#include "kernel_wrapper.h"
extern "C" void kern_sdiv_sign();
int main() {
    constexpr int N = 20;
    // All four sign combinations, |d|=1, divisor > |dividend|, INT_MIN/INT_MAX
    // edges, and the exact idivide-cuda H200 failure case (33664 / -1).
    int A[N] = {33664, 33664, -33664, -33664, 100, -100, 100, -100,
                7, -7, 0, 0, INT_MAX, INT_MIN + 1, INT_MAX, 5,
                1024, -1024, 123456789, -123456789};
    int B[N] = {-1, 1, 1, -1, 7, 7, -7, -7,
                100, 100, 5, -5, -1, -1, INT_MAX, INT_MIN + 1,
                -1024, 1024, -3, 3};
    int* q = new int[1024]();
    int* r = new int[1024]();
    int* a = new int[1024]();
    int* b = new int[1024]();
    for (int i = 0; i < N; i++) { a[i] = A[i]; b[i] = B[i]; }
    launchKernel(kern_sdiv_sign, 1, N, ArgAt{0x210}, q, ArgAt{0x218}, r,
                 ArgAt{0x220}, a, ArgAt{0x228}, b, ArgAt{0x230}, N);
    int bad = 0;
    for (int i = 0; i < N; i++) {
        int eq = A[i] / B[i], er = A[i] % B[i];
        bool ok = (q[i] == eq) && (r[i] == er);
        if (!ok) bad++;
        printf("%2d: %11d / %11d = lifted (%d, %d), native (%d, %d)  %s\n",
               i, A[i], B[i], q[i], r[i], eq, er, ok ? "ok" : "MISMATCH");
    }
    printf("bad=%d\n", bad);
    if (bad == 0) printf("TEST PASSED\n");
    return bad ? 1 : 0;
}
