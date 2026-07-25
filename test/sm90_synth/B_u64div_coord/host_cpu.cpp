#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "kernel_wrapper.h"
extern "C" void kern_coord_div();

int main() {
    constexpr int N = 16;
    constexpr int BUFSZ = 64;            // out[] capacity; coords are 1..16
    constexpr long long STRIDE = 1;      // out[coord*1] = out[coord]
    // Each coord = A[i]/B[i] is constructed to be a DISTINCT small index i+1:
    // A[i] = (i+1)*B[i] + R[i] with 0 <= R[i] < B[i].  The divisors B[i] span
    // small (3) to large (~1e12) magnitudes so the 64-bit reciprocal /
    // Newton-Raphson refinement is genuinely exercised, and the varied
    // remainders R[i] (0 .. B-1) stress the Granlund-Möller +1 correction.
    long long B[N] = {
        1000000007LL, 999999999937LL, 3LL, 2500000000LL,
        700000000LL, 50000017LL, 32000000LL, 1073741824LL,
        999999999989LL, 125000011LL, 4000037LL, 21000000LL,
        1000003LL, 300000007LL, 35000000LL, 4500000000LL };
    long long R[N] = {
        0, 1, 2, 7, 123456LL, 49999999LL, 0, 5,
        777LL, 100LL, 4000000LL, 20999999LL, 1000002LL, 0, 34999999LL,
        4499999999LL };
    long long A[N];
    for (int i = 0; i < N; i++) A[i] = (long long)(i + 1) * B[i] + R[i];
    long long* out = new long long[BUFSZ]();
    long long* a = new long long[1024]();
    long long* b = new long long[1024]();
    // Compute native coords, verify distinct + in-bounds (sanity on the test
    // vectors themselves, so a buggy lift can't be excused by a bad oracle).
    long long coord[N];
    bool seen[BUFSZ] = {false};
    bool setup_ok = true;
    for (int i = 0; i < N; i++) {
        a[i] = A[i]; b[i] = B[i];
        coord[i] = A[i] / B[i];
        if (coord[i] < 0 || coord[i] >= BUFSZ) { setup_ok = false; }
        else if (seen[coord[i]]) { setup_ok = false; }     // collision
        else seen[coord[i]] = true;
    }
    if (!setup_ok) {
        printf("SETUP ERROR: coords not distinct/in-bounds\n");
        for (int i = 0; i < N; i++)
            printf("  coord[%d]=%lld\n", i, coord[i]);
        return 2;
    }
    const long long SENT = -777;       // sentinel: untouched slot
    for (int j = 0; j < BUFSZ; j++) out[j] = SENT;

    launchKernel(kern_coord_div, 1, N, ArgAt{0x210}, out, ArgAt{0x218}, a,
                 ArgAt{0x220}, b, ArgAt{0x228}, STRIDE, ArgAt{0x230}, N);

    int bad = 0;
    for (int i = 0; i < N; i++) {
        long long got = out[coord[i] * STRIDE];
        bool ok = (got == A[i]);
        if (!ok) bad++;
        printf("%2d: %21lld / %21lld -> coord %5lld : out[coord]=%lld expected %lld  %s\n",
               i, A[i], B[i], coord[i], got, A[i], ok ? "ok" : "MISMATCH");
    }
    printf("bad=%d\n", bad);
    if (bad == 0) printf("TEST PASSED\n");
    return bad ? 1 : 0;
}
