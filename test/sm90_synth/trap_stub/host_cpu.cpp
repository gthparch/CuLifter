#include <cstdio>
#include <cstring>
#include "kernel_wrapper.h"

// Only the cold-trap kernel is launched. kern_trap_stub (vprintf + BPT.TRAP
// arch-guard shape) is linked into the object but must never be driven; an
// execution harness must exclude trap-only entry points.
extern "C" void kern_trap_cold();

int main() {
    constexpr int N = 256;
    int h_in[N], h_out[N];
    for (int i = 0; i < N; i++) { h_in[i] = i; h_out[i] = 0; }

    launchKernel(kern_trap_cold, 1, 256, h_out, h_in, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != i + 1) {
            printf("FAIL at %d: got %d want %d\n", i, h_out[i], i + 1);
            ok = false;
            break;
        }
    }
    if (ok) printf("TEST PASSED\n");
    return ok ? 0 : 1;
}
