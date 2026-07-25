// x86 driver for kern_shm_extent — verifies the wrapper's shared-memory
// sizing + overflow guard:
//   1. 192KB (old array size) and 224KB (valid on sm_90, silently corrupted
//      the old 192KB wrapper) must both compute the EXACT strided sums.
//   2. 256KB exceeds the sm_90 carveout: the PROT_NONE guard must convert
//      the overflow into the explicit SKIP_SHMEM verdict (child exit 4) —
//      never a silent garbage pass. Run in a fork so this process survives.
#include <cstdio>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>
#include "kernel_wrapper.h"

extern "C" void kern_shm_extent();

static bool run_case(int words, float* out) {
    float* in = new float[words];
    for (int i = 0; i < words; i++) in[i] = float(i % 8);
    std::memset(out, 0, 64 * sizeof(float));
    launchKernel(kern_shm_extent, 1, 64, out, in, words);
    delete[] in;
    return true;
}

int main() {
    float out[64];
    bool ok = true;
    for (int words : {49152, 57344}) {  // 192KB, 224KB
        run_case(words, out);
        for (int t = 0; t < 64 && ok; t++) {
            float want = float(words / 64) * float(t % 8);
            if (out[t] != want) {
                printf("words=%d t=%d got=%f want=%f\n", words, t, out[t], want);
                ok = false;
            }
        }
        printf("words=%d %s\n", words, ok ? "ok" : "MISMATCH");
    }

    // 256KB > 228KB carveout: expect the guard verdict in a child process.
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        run_case(65536, out);
        _exit(0);  // reaching here means the guard did NOT fire
    }
    int st = 0;
    waitpid(pid, &st, 0);
    bool guarded = WIFEXITED(st) && WEXITSTATUS(st) == 4;
    printf("words=65536 (256KB) guard: %s\n",
           guarded ? "SKIP_SHMEM as expected" : "MISSED (no SKIP_SHMEM exit)");
    ok = ok && guarded;

    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    return ok ? 0 : 1;
}
