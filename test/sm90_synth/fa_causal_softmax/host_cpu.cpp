#include <cstdio>
#include <cstring>
#include <cmath>
#include "kernel_wrapper.h"
// Runs the LIFTED single-warp CAUSAL FA-forward kernel and verifies against a
// CPU causal-attention oracle. No GPU / ref bin needed.
extern "C" void fa_causal_softmax();

int main() {
    const int D = 16, SQ = 32, SK = 64;
    float Q[SQ * D], K[SK * D], Vv[SK * D], O[SQ * D];
    for (int i = 0; i < SQ * D; i++) Q[i]  = (float)(((i * 7 + 5) % 13) - 6) * 0.1f;
    for (int i = 0; i < SK * D; i++) K[i]  = (float)(((i * 5 + 3) % 11) - 5) * 0.1f;
    for (int i = 0; i < SK * D; i++) Vv[i] = (float)(((i * 3 + 1) % 9)  - 4) * 0.1f;
    memset(O, 0, sizeof(O));

    launchKernel(fa_causal_softmax, 1, 32, Q, K, Vv, O, SK);

    const float scale = 0.25f;
    bool ok = true;
    float maxerr = 0.0f;
    for (int i = 0; i < SQ && ok; i++) {
        float s[SK], mx = -1e30f;
        for (int j = 0; j < SK; j++) {
            if (j > i) { s[j] = -1e30f; continue; }   // causal mask
            float d = 0;
            for (int k = 0; k < D; k++) d += Q[i * D + k] * K[j * D + k];
            s[j] = d * scale;
            if (s[j] > mx) mx = s[j];
        }
        float den = 0;
        for (int j = 0; j < SK; j++) { s[j] = (j > i) ? 0.0f : expf(s[j] - mx); den += s[j]; }
        for (int k = 0; k < D; k++) {
            float o = 0;
            for (int j = 0; j < SK; j++) o += s[j] * Vv[j * D + k];
            o /= den;
            float e = fabsf(o - O[i * D + k]);
            if (e > maxerr) maxerr = e;
            if (e > 1e-3f) {
                printf("FAIL [%d][%d]: got %f exp %f\n", i, k, O[i * D + k], o);
                ok = false; break;
            }
        }
    }
    printf("max_err=%g\n", maxerr);
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    return ok ? 0 : 1;
}
