// nvlib_compare.h — shared tolerance comparison for nv_libraries CPU harnesses.
//
// Why tolerance: when a kernel's output derives from floating point, the lifted
// IR may round differently from a host reference by a few ULP (FFMA fusion,
// contract FMF, intrinsic vs libm). Bit-exact equality then gives spurious
// FAILs. Use:
//   cmp_u8_tol  — integer outputs computed via FP: allow +/- `tol` (default 1).
//   cmp_f32_tol — float  outputs: pass if |a-b| <= atol + rtol*|b|.
//   cmp_f64_tol — double outputs: same, f64.
// For pure data-movement / integer-only kernels use exact equality (tol=0).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cmath>

template<class T>
static int cmp_int_tol(const T* got, const T* ref, int n, long tol, const char* tag) {
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        long d = (long)got[i] - (long)ref[i]; if (d < 0) d = -d;
        if (d > tol) { if (bad < 8) printf("  %s[%d]: got %ld want %ld (|d|=%ld>tol %ld)\n",
                                           tag, i, (long)got[i], (long)ref[i], d, tol); ++bad; }
    }
    return bad;
}
static int cmp_u8_tol(const uint8_t* g, const uint8_t* r, int n, long tol=1, const char* tag="u8")
{ return cmp_int_tol<uint8_t>(g, r, n, tol, tag); }

static int cmp_f32_tol(const float* g, const float* r, int n,
                       float atol=1e-4f, float rtol=1e-4f, const char* tag="f32") {
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(g[i]-r[i]), thr = atol + rtol*fabsf(r[i]);
        if (!(d <= thr)) { if (bad<8) printf("  %s[%d]: got %.7g want %.7g (|d|=%.3g>thr %.3g)\n",
                                             tag,i,g[i],r[i],d,thr); ++bad; }
    }
    return bad;
}
static int cmp_f64_tol(const double* g, const double* r, int n,
                       double atol=1e-9, double rtol=1e-9, const char* tag="f64") {
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        double d = fabs(g[i]-r[i]), thr = atol + rtol*fabs(r[i]);
        if (!(d <= thr)) { if (bad<8) printf("  %s[%d]: got %.15g want %.15g (|d|=%.3g>thr %.3g)\n",
                                             tag,i,g[i],r[i],d,thr); ++bad; }
    }
    return bad;
}

static int report(int bad, int n) {
    printf("mismatches = %d / %d\n", bad, n);
    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL\n"); return 1;
}
