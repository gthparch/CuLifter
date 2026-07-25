#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void MU_softmax();
int main() {
    constexpr int N = 1024;
    float *h_in = new float[N], *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)((i%7)-3) * 0.5f;  // [-1.5, 1.5]
    launchKernel(MU_softmax, (N+255)/256, 256, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        int base = (i/8)*8, end = base+8; if (end>N) end=N;
        float maxv = h_in[base];
        for (int j=base+1;j<end;j++) maxv=fmaxf(maxv,h_in[j]);
        float sum=0; for (int j=base;j<end;j++) sum+=expf(h_in[j]-maxv);
        float exp_val = expf(h_in[i]-maxv)/sum;
        if (fabsf(h_out[i]-exp_val) > 1e-4f) {
            std::cout << "FAIL at " << i << ": got " << h_out[i] << " exp " << exp_val << std::endl;
            ok=false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
