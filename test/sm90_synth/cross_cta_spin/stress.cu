// stress: N producer CTAs (1..N-1) each atomicAdd contrib into partials[0] then
// bump a counter; OWNER CTA0 spins until counter == N-1, then out[0]=partials[0]+base.
// Exercises MULTIPLE concurrent producers + a counting barrier (the real streamk
// shape: owner waits for ALL contributors, not just one).
extern "C" __global__ void streamk_stress(float* out, float* partials, int* done,
                                          float base, float contrib, int nprod) {
    int cta = blockIdx.x;
    if (cta == 0) {
        while (atomicAdd(&done[0], 0) < nprod) { }   // wait for ALL producers
        __threadfence();
        out[0] = partials[0] + base;
    } else {
        atomicAdd(&partials[0], contrib);             // deposit partial
        __threadfence();
        atomicAdd(&done[0], 1);                       // signal arrival
    }
}
