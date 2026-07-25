// MINIMAL reduction #2: runtime div/mod of a flat tid by a RUNTIME divisor
// (passed as arg, non-pow2), feeding a gather — exactly the conv kernel's
// npq_row=tid/CRS, crs_step=tid%CRS decompose. If THIS diverges, the bug is
// the magic-divide recognition for the uniform-divisor (I2F.U32.RP) variant.
extern "C" __global__ void kern_runtime_divmod_gather(
    const float* __restrict__ src,
    float* __restrict__ out,
    int total, int D, int SRC)      // D = runtime divisor (e.g. 18, non-pow2)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int qo = tid / D;     // quotient  (npq_row)
    int rem = tid % D;    // remainder (crs_step)
    int idx = (qo * D + rem) % SRC;   // == tid%SRC if div/mod correct
    out[tid] = src[idx] + (float)qo * 1000.0f + (float)rem;  // expose both q and r
}
