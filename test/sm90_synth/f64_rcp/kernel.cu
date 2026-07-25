// Microbench: f64 reciprocal / division -> MUFU.RCP64H + Newton-Raphson.
// Probes the NaN cluster (lci/kalman/...): what does RCP64H read (upper half
// vs full f64) and what does the lifted x86 code compute?
extern "C" __global__ void f64_rcp(const double* x, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.0 / x[i];
}
