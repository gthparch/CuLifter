#include <cstdio>
#include <cuda_runtime.h>
// Exercises F2F.F64.F32 (fpext) + F2F.F32.F64 (fptrunc) round-trips with f64
// arithmetic in between — d3q19-bgk's distinctive mixed-precision pattern
// (energy computed in f64 from f32 lattice data, lifted -> energy=0).
__global__ void k(const float* in, double* dacc, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float f = in[i];
  double d = (double)f;            // F2F.F64.F32
  double e = d * d + 1.5 * d - 0.25;
  e = e / (d + 2.0);               // f64 div (RCP64H)
  dacc[i] = e;                     // STG.E.64
  out[i] = (float)e;               // F2F.F32.F64
}
int main() {
  const int N = 4096; float *in, *out; double *dacc;
  cudaMallocManaged(&in, N*sizeof(float));
  cudaMallocManaged(&out, N*sizeof(float));
  cudaMallocManaged(&dacc, N*sizeof(double));
  for (int i=0;i<N;i++) in[i] = (float)((i%97)-48) * 0.3125f;
  k<<<(N+255)/256,256>>>(in, dacc, out, N);
  cudaDeviceSynchronize();
  double s=0; for (int i=0;i<N;i++) s += dacc[i] + out[i];
  printf("sum=%.6f first=%.6f %.6f last=%.6f %.6f\n", s, dacc[0], (double)out[0], dacc[N-1], (double)out[N-1]);
  return 0;
}
