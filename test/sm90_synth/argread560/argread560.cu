#include <cstdio>
#include <cuda_runtime.h>
// Reproduce degrid's img_dim: 6 args (4 ptr + 2 size_t) -> const_mem offsets
// 528..568. Read the 5th arg (size_t at offset 560) LATE (after a loop), like
// degrid reads img_dim deep in its body. Lifted reportedly reads it as 0.
__global__ void k(double* out, const double* a, size_t npts,
                  const double* b, size_t dimv, const double* c) {
  long i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < (long)npts) {
    double acc = 0;
    for (long j = 0; j < 4; j++) acc += a[i] + b[i] + c[i];  // touch other args/mem
    out[i] = (double)dimv + acc*0;   // isolate dimv (the 5th arg)
  }
}
int main(){
  const long N=4096; size_t dimv=8192, npts=N;
  double *out,*a,*b,*c;
  cudaMallocManaged(&out,N*8); cudaMallocManaged(&a,N*8); cudaMallocManaged(&b,N*8); cudaMallocManaged(&c,N*8);
  for(long i=0;i<N;i++){a[i]=1;b[i]=2;c[i]=3;out[i]=-1;}
  k<<<(N+255)/256,256>>>(out,a,npts,b,dimv,c); cudaDeviceSynchronize();
  int bad=0; for(long i=0;i<N;i++) if(out[i]!=(double)dimv) bad++;
  printf("dimv_read=%g bad=%d %s\n", out[0], bad, bad==0?"PASS":"FAIL");
  return 0;
}
