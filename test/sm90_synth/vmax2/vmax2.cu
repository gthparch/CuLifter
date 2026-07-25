#include <cstdio>
#include <cuda_runtime.h>
// packed 2x16 signed max/min (__vmaxs2/__vmins2 -> SASS VIMNMX.S16) + add-min (__viaddmin)
__global__ void k(const unsigned* a, const unsigned* b, unsigned* o){
  unsigned x=a[0], y=b[0];
  o[0]=__vmaxs2(x,y);
  o[1]=__vmins2(x,y);
  o[2]=__vmaxs4(x,y);   // packed 4x8
  o[3]=__vmins4(x,y);
}
int main(){
  unsigned *a,*b,*o; cudaMallocManaged(&a,4);cudaMallocManaged(&b,4);cudaMallocManaged(&o,16);
  // lane-wise differs from scalar: hi16 a>b, lo16 a<b
  a[0]=0x0005FFFEu; b[0]=0x00020003u; // lanes a=(5,-2) b=(2,3) -> max=(5,3)=0x00050003 min=(2,-2)=0x0002FFFE
  k<<<1,1>>>(a,b,o); cudaDeviceSynchronize();
  printf("vmaxs2=%08x vmins2=%08x vmaxs4=%08x vmins4=%08x\n",o[0],o[1],o[2],o[3]);
  return 0;
}
