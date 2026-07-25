#include <cstdio>
#include <cmath>
__noinline__ __device__ double leaf(double x){ double y=x*2.0; if(x<0.0)y=-y; return y+0.5; }
__noinline__ __device__ double mid(double x){
    double a = leaf(x);          // nested cross-function call
    double b = leaf(x + 1.0);
    return a + b;                // return value flows through 2 inline levels
}
__global__ void kernel(const double* __restrict__ in, double* __restrict__ out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) out[i]=mid(in[i]);
}
int main(){
    const int n=256; double *in,*out;
    cudaMallocManaged(&in,n*sizeof(double)); cudaMallocManaged(&out,n*sizeof(double));
    for(int i=0;i<n;i++) in[i]=(double)(i-128)*0.25;
    kernel<<<(n+63)/64,64>>>(in,out,n); cudaDeviceSynchronize();
    int bad=0;
    for(int i=0;i<n;i++){ double x=in[i];
        double a=x*2.0; if(x<0.0)a=-a; a+=0.5;
        double x1=x+1.0,b=x1*2.0; if(x1<0.0)b=-b; b+=0.5;
        double r=a+b;
        if(fabs(r-out[i])>1e-9){ if(bad<5)printf("MISMATCH i=%d x=%f ref=%.9e got=%.9e\n",i,x,r,out[i]); bad++; }
    }
    printf("%s\n", bad==0?"PASS":"FAIL"); return 0;
}
