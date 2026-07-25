#include <cstdio>
#include <cmath>
// Exercise cos() (trig reduction slowpath) inside a non-inlined device fn,
// like fresnel's xFresnel_Sine_Integral large-x path.
__noinline__ __device__ double work(double x){
    double x2 = x*x;
    double f = 0.3 / (x + 1.0);   // forces __cuda_sm20_div
    double s = 0.5 - cos(x2)*f;
    return (x < 0.0) ? -s : s;
}
__global__ void kernel(const double* __restrict__ in, double* __restrict__ out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) out[i]=work(in[i]);
}
int main(){
    const int n=256; double *in,*out;
    cudaMallocManaged(&in,n*sizeof(double)); cudaMallocManaged(&out,n*sizeof(double));
    for(int i=0;i<n;i++) in[i]=(double)(i)*0.05;  // up to ~12
    kernel<<<(n+63)/64,64>>>(in,out,n); cudaDeviceSynchronize();
    int bad=0;
    for(int i=0;i<n;i++){ double x=in[i],x2=x*x,f=0.3/(x+1.0),s=0.5-cos(x2)*f; double r=(x<0.0)?-s:s;
        if(fabs(r-out[i])>1e-7){ if(bad<5)printf("MISMATCH i=%d x=%f ref=%.9e got=%.9e\n",i,x,r,out[i]); bad++; }
    }
    printf("%s\n", bad==0?"PASS":"FAIL"); return 0;
}
