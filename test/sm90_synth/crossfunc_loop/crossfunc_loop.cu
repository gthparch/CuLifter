#include <cstdio>
#include <cmath>
#include <float.h>
// Mimic Power_Series_S: a convergence loop in a non-inlined device func.
__noinline__ __device__ double series(double x) {
    double x2 = x * x;
    double x3 = x * x2;
    double x4 = -x2 * x2;
    double xn = 1.0, Sn = 1.0, Sm1 = 0.0, term, factorial = 1.0;
    double c = 7.978845608028653558798921198687637369517e-1;
    int y = 0;
    if (x == 0.0) return 0.0;
    Sn /= 3.0;
    while (fabs(Sn - Sm1) > DBL_EPSILON * fabs(Sm1)) {
        Sm1 = Sn; y += 1;
        factorial *= (double)(y + y);
        factorial *= (double)(y + y + 1);
        xn *= x4;
        term = xn / factorial;
        term /= (double)(y + y + y + y + 3);
        Sn += term;
    }
    return x3 * c * Sn;
}
__global__ void kernel(const double* __restrict__ in, double* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = series(in[i]);
}
int main() {
    const int n = 256; double *in, *out;
    cudaMallocManaged(&in, n*sizeof(double)); cudaMallocManaged(&out, n*sizeof(double));
    for (int i=0;i<n;i++) in[i] = (double)i * 0.001;  // small x, Power_Series regime
    kernel<<<(n+63)/64,64>>>(in,out,n); cudaDeviceSynchronize();
    int bad=0;
    // host recompute inline
    for(int i=0;i<n;i++){
        double x=in[i], x2=x*x, x3=x*x2, x4=-x2*x2, xn=1,Sn=1,Sm1=0,term,fac=1;
        double c=7.978845608028653558798921198687637369517e-1; int y=0;
        double r;
        if(x==0.0) r=0.0; else { Sn/=3.0;
          while(fabs(Sn-Sm1)>DBL_EPSILON*fabs(Sm1)){Sm1=Sn;y++;fac*=(double)(y+y);fac*=(double)(y+y+1);xn*=x4;term=xn/fac;term/=(double)(y+y+y+y+3);Sn+=term;}
          r=x3*c*Sn; }
        if(fabs(r-out[i])>1e-9){ if(bad<5) printf("MISMATCH i=%d x=%f ref=%.9e got=%.9e\n",i,x,r,out[i]); bad++; }
    }
    printf("%s\n", bad==0?"PASS":"FAIL"); return 0;
}
