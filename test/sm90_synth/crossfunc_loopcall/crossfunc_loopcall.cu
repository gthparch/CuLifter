#include <cstdio>
#include <cmath>
#include <float.h>
__noinline__ __device__ double aux(double x){ return cos(x) + 0.1/(x+2.0); }
// device fn with BOTH a convergence loop AND nested calls + early return,
// mirroring fresnel's xFresnel_Sine_Integral structure.
__noinline__ __device__ double work(double x){
    if (fabs(x) < 0.5) {
        // power-series-like loop (small-x path)
        double x2=x*x, x3=x*x2, x4=-x2*x2, xn=1,Sn=1,Sm1=0,term,fac=1,c=0.7978845608;
        int y=0;
        if (x==0.0) return 0.0;
        Sn/=3.0;
        while(fabs(Sn-Sm1)>DBL_EPSILON*fabs(Sm1)){Sm1=Sn;y++;fac*=(double)(y+y);fac*=(double)(y+y+1);xn*=x4;term=xn/fac;term/=(double)(y+y+y+y+3);Sn+=term;}
        return x3*c*Sn;
    }
    // large-x path with nested calls
    double f = aux(fabs(x));
    double g = aux(fabs(x)+0.3);
    double x2 = x*x;
    double s = 0.5 - cos(x2)*f - cos(x2)*g;
    return (x<0.0)?-s:s;
}
__global__ void kernel(const double* __restrict__ in, double* __restrict__ out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=work(in[i]);
}
int main(){
    const int n=256; double *in,*out;
    cudaMallocManaged(&in,n*sizeof(double)); cudaMallocManaged(&out,n*sizeof(double));
    for(int i=0;i<n;i++) in[i]=(double)(i)*0.01;  // covers both <0.5 and >0.5
    kernel<<<(n+63)/64,64>>>(in,out,n); cudaDeviceSynchronize();
    int bad=0;
    for(int i=0;i<n;i++){ double x=in[i],r;
        if(fabs(x)<0.5){ double x2=x*x,x3=x*x2,x4=-x2*x2,xn=1,Sn=1,Sm1=0,term,fac=1,c=0.7978845608; int y=0;
          if(x==0.0)r=0.0; else{Sn/=3.0; while(fabs(Sn-Sm1)>DBL_EPSILON*fabs(Sm1)){Sm1=Sn;y++;fac*=(double)(y+y);fac*=(double)(y+y+1);xn*=x4;term=xn/fac;term/=(double)(y+y+y+y+3);Sn+=term;} r=x3*c*Sn;}
        } else { double f=cos(fabs(x))+0.1/(fabs(x)+2.0),g=cos(fabs(x)+0.3)+0.1/(fabs(x)+0.3+2.0),x2=x*x,s=0.5-cos(x2)*f-cos(x2)*g; r=(x<0.0)?-s:s; }
        if(fabs(r-out[i])>1e-7){ if(bad<5)printf("MISMATCH i=%d x=%f ref=%.9e got=%.9e\n",i,x,r,out[i]); bad++; }
    }
    printf("%s\n", bad==0?"PASS":"FAIL"); return 0;
}
