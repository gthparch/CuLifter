#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define TILE_SIZE 256

__global__ void kern_tensorT_full(
    const int dim_input, const int dim_output, const int nblocks, const int tile_size,
    const int *shape_input, const int *shape_output,
    const float *shape_input_r, const float *shape_output_r,
    const int *stride_input, const int *stride_output_local, const int *stride_output_global,
    const double *input, double *output)
{
    __shared__ double tile[TILE_SIZE];
    for (int block_idx = blockIdx.x; block_idx < nblocks; block_idx += gridDim.x) {
        int it=block_idx, im=0, offset1=0;
        for(int i=0;i<dim_input;i++){
            im=(int)(it*shape_input_r[i]);
            offset1+=stride_input[i]*(it-im*shape_input[i]);
            it=im;
        }
        for(int i=threadIdx.x;i<tile_size;i+=blockDim.x)
            tile[i]=input[i+block_idx*tile_size];
        __syncthreads();
        for(int i=threadIdx.x;i<tile_size;i+=blockDim.x){
            it=i; int offset2=0, local_offset=0;
            for(int j=0;j<dim_output;j++){
                im=(int)(it*shape_output_r[j]);
                int tmp=it-im*shape_output[j];
                offset2+=stride_output_global[j]*tmp;
                local_offset+=stride_output_local[j]*tmp;
                it=im;
            }
            output[offset1+offset2]=tile[local_offset];
        }
        __syncthreads();
    }
}

int main() {
    constexpr int d1=4,d2=3,d3=2, d4=2,d5=3,d6=2;
    constexpr int N=d1*d2*d3*d4*d5*d6; // 288
    double *hi=new double[N], *ho=new double[N];
    for(int i=0;i<N;i++) hi[i]=(double)i;
    int h_si[]={d4,d5,d6}, h_so[]={d2,d3,d1};
    float h_sir[]={1.f/d4,1.f/d5,1.f/d6}, h_sor[]={1.f/d2,1.f/d3,1.f/d1};
    int h_stri[]={d2*d3,d2*d3*d4*d6*d1,d2*d3*d4};
    int h_strol[]={d1,d1*d2,1}, h_strog[]={1,d2,d2*d3*d4*d6};

    double*di,*do_; int*dsi,*dso,*dstri,*dstrol,*dstrog; float*dsir,*dsor;
    cudaMalloc(&di,N*8);cudaMalloc(&do_,N*8);
    cudaMalloc(&dsi,12);cudaMalloc(&dso,12);cudaMalloc(&dsir,12);cudaMalloc(&dsor,12);
    cudaMalloc(&dstri,12);cudaMalloc(&dstrol,12);cudaMalloc(&dstrog,12);
    cudaMemcpy(di,hi,N*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dsi,h_si,12,cudaMemcpyHostToDevice);
    cudaMemcpy(dso,h_so,12,cudaMemcpyHostToDevice);
    cudaMemcpy(dsir,h_sir,12,cudaMemcpyHostToDevice);
    cudaMemcpy(dsor,h_sor,12,cudaMemcpyHostToDevice);
    cudaMemcpy(dstri,h_stri,12,cudaMemcpyHostToDevice);
    cudaMemcpy(dstrol,h_strol,12,cudaMemcpyHostToDevice);
    cudaMemcpy(dstrog,h_strog,12,cudaMemcpyHostToDevice);
    int tile_size=d1*d2*d3, nblocks=d4*d5*d6;
    kern_tensorT_full<<<nblocks,256>>>(3,3,nblocks,tile_size,dsi,dso,dsir,dsor,dstri,dstrol,dstrog,di,do_);
    cudaDeviceSynchronize();
    cudaMemcpy(ho,do_,N*8,cudaMemcpyDeviceToHost);
    FILE*f=fopen("ref_output.bin","wb");fwrite(ho,8,N,f);fclose(f);
    printf("TEST PASSED\n");
    delete[]hi;delete[]ho;
    cudaFree(di);cudaFree(do_);cudaFree(dsi);cudaFree(dso);
    cudaFree(dsir);cudaFree(dsor);cudaFree(dstri);cudaFree(dstrol);cudaFree(dstrog);
    return 0;
}
