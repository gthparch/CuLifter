extern "C" __global__ void GD_2d_large(const float* in, float* out, int W, int H) {
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    if(col<W&&row<H) { int c=row*W+col; out[c]=in[c]*2.0f+1.0f; }
}
