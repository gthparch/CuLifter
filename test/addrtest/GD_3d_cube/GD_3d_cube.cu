extern "C" __global__ void GD_3d_cube(const float* in, float* out, int X, int Y, int Z) {
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    int z=blockIdx.z*blockDim.z+threadIdx.z;
    if(x<X&&y<Y&&z<Z) { int c=(z*Y+y)*X+x; out[c]=in[c]+1.0f; }
}
