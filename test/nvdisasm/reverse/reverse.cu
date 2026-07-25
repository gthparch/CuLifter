__global__ void reverse (int *d, const int len)
{
  __shared__ int s[256];
  int t = threadIdx.x;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[len-t-1];
}