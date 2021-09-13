#include <iostream>
#include <cuda.h>

// QUICK TEST TO SEE IF CUDAMALLOC AND CUDAMEMCPY WORK

int main()
{
  double* x;
  x=(double*)malloc(sizeof(double));
  x[0]=2.1;
  double* dev_x;

  cudaError_t err=cudaMalloc(&dev_x, sizeof(double));

  std::cout<<cudaGetErrorName(err)<<std::endl;

  err=cudaMemcpy(dev_x, x, sizeof(double), cudaMemcpyHostToDevice);

  std::cout<<cudaGetErrorName(err)<<std::endl;

  double* y;
  y=(double*) malloc(sizeof(double));
  err=cudaMemcpy(y, dev_x, sizeof(double), cudaMemcpyDeviceToHost);

  std::cout<<cudaGetErrorName(err)<<std::endl;

  std::cout<<y<<std::endl;

  return 0;
  
}
