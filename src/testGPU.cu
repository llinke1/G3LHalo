#include <iostream>
#include <cuda.h>
#include "cuda_helpers.h"

// QUICK TEST TO SEE IF CUDAMALLOC AND CUDAMEMCPY WORK

int main()
{
#if GPU
  double* x;
  x=(double*)malloc(sizeof(double));
  x[0]=2.1;
  double* dev_x;

  CUDA_SAFE_CALL(cudaMalloc(&dev_x, sizeof(double)));
  std::cerr<<"Could allocate memory on device"<<std::endl;
  
  CUDA_SAFE_CALL(cudaMemcpy(dev_x, x, sizeof(double), cudaMemcpyHostToDevice));
  std::cerr<<"Could copy value to device"<<std::endl;
  
  double* y;
  y=(double*) malloc(sizeof(double));
  CUDA_SAFE_CALL(cudaMemcpy(y, dev_x, sizeof(double), cudaMemcpyDeviceToHost));
  std::cerr<<"Could copy value from device"<<std::endl;

  if(x[0]==y[0])
    {
      std::cerr<<"Value copied from device is the same as on host"<<std::endl;
    }
  else
    {
      std::cerr<<"Value copied from device is not the same as on host"<<std::endl;
    };
#endif
  return 0;
  
}
