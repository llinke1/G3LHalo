#ifndef G3LHALO_CUDA_HELPERS_H
#define G3LHALO_CUDA_HELPERS_H

//For GPU Parallelisation, match this to maximum of computing GPU
#define THREADS 256 //Maximum Threads per Block
#define BLOCKS 184 //Maximum blocks for all SMs in GPU


// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif //G3LHALO_CUDA_HELPERS_H
