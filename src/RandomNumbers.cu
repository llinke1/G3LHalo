#include "RandomNumbers.h"

/**
 * Initialization of random numbers
 * @param seed Seed for random number generation
 * @param states Pointer to state of the random number generator
 */
__global__ void init_rand(unsigned int seed, curandState *states)
  {
  //Global ID of this thread
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  //Initialize states with seed
  curand_init(seed, thread_index, 0, &states[thread_index]);
}

/**
 * Generation of random numbers
 * @param states Pointer to states of random number generator
 * @param index Index of random number that is to be generated
 * @param min Minimum of random number to be generated
 * @param max Maximum of random number to be generated
 * @retval random number
 */
__device__ double gen_rand(curandState *states, int index, double min, double max)
  {
  //Read out state
  curandState localState = states[index];

  //Generate random number
  double random=curand_uniform(&localState)*(max-min)+min;

  //Set state to new state
  states[index]=localState;

  return random;
}
