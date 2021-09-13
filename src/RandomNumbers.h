/**
* Functions for Random Number Generation
*/

#ifndef RANDOMNUMBERS_CUH
#define RANDOMNUMBERS_CUH

//Cuda headers for random number generation
#include <curand.h>
#include <curand_kernel.h>

/**
 * Initialization of random numbers
 * @param seed Seed for random number generation
 * @param states Pointer to state of the random number generator
 */
__global__ void init_rand(unsigned int seed, curandState *states);

/**
 * Generation of random numbers
 * @param states Pointer to states of random number generator
 * @param index Index of random number that is to be generated
 * @param min Minimum of random number to be generated
 * @param max Maximum of random number to be generated
 * @retval random number
 */
__device__ double gen_rand(curandState *states, int index, double min, double max);

#endif //RANDOMNUMBERS_CUH
