#include "HOD.h"
#include "Params.h"
#include <iostream>

#if GPU
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

g3lhalo::HOD::HOD(Params *params_)
{
    params = params_;

#if GPU
    param_arr[0] = params->f;
    param_arr[1] = params->alpha;
    param_arr[2] = params->mmin;
    param_arr[3] = params->sigma;
    param_arr[4] = params->mprime;
    param_arr[5] = params->beta;

    CUDA_SAFE_CALL(cudaMalloc(&dev_params, 6 * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpy(dev_params, param_arr, 6 * sizeof(double), cudaMemcpyHostToDevice));

#endif
}

g3lhalo::HOD::~HOD()
{
#if GPU
    cudaFree(dev_params);
#endif
}

__host__ __device__ double g3lhalo::HOD::Nsat(double m, double *d_params)
{
#if __CUDA_ARCH__
    if (d_params == NULL)
    {
        printf("Problem in g3lhalo::HOD::Nsat\n");
        printf("Returning Nsat=0, but check this!\n");
        return 0;
    }

    double mth = d_params[2];
    double sigma = d_params[3];
    double mprime = d_params[4];
    double beta = d_params[5];
#else
    double mth = params->mmin;
    double sigma = params->sigma;
    double mprime = params->mprime;
    double beta = params->beta;
#endif
    return 0.5 * (1 + erf(log(m / mth) / sigma / 1.414213562)) * pow(m / mprime, beta);
}

__host__ __device__ double g3lhalo::HOD::Ncen(double m, double *d_params)
{
#if __CUDA_ARCH__
    if (d_params == NULL)
    {
        printf("Problem in g3lhalo::HOD::Ncen\n");
        printf("Returning Ncen=0, but check this!\n");
        return 0;
    }
    double alpha = d_params[1];
    double mth = d_params[2];
    double sigma = d_params[3]; 
#else
    double alpha = params->alpha;
    double mth = params->mmin;
    double sigma = params->sigma;
#endif
    return 0.5 * alpha * (1 + erf(log(m / mth) / sigma / 1.414213562));
}

__host__ __device__ double g3lhalo::NsatNsat(double m, HOD *hod1, HOD *hod2, double* d_params1, double* d_params2,
 double A, double epsilon, double scale1, double scale2)
{
    bool sameType = false;
    if (hod1 == hod2)
        sameType = true;

    double Ns1 = hod1->Nsat(m, d_params1);
    if (sameType)
        return (Ns1 + scale1 * scale1 - 1) * Ns1;

    double Ns2 = hod2->Nsat(m, d_params2);
    double result = Ns1 * Ns2 + A * pow(m, epsilon) * sqrt(Ns1 * Ns2);
    if (result < 0)
        return 0;
    return result;
}