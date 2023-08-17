#include "generalFunctions.h"
#include "HOD.h"

#include <iostream>

__host__ __device__ double g3lhalo::u_NFW(double k, double m, double z, double f,
                                          double zmin, double zmax, double mmin, double mmax,
                                          int Nbins, const double *rho_bar, const double *concentration)
{
  // Get indices of z and m
  int z_ix = int((z - zmin) * Nbins / (zmax - zmin));
  int m_ix = int(std::log(m / mmin) * Nbins / std::log(mmax / mmin));

  // Get concentration
  double c = f * concentration[z_ix * Nbins + m_ix];

  double arg1 = k * r_200(m, z, zmin, zmax, Nbins, rho_bar) / c;
  double arg2 = arg1 * (1 + c);

  double si1, ci1, si2, ci2;
  SiCi(arg1, si1, ci1);
  SiCi(arg2, si2, ci2);

  double term1 = sin(arg1) * (si2 - si1);
  double term2 = cos(arg1) * (ci2 - ci1);
  double term3 = -sin(arg1 * c) / arg2;
  double F = std::log(1. + c) - c / (1. + c);

  return (term1 + term2 + term3) / F;
}

__host__ __device__ double g3lhalo::r_200(double m, double z, double zmin, double zmax, int Nbins,
                                          const double *rho_bar)
{
  int z_ix = int((z - zmin) * Nbins / (zmax - zmin));
  return pow(0.239 * m / rho_bar[z_ix] / 200, 1. / 3.);
}

__host__ __device__ double g3lhalo::apertureFilter(double eta)
{
  return 0.5 * eta * eta * exp(-0.5 * eta * eta);
}

__host__ __device__ void g3lhalo::SiCi(double x, double &si, double &ci)
{
  double x2 = x * x;
  double x4 = x2 * x2;
  double x6 = x2 * x4;
  double x8 = x4 * x4;
  double x10 = x8 * x2;
  double x12 = x6 * x6;
  double x14 = x12 * x2;

  if (x < 4)
  {

    double a = 1 - 4.54393409816329991e-2 * x2 + 1.15457225751016682e-3 * x4 - 1.41018536821330254e-5 * x6 + 9.43280809438713025e-8 * x8 - 3.53201978997168357e-10 * x10 + 7.08240282274875911e-13 * x12 - 6.05338212010422477e-16 * x14;

    double b = 1 + 1.01162145739225565e-2 * x2 + 4.99175116169755106e-5 * x4 + 1.55654986308745614e-7 * x6 + 3.28067571055789734e-10 * x8 + 4.5049097575386581e-13 * x10 + 3.21107051193712168e-16 * x12;

    si = x * a / b;

    double gamma = 0.5772156649;
    a = -0.25 + 7.51851524438898291e-3 * x2 - 1.27528342240267686e-4 * x4 + 1.05297363846239184e-6 * x6 - 4.68889508144848019e-9 * x8 + 1.06480802891189243e-11 * x10 - 9.93728488857585407e-15 * x12;

    b = 1 + 1.1592605689110735e-2 * x2 + 6.72126800814254432e-5 * x4 + 2.55533277086129636e-7 * x6 + 6.97071295760958946e-10 * x8 + 1.38536352772778619e-12 * x10 + 1.89106054713059759e-15 * x12 + 1.39759616731376855e-18 * x14;

    ci = gamma + std::log(x) + x2 * a / b;
  }
  else
  {
    double x16 = x8 * x8;
    double x18 = x16 * x2;
    double x20 = x10 * x10;
    double cos_x = cos(x);
    double sin_x = sin(x);

    double f = (1 + 7.44437068161936700618e2 / x2 + 1.96396372895146869801e5 / x4 + 2.37750310125431834034e7 / x6 + 1.43073403821274636888e9 / x8 + 4.33736238870432522765e10 / x10 + 6.40533830574022022911e11 / x12 + 4.20968180571076940208e12 / x14 + 1.00795182980368574617e13 / x16 + 4.94816688199951963482e12 / x18 - 4.94701168645415959931e11 / x20) / (1 + 7.46437068161927678031e2 / x2 + 1.97865247031583951450e5 / x4 + 2.41535670165126845144e7 / x6 + 1.47478952192985464958e9 / x8 + 4.58595115847765779830e10 / x10 + 7.08501308149515401563e11 / x12 + 5.06084464593475076774e12 / x14 + 1.43468549171581016479e13 / x16 + 1.11535493509914254097e13 / x18) / x;

    double g = (1 + 8.1359520115168615e2 / x2 + 2.35239181626478200e5 / x4 + 3.12557570795778731e7 / x6 + 2.06297595146763354e9 / x8 + 6.83052205423625007e10 / x10 + 1.09049528450362786e12 / x12 + 7.57664583257834349e12 / x14 + 1.81004487464664575e13 / x16 + 6.43291613143049485e12 / x18 - 1.36517137670871689e12 / x20) /
               (1 + 8.19595201151451564e2 / x2 + 2.40036752835578777e5 / x4 + 3.26026661647090822e7 / x6 + 2.23355543278099360e9 / x8 + 7.87465017341829930e10 / x10 + 1.39866710696414565e12 / x12 + 1.17164723371736605e13 / x14 + 4.01839087307656620e13 / x16 + 3.99653257887490811e13 / x18) / x2;

    si = 0.5 * M_PI - f * cos_x - g * sin_x;
    ci = f * sin_x - g * cos_x;
  };
  return;
}

__host__ __device__ double g3lhalo::F(double k1, double k2, double cosphi)
{
  return 0.7143 + 0.2857 * cosphi * cosphi + 0.5 * cosphi * (k1 / k2 + k2 / k2);
}

double g3lhalo::Bi_lin(double k1, double k2, double cosphi, double z,
                       double kmin, double kmax, double zmin, double zmax,
                       int Nbins, const double *P_lin)
{
  // Get k3 = |\vec{k1}+\vec{k2}|
  double k3 = sqrt(k1 * k1 + k2 * k2 + 2 * k1 * k2 * cosphi);

  // Get Cosine of angle between k1 and k2
  double c_phi12 = cosphi; //(k3*k3-k1*k1-k2*k2)/(2*k1*k2);

  // Get cosine of angle between k1 and k3
  double c_phi13 = (k2 * k2 - k3 * k3 - k1 * k1) / (2 * k1 * k3);

  // Get cosine of angle between k2 and k3
  double c_phi23 = (k1 * k1 - k2 * k2 - k3 * k3) / (2 * k3 * k2);

  int z_ix = std::round((z - zmin) * Nbins / (zmax - zmin));

  int k1_ix = std::round(std::log(k1 / kmin) * Nbins / std::log(kmax / kmin));
  int k2_ix = std::round(std::log(k2 / kmin) * Nbins / std::log(kmax / kmin));
  int k3_ix = std::round(std::log(k3 / kmin) * Nbins / std::log(kmax / kmin));

  // Set Powerspectrum
  double Pk1, Pk2, Pk3;
  if (k1_ix >= 0)
  {
    Pk1 = P_lin[z_ix * Nbins + k1_ix];
  }
  else // Use linear approx for very small ks (should work reasonably well)
  {
    Pk1 = P_lin[z_ix * Nbins] / kmin * k1;
  };

  if (k2_ix >= 0)
  {
    Pk2 = P_lin[z_ix * Nbins + k2_ix];
  }
  else // Use linear approx for very small ks (should work reasonably well)
  {
    Pk2 = P_lin[z_ix * Nbins] / kmin * k2;
  };

  if (k3_ix >= 0)
  {
    Pk3 = P_lin[z_ix * Nbins + k3_ix];
  }
  else // Use linear approx for very small ks (should work reasonably well)
  {
    Pk3 = P_lin[z_ix * Nbins] / kmin * k3;
  };

  // Get B(k1, k2)
  return 2 * (F(k1, k2, c_phi12) * Pk1 * Pk2 + F(k1, k3, c_phi13) * Pk1 * Pk3 + F(k2, k3, c_phi23) * Pk2 * Pk3);
}

int g3lhalo::integrand_nbar(unsigned ndim, size_t npts, const double *m, void *thisPtr, unsigned fdim, double *value)
{

  // Check if fdim is correct
  if (fdim != 1)
  {
    std::cerr << "NNMap_Model::integrand_nbar: Wrong fdim" << std::endl;
    exit(1);
  };
  // Read out Container
  n_z_container *container = (n_z_container *)thisPtr;
  HOD *hod = container->hod;
  double z = container->z;

  double zmin = container->zmin;
  double zmax = container->zmax;
  double mmin = container->mmin;
  double mmax = container->mmax;
  int Nbins = container->Nbins;

  double *hmf;

#if GPU
  hmf = container->dev_hmf;
#else
  hmf = container->hmf;
#endif

#if GPU // Calculation with GPU

  // Allocate on device for integration values
  double *dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));

  // Allocate on device for masses
  double *dev_ms;
  CUDA_SAFE_CALL(cudaMalloc(&dev_ms, npts * ndim * sizeof(double)));

  // Copy masses to device
  CUDA_SAFE_CALL(cudaMemcpy(dev_ms, m, npts * ndim * sizeof(double), cudaMemcpyHostToDevice));

  // Do calculation
  g3lhalo::GPUkernel_nbar<<<BLOCKS, THREADS>>>(dev_ms, z, npts, hod, hod->dev_params, zmin, zmax, mmin, mmax, Nbins, hmf, dev_value);

 cudaFree(dev_ms); // Clean up

  // Copy results from device to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

 cudaFree(dev_value); // Clean up

#else // Calculation on CPU

#pragma omp parallel for // Omp parallelization
  for (unsigned int i = 0; i < npts; i++)
  {
    double m_ = pow(10, m[i * ndim]);
    value[i] = kernel_function_nbar(m_, z, hod, NULL, zmin, zmax, mmin, mmax, Nbins, hmf);
  };
#endif                   // GPU

  return 0; // Success :)
}

#if GPU

// __global__ void g3lhalo::GPUkernel_nbar(const double *ms, double z, int npts, double alpha,
//                                         double mth, double sigma, double mprime, double beta,
//                                         double zmin, double zmax,
//                                         double mmin, double mmax, int Nbins, const double *hmf,
//                                         double *value)
// {
//   /// Index of thread
//   int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

//   // Grid-Stride Loop, so I get npts evaluations
//   for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
//   {
//     double m = pow(10, ms[i]);

//     value[i] = kernel_function_nbar(m, z, alpha, mth, sigma, mprime, beta, zmin, zmax, mmin, mmax, Nbins, hmf);
//   };
// }

__global__ void g3lhalo::GPUkernel_nbar(const double *ms, double z, int npts, HOD *hod, double * dev_params,
                                        double zmin, double zmax,
                                        double mmin, double mmax, int Nbins, const double *hmf,
                                        double *value)
{
  /// Index of thread
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  // Grid-Stride Loop, so I get npts evaluations
  for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
  {
    double m = pow(10, ms[i]);

    value[i] = kernel_function_nbar(m, z, hod, dev_params, zmin, zmax, mmin, mmax, Nbins, hmf);
  };
}
#endif

__device__ __host__ double g3lhalo::kernel_function_nbar(double m, double z, g3lhalo::HOD *hod, double * dev_params,
                                                         double zmin, double zmax,
                                                         double mmin, double mmax, int Nbins, const double *hmf)
{

  double Nc =  hod->Ncen(m, dev_params); //<Number of central galaxies
  double Ns = hod->Nsat(m, dev_params); //< Number of satellite galaxies

  // Read indices for mass and redshift
  int m_ix = std::round(std::log(m / mmin) * Nbins / std::log(mmax / mmin));
  int z_ix = std::round((z - zmin) * Nbins / (zmax - zmin));
  return m * hmf[int(z_ix * Nbins + m_ix)] * (Nc + Ns);
}