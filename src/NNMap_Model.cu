#include "NNMap_Model.h"
#include "constants.h"
#include "cubature.h"
#include <iostream>
#include <fstream>

#if GPU
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

#include <algorithm>

g3lhalo::NNMap_Model::NNMap_Model(Cosmology *cosmology_, const double &zmin_, const double &zmax_, const double &kmin_, const double &kmax_,
                                  const double &mmin_, const double &mmax_, const int &Nbins_,
                                  double *g_, double *p_lens1_, double *p_lens2_, double *w_,
                                  double *dwdz_, double *hmf_, double *P_lin_, double *b_h_, double *concentration_,
                                  HOD *hod1_, HOD *hod2_, double A_, double epsilon_)
{

  // Set parameters
  cosmology = cosmology_;

  zmin = zmin_;
  zmax = zmax_;
  kmin = kmin_;
  kmax = kmax_;
  mmin = mmin_;
  mmax = mmax_;
  Nbins = Nbins_;
  g = g_;
  p_lens1 = p_lens1_;
  p_lens2 = p_lens2_;
  w = w_;
  dwdz = dwdz_;
  hmf = hmf_;
  P_lin = P_lin_;
  b_h = b_h_;
  concentration = concentration_;
  hod1 = hod1_;
  hod2 = hod2_;
  A = A_;
  epsilon = epsilon_;

  zmin_integral = zmin;
  zmax_integral = zmax - (zmax - zmin) / Nbins;
  mmin_integral = mmin;
  mmax_integral = exp(log(mmin) + log(mmax / mmin) * (Nbins - 1) / Nbins);

#if GPU
  // Allocation of memory for precomputed functions on device
  CUDA_SAFE_CALL(cudaMalloc(&dev_g, Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_p_lens1, Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_p_lens2, Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_w, Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_dwdz, Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_hmf, Nbins * Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_P_lin, Nbins * Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_b_h, Nbins * Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_concentration, Nbins * Nbins * sizeof(double)));

  // Copying of precomputed functions to device
  CUDA_SAFE_CALL(cudaMemcpy(dev_g, g, Nbins * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_p_lens1, p_lens1, Nbins * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_p_lens2, p_lens2, Nbins * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_w, w, Nbins * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_dwdz, dwdz, Nbins * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_hmf, hmf, Nbins * Nbins * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_P_lin, P_lin, Nbins * Nbins * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_b_h, b_h, Nbins * Nbins * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_concentration, concentration, Nbins * Nbins * sizeof(double), cudaMemcpyHostToDevice));

  // Allocation of memory for densities on device
  CUDA_SAFE_CALL(cudaMalloc(&dev_rho_bar, Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_n_bar1, Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_n_bar2, Nbins * sizeof(double)));

#endif // GPU

  // Allocattion of memory for densities on Host
  rho_bar = (double *)malloc(Nbins * sizeof(double));
  n_bar1 = (double *)malloc(Nbins * sizeof(double));
  n_bar2 = (double *)malloc(Nbins * sizeof(double));

#if VERBOSE
  std::cerr << "Finished memory setting" << std::endl;
#endif // VERBOSE

  // Calculate densities (stored in rho_bar, n_bar1 and n_bar2)
  updateDensities();

#if GPU
  // Copying of densities to device
  CUDA_SAFE_CALL(cudaMemcpy(dev_rho_bar, rho_bar, Nbins * sizeof(double), cudaMemcpyHostToDevice));
#endif
#if VERBOSE
  std::cerr << "Finished initalizing NNMap" << std::endl;
#endif // VERBOSE
}

#if SCALING
g3lhalo::NNMap_Model::NNMap_Model(Cosmology *cosmology_, const double &zmin_, const double &zmax_, const double &kmin_, const double &kmax_,
                                  const double &mmin_, const double &mmax_, const int &Nbins_,
                                  double *g_, double *p_lens1_, double *p_lens2_, double *w_,
                                  double *dwdz_, double *hmf_, double *P_lin_, double *b_h_, double *concentration_,
                                  double *scaling1_, double *scaling2_, HOD *hod1_, HOD *hod2_, double A_, double epsilon_)
{

  // Set parameters
  cosmology = cosmology_;

  zmin = zmin_;
  zmax = zmax_;
  kmin = kmin_;
  kmax = kmax_;
  mmin = mmin_;
  mmax = mmax_;
  Nbins = Nbins_;
  g = g_;
  p_lens1 = p_lens1_;
  p_lens2 = p_lens2_;
  w = w_;
  dwdz = dwdz_;
  hmf = hmf_;
  P_lin = P_lin_;
  b_h = b_h_;
  concentration = concentration_;
  hod1 = hod1_;
  hod2 = hod2_;
  A = A_;
  epsilon = epsilon_;

#if SCALING
  scaling1 = scaling1_;
  scaling2 = scaling2_;
#endif

  zmin_integral = zmin;
  zmax_integral = zmax - (zmax - zmin) / Nbins;
  mmin_integral = mmin;
  mmax_integral = exp(log(mmin) + log(mmax / mmin) * (Nbins - 1) / Nbins);

#if GPU
  // Allocation of memory for precomputed functions on device
  CUDA_SAFE_CALL(cudaMalloc(&dev_g, Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_p_lens1, Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_p_lens2, Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_w, Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_dwdz, Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_hmf, Nbins * Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_P_lin, Nbins * Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_b_h, Nbins * Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_concentration, Nbins * Nbins * sizeof(double)));

#if SCALING
  CUDA_SAFE_CALL(cudaMalloc(&dev_scaling1, Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_scaling2, Nbins * sizeof(double)));
#endif

  // Copying of precomputed functions to device
  CUDA_SAFE_CALL(cudaMemcpy(dev_g, g, Nbins * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_p_lens1, p_lens1, Nbins * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_p_lens2, p_lens2, Nbins * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_w, w, Nbins * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_dwdz, dwdz, Nbins * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_hmf, hmf, Nbins * Nbins * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_P_lin, P_lin, Nbins * Nbins * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_b_h, b_h, Nbins * Nbins * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_concentration, concentration, Nbins * Nbins * sizeof(double), cudaMemcpyHostToDevice));

#if SCALING
  CUDA_SAFE_CALL(cudaMemcpy(dev_scaling1, scaling1, Nbins * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_scaling2, scaling2, Nbins * sizeof(double), cudaMemcpyHostToDevice));
#endif

  // Allocation of memory for densities on device
  CUDA_SAFE_CALL(cudaMalloc(&dev_rho_bar, Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_n_bar1, Nbins * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_n_bar2, Nbins * sizeof(double)));

#endif // GPU

  // Allocattion of memory for densities on Host
  rho_bar = (double *)malloc(Nbins * sizeof(double));
  n_bar1 = (double *)malloc(Nbins * sizeof(double));
  n_bar2 = (double *)malloc(Nbins * sizeof(double));

#if VERBOSE
  std::cerr << "Finished memory setting" << std::endl;
#endif // VERBOSE

  // Calculate densities (stored in rho_bar, n_bar1 and n_bar2)
  updateDensities();

#if GPU
  // Copying of densities to device
  CUDA_SAFE_CALL(cudaMemcpy(dev_rho_bar, rho_bar, Nbins * sizeof(double), cudaMemcpyHostToDevice));
#endif
#if VERBOSE
  std::cerr << "Finished initalizing NNMap" << std::endl;
#endif // VERBOSE
}
#endif
g3lhalo::NNMap_Model::~NNMap_Model()
{
#if GPU // Free device memory
  cudaFree(dev_g);
  cudaFree(dev_p_lens1);
  cudaFree(dev_p_lens2);
  cudaFree(dev_w);
  cudaFree(dev_dwdz);
  cudaFree(dev_hmf);
  cudaFree(dev_P_lin);
  cudaFree(dev_b_h);
  cudaFree(dev_concentration);
  cudaFree(dev_rho_bar);
  cudaFree(dev_n_bar1);
  cudaFree(dev_n_bar2);

#if SCALING
  cudaFree(dev_scaling1);
  cudaFree(dev_scaling2);
#endif
#endif // GPU
  // Free host memory
  free(rho_bar);
  free(n_bar1);
  free(n_bar2);
#if SCALING
  free(scaling1);
  free(scaling2);
#endif
}

void g3lhalo::NNMap_Model::updateHODs(HOD *hod1_, HOD *hod2_)
{
  hod1 = hod1_;
  hod2 = hod2_;
  updateDensities();
}

void g3lhalo::NNMap_Model::updateDensities()
{
  // Set integral borders (logarithmic)
  double m_min[1] = {std::log10(mmin_integral)};
  double m_max[1] = {std::log10(mmax_integral)};
  double error, result;

  for (int i = 0; i < Nbins; i++) // Go over all redshift bins
  {
    double z = zmin + i * (zmax - zmin) / Nbins; // Get redshift bin

    // \f$\bar{\rho}(z) = \Omega_m(z) \rho_{crit}$\f
    rho_bar[i] = cosmology->Omega_m_(z) * cosmology->rho_crit;

    // Set container for integral
    n_z_container container;
    container.z = z;
    container.zmin = zmin;
    container.zmax = zmax;
    container.mmin = mmin;
    container.mmax = mmax;
    container.Nbins = Nbins;
    container.hmf = hmf;
#if GPU
    container.dev_hmf = dev_hmf;
#endif

    container.hod = hod1;
    // Calculate integral for type 1
    hcubature_v(1, integrand_nbar, &container, 1, m_min, m_max, 0, 0, 1e-4, ERROR_L1, &result, &error);
    // Set n_bar1 (ln10 is to account for logarithmic integral)
    n_bar1[i] = g3lhalo::ln10 * result;

    container.hod = hod2;
    // Calculate integral for type 2
    hcubature_v(1, integrand_nbar, &container, 1, m_min, m_max, 0, 0, 1e-4, ERROR_L1, &result, &error);
    // Set n_bar2 (ln10 is to account for logarithmic integral)
    n_bar2[i] = g3lhalo::ln10 * result;
  };

#if VERBOSE
  std::cerr << "Finished calculating densities" << std::endl;
  std::cerr << n_bar1[0] << std::endl;
#endif // VERBOSE

#if GPU // Copy to device
  CUDA_SAFE_CALL(cudaMemcpy(dev_n_bar1, n_bar1, Nbins * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_n_bar2, n_bar2, Nbins * sizeof(double), cudaMemcpyHostToDevice));
#endif // GPU

#if VERBOSE
  std::cerr << "Finished density update" << std::endl;
#endif // VERBOSE
}

double g3lhalo::NNMap_Model::NNMap_1h(const double &theta1, const double &theta2, const double &theta3, const int &type1, const int &type2)
{

  // Set Container
  nnmap_container container;
  container.nnmap = this;
  container.theta1 = theta1;
  container.theta2 = theta2;
  container.theta3 = theta3;
  container.type1 = type1;
  container.type2 = type2;

  // Set Integral borders
  // Logarithmic over l1, l2, and m
  double params_min[5] = {std::log10(lmin_integral), std::log10(lmin_integral), phimin_integral, std::log10(mmin_integral), zmin_integral};
  double params_max[5] = {std::log10(lmax_integral), std::log10(lmax_integral), phimax_integral, std::log10(mmax_integral), zmax_integral};

  // Do calculation
  double result, error;
  hcubature_v(1, NNM::integrand_1halo, &container, 5, params_min, params_max, 5000000, 0, 1e-3, ERROR_L1, &result, &error);

  // Return result (Prefactor includes three ln10 terms due to logarithmic integral)
  return result * 3 * cosmology->H0 * cosmology->H0 * cosmology->Omega_m / 2 / g3lhalo::c / g3lhalo::c * g3lhalo::ln10 * g3lhalo::ln10 * g3lhalo::ln10 / (2 * g3lhalo::pi) / (2 * g3lhalo::pi) / (2 * g3lhalo::pi);
}

double g3lhalo::NNMap_Model::NNMap_2h(const double &theta1, const double &theta2, const double &theta3, const int &type1, const int &type2)
{
  // Set Container
  nnmap_container container;
  container.nnmap = this;
  container.theta1 = theta1;
  container.theta2 = theta2;
  container.theta3 = theta3;
  container.type1 = type1;
  container.type2 = type2;

  // Set integral borders
  // Logarithmic over l1, l2, m1, and m2
  double params_min[6] = {std::log10(lmin_integral), std::log10(lmin_integral), phimin_integral, std::log10(mmin_integral), std::log10(mmin_integral), zmin_integral};
  double params_max[6] = {std::log10(lmax_integral), std::log10(lmax_integral), phimax_integral, std::log10(mmax_integral), std::log10(mmax_integral), zmax_integral};

  // Do calculation
  double result, error;
  hcubature_v(1, NNM::integrand_2halo, &container, 6, params_min, params_max, 5000000, 0, 1e-3, ERROR_L1, &result, &error);

  // Return result (Prefactor includes four ln10 terms due to logarithmic integral)
  return result * 3 * cosmology->H0 * cosmology->H0 * cosmology->Omega_m / 2 / g3lhalo::c / g3lhalo::c * g3lhalo::ln10 * g3lhalo::ln10 * g3lhalo::ln10 * g3lhalo::ln10 / (2 * g3lhalo::pi) / (2 * g3lhalo::pi) / (2 * g3lhalo::pi);
}

double g3lhalo::NNMap_Model::NNMap_3h(const double &theta1, const double &theta2, const double &theta3, const int &type1, const int &type2)
{
  // Set container
  nnmap_container container;
  container.nnmap = this;
  container.theta1 = theta1;
  container.theta2 = theta2;
  container.theta3 = theta3;
  container.type1 = type1;
  container.type2 = type2;

  // Set integral borders
  // Logarithmic over l1, l2, m1, m2, and m3
  double params_min[7] = {std::log10(lmin_integral), std::log10(lmin_integral), phimin_integral, std::log10(mmin_integral), std::log10(mmin_integral), std::log10(mmin_integral), zmin_integral};
  double params_max[7] = {std::log10(lmax_integral), std::log10(lmax_integral), phimax_integral, std::log10(mmax_integral), std::log10(mmax_integral), std::log10(mmax_integral), zmax_integral};

  // Do calculation
  double result, error;
  hcubature_v(1, NNM::integrand_3halo, &container, 7, params_min, params_max, 5000000, 0, 1e-4, ERROR_L1, &result, &error);

  // Return result (Prefactor includes five ln10 terms due to logarithmic integral)
  return result * 3 * cosmology->H0 * cosmology->H0 * cosmology->Omega_m / 2 / g3lhalo::c / g3lhalo::c * g3lhalo::ln10 * g3lhalo::ln10 * g3lhalo::ln10 * g3lhalo::ln10 * g3lhalo::ln10 / (2 * g3lhalo::pi) / (2 * g3lhalo::pi) / (2 * g3lhalo::pi);
}

double g3lhalo::NNMap_Model::NNMap(const double &theta1, const double &theta2, const double &theta3, const int &type1, const int &type2)
{
  return NNMap_1h(theta1, theta2, theta3, type1, type2)    // 1-halo term
         + NNMap_2h(theta1, theta2, theta3, type1, type2)  // 2-halo term
         + NNMap_3h(theta1, theta2, theta3, type1, type2); // 3-halo term
}

void g3lhalo::NNMap_Model::calculateAll(const std::vector<double> &thetas1, const std::vector<double> &thetas2, const std::vector<double> &thetas3,
                                        HOD *hod1, HOD *hod2, double A, double epsilon, std::vector<double> &results)
{
  updateHODs(hod1, hod2);
  int n = thetas1.size();
  // N1N2
  for (int i = 0; i < n; i++)
  {
    double theta1 = thetas1.at(i); // Read Thetas
    double theta2 = thetas2.at(i);
    double theta3 = thetas3.at(i);

    results.push_back(NNMap(theta1, theta2, theta3, 1, 2));
  }
  // N1N1
  for (int i = 0; i < n; i++)
  {
    double theta1 = thetas1.at(i); // Read Thetas
    double theta2 = thetas2.at(i);
    double theta3 = thetas3.at(i);

    results.push_back(NNMap(theta1, theta2, theta3, 1, 1));
  }
  // N2N2
  for (int i = 0; i < n; i++)
  {
    double theta1 = thetas1.at(i); // Read Thetas
    double theta2 = thetas2.at(i);
    double theta3 = thetas3.at(i);

    results.push_back(NNMap(theta1, theta2, theta3, 2, 2));
  }
  return;
}

/********************************************* 1 HALO TERM *****************************************************************/
int g3lhalo::NNM::integrand_1halo(unsigned ndim, size_t npts, const double *params, void *thisPtr, unsigned fdim, double *value)
{
  // Check if fdim is correct
  if (fdim != 1)
  {
    std::cerr << "NNMap_Model::integrand_nbar: Wrong fdim" << std::endl;
    exit(1);
  };

  // Read out container
  nnmap_container *container = (nnmap_container *)thisPtr;
  NNMap_Model *nnmap = container->nnmap;
  double theta1 = container->theta1;
  double theta2 = container->theta2;
  double theta3 = container->theta3;

  double type1 = container->type1;
  double type2 = container->type2;

  HOD *hod1, *hod2;
  double *nbar1, *nbar2;
  if (type1 == 1)
  {
    hod1 = nnmap->hod1;
#if GPU
    nbar1 = nnmap->dev_n_bar1;
#else
    nbar1 = nnmap->n_bar1;
#endif
  }
  else
  {
    hod1 = nnmap->hod2;
#if GPU
    nbar1 = nnmap->dev_n_bar2;
#else
    nbar1 = nnmap->n_bar2;
#endif
  }

  if (type2 == 1)
  {
    hod2 = nnmap->hod1;
#if GPU
    nbar2 = nnmap->dev_n_bar1;
#else
    nbar2 = nnmap->n_bar1;
#endif
  }
  else
  {
    hod2 = nnmap->hod2;
#if GPU
    nbar2 = nnmap->dev_n_bar2;
#else
    nbar2 = nnmap->n_bar2;
#endif
  }

  // Set HOD parameters
  double A = nnmap->A;
  double epsilon = nnmap->epsilon;

  // std::cerr<<f1<<std::endl;
#if GPU // Calculation on GPU

  // Allocate for integration values on device
  double *dev_value;
  CUDA_SAFE_CALL(cudaMalloc(&dev_value, fdim * npts * sizeof(double)));

  // Allocate for integration variables on device
  double *dev_params;
  CUDA_SAFE_CALL(cudaMalloc(&dev_params, npts * ndim * sizeof(double)));

  // Copy integration variables to device
  CUDA_SAFE_CALL(cudaMemcpy(dev_params, params, npts * ndim * sizeof(double), cudaMemcpyHostToDevice));

// Do calculation
#if SCALING
  GPUkernel_1Halo<<<BLOCKS, THREADS>>>(dev_params, theta1, theta2, theta3, npts, hod1, hod2, hod1->dev_params, hod2->dev_params, A, epsilon, nnmap->zmin,
                                       nnmap->zmax,
                                       nnmap->mmin, nnmap->mmax, nnmap->Nbins, nnmap->dev_g, nnmap->dev_p_lens1, nnmap->dev_p_lens2,
                                       nnmap->dev_w, nnmap->dev_dwdz, nnmap->dev_hmf, nnmap->dev_concentration, nnmap->dev_rho_bar,
                                       nbar1, nbar2,
                                       dev_value,
                                       nnmap->dev_scaling1, nnmap->dev_scaling2);
#else
  GPUkernel_1Halo<<<BLOCKS, THREADS>>>(dev_params, theta1, theta2, theta3, npts, hod1, hod2, hod1->dev_params, hod2->dev_params, A, epsilon, nnmap->zmin,
                                       nnmap->zmax,
                                       nnmap->mmin, nnmap->mmax, nnmap->Nbins, nnmap->dev_g, nnmap->dev_p_lens1, nnmap->dev_p_lens2,
                                       nnmap->dev_w, nnmap->dev_dwdz, nnmap->dev_hmf, nnmap->dev_concentration, nnmap->dev_rho_bar,
                                       nbar1, nbar2, dev_value);
#endif
  cudaFree(dev_params); // Clean up

  // Copy values from device to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); // Clean up

#else // Calculation on CPU

#pragma omp parallel for
  for (unsigned int i = 0; i < npts; i++)
  {
    double l1 = pow(10, params[i * ndim]);
    double l2 = pow(10, params[i * ndim + 1]);
    double phi = params[i * ndim + 2];
    double m = pow(10, params[i * ndim + 3]);
    double z = params[i * ndim + 4];
#if SCALING
    value[i] = kernel_function_1halo(theta1, theta2, theta3, l1, l2, phi, m, z, nnmap->zmin, nnmap->zmax, nnmap->mmin, nnmap->mmax,
                                     nnmap->Nbins, hod1, hod2, NULL, NULL, A, epsilon, nnmap->g,
                                     nnmap->p_lens1, nnmap->p_lens2, nnmap->w, nnmap->dwdz, nnmap->hmf, nnmap->concentration,
                                     nnmap->rho_bar, nbar1, nbar2, nnmap->scaling1, nnmap->scaling2);
#else
    value[i] = kernel_function_1halo(theta1, theta2, theta3, l1, l2, phi, m, z, nnmap->zmin, nnmap->zmax, nnmap->mmin, nnmap->mmax,
                                     nnmap->Nbins, hod1, hod2,NULL, NULL, A, epsilon, nnmap->g,
                                     nnmap->p_lens1, nnmap->p_lens2, nnmap->w, nnmap->dwdz, nnmap->hmf, nnmap->concentration,
                                     nnmap->rho_bar, nbar1, nbar2);
#endif
  };

#endif
  return 0; // Success :)
}

__device__ __host__ double g3lhalo::NNM::kernel_function_1halo(double theta1, double theta2,
                                                          double theta3, double l1, double l2, double phi, double m, double z,
                                                          double zmin, double zmax, double mmin, double mmax, int Nbins,
                                                          g3lhalo::HOD *hod1, g3lhalo::HOD *hod2, double *dev_params1, double *dev_params2,
                                                          double A, double epsilon,
                                                          const double *g, const double *p_lens1,
                                                          const double *p_lens2, const double *w, const double *dwdz,
                                                          const double *hmf, const double *concentration,
                                                          const double *rho_bar, const double *n_bar1, const double *n_bar2, const double *scaling1, const double *scaling2)
{
  // Get inidices of z and m
  int z_ix = std::round(((z - zmin) * Nbins / (zmax - zmin)));
  int m_ix = std::round((std::log(m / mmin) * Nbins / std::log(mmax / mmin)));

  // Get lens galaxy densities
  double nbar1, nbar2;
  double scale1, scale2;

  nbar1 = n_bar1[z_ix];
  scale1 = (scaling1 == NULL ? 1 : scaling1[m_ix]);

  nbar2 = n_bar2[z_ix];
  scale2 = (scaling2 == NULL ? 1 : scaling2[m_ix]);

  double w_ = w[z_ix]; //< Comoving distance [Mpc]

  double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi));
  double k1 = l1 / w_;
  double k2 = l2 / w_;
  double k3 = l3 / w_;

  // 3D Galaxy-Galaxy-Matter Bispectrum
  double Bggd = 1. / nbar1 / nbar2 / rho_bar[z_ix] * hmf[z_ix * Nbins + m_ix] * m * u_NFW(k3, m, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration);

  if (scaling1 == NULL)
  {
    Bggd *= G_gg(k1, k2, m, z, hod1, hod2, dev_params1, dev_params2, A, epsilon, zmin, zmax,
                 mmin, mmax, Nbins, rho_bar, concentration);
  }
  else
  {
    Bggd *= G_gg(k1, k2, m, z, hod1, hod2, dev_params1, dev_params2, A, epsilon, zmin, zmax,
                 mmin, mmax, Nbins, rho_bar, concentration, scale1, scale2);
  };

  // 2D Galaxy-Galaxy-Convergence Bispectrum
  double bggk = Bggd / dwdz[z_ix] * g[z_ix] * p_lens1[z_ix] * p_lens2[z_ix] / w_ / w_ / w_ * (1. + z);

  return l1 * l2 * m * l1 * l2 * apertureFilter(theta1 * l1) * apertureFilter(theta2 * l2) * apertureFilter(theta3 * l3) * bggk;
}

#if GPU
__global__ void g3lhalo::NNM::GPUkernel_1Halo(const double *params, double theta1, double theta2, double theta3,
                                         int npts, g3lhalo::HOD *hod1, g3lhalo::HOD *hod2, double *dev_params1, double *dev_params2,
                                         double A, double epsilon,
                                         double zmin, double zmax, double mmin, double mmax,
                                         int Nbins, const double *g, const double *p_lens1, const double *p_lens2,
                                         const double *w, const double *dwdz, const double *hmf, const double *concentration,
                                         const double *rho_bar, const double *n_bar1, const double *n_bar2,
                                         double *value, const double *scaling1, const double *scaling2)
{
  /// Index of thread
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  // Grid-Stride Loop, so I get npts evaluations
  for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
  {

    double l1 = pow(10, params[i * 5]);
    double l2 = pow(10, params[i * 5 + 1]);
    double phi = params[i * 5 + 2];
    double m = pow(10, params[i * 5 + 3]);
    double z = params[i * 5 + 4];

    if (scaling1 == NULL)
    {
      value[i] = kernel_function_1halo(theta1, theta2, theta3, l1, l2, phi, m, z, zmin, zmax, mmin, mmax, Nbins, hod1, hod2, dev_params1, dev_params2,
                                       A, epsilon, g,
                                       p_lens1, p_lens2, w, dwdz, hmf, concentration, rho_bar, n_bar1, n_bar2);
    }
    else
    {
      value[i] = kernel_function_1halo(theta1, theta2, theta3, l1, l2, phi, m, z, zmin, zmax, mmin, mmax, Nbins, hod1, hod2, dev_params1, dev_params2,
                                       A, epsilon, g,
                                       p_lens1, p_lens2, w, dwdz, hmf, concentration, rho_bar, n_bar1, n_bar2, scaling1, scaling2);
    };
  };
}

#endif
/********************************************* 2 HALO TERM *****************************************************************/

int g3lhalo::NNM::integrand_2halo(unsigned ndim, size_t npts, const double *params, void *thisPtr, unsigned fdim, double *value)
{
  if (fdim != 1) // Check fdim
  {
    std::cerr << "NNMap_Model::integrand_nbar: Wrong fdim" << std::endl;
    exit(1);
  };

  // Read out Container
  nnmap_container *container = (nnmap_container *)thisPtr;
  NNMap_Model *nnmap = container->nnmap;
  double theta1 = container->theta1;
  double theta2 = container->theta2;
  double theta3 = container->theta3;
  double type1 = container->type1;
  double type2 = container->type2;

  double A = nnmap->A;
  double epsilon = nnmap->epsilon;

  HOD *hod1, *hod2;
  double *nbar1, *nbar2;
  if (type1 == 1)
  {
    hod1 = nnmap->hod1;
#if GPU
    nbar1 = nnmap->dev_n_bar1;
#else
    nbar1 = nnmap->n_bar1;
#endif
  }
  else
  {
    hod1 = nnmap->hod2;
#if GPU
    nbar1 = nnmap->dev_n_bar2;
#else
    nbar1 = nnmap->n_bar2;
#endif
  }

  if (type2 == 1)
  {
    hod2 = nnmap->hod1;
#if GPU
    nbar2 = nnmap->dev_n_bar1;
#else
    nbar2 = nnmap->n_bar1;
#endif
  }
  else
  {
    hod2 = nnmap->hod2;
#if GPU
    nbar2 = nnmap->dev_n_bar2;
#else
    nbar2 = nnmap->n_bar2;
#endif
  }
#if GPU // Do calculation on GPU

  int npts_it = 100000;                    //< Maximal number of integration points simultaneously executed (limited by device RAM!)
  int number_it = int(npts / npts_it) + 1; //< Number of iterations necessary to get npts executions

  for (int i = 0; i < number_it; i++)
  {
    // Set points for which this iteration goes through
    int start = i * npts_it;
    int end = std::min<int>((i + 1) * npts_it, npts);
    int n_it = end - start;

    // Set parameters for which this iteration goes through
    double params_it[n_it * ndim];
#pragma omp parallel for
    for (int j = 0; j < n_it * ndim; j++)
    {
      params_it[j] = params[start * ndim + j];
    };

    // Allocate device memory for parameters
    double *dev_params;
    CUDA_SAFE_CALL(cudaMalloc(&dev_params, n_it * ndim * sizeof(double)));

    // Copy parameters to device memory
    CUDA_SAFE_CALL(cudaMemcpy(dev_params, &params_it, n_it * ndim * sizeof(double), cudaMemcpyHostToDevice));

    // Allocate memory for results on device
    double *dev_value;
    CUDA_SAFE_CALL(cudaMalloc(&dev_value, fdim * n_it * sizeof(double)));

// Do calculation
#if SCALING
    GPUkernel_2Halo<<<BLOCKS, THREADS>>>(dev_params, theta1, theta2, theta3, n_it, hod1, hod2, hod1->dev_params, hod2->dev_params, A, epsilon, nnmap->zmin,
                                         nnmap->zmax, nnmap->mmin, nnmap->mmax, nnmap->kmin, nnmap->kmax, nnmap->Nbins,
                                         nnmap->dev_g, nnmap->dev_p_lens1, nnmap->dev_p_lens2, nnmap->dev_w, nnmap->dev_dwdz,
                                         nnmap->dev_hmf, nnmap->dev_P_lin, nnmap->dev_b_h, nnmap->dev_concentration,
                                         nnmap->dev_rho_bar, nbar1, nbar2, dev_value,
                                         nnmap->dev_scaling1, nnmap->dev_scaling2);
#else
    GPUkernel_2Halo<<<BLOCKS, THREADS>>>(dev_params, theta1, theta2, theta3, n_it, hod1, hod2, hod1->dev_params, hod2->dev_params, A, epsilon, nnmap->zmin,
                                         nnmap->zmax, nnmap->mmin, nnmap->mmax, nnmap->kmin, nnmap->kmax, nnmap->Nbins,
                                         nnmap->dev_g, nnmap->dev_p_lens1, nnmap->dev_p_lens2, nnmap->dev_w, nnmap->dev_dwdz,
                                         nnmap->dev_hmf, nnmap->dev_P_lin, nnmap->dev_b_h, nnmap->dev_concentration,
                                         nnmap->dev_rho_bar, nbar1, nbar2, dev_value);
#endif

    cudaFree(dev_params); //< Clean up

    // Copy of results to host
    double value_it[fdim * n_it];
    CUDA_SAFE_CALL(cudaMemcpy(&value_it, dev_value, fdim * n_it * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dev_value); //< Clean up

    // Add result to overall vector
#pragma omp parallel for
    for (int j = 0; j < fdim * n_it; j++)
    {
      value[start * fdim + j] = value_it[j];
    };
  };

#else // Do calculation on CPU

#pragma omp parallel for
  for (unsigned int i = 0; i < npts; i++)
  {
    double l1 = pow(10, params[i * ndim]);
    double l2 = pow(10, params[i * ndim + 1]);
    double phi = params[i * ndim + 2];
    double m1 = pow(10, params[i * ndim + 3]);
    double m2 = pow(10, params[i * ndim + 4]);
    double z = params[i * ndim + 5];
#if SCALING
    value[i] = kernel_function_2halo(theta1, theta2, theta3, l1, l2, phi, m1, m2, z, nnmap->zmin, nnmap->zmax, nnmap->mmin,
                                     nnmap->mmax, nnmap->kmin, nnmap->kmax, nnmap->Nbins, hod1, hod2, NULL, NULL, A, epsilon, nnmap->g,
                                     nnmap->p_lens1, nnmap->p_lens2, nnmap->w, nnmap->dwdz, nnmap->hmf, nnmap->P_lin, nnmap->b_h,
                                     nnmap->concentration, nnmap->rho_bar, nbar1, nbar2,
                                     nnmap->scaling1, nnmap->scaling2);
#else
    value[i] = kernel_function_2halo(theta1, theta2, theta3, l1, l2, phi, m1, m2, z, nnmap->zmin, nnmap->zmax, nnmap->mmin,
                                     nnmap->mmax, nnmap->kmin, nnmap->kmax, nnmap->Nbins, hod1, hod2, NULL, NULL, A, epsilon, nnmap->g,
                                     nnmap->p_lens1, nnmap->p_lens2, nnmap->w, nnmap->dwdz, nnmap->hmf, nnmap->P_lin, nnmap->b_h,
                                     nnmap->concentration, nnmap->rho_bar, nbar1, nbar2);
#endif
  };

#endif
  return 0; // Success :)
}

__device__ __host__ double g3lhalo::NNM::kernel_function_2halo(double theta1, double theta2, double theta3, double l1, double l2,
                                                          double phi, double m1, double m2, double z, double zmin, double zmax,
                                                          double mmin, double mmax, double kmin, double kmax, int Nbins, g3lhalo::HOD *hod1,
                                                          g3lhalo::HOD *hod2, double *dev_params1, double *dev_params2, double A, double epsilon,
                                                          const double *g, const double *p_lens1, const double *p_lens2,
                                                          const double *w, const double *dwdz, const double *hmf, const double *P_lin,
                                                          const double *b_h, const double *concentration, const double *rho_bar,
                                                          const double *n_bar1, const double *n_bar2,
                                                          const double *scaling1, const double *scaling2)
{

  // Get Indices of z, m1, and m2
  int z_ix = std::round((z - zmin) * Nbins / (zmax - zmin));
  int m1_ix = std::round(std::log(m1 / mmin) * Nbins / std::log(mmax / mmin));
  int m2_ix = std::round(std::log(m2 / mmin) * Nbins / std::log(mmax / mmin));

  // Get lens galaxy densities
  double nbar1, nbar2;
  double scale1, scale2;

  nbar1 = n_bar1[z_ix];
  scale1 = (scaling1 == NULL ? 1 : scaling1[m2_ix]);

  nbar2 = n_bar2[z_ix];
  scale2 = (scaling2 == NULL ? 1 : scaling2[m2_ix]);

  double w_ = w[z_ix]; // Comoving distance [Mpc]
  double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi));
  double k1 = l1 / w_;
  double k2 = l2 / w_;
  double k3 = l3 / w_;

  int k1_ix = std::round(std::log(k1 / kmin) * Nbins / std::log(kmax / kmin));
  int k2_ix = std::round(std::log(k2 / kmin) * Nbins / std::log(kmax / kmin));
  int k3_ix = std::round(std::log(k3 / kmin) * Nbins / std::log(kmax / kmin));

  // Set Powerspectrum
  // Use linear approx for very small ks (should work reasonably well)
  double Pk1, Pk2, Pk3;
  if (k1_ix >= 0)
    Pk1 = P_lin[z_ix * Nbins + k1_ix];
  else
    Pk1 = P_lin[z_ix * Nbins] / kmin * k1;

  if (k2_ix >= 0)
    Pk2 = P_lin[z_ix * Nbins + k2_ix];
  else
    Pk2 = P_lin[z_ix * Nbins] / kmin * k2;

  if (k3_ix >= 0)
    Pk3 = P_lin[z_ix * Nbins + k3_ix];
  else
    Pk3 = P_lin[z_ix * Nbins] / kmin * k3;

  // 3D Bispectrum
  double Bggd;
  if (scaling1 == NULL)
  {
    Bggd = 1. / nbar1 / nbar2 / rho_bar[z_ix] * hmf[z_ix * Nbins + m1_ix] * hmf[z_ix * Nbins + m2_ix] * b_h[z_ix * Nbins + m1_ix] * b_h[z_ix * Nbins + m2_ix] * (m1 * u_NFW(k3, m1, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * G_gg(k1, k2, m2, z, hod1, hod2, dev_params1, dev_params2, A, epsilon, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * Pk3 + m2 * u_NFW(k3, m2, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * G_g(k1, m1, z, hod1, dev_params1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * G_g(k2, m2, z, hod2, dev_params2, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * Pk1 + m2 * u_NFW(k3, m2, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * G_g(k2, m1, z, hod2, dev_params2, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * G_g(k1, m2, z, hod1, dev_params1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * Pk2);
  }
  else
  {
    Bggd = 1. / nbar1 / nbar2 / rho_bar[z_ix] * hmf[z_ix * Nbins + m1_ix] * hmf[z_ix * Nbins + m2_ix] * b_h[z_ix * Nbins + m1_ix] * b_h[z_ix * Nbins + m2_ix] * (m1 * u_NFW(k3, m1, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * G_gg(k1, k2, m2, z, hod1, hod2, dev_params1, dev_params2, A, epsilon, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration, scale1, scale2) * Pk3 + m2 * u_NFW(k3, m2, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * G_g(k1, m1, z, hod1, dev_params1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * G_g(k2, m2, z, hod2, dev_params2, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * Pk1 + m2 * u_NFW(k3, m2, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * G_g(k2, m1, z, hod2, dev_params2, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * G_g(k1, m2, z, hod1, dev_params1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * Pk2);
  };

  // 2D Bispectrum
  double bggk = 1. / dwdz[z_ix] * g[z_ix] * p_lens1[z_ix] * p_lens2[z_ix] / w_ / w_ / w_ * (1. + z) * Bggd;

  return l1 * l2 * m1 * m2 * l1 * l2 * apertureFilter(theta1 * l1) * apertureFilter(theta2 * l2) * apertureFilter(theta3 * l3) * bggk;
}

#if GPU

__global__ void g3lhalo::NNM::GPUkernel_2Halo(const double *params, double theta1, double theta2, double theta3, int npts, g3lhalo::HOD *hod1, g3lhalo::HOD *hod2,
                                         double *dev_params1, double *dev_params2,
                                         double A, double epsilon, double zmin, double zmax, double mmin, double mmax, double kmin,
                                         double kmax, int Nbins, const double *g, const double *p_lens1, const double *p_lens2,
                                         const double *w, const double *dwdz, const double *hmf, const double *P_lin,
                                         const double *b_h, const double *concentration, const double *rho_bar, const double *n_bar1,
                                         const double *n_bar2, double *value, const double *scaling1, const double *scaling2)
{
  /// Index of thread
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  // Grid-Stride Loop, so I get npts evaluations
  for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
  {
    double l1 = pow(10, params[i * 6]);
    double l2 = pow(10, params[i * 6 + 1]);
    double phi = params[i * 6 + 2];
    double m1 = pow(10, params[i * 6 + 3]);
    double m2 = pow(10, params[i * 6 + 4]);
    double z = params[i * 6 + 5];

    if (scaling1 == NULL)
    {
      value[i] = kernel_function_2halo(theta1, theta2, theta3, l1, l2, phi, m1, m2, z, zmin, zmax, mmin, mmax, kmin, kmax, Nbins, hod1, hod2, dev_params1, dev_params2, A,
                                       epsilon, g, p_lens1, p_lens2, w, dwdz, hmf, P_lin, b_h, concentration, rho_bar, n_bar1, n_bar2);
    }
    else
    {
      value[i] = kernel_function_2halo(theta1, theta2, theta3, l1, l2, phi, m1, m2, z, zmin, zmax, mmin, mmax, kmin, kmax, Nbins, hod1, hod2, dev_params1, dev_params2, A,
                                       epsilon, g, p_lens1, p_lens2, w, dwdz, hmf, P_lin, b_h, concentration, rho_bar, n_bar1, n_bar2,
                                       scaling1, scaling2);
    }
  };
}

#endif
/********************************************* 3 HALO TERM *****************************************************************/

int g3lhalo::NNM::integrand_3halo(unsigned ndim, size_t npts, const double *params, void *thisPtr, unsigned fdim, double *value)
{
  if (fdim != 1) //< Check fdim
  {
    std::cerr << "NNMap_Model::integrand_3Halo: Wrong fdim" << std::endl;
    exit(1);
  };
  // Read out Container
  nnmap_container *container = (nnmap_container *)thisPtr;
  NNMap_Model *nnmap = container->nnmap;
  double theta1 = container->theta1;
  double theta2 = container->theta2;
  double theta3 = container->theta3;
  double type1 = container->type1;
  double type2 = container->type2;

  HOD *hod1, *hod2;
  double *nbar1, *nbar2;
  if (type1 == 1)
  {
    hod1 = nnmap->hod1;
#if GPU
    nbar1 = nnmap->dev_n_bar1;
#else
    nbar1 = nnmap->n_bar1;
#endif
  }
  else
  {
    hod1 = nnmap->hod2;
#if GPU
    nbar1 = nnmap->dev_n_bar2;
#else
    nbar1 = nnmap->n_bar2;
#endif
  }

  if (type2 == 1)
  {
    hod2 = nnmap->hod1;
#if GPU
    nbar2 = nnmap->dev_n_bar1;
#else
    nbar2 = nnmap->n_bar1;
#endif
  }
  else
  {
    hod2 = nnmap->hod2;
#if GPU
    nbar2 = nnmap->dev_n_bar2;
#else
    nbar2 = nnmap->n_bar2;
#endif
  }

#if GPU // Do calculation on GPU

  // Allocate for integration values on device
  double *dev_value;
  CUDA_SAFE_CALL(cudaMalloc(&dev_value, fdim * npts * sizeof(double)));

  // Allocate for integration variables on device
  double *dev_params;
  CUDA_SAFE_CALL(cudaMalloc(&dev_params, npts * ndim * sizeof(double)));

  // Copy integration variables to device
  CUDA_SAFE_CALL(cudaMemcpy(dev_params, params, npts * ndim * sizeof(double), cudaMemcpyHostToDevice));

  // Do calculation
  GPUkernel_3Halo<<<BLOCKS, THREADS>>>(dev_params, theta1, theta2, theta3, npts, hod1, hod2, hod1->dev_params, hod2->dev_params, nnmap->zmin, nnmap->zmax, nnmap->mmin,
                                       nnmap->mmax, nnmap->kmin, nnmap->kmax, nnmap->Nbins, nnmap->dev_g, nnmap->dev_p_lens1,
                                       nnmap->dev_p_lens2, nnmap->dev_w, nnmap->dev_dwdz, nnmap->dev_hmf, nnmap->dev_P_lin,
                                       nnmap->dev_b_h, nnmap->dev_concentration, nnmap->dev_rho_bar, nbar1,
                                       nbar2, dev_value);

  cudaFree(dev_params); //< Clean up

  // Copy values from device to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); //< Clean up

#else // Calculation on CPU

#pragma omp parallel for
  for (unsigned int i = 0; i < npts; i++)
  {
    double l1 = pow(10, params[i * ndim]);
    double l2 = pow(10, params[i * ndim + 1]);
    double phi = params[i * ndim + 2];
    double m1 = pow(10, params[i * ndim + 3]);
    double m2 = pow(10, params[i * ndim + 4]);
    double m3 = pow(10, params[i * ndim + 5]);
    double z = params[i * ndim + 6];
    value[i] = kernel_function_3halo(theta1, theta2, theta3, l1, l2, phi, m1, m2, m3, z, nnmap->zmin, nnmap->zmax, nnmap->mmin,
                                     nnmap->mmax, nnmap->kmin, nnmap->kmax, nnmap->Nbins, hod1, hod2, NULL, NULL, nnmap->g, nnmap->p_lens1,
                                     nnmap->p_lens2, nnmap->w, nnmap->dwdz, nnmap->hmf, nnmap->P_lin, nnmap->b_h,
                                     nnmap->concentration, nnmap->rho_bar, nbar1, nbar2);
  };

#endif
  return 0; // Success :)
}

__device__ __host__ double g3lhalo::NNM::kernel_function_3halo(double theta1, double theta2, double theta3, double l1, double l2,
                                                          double phi, double m1, double m2, double m3, double z, double zmin,
                                                          double zmax, double mmin, double mmax, double kmin, double kmax, int Nbins,
                                                          g3lhalo::HOD *hod1, g3lhalo::HOD *hod2, double *dev_params1, double *dev_params2, const double *g,
                                                          const double *p_lens1, const double *p_lens2, const double *w,
                                                          const double *dwdz, const double *hmf, const double *P_lin,
                                                          const double *b_h, const double *concentration, const double *rho_bar,
                                                          const double *n_bar1, const double *n_bar2)
{
  // Index of z, m1, m2, and m3
  int z_ix = std::round((z - zmin) * Nbins / (zmax - zmin));
  int m1_ix = std::round(std::log(m1 / mmin) * Nbins / std::log(mmax / mmin));
  int m2_ix = std::round(std::log(m2 / mmin) * Nbins / std::log(mmax / mmin));
  int m3_ix = std::round(std::log(m3 / mmin) * Nbins / std::log(mmax / mmin));

  double nbar1, nbar2;
  nbar1 = n_bar1[z_ix];

  nbar2 = n_bar2[z_ix];

  double w_ = w[z_ix]; // Comoving distance [Mpc]
  double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi));
  double k1 = l1 / w_;
  double k2 = l2 / w_;
  double k3 = l3 / w_;

  // 3D Bispectrum
  double Bggd = 1. / nbar1 / nbar2 / rho_bar[z_ix] * hmf[z_ix * Nbins + m1_ix] * hmf[z_ix * Nbins + m2_ix] * hmf[z_ix * Nbins + m3_ix] * m3 * u_NFW(k3, m3, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * G_g(k1, m1, z, hod1, dev_params1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * G_g(k2, m2, z, hod2, dev_params2, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * Bi_lin(k1, k2, cos(phi), z, kmin, kmax, zmin, zmax, Nbins, P_lin) * b_h[z_ix * Nbins + m1_ix] * b_h[z_ix * Nbins + m2_ix] * b_h[z_ix * Nbins + m3_ix];

  // 2D bispcetrum
  double bggk = Bggd / dwdz[z_ix] * g[z_ix] * p_lens1[z_ix] * p_lens2[z_ix] / w_ / w_ / w_ * (1. + z);

  return l1 * l2 * m1 * m2 * m3 * l1 * l2 * apertureFilter(theta1 * l1) * apertureFilter(theta2 * l2) * apertureFilter(theta3 * l3) * bggk;
}

#if GPU

__global__ void g3lhalo::NNM::GPUkernel_3Halo(const double *params, double theta1, double theta2, double theta3, int npts, HOD *hod1, HOD *hod2,
                                         double *dev_params1, double *dev_params2,
                                         double zmin, double zmax, double mmin, double mmax, double kmin, double kmax, int Nbins,
                                         const double *g, const double *p_lens1, const double *p_lens2, const double *w,
                                         const double *dwdz, const double *hmf, const double *P_lin, const double *b_h,
                                         const double *concentration, const double *rho_bar, const double *n_bar1,
                                         const double *n_bar2, double *value)

{
  /// Index of thread
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  // Grid-Stride Loop, so I get npts evaluations
  for (int i = thread_index; i < npts; i += blockDim.x * gridDim.x)
  {
    double l1 = pow(10, params[i * 7]);
    double l2 = pow(10, params[i * 7 + 1]);
    double phi = params[i * 7 + 2];
    double m1 = pow(10, params[i * 7 + 3]);
    double m2 = pow(10, params[i * 7 + 4]);
    double m3 = pow(10, params[i * 7 + 5]);
    double z = params[i * 7 + 6];

    value[i] = kernel_function_3halo(theta1, theta2, theta3, l1, l2, phi, m1, m2, m3, z, zmin, zmax, mmin, mmax, kmin, kmax, Nbins,
                                     hod1, hod2, dev_params1, dev_params2, g, p_lens1, p_lens2, w, dwdz, hmf, P_lin, b_h, concentration, rho_bar, n_bar1,
                                     n_bar2);
  };
}
#endif

__host__ __device__ double g3lhalo::NNM::G_g(double k, double m, double z, g3lhalo::HOD *hod, double *dev_params,
                                        double zmin, double zmax, double mmin, double mmax,
                                        int Nbins, const double *rho_bar, const double *concentration)
{
  double Nc = hod->Ncen(m, dev_params);
  double Ns = hod->Nsat(m, dev_params);
  double f;
#if __CUDA_ARCH__
  f = dev_params[0];
#else
  f = hod->params->f1;
#endif
  return Nc + Ns * u_NFW(k, m, z, f, zmin, zmax, mmin, mmax, Nbins, rho_bar,
                         concentration);
}

__host__ __device__ double g3lhalo::NNM::G_gg(double k1, double k2, double m, double z, g3lhalo::HOD *hod1, g3lhalo::HOD *hod2, double *dev_params1, double *dev_params2,
                                         double A, double epsilon,
                                         double zmin, double zmax, double mmin, double mmax,
                                         int Nbins, const double *rho_bar, const double *concentration, double scale1, double scale2)
{
  double Nc1 = hod1->Ncen(m, dev_params1);
  double Nc2 = hod2->Ncen(m, dev_params2);
  double Ns1 = hod1->Nsat(m, dev_params1);
  double Ns2 = hod2->Nsat(m, dev_params2);
  double Nss = g3lhalo::NsatNsat(m, hod1, hod2, dev_params1, dev_params2, A, epsilon, scale1, scale2);

  double f1, f2;
#if __CUDA_ARCH__
  f1 = dev_params1[0];
  f2 = dev_params2[0];
#else
  f1 = hod1->params->f1;
  f2 = hod2->params->f1;
#endif

  return Nc1 * Ns2 * u_NFW(k2, m, z, f2, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) + Nc2 * Ns1 * u_NFW(k1, m, z, f1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) + Nss * u_NFW(k1, m, z, f1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * u_NFW(k2, m, z, f2, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration);
}
