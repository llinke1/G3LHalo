#include "NMapMap_Model.h"
#include "constants.h"
#include "cubature.h"
#include <iostream>
#include <fstream>

#if GPU
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

#include <algorithm>

g3lhalo::NMapMap_Model::NMapMap_Model(Cosmology *cosmology_, const double &zmin_, const double &zmax_, const double &kmin_, const double &kmax_,
                                      const double &mmin_, const double &mmax_, const int &Nbins_,
                                      double *g_, double *p_lens_, double *w_,
                                      double *dwdz_, double *hmf_, double *P_lin_, double *b_h_, double *concentration_,
                                      HOD *hod_)
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
    p_lens = p_lens_;
    w = w_;
    dwdz = dwdz_;
    hmf = hmf_;
    P_lin = P_lin_;
    b_h = b_h_;
    concentration = concentration_;
    hod = hod_;

    zmin_integral = zmin;
    zmax_integral = zmax - (zmax - zmin) / Nbins;
    mmin_integral = mmin;
    mmax_integral = exp(log(mmin) + log(mmax / mmin) * (Nbins - 1) / Nbins);

#if GPU
    // Allocation of memory for precomputed functions on device
    CUDA_SAFE_CALL(cudaMalloc(&dev_g, Nbins * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(&dev_p_lens, Nbins * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(&dev_w, Nbins * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(&dev_dwdz, Nbins * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(&dev_hmf, Nbins * Nbins * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(&dev_P_lin, Nbins * Nbins * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(&dev_b_h, Nbins * Nbins * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(&dev_concentration, Nbins * Nbins * sizeof(double)));

    // Copying of precomputed functions to device
    CUDA_SAFE_CALL(cudaMemcpy(dev_g, g, Nbins * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dev_p_lens, p_lens, Nbins * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dev_w, w, Nbins * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dev_dwdz, dwdz, Nbins * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dev_hmf, hmf, Nbins * Nbins * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dev_P_lin, P_lin, Nbins * Nbins * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dev_b_h, b_h, Nbins * Nbins * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dev_concentration, concentration, Nbins * Nbins * sizeof(double), cudaMemcpyHostToDevice));

    // Allocation of memory for densities on device
    CUDA_SAFE_CALL(cudaMalloc(&dev_rho_bar, Nbins * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(&dev_n_bar, Nbins * sizeof(double)));

#endif // GPU

    // Allocattion of memory for densities on Host
    rho_bar = (double *)malloc(Nbins * sizeof(double));
    n_bar = (double *)malloc(Nbins * sizeof(double));

#if VERBOSE
    std::cerr << "Finished memory setting" << std::endl;
#endif // VERBOSE

    // Calculate densities (stored in rho_bar, and nbar)
    updateDensities();

#if GPU
    // Copying of densities to device
    CUDA_SAFE_CALL(cudaMemcpy(dev_rho_bar, rho_bar, Nbins * sizeof(double), cudaMemcpyHostToDevice));
#endif
#if VERBOSE
    std::cerr << "Finished initalizing NMapMap" << std::endl;
#endif // VERBOSE
}

g3lhalo::NMapMap_Model::~NMapMap_Model()
{
#if GPU // Free device memory
    cudaFree(dev_g);
    cudaFree(dev_p_lens);
    cudaFree(dev_w);
    cudaFree(dev_dwdz);
    cudaFree(dev_hmf);
    cudaFree(dev_P_lin);
    cudaFree(dev_b_h);
    cudaFree(dev_concentration);
    cudaFree(dev_rho_bar);
    cudaFree(dev_n_bar);
#endif // GPU
    // Free host memory
    free(rho_bar);
    free(n_bar);
#if SCALING
    free(scaling1);
    free(scaling2);
#endif
}

void g3lhalo::NMapMap_Model::updateHOD(g3lhalo::HOD *hod_)
{
    hod = hod_;
    updateDensities();
}

void g3lhalo::NMapMap_Model::updateDensities()
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

        container.hod = hod;
        hcubature_v(1, integrand_nbar, &container, 1, m_min, m_max, 0, 0, 1e-4, ERROR_L1, &result, &error);
        // Set n_bar1 (ln10 is to account for logarithmic integral)
        n_bar[i] = g3lhalo::ln10 * result;
    };

#if VERBOSE
    std::cerr << "Finished calculating densities" << std::endl;
    std::cerr << n_bar[0] << std::endl;
#endif // VERBOSE

#if GPU // Copy to device
    CUDA_SAFE_CALL(cudaMemcpy(dev_n_bar, n_bar, Nbins * sizeof(double), cudaMemcpyHostToDevice));
#endif // GPU

#if VERBOSE
    std::cerr << "Finished density update" << std::endl;
#endif // VERBOSE
}

double g3lhalo::NMapMap_Model::NMapMap_1h(const double &theta1, const double &theta2, const double &theta3)
{

    // Set Container
    nmapmap_container container;
    container.nmapmap = this;
    container.theta1 = theta1;
    container.theta2 = theta2;
    container.theta3 = theta3;

    // Set Integral borders
    // Logarithmic over l1, l2, and m
    double params_min[5] = {std::log10(lmin_integral), std::log10(lmin_integral), phimin_integral, std::log10(mmin_integral), zmin_integral};
    double params_max[5] = {std::log10(lmax_integral), std::log10(lmax_integral), phimax_integral, std::log10(mmax_integral), zmax_integral};

    // Do calculation
    double result, error;
    hcubature_v(1, NMM::integrand_1halo, &container, 5, params_min, params_max, 5000000, 0, 1e-3, ERROR_L1, &result, &error);

    // Return result (Prefactor includes three ln10 terms due to logarithmic integral)
    return result 
    * 9 * pow(cosmology->H0, 4) * pow(cosmology->Omega_m, 2) / 4 / pow(g3lhalo::c, 4) 
    * g3lhalo::ln10 * g3lhalo::ln10 * g3lhalo::ln10 / (2 * g3lhalo::pi) / (2 * g3lhalo::pi) / (2 * g3lhalo::pi);
}

double g3lhalo::NMapMap_Model::NMapMap_2h(const double &theta1, const double &theta2, const double &theta3)
{

    // Set Container
    nmapmap_container container;
    container.nmapmap = this;
    container.theta1 = theta1;
    container.theta2 = theta2;
    container.theta3 = theta3;

    // Set Integral borders
    // Logarithmic over l1, l2, and m
    double params_min[6] = {std::log10(lmin_integral), std::log10(lmin_integral), phimin_integral, std::log10(mmin_integral), std::log10(mmin_integral), zmin_integral};
    double params_max[6] = {std::log10(lmax_integral), std::log10(lmax_integral), phimax_integral, std::log10(mmax_integral), std::log10(mmax_integral), zmax_integral};

    // Do calculation
    double result, error;
    hcubature_v(1, NMM::integrand_2halo, &container, 6, params_min, params_max, 5000000, 0, 1e-3, ERROR_L1, &result, &error);

    // Return result (Prefactor includes four ln10 terms due to logarithmic integral)
    return result * 9 * pow(cosmology->H0, 4) * pow(cosmology->Omega_m, 2) / 4 / pow(g3lhalo::c, 4) * g3lhalo::ln10 * g3lhalo::ln10 * g3lhalo::ln10 * g3lhalo::ln10 / (2 * g3lhalo::pi) / (2 * g3lhalo::pi) / (2 * g3lhalo::pi);
}

double g3lhalo::NMapMap_Model::NMapMap_3h(const double &theta1, const double &theta2, const double &theta3)
{

    // Set Container
    nmapmap_container container;
    container.nmapmap = this;
    container.theta1 = theta1;
    container.theta2 = theta2;
    container.theta3 = theta3;

    // Set Integral borders
    // Logarithmic over l1, l2, and m
    double params_min[7] = {std::log10(lmin_integral), std::log10(lmin_integral), phimin_integral, std::log10(mmin_integral), std::log10(mmin_integral), std::log10(mmin_integral), zmin_integral};
    double params_max[7] = {std::log10(lmax_integral), std::log10(lmax_integral), phimax_integral, std::log10(mmax_integral), std::log10(mmax_integral), std::log10(mmax_integral), zmax_integral};

    // Do calculation
    double result, error;
    hcubature_v(1, NMM::integrand_3halo, &container, 7, params_min, params_max, 5000000, 0, 1e-3, ERROR_L1, &result, &error);

    // Return result (Prefactor includes five ln10 terms due to logarithmic integral)
    return result * 9 * pow(cosmology->H0, 4) * pow(cosmology->Omega_m, 2) / 4 / pow(g3lhalo::c, 4) * g3lhalo::ln10 * g3lhalo::ln10 * g3lhalo::ln10 * g3lhalo::ln10 * g3lhalo::ln10 / (2 * g3lhalo::pi) / (2 * g3lhalo::pi) / (2 * g3lhalo::pi);
}

double g3lhalo::NMapMap_Model::NMapMap(const double &theta1, const double &theta2, const double &theta3)
{
    return NMapMap_1h(theta1, theta2, theta3)    // 1-halo term
           + NMapMap_2h(theta1, theta2, theta3)  // 2-halo term
           + NMapMap_3h(theta1, theta2, theta3); // 3-halo term
}

void g3lhalo::NMapMap_Model::calculateAll(const std::vector<double> &thetas1, const std::vector<double> &thetas2, const std::vector<double> &thetas3,
                                          HOD *hod_, std::vector<double> &results)
{
    updateHOD(hod_);
    int n = thetas1.size();
    for (int i = 0; i < n; i++)
    {
        double theta1 = thetas1.at(i); // Read Thetas
        double theta2 = thetas2.at(i);
        double theta3 = thetas3.at(i);

        results.push_back(NMapMap(theta1, theta2, theta3));
    }
    return;
}

__device__ __host__ double g3lhalo::NMM::kernel_function_1halo(double theta1, double theta2,
                                                               double theta3, double l1, double l2, double phi, double m, double z,
                                                               double zmin, double zmax, double mmin, double mmax, int Nbins,
                                                               g3lhalo::HOD *hod, double *dev_params,
                                                               const double *g, const double *p_lens, const double *w, const double *dwdz,
                                                               const double *hmf, const double *concentration,
                                                               const double *rho_bar, const double *n_bar)
{
    // Get inidices of z and m
    int z_ix = std::round(((z - zmin) * Nbins / (zmax - zmin)));
    int m_ix = std::round((std::log(m / mmin) * Nbins / std::log(mmax / mmin)));

    // Get lens galaxy densities
    double nbar = n_bar[z_ix];

    double w_ = w[z_ix]; //< Comoving distance [Mpc]

    double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi));
    double k1 = l1 / w_;
    double k2 = l2 / w_;
    double k3 = l3 / w_;

    double Nc = hod->Ncen(m, dev_params);
    double Ns = hod->Nsat(m, dev_params);
    double f;
#if __CUDA_ARCH__
    f = dev_params[0];
#else
    f = hod->params->f;
#endif

    // 3D Galaxy-Matter-Matter Bispectrum
    double Bgdd = 1. / nbar / rho_bar[z_ix] / rho_bar[z_ix] * hmf[z_ix * Nbins + m_ix] * m * m * (Nc + Ns * u_NFW(k1, m, z, f, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration)) * u_NFW(k2, m, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * u_NFW(k3, m, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration);

    // 2D Galaxy-Convergence-Convergence Bispectrum
    double bgkk = Bgdd / dwdz[z_ix] * g[z_ix] * g[z_ix] * p_lens[z_ix] / w_ / w_ * (1. + z) * (1. + z);

    return l1 * l2 * m * l1 * l2 * apertureFilter(theta1 * l1) * apertureFilter(theta2 * l2) * apertureFilter(theta3 * l3) * bgkk / nbar;
}

__global__ void g3lhalo::NMM::GPUkernel_1Halo(const double *params, double theta1, double theta2, double theta3,
                                              int npts, g3lhalo::HOD *hod, double *dev_params,
                                              double zmin, double zmax, double mmin, double mmax,
                                              int Nbins, const double *g, const double *p_lens,
                                              const double *w, const double *dwdz, const double *hmf, const double *concentration,
                                              const double *rho_bar, const double *n_bar,
                                              double *value)
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

        value[i] = kernel_function_1halo(theta1, theta2, theta3, l1, l2, phi, m, z, zmin, zmax, mmin, mmax, Nbins, hod, dev_params, g,
                                         p_lens, w, dwdz, hmf, concentration, rho_bar, n_bar);
    };
}

int g3lhalo::NMM::integrand_1halo(unsigned ndim, size_t npts, const double *params, void *thisPtr, unsigned fdim, double *value)
{
    // Check if fdim is correct
    if (fdim != 1)
    {
        std::cerr << "g3lhalo::NMM::integrand_1halo: Wrong fdim" << std::endl;
        exit(1);
    };

    // Read out container
    nmapmap_container *container = (nmapmap_container *)thisPtr;
    NMapMap_Model *nmapmap = container->nmapmap;
    double theta1 = container->theta1;
    double theta2 = container->theta2;
    double theta3 = container->theta3;

    HOD *hod = nmapmap->hod;
    double *nbar;
#if GPU
    nbar = nmapmap->dev_n_bar;
#else
    nbar = nmapmap->n_bar;
#endif

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

    GPUkernel_1Halo<<<BLOCKS, THREADS>>>(dev_params, theta1, theta2, theta3, npts, hod, hod->dev_params, nmapmap->zmin,
                                         nmapmap->zmax,
                                         nmapmap->mmin, nmapmap->mmax, nmapmap->Nbins, nmapmap->dev_g, nmapmap->dev_p_lens,
                                         nmapmap->dev_w, nmapmap->dev_dwdz, nmapmap->dev_hmf, nmapmap->dev_concentration, nmapmap->dev_rho_bar,
                                         nbar, dev_value);

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

        value[i] = kernel_function_1halo(theta1, theta2, theta3, l1, l2, phi, m, z, nmapmap->zmin, nmapmap->zmax, nmapmap->mmin, nmapmap->mmax,
                                         nmapmap->Nbins, hod, NULL, nmapmap->g,
                                         nmapmap->p_lens, nmapmap->w, nmapmap->dwdz, nmapmap->hmf, nmapmap->concentration,
                                         nmapmap->rho_bar, nbar);
    };

#endif
    return 0; // Success :)
}

__device__ __host__ double g3lhalo::NMM::kernel_function_2halo(double theta1, double theta2, double theta3, double l1, double l2,
                                                               double phi, double m1, double m2, double z, double zmin, double zmax,
                                                               double mmin, double mmax, double kmin, double kmax, int Nbins, g3lhalo::HOD *hod,
                                                               double *dev_params,
                                                               const double *g, const double *p_lens,
                                                               const double *w, const double *dwdz, const double *hmf, const double *P_lin,
                                                               const double *b_h, const double *concentration, const double *rho_bar,
                                                               const double *n_bar)
{

    // Get Indices of z, m1, and m2
    int z_ix = std::round((z - zmin) * Nbins / (zmax - zmin));
    int m1_ix = std::round(std::log(m1 / mmin) * Nbins / std::log(mmax / mmin));
    int m2_ix = std::round(std::log(m2 / mmin) * Nbins / std::log(mmax / mmin));

    // Get lens galaxy densities
    double nbar = n_bar[z_ix];

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

    double Nc1 = hod->Ncen(m1, dev_params);
    double Ns1 = hod->Nsat(m1, dev_params);
    double Nc2 = hod->Ncen(m2, dev_params);
    double Ns2 = hod->Nsat(m2, dev_params);
    double f;
#if __CUDA_ARCH__
    f = dev_params[0];
#else
    f = hod->params->f;
#endif

    // 3D Bispectrum

    double Bgdd;
    Bgdd = 1. / nbar / rho_bar[z_ix] / rho_bar[z_ix] * hmf[z_ix * Nbins + m1_ix] * hmf[z_ix * Nbins + m2_ix] * b_h[z_ix * Nbins + m1_ix] * b_h[z_ix * Nbins + m2_ix] * (m1 * m2 * u_NFW(k2, m1, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * u_NFW(k3, m2, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * (Nc2 + Ns2 * u_NFW(k1, m2, z, f, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration)) * Pk2 + m1 * m2 * u_NFW(k3, m1, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * u_NFW(k2, m2, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * (Nc2 + Ns2 * u_NFW(k1, m2, z, f, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration)) * Pk3 + m2 * m2 * u_NFW(k2, m1, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * u_NFW(k3, m2, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * (Nc1 + Ns1 * u_NFW(k1, m1, z, f, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration)) * Pk1);

    // 2D Galaxy-Convergence-Convergence Bispectrum
    double bgkk = Bgdd / dwdz[z_ix] * g[z_ix] * g[z_ix] * p_lens[z_ix] / w_ / w_ * (1. + z) * (1. + z);

    return l1 * l2 * m1 * m2 * l1 * l2 * apertureFilter(theta1 * l1) * apertureFilter(theta2 * l2) * apertureFilter(theta3 * l3) * bgkk / nbar;
}

__global__ void g3lhalo::NMM::GPUkernel_2Halo(const double *params, double theta1, double theta2, double theta3, int npts, g3lhalo::HOD *hod,
                                              double *dev_params, double zmin, double zmax, double mmin, double mmax, double kmin,
                                              double kmax, int Nbins, const double *g, const double *p_lens,
                                              const double *w, const double *dwdz, const double *hmf, const double *P_lin,
                                              const double *b_h, const double *concentration, const double *rho_bar, const double *n_bar, double *value)
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

        value[i] = kernel_function_2halo(theta1, theta2, theta3, l1, l2, phi, m1, m2, z, zmin, zmax, mmin, mmax, kmin, kmax, Nbins, hod, dev_params, g, p_lens,
                                         w, dwdz, hmf, P_lin, b_h, concentration, rho_bar, n_bar);
    };
}

int g3lhalo::NMM::integrand_2halo(unsigned ndim, size_t npts, const double *params, void *thisPtr, unsigned fdim, double *value)
{
    if (fdim != 1) // Check fdim
    {
        std::cerr << "NNMap_Model::integrand_nbar: Wrong fdim" << std::endl;
        exit(1);
    };

    // Read out Container
    nmapmap_container *container = (nmapmap_container *)thisPtr;
    NMapMap_Model *nmapmap = container->nmapmap;
    double theta1 = container->theta1;
    double theta2 = container->theta2;
    double theta3 = container->theta3;
    HOD *hod = nmapmap->hod;
    double *nbar;
#if GPU
    nbar = nmapmap->dev_n_bar;
#else
    nbar = nmapmap->n_bar;
#endif

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

        GPUkernel_2Halo<<<BLOCKS, THREADS>>>(dev_params, theta1, theta2, theta3, n_it, hod, hod->dev_params, nmapmap->zmin,
                                             nmapmap->zmax, nmapmap->mmin, nmapmap->mmax, nmapmap->kmin, nmapmap->kmax, nmapmap->Nbins,
                                             nmapmap->dev_g, nmapmap->dev_p_lens, nmapmap->dev_w, nmapmap->dev_dwdz,
                                             nmapmap->dev_hmf, nmapmap->dev_P_lin, nmapmap->dev_b_h, nmapmap->dev_concentration,
                                             nmapmap->dev_rho_bar, nbar, dev_value);

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
        value[i] = kernel_function_2halo(theta1, theta2, theta3, l1, l2, phi, m1, m2, z, nmapmap->zmin, nmapmap->zmax, nmapmap->mmin,
                                         nmapmap->mmax, nmapmap->kmin, nmapmap->kmax, nmapmap->Nbins, hod, NULL, nmapmap->g,
                                         nmapmap->p_lens, nmapmap->w, nmapmap->dwdz, nmapmap->hmf, nmapmap->P_lin, nmapmap->b_h,
                                         nmapmap->concentration, nmapmap->rho_bar, nbar);
    };

#endif
    return 0; // Success :)
}

__device__ __host__ double g3lhalo::NMM::kernel_function_3halo(double theta1, double theta2, double theta3, double l1, double l2,
                                                               double phi, double m1, double m2, double m3, double z, double zmin,
                                                               double zmax, double mmin, double mmax, double kmin, double kmax, int Nbins,
                                                               g3lhalo::HOD *hod, double *dev_params, const double *g,
                                                               const double *p_lens, const double *w,
                                                               const double *dwdz, const double *hmf, const double *P_lin,
                                                               const double *b_h, const double *concentration, const double *rho_bar,
                                                               const double *n_bar)
{
    // Index of z, m1, m2, and m3
    int z_ix = std::round((z - zmin) * Nbins / (zmax - zmin));
    int m1_ix = std::round(std::log(m1 / mmin) * Nbins / std::log(mmax / mmin));
    int m2_ix = std::round(std::log(m2 / mmin) * Nbins / std::log(mmax / mmin));
    int m3_ix = std::round(std::log(m3 / mmin) * Nbins / std::log(mmax / mmin));

    double nbar;
    nbar = n_bar[z_ix];

    double w_ = w[z_ix]; // Comoving distance [Mpc]
    double l3 = sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * cos(phi));
    double k1 = l1 / w_;
    double k2 = l2 / w_;
    double k3 = l3 / w_;

    double Nc = hod->Ncen(m3, dev_params);
    double Ns = hod->Nsat(m3, dev_params);
    double f;
#if __CUDA_ARCH__
    f = dev_params[0];
#else
    f = hod->params->f;
#endif

    // 3D Bispectrum
    double Bgdd = 1. / nbar / rho_bar[z_ix] / rho_bar[z_ix] * hmf[z_ix * Nbins + m1_ix] * hmf[z_ix * Nbins + m2_ix] * hmf[z_ix * Nbins + m3_ix] * b_h[z_ix * Nbins + m1_ix] * b_h[z_ix * Nbins + m2_ix] * b_h[z_ix * Nbins + m3_ix] * m1 * m2 * u_NFW(k2, m1, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * u_NFW(k3, m2, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration) * (Nc + Ns * u_NFW(k1, m2, z, f, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration)) * Bi_lin(k1, k2, cos(phi), z, kmin, kmax, zmin, zmax, Nbins, P_lin);

    // 2D bispcetrum
    double bgkk = Bgdd / dwdz[z_ix] * g[z_ix] * g[z_ix] * p_lens[z_ix] / w_ / w_ * (1. + z) * (1. + z);

    return l1 * l2 * m1 * m2 * m3 * l1 * l2 * apertureFilter(theta1 * l1) * apertureFilter(theta2 * l2) * apertureFilter(theta3 * l3) * bgkk / nbar;
}

__global__ void g3lhalo::NMM::GPUkernel_3Halo(const double *params, double theta1, double theta2, double theta3, int npts, HOD *hod,
                                              double *dev_params,
                                              double zmin, double zmax, double mmin, double mmax, double kmin, double kmax, int Nbins,
                                              const double *g, const double *p_lens, const double *w,
                                              const double *dwdz, const double *hmf, const double *P_lin, const double *b_h,
                                              const double *concentration, const double *rho_bar, const double *n_bar, double *value)

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
                                         hod, dev_params, g, p_lens, w, dwdz, hmf, P_lin, b_h, concentration, rho_bar, n_bar);
    };
}

int g3lhalo::NMM::integrand_3halo(unsigned ndim, size_t npts, const double *params, void *thisPtr, unsigned fdim, double *value)
{
    if (fdim != 1) //< Check fdim
    {
        std::cerr << "NMMap::integrand_3Halo: Wrong fdim" << std::endl;
        exit(1);
    };
    // Read out Container
    nmapmap_container *container = (nmapmap_container *)thisPtr;
    NMapMap_Model *nmapmap = container->nmapmap;
    double theta1 = container->theta1;
    double theta2 = container->theta2;
    double theta3 = container->theta3;

    HOD *hod = nmapmap->hod;
    double *nbar;
#if GPU
    nbar = nmapmap->dev_n_bar;
#else
    nbar = nmapmap->n_bar;
#endif

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
    GPUkernel_3Halo<<<BLOCKS, THREADS>>>(dev_params, theta1, theta2, theta3, npts, hod, hod->dev_params, nmapmap->zmin, nmapmap->zmax, nmapmap->mmin,
                                         nmapmap->mmax, nmapmap->kmin, nmapmap->kmax, nmapmap->Nbins, nmapmap->dev_g, nmapmap->dev_p_lens,
                                         nmapmap->dev_w, nmapmap->dev_dwdz, nmapmap->dev_hmf, nmapmap->dev_P_lin,
                                         nmapmap->dev_b_h, nmapmap->dev_concentration, nmapmap->dev_rho_bar, nbar, dev_value);

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
        value[i] = kernel_function_3halo(theta1, theta2, theta3, l1, l2, phi, m1, m2, m3, z, nmapmap->zmin, nmapmap->zmax, nmapmap->mmin,
                                         nmapmap->mmax, nmapmap->kmin, nmapmap->kmax, nmapmap->Nbins, hod, NULL, nmapmap->g, nmapmap->p_lens,
                                         nmapmap->w, nmapmap->dwdz, nmapmap->hmf, nmapmap->P_lin, nmapmap->b_h,
                                         nmapmap->concentration, nmapmap->rho_bar, nbar);
    };

#endif
    return 0; // Success :)
}
