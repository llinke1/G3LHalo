#ifndef G3LHALO_GENERALFUNCTIONS_H
#define G3LHALO_GENERALFUNCTIONS_H

#include "HOD.h"

namespace g3lhalo
{
  /**
   * Fourier Transform of NFW profile
   * @param k wave vector [1/Mpc]
   * @param m Halo mass [Msun]
   * @param z redshift
   * @param f concentration factor
   * @param zmin Minimal Redshift of binning
   * @param zmax Maximal redshift of binning
   * @param mmin1 Parameter mmin of HOD1 [Msun]
   * @param mmin2 Parameter mmin of HOD2 [Msun]
   * @param Nbins Number of bins
   * @param Array contain matter density
   * @param Array containing concentration
   */
  __host__ __device__ double u_NFW(double k, double m, double z, double f,
                                   double zmin, double zmax, double mmin, double mmax,
                                   int Nbins, const double *rho_bar, const double *concentration);

  /**
   * Virial radius (defined with 200 rho_crit overdensity)
   * @param m halo mass [Msun]
   * @param z redshift
   * @param zmin Minimal Redshift of binning
   * @param zmax Maximal redshift of binning
   * @param Nbins Number of bins
   * @param Array contain matter density
   */
  __host__ __device__ double r_200(double m, double z, double zmin, double zmax, int Nbins, const double *rho_bar);

  /**
   * Fouriertransform of exponential Aperture Filter (Crittenden 2002)
   * @param eta Parameter, should be Theta*k
   */
  __host__ __device__ double apertureFilter(double eta);

  /**
   * Approximation to Si(x) and Ci(x) Functions
   * Same as GSL implementation, because they are not implemented in CUDA
   * @param x x value
   * @param si will contain Si(x)
   * @param ci will contain Ci(x)
   */
  __host__ __device__ void SiCi(double x, double &si, double &ci);

  /**
   * Kernel for Treelevel bispectrum
   * @param k1 k1 [1/Mpc]
   * @param k2 k2 [1/Mpc]
   * @param cosphi Cosine of angle between k1 and k2
   */
  __host__ __device__ double F(double k1, double k2, double cosphi);

  /**
   * Treelevel Bispectrum
   * \f $2 ( F(k_1, k_2) P(k_1) P(k_2) + F(k_1, k_3) P(k_1) P(k_3) + F(k_2, k_3) P(k_2) P(k_3) )$\f
   * @param k1 k1 [1/Mpc]
   * @param k2 k2 [1/Mpc]
   * @param cosphi Cosine of angle between k1 and k2
   * @param z redshift
   * @param kmin Minimal k for binning [1/Mpc]
   * @param kmax Maximal k for binning [1/Mpc]
   * @param zmin Minimal Redshift of binning
   * @param zmax Maximal Redshift of binning
   * @param Nbins Number of bins
   * @param P_lin array containing linear Powerspectrum
   */
  __host__ __device__ double Bi_lin(double k1, double k2, double cosphi, double z,
                                    double kmin, double kmax, double zmin, double zmax,
                                    int Nbins, const double *P_lin);

#if GPU

  /**
   * GPU Kernel function for calculation of galaxy number density
   * @param ms halo masses over which to integrate [Msun]
   * @param z redshift at which to evaluate number density
   * @param npts Number of integration points
   * @param hod HOD object containing parameters
   * @param dev_params Pointer to HOD parameters on device, should be the same as in hod object
   * @param zmin Minimal redshift of binning
   * @param zmax Maximal redshift of binning
   * @param mmin Minimal halomass of binning [Msun]
   * @param mmax Maximal halomass of binning [Msun]
   * @param Nbins Number of bins
   * @param hmf tabularized halo mass function
   * @param value Output parameter, will contain galaxy number density
   * */
  __global__ void GPUkernel_nbar(const double *ms, double z, int npts, g3lhalo::HOD *hod, double *dev_params,
                                 double zmin, double zmax,
                                 double mmin, double mmax, int Nbins, const double *hmf,
                                 double *value);
#endif

  /**
   * Kernel function for calculation of galaxy number density
   * @param m halo mass [Msun]
   * @param z redshift
   * @param hod HOD object containing parameters
   * @param dev_params Pointer to HOD parameters on device, should be the same as in hod object
   * @param zmin Minimal redshift of binning
   * @param zmax Maximal redshift of binning
   * @param mmin Minimal halomass of binning [Msun]
   * @param mmax Maximal halomass of binning [Msun]
   * @param Nbins Number of bins
   * @param hmf tabularized halo mass function
   */
  __device__ __host__ double kernel_function_nbar(double m, double z, g3lhalo::HOD *hod, double *dev_params,
                                                  double zmin, double zmax,
                                                  double mmin, double mmax, int Nbins, const double *hmf);

/**
 * Integrand for galaxy number density for the use with cubature
 * @param ndim Dimensionality of integral (must be 1 here)
 * @param npts Number of integration points
 * @param m Array containing halo masses of integration points
 * @param thisPtr Pointer to integration container (of form n_z_container)
 * @param fdim Dimension of function (must be 1 here)
 * @param value Will contain integrand values
*/
  int integrand_nbar(unsigned ndim, size_t npts, const double *m, void *thisPtr, unsigned fdim, double *value);

/**
 * Structure for galaxy number density integration
*/
  struct n_z_container
  {
    HOD *hod;
    double z;
    double zmin, zmax, mmin, mmax;
    int Nbins;
    double *hmf, *dev_hmf;
  };
}

#endif // G3LHALO_GENERALFUNCTIONS_H