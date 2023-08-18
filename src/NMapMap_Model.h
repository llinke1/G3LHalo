#ifndef G3LHALO_NMAPMAP_MODEL_H
#define G3LHALO_NMAPMAP_MODEL_H

#include "Params.h"
#include "Function.h"
#include "constants.h"
#include "HOD.h"
#include "generalFunctions.h"

#if GPU
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cuda_helpers.h"
#endif

#include <curand.h>
#include <curand_kernel.h>
#include "Cosmology.h"
#include <fstream>

namespace g3lhalo
{
  /**
   * Class calculating NMapMap from G3L Halomodel
   *
   * @author Laila Linke laila.linke@uibk.ac.at
   */
  class NMapMap_Model
  {
  public:
    /// Cosmological parameters
    Cosmology *cosmology;
    /// Binning for lookup tables
    double zmin, zmax; //<Redshifts
    double kmin, kmax; //< wavevector [1/Mpc]
    double mmin, mmax; //< Halo mass [Msun]
    int Nbins;         //< Number of bins

    /// Integral borders
    double zmin_integral, zmax_integral;      //< Redshifts
    double mmin_integral, mmax_integral;      //< Halo mass [Msun]
    double phimin_integral = 0;               //< Phi [rad]
    double phimax_integral = 2 * g3lhalo::pi; //< Phi [rad]
    double lmin_integral = 0.1;               //< l [1/rad]
    double lmax_integral = 1e6;               //< l [1/rad]

    /// Precomputed Lookup tables
    double *g;             //< Lensing efficiency
    double *p_lens;        //< Lens redshift distribution
    double *w;             //< Comoving distance [Mpc]
    double *dwdz;          //< Derivative of Comoving distance wrt redshift [Mpc]
    double *hmf;           //< Halo mass function [1/Mpc^3/Msun]
    double *P_lin;         //< Linear Power spectrum [1/Mpc^3]
    double *b_h;           //< Linear Halo bias
    double *concentration; //< Concentration of NFW profiles

    /// Lookup tables that are updated in initialization
    double *rho_bar; //< Matter density [Msun/Mpc^3]
    double *n_bar;   //< Galaxy number density [1/Mpc^3]

#if GPU
    /// Precomputed Lookup tables on device (are set in initialization)
    double *dev_g;                     //< Lensing efficiency
    double *dev_p_lens; //< Lens redshift distribution
    double *dev_w;                     //< Comoving distance [Mpc]
    double *dev_dwdz;                  //< Derivative of comoving distance wrt redshift [Mpc]
    double *dev_hmf;                   //< Halo mass function [1/Mpc^3/Msun]
    double *dev_P_lin;                 //< Linear Power spectrum [1/Mpc^3]
    double *dev_b_h;                   //< Linear Halo bias
    double *dev_concentration;         //< Concentration of NFW profiles


    /// Lookup tables that are updated in initialization
    double *dev_rho_bar;             //< Matter density [Msun /Mpc^3]
    double *dev_n_bar; //< Galaxy number density [1/Mpc^3]
#endif

    HOD *hod;

    NMapMap_Model(){};

    NMapMap_Model(Cosmology *cosmology_, const double &zmin_, const double &zmax_, const double &kmin_, const double &kmax_,
                  const double &mmin_, const double &mmax_, const int &Nbins_,
                  double *g_, double *p_lens_, double *w_,
                  double *dwdz_, double *hmf_, double *P_lin_, double *b_h_, double *concentration_,
                  HOD *hod_);

    ~NMapMap_Model();

    void updateHOD(HOD *hod_);

    void updateDensities();

    double NMapMap_1h(const double &theta1, const double &theta2, const double &theta3);
    double NMapMap_2h(const double &theta1, const double &theta2, const double &theta3);
    double NMapMap_3h(const double &theta1, const double &theta2, const double &theta3);

    double NMapMap(const double &theta1, const double &theta2, const double &theta3);

    void calculateAll(const std::vector<double> &thetas1, const std::vector<double> &thetas2, const std::vector<double> &thetas3,
                      HOD *hod, std::vector<double> &results);
  };

  namespace NMM
  {
    __device__ __host__ double kernel_function_1halo(double theta1, double theta2, double theta3, double l1, double l2, double phi, double m, double z,
                                                     double zmin, double zmax, double mmin, double mmax, int Nbins,
                                                     HOD *hod, double *dev_params, 
                                                     const double *g, const double *p_lens, const double *w, const double *dwdz,
                                                     const double *hmf, const double *concentration,
                                                     const double *rho_bar, const double *n_bar);

    __global__ void GPUkernel_1Halo(const double *params, double theta1, double theta2, double theta3, int npts,
                                    HOD *hod, double *dev_params,
                                    double zmin, double zmax, double mmin, double mmax,
                                    int Nbins, const double *g, const double *p_lens, const double *w,
                                    const double *dwdz, const double *hmf, const double *concentration, const double *rho_bar, const double *n_bar,
                                    double *value);

    int integrand_1halo(unsigned ndim, size_t npts, const double *params, void *thisPtr, unsigned fdim, double *value);

    __device__ __host__ double kernel_function_2halo(double theta1, double theta2, double theta3, double l1, double l2, double phi,
                                                     double m1, double m2, double z, double zmin, double zmax, double mmin, double mmax,
                                                     double kmin, double kmax, int Nbins, HOD *hod, double *dev_params,
                                                     const double *g, const double *p_lens, const double *w, const double *dwdz, const double *hmf,
                                                     const double *P_lin, const double *b_h, const double *concentration,
                                                     const double *rho_bar, const double *n_bar);

    __global__ void GPUkernel_2Halo(const double *params, double theta1, double theta2, double theta3, int npts, HOD *hod, double *dev_params,
                                    double zmin, double zmax, double mmin, double mmax, double kmin, double kmax, int Nbins,
                                    const double *g, const double *p_lens, const double *w, const double *dwdz,
                                    const double *hmf, const double *P_lin, const double *b_h, const double *concentration,
                                    const double *rho_bar, const double *n_bar,
                                    double *value);

    int integrand_2halo(unsigned ndim, size_t npts, const double *params, void *thisPtr, unsigned fdim, double *value);

    __device__ __host__ double kernel_function_3halo(double theta1, double theta2, double theta3, double l1, double l2, double phi,
                                                     double m1, double m2, double m3, double z, double zmin, double zmax, double mmin,
                                                     double mmax, double kmin, double kmax, int Nbins, HOD *hod, double *dev_params,
                                                     const double *g, const double *p_lens, const double *w,
                                                     const double *dwdz, const double *hmf, const double *P_lin, const double *b_h,
                                                     const double *concentration, const double *rho_bar, const double *n_bar1);

    __global__ void GPUkernel_3Halo(const double *params, double theta1, double theta2, double theta3, int npts, HOD *hod, double *dev_params,
                                    double zmin, double zmax,
                                    double mmin, double mmax, double kmin, double kmax, int Nbins, const double *g,
                                    const double *p_lens, const double *w, const double *dwdz, const double *hmf,
                                    const double *P_lin, const double *b_h, const double *concentration, const double *rho_bar,
                                    const double *n_bar, double *value);

    int integrand_3halo(unsigned ndim, size_t npts, const double *params, void *thisPtr, unsigned fdim, double *value);

  }

  struct nmapmap_container
  {
    NMapMap_Model *nmapmap;
    double theta1, theta2, theta3;
  };

}

#endif // G3LHALO_NMAPMAP_MODEL_H