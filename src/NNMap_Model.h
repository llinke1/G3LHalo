#ifndef G3LHALO_NNMAP_MODEL_H
#define G3LHALO_NNMAP_MODEL_H

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
   * Class calculating NNMap from G3L Halomodel
   *
   * @author Laila Linke llinke@astro.uni-bonn.de
   */
  class NNMap_Model
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
    double *g;                 //< Lensing efficiency
    double *p_lens1, *p_lens2; //< Lens redshift distribution
    double *w;                 //< Comoving distance [Mpc]
    double *dwdz;              //< Derivative of Comoving distance wrt redshift [Mpc]
    double *hmf;               //< Halo mass function [1/Mpc^3/Msun]
    double *P_lin;             //< Linear Power spectrum [1/Mpc^3]
    double *b_h;               //< Linear Halo bias
    double *concentration;     //< Concentration of NFW profiles
#if SCALING
    double *scaling1; //<Values for rescaling of sigma^2 for a type satellites
    double *scaling2; //<Values for rescaling of sigma^2 for b type satellites
#endif

    /// Lookup tables that are updated in initialization
    double *rho_bar;         //< Matter density [Msun/Mpc^3]
    double *n_bar1, *n_bar2; //< Galaxy number density [1/Mpc^3]

#if GPU
    /// Precomputed Lookup tables on device (are set in initialization)
    double *dev_g;                     //< Lensing efficiency
    double *dev_p_lens1, *dev_p_lens2; //< Lens redshift distribution
    double *dev_w;                     //< Comoving distance [Mpc]
    double *dev_dwdz;                  //< Derivative of comoving distance wrt redshift [Mpc]
    double *dev_hmf;                   //< Halo mass function [1/Mpc^3/Msun]
    double *dev_P_lin;                 //< Linear Power spectrum [1/Mpc^3]
    double *dev_b_h;                   //< Linear Halo bias
    double *dev_concentration;         //< Concentration of NFW profiles
#if SCALING
    double *dev_scaling1; //<Values for rescaling of sigma^2 for a type satellites
    double *dev_scaling2; //<Values for rescaling of sigma^2 for b type satellites
#endif

    /// Lookup tables that are updated in initialization
    double *dev_rho_bar;             //< Matter density [Msun /Mpc^3]
    double *dev_n_bar1, *dev_n_bar2; //< Galaxy number density [1/Mpc^3]
#endif

    /// Model parameters
    // Params *params;

    double A;
    double epsilon;

    HOD *hod1;
    HOD *hod2;

    /// Empty constructor
    NNMap_Model(){};

    /**
     * Constructor from values
     * Sets Parameter members to the values provided and calculates matter- and galaxy densities
     * If GPU is set to true, copies look-up tables to device memory (including memory allocation)
     *
     * @param cosmology_ Cosmological parameters (flat LCDM)
     * @param zmin_ Minimal redshift for binning
     * @param zmax_ Maximal redshift for binning
     * @param kmin_ Minimal k for binning [1/Mpc]
     * @param kmax_ Maximal k for binning [1/Mpc]
     * @param mmin_ Minimal halomass for binning [Msun]
     * @param mmax_ Maximal halomass for binning [Msun]
     * @param Nbins_ Number of bins
     * @param g_ Lensing efficiency
     * @param p_lens1_ Lens galaxy redshift distribution
     * @param p_lens2_ Lens galaxy redshift distribution
     * @param w_ Comoving distance [Mpc]
     * @param dwdz_ Derivative of comoving distance wrt redshift [Mpc]
     * @param hmf_ Halo mass function [1/Mpc^3/Msun]
     * @param P_lin_ Linear Power spectrum [1/Mpc^3]
     * @param b_h_ Linear Halo bias
     * @param concentration_ Concentration of NFW profiles
     * @param params_ HOD parameters
     */
    NNMap_Model(Cosmology *cosmology_, const double &zmin_, const double &zmax_, const double &kmin_, const double &kmax_,
                const double &mmin_, const double &mmax_, const int &Nbins_,
                double *g_, double *p_lens1_, double *p_lens2_, double *w_,
                double *dwdz_, double *hmf_, double *P_lin_, double *b_h_, double *concentration_,
                HOD *hod1_, HOD *hod2_, double A_, double epsilon_);

#if SCALING

    /**
     * Constructor from values
     * Sets Parameter members to the values provided and calculates matter- and galaxy densities
     * If GPU is set to true, copies look-up tables to device memory (including memory allocation)
     *
     * @param cosmology_ Cosmological parameters (flat LCDM)
     * @param zmin_ Minimal redshift for binning
     * @param zmax_ Maximal redshift for binning
     * @param kmin_ Minimal k for binning [1/Mpc]
     * @param kmax_ Maximal k for binning [1/Mpc]
     * @param mmin_ Minimal halomass for binning [Msun]
     * @param mmax_ Maximal halomass for binning [Msun]
     * @param Nbins_ Number of bins
     * @param g_ Lensing efficiency
     * @param p_lens1_ Lens galaxy redshift distribution
     * @param p_lens2_ Lens galaxy redshift distribution
     * @param w_ Comoving distance [Mpc]
     * @param dwdz_ Derivative of comoving distance wrt redshift [Mpc]
     * @param hmf_ Halo mass function [1/Mpc^3/Msun]
     * @param P_lin_ Linear Power spectrum [1/Mpc^3]
     * @param b_h_ Linear Halo bias
     * @param concentration_ Concentration of NFW profiles
     * @param scaling1_ Rescaling for type a satellites
     * @param scaling2_ Rescaling for type b satellites
     * @param params_ HOD parameters
     */
    // NNMap_Model(Cosmology *cosmology_, const double &zmin_, const double &zmax_, const double &kmin_, const double &kmax_, const double &mmin_, const double &mmax_, const int &Nbins_,
    //             double *g_, double *p_lens1_, double *p_lens2_, double *w_, double *dwdz_, double *hmf_, double *P_lin_, double *b_h_, double *concentration_, double *scaling1_, double *scaling2_, Params *params_);

    NNMap_Model(Cosmology *cosmology_, const double &zmin_, const double &zmax_, const double &kmin_, const double &kmax_,
                const double &mmin_, const double &mmax_, const int &Nbins_,
                double *g_, double *p_lens1_, double *p_lens2_, double *w_,
                double *dwdz_, double *hmf_, double *P_lin_, double *b_h_, double *concentration_,
                double *scaling1_, double *scaling2_, HOD *hod1_, HOD *hod2_, double A_, double epsilon_);
#endif

    /**
     * Destructor
     * Frees memory for densities and, if GPU is set to true, all device memory
     */
    ~NNMap_Model();

    /**
     * Changes HOD params and updates the galaxy and matter densities
     * @params params_ New HOD parameters
     */
    // void updateParams(Params *params_);

    void updateHODs(HOD *hod1_, HOD *hod2_);

    /**
     * Calculates matter and galaxy densities for current HOD parameters
     * Uses logarithmic integral over halo masses
     */
    void updateDensities();

  public:
    /**
     * Calculates 1 Halo term of NNMap
     * @param theta1 First aperture radius [rad]
     * @param theta2 Second aperture radius [rad]
     * @param theta3 Third aperture radius [rad]
     * @param type1 Type of first galaxy (either 1 or 2)
     * @param type2 Type of second galaxy (either 1 or 2)
     * @return 1 Halo term of NNMap
     */
    double NNMap_1h(const double &theta1, const double &theta2, const double &theta3, const int &type1, const int &type2);

    /**
     * Calculates 2 Halo term of NNMap
     * @param theta1 First aperture radius [rad]
     * @param theta2 Second aperture radius [rad]
     * @param theta3 Third aperture radius [rad]
     * @param type1 Type of first galaxy (either 1 or 2)
     * @param type2 Type of second galaxy (either 1 or 2)
     * @return 2 Halo term of NNMap
     */
    double NNMap_2h(const double &theta1, const double &theta2, const double &theta3, const int &type1, const int &type2);

    /**
     * Calculates 3 Halo term of NNMap
     * @param theta1 First aperture radius [rad]
     * @param theta2 Second aperture radius [rad]
     * @param theta3 Third aperture radius [rad]
     * @param type1 Type of first galaxy (either 1 or 2)
     * @param type2 Type of second galaxy (either 1 or 2)
     * @return 3 Halo term of NNMap
     */
    double NNMap_3h(const double &theta1, const double &theta2, const double &theta3, const int &type1, const int &type2);

    /**
     * Calculates complete NNMap
     * @param theta1 First aperture radius [rad]
     * @param theta2 Second aperture radius [rad]
     * @param theta3 Third aperture radius [rad]
     * @param type1 Type of first galaxy (either 1 or 2)
     * @param type2 Type of second galaxy (either 1 or 2)
     * @return Complete NNMap
     */
    double NNMap(const double &theta1, const double &theta2, const double &theta3, const int &type1, const int &type2);

    /**
     * Calculates complete NNMap (N1N2, N1N1, and N2N2) for range of aperture radii and specified params
     * First updates Params and densities, then does calculation and writes it into results-vector
     * @param thetas1 First aperture radii [rad]
     * @param thetas2 Second aperture radii [rad]
     * @param thetas3 Third aperture radii [rad]
     * @param results COntainer for results (N1N2M, N1N1M, N2N2M)
     */
    void calculateAll(const std::vector<double> &thetas1, const std::vector<double> &thetas2, const std::vector<double> &thetas3,
                      HOD *hod1, HOD *hod2, double A, double epsilon, std::vector<double> &results);


  };

  namespace NNM
  {
    __device__ __host__ double kernel_function_1halo(double theta1, double theta2, double theta3, double l1, double l2, double phi, double m, double z,
                                                     double zmin, double zmax, double mmin, double mmax, int Nbins,
                                                     HOD *hod1, HOD *hod2, double *dev_params1, double *dev_params2, double A, double epsilon,
                                                     const double *g, const double *p_lens1,
                                                     const double *p_lens2, const double *w, const double *dwdz,
                                                     const double *hmf, const double *concentration,
                                                     const double *rho_bar, const double *n_bar1, const double *n_bar2, const double *scaling1 = NULL, const double *scaling2 = NULL);

    __global__ void GPUkernel_1Halo(const double *params, double theta1, double theta2, double theta3, int npts,
                                    HOD *hod1, HOD *hod2, double *dev_params1, double *dev_params2, double A, double epsilon,
                                    double zmin, double zmax, double mmin, double mmax,
                                    int Nbins, const double *g, const double *p_lens1, const double *p_lens2, const double *w,
                                    const double *dwdz, const double *hmf, const double *concentration, const double *rho_bar, const double *n_bar1, const double *n_bar2,
                                    double *value, const double *scaling1 = NULL, const double *scaling2 = NULL);

    /**
     * Integrand for 1Halo term of NNM for use with cubature
     * @param ndim Dimension of integral (5 in this case)
     * @param npts Number of integration points
     * @param params Array of integration x-values
     * @param thisPtr Pointer to integration container
     * @param fdim Dimension of integrand function (1 in this case)
     * @param value Array, which will contain the integration values
     */
    int integrand_1halo(unsigned ndim, size_t npts, const double *params, void *thisPtr, unsigned fdim, double *value);

    /**
     * Integrand for 2Halo term of NNM for use with cubature
     * @param ndim Dimension of integral (6 in this case)
     * @param npts Number of integration points
     * @param params Array of integration x-values
     * @param thisPtr Pointer to integration container
     * @param fdim Dimension of integrand function (1 in this case)
     * @param value Array, which will contain the integration values
     */
    int integrand_2halo(unsigned ndim, size_t npts, const double *params, void *thisPtr, unsigned fdim, double *value);

    /**
     * Kernel function for 2Halo term of NNM
     * @param theta1 Aperture Radius [rad]
     * @param theta2 Aperture Radius [rad]
     * @param theta3 Aperture Radius [rad]
     * @param l1 l parameter [1/rad]
     * @param l2 l parameter [1/rad]
     * @param phi phi parameter [rad]
     * @param m1 Halo mass1 [Msun]
     * @param m2 Halo mass2 [Msun]
     * @param z Redshift
     * @param zmin Minimal Redshift of binning
     * @param zmax Maximal Redshift of binning
     * @param mmin Minimal Halomass for binning [Msun]
     * @param mmax Maximal Halomass for binning [Msun]
     * @param Nbins Number of bins
     * @param type1 Galaxy type (1 or 2)
     * @param type2 Galaxy type (1 or 2)
     * @param f1 Parameter f of HOD 1
     * @param f2 Parameter f of HOD2
     * @param alpha1 Parameter alpha of HOD1
     * @param alpha2 Parameter alpha of HOD2
     * @param mmin1 Parameter mmin of HOD1 [Msun]
     * @param mmin2 Parameter mmin of HOD2 [Msun]
     * @param sigma1 Parameter sigma of HOD1
     * @param sigma2 Parameter sigma of HOD2
     * @param mprime1 Parameter mprime of HOD1 [Msun]
     * @param mprime2 Parameter mprime of HOD2 [Msun]
     * @param beta1 Parameter beta of HOD
     * @param beta2 Parameter beta of HOD
     * @param A Parameter A of halo model
     * @param epsilon Parameter epsilon of halo model
     * @param g Array containing precomputed lensing efficiency
     * @param p_lens1 Array containing lens redshift distribution
     * @param p_lens2 Array containing lens redshift distribution
     * @param w Array containing comoving distance [Mpc]
     * @param dwdz Array containing derivative of comoving distance [Mpc]
     * @param hmf Array containing precomputed HMF [1/Mpc^3/Msun]
     * @param concentration Array containing concentration of NFW profiles
     * @param rho_bar Array containing matter density [Msun/Mpc^3]
     * @param n_bar1 Array containing galaxy number density [1/Mpc^3]
     * @param n_bar2 Array containing galaxy number density [1/Mpc^3]
     * @param scaling1 Array containing scaling factors for satellite variances type a(optional)
     * @param scaling2 Array containing scaling factors for satellite variances type b(optional)
     */
    __device__ __host__ double kernel_function_2halo(double theta1, double theta2, double theta3, double l1, double l2, double phi,
                                                     double m1, double m2, double z, double zmin, double zmax, double mmin, double mmax,
                                                     double kmin, double kmax, int Nbins, HOD *hod1, HOD *hod2, double *dev_params1, double *dev_params2,
                                                     double A, double epsilon, const double *g, const double *p_lens1,
                                                     const double *p_lens2, const double *w, const double *dwdz, const double *hmf,
                                                     const double *P_lin, const double *b_h, const double *concentration,
                                                     const double *rho_bar, const double *n_bar1, const double *n_bar2,
                                                     const double *scaling1 = NULL, const double *scaling2 = NULL);

#if GPU
    /**
     * GPU Kernel for 2Halo term of NNM
     * @param params Integration variables
     * @param theta1 Aperture Radius [rad]
     * @param theta2 Aperture Radius [rad]
     * @param theta3 Aperture Radius [rad]
     * @param npts Number of integration points
     * @param type1 Galaxy type (1 or 2)
     * @param type2 Galaxy type (1 or 2)
     * @param f1 Parameter f of HOD 1
     * @param f2 Parameter f of HOD2
     * @param alpha1 Parameter alpha of HOD1
     * @param alpha2 Parameter alpha of HOD2
     * @param mmin1 Parameter mmin of HOD1 [Msun]
     * @param mmin2 Parameter mmin of HOD2 [Msun]
     * @param sigma1 Parameter sigma of HOD1
     * @param sigma2 Parameter sigma of HOD2
     * @param mprime1 Parameter mprime of HOD1 [Msun]
     * @param mprime2 Parameter mprime of HOD2 [Msun]
     * @param beta1 Parameter beta of HOD
     * @param beta2 Parameter beta of HOD
     * @param A Parameter A of halo model
     * @param epsilon Parameter epsilon of halo model
     * @param zmin Minimal Redshift of binning
     * @param zmax Maximal Redshift of binning
     * @param mmin Minimal Halomass for binning [Msun]
     * @param mmax Maximal Halomass for binning [Msun]
     * @param kmin Minimal k for binning [1/Mpc]
     * @param kmax Maximal k for binning [1/Mpc]
     * @param Nbins Number of bins
     * @param g Array containing precomputed lensing efficiency
     * @param p_lens1 Array containing lens redshift distribution
     * @param p_lens2 Array containing lens redshift distribution
     * @param w Array containing comoving distance [Mpc]
     * @param dwdz Array containing derivative of comoving distance [Mpc]
     * @param hmf Array containing precomputed HMF [1/Mpc^3/Msun]
     * @param concentration Array containing concentration of NFW profiles
     * @param rho_bar Array containing matter density [Msun/Mpc^3]
     * @param n_bar1 Array containing galaxy number density [1/Mpc^3]
     * @param n_bar2 Array containing galaxy number density [1/Mpc^3]
     * @param value Array which will contain results
     * @param scaling1 Array containing scaling factors for satellite variances type a(optional)
     * @param scaling2 Array containing scaling factors for satellite variances type b(optional)
     */
    __global__ void GPUkernel_2Halo(const double *params, double theta1, double theta2, double theta3, int npts, HOD *hod1, HOD *hod2, double *dev_params1, double *dev_params2,
                                    double A, double epsilon,
                                    double zmin, double zmax, double mmin, double mmax, double kmin, double kmax, int Nbins,
                                    const double *g, const double *p_lens1, const double *p_lens2, const double *w, const double *dwdz,
                                    const double *hmf, const double *P_lin, const double *b_h, const double *concentration,
                                    const double *rho_bar, const double *n_bar1, const double *n_bar2,
                                    double *value, const double *scaling1 = NULL, const double *scaling2 = NULL);
#endif

    /**
     * Integrand for 3Halo term of NNM for use with cubature
     * @param ndim Dimension of integral (7 in this case)
     * @param npts Number of integration points
     * @param params Array of integration x-values
     * @param thisPtr Pointer to integration container
     * @param fdim Dimension of integrand function (1 in this case)
     * @param value Array, which will contain the integration values
     */
    int integrand_3halo(unsigned ndim, size_t npts, const double *params, void *thisPtr, unsigned fdim, double *value);

    /**
     * Kernel function for 2Halo term of NNM
     * @param theta1 Aperture Radius [rad]
     * @param theta2 Aperture Radius [rad]
     * @param theta3 Aperture Radius [rad]
     * @param l1 l parameter [1/rad]
     * @param l2 l parameter [1/rad]
     * @param phi phi parameter [rad]
     * @param m1 Halo mass1 [Msun]
     * @param m2 Halo mass2 [Msun]
     * @param m3 Halo mass3 [Msun]
     * @param z Redshift
     * @param zmin Minimal Redshift of binning
     * @param zmax Maximal Redshift of binning
     * @param mmin Minimal Halomass for binning [Msun]
     * @param mmax Maximal Halomass for binning [Msun]
     * @param kmin Minimal k for binning [1/Mpc]
     * @param kmax Maximal k for binning [1/Mpc]
     * @param Nbins Number of bins
     * @param type1 Galaxy type (1 or 2)
     * @param type2 Galaxy type (1 or 2)
     * @param f1 Parameter f of HOD 1
     * @param f2 Parameter f of HOD2
     * @param alpha1 Parameter alpha of HOD1
     * @param alpha2 Parameter alpha of HOD2
     * @param mmin1 Parameter mmin of HOD1 [Msun]
     * @param mmin2 Parameter mmin of HOD2 [Msun]
     * @param sigma1 Parameter sigma of HOD1
     * @param sigma2 Parameter sigma of HOD2
     * @param mprime1 Parameter mprime of HOD1 [Msun]
     * @param mprime2 Parameter mprime of HOD2 [Msun]
     * @param beta1 Parameter beta of HOD
     * @param beta2 Parameter beta of HOD
     * @param A Parameter A of halo model
     * @param epsilon Parameter epsilon of halo model
     * @param g Array containing precomputed lensing efficiency
     * @param p_lens1 Array containing lens redshift distribution
     * @param p_lens2 Array containing lens redshift distribution
     * @param w Array containing comoving distance [Mpc]
     * @param dwdz Array containing derivative of comoving distance [Mpc]
     * @param hmf Array containing precomputed HMF [1/Mpc^3/Msun]
     * @param concentration Array containing concentration of NFW profiles
     * @param rho_bar Array containing matter density [Msun/Mpc^3]
     * @param n_bar1 Array containing galaxy number density [1/Mpc^3]
     * @param n_bar2 Array containing galaxy number density [1/Mpc^3]
     * @param H0 Hubble constant [km/s/Mpc]
     * @param OmM Omega_m
     */
    __device__ __host__ double kernel_function_3halo(double theta1, double theta2, double theta3, double l1, double l2, double phi,
                                                     double m1, double m2, double m3, double z, double zmin, double zmax, double mmin,
                                                     double mmax, double kmin, double kmax, int Nbins, HOD *hod1, HOD *hod2, double *dev_params1, double *dev_params2,
                                                     const double *g, const double *p_lens1, const double *p_lens2, const double *w,
                                                     const double *dwdz, const double *hmf, const double *P_lin, const double *b_h,
                                                     const double *concentration, const double *rho_bar, const double *n_bar1,
                                                     const double *n_bar2);

#if GPU
    /**
     * GPU Kernel for 2Halo term of NNM
     * @param params Integration variables
     * @param theta1 Aperture Radius [rad]
     * @param theta2 Aperture Radius [rad]
     * @param theta3 Aperture Radius [rad]
     * @param npts Number of integration points
     * @param type1 Galaxy type (1 or 2)
     * @param type2 Galaxy type (1 or 2)
     * @param f1 Parameter f of HOD 1
     * @param f2 Parameter f of HOD2
     * @param alpha1 Parameter alpha of HOD1
     * @param alpha2 Parameter alpha of HOD2
     * @param mmin1 Parameter mmin of HOD1 [Msun]
     * @param mmin2 Parameter mmin of HOD2 [Msun]
     * @param sigma1 Parameter sigma of HOD1
     * @param sigma2 Parameter sigma of HOD2
     * @param mprime1 Parameter mprime of HOD1 [Msun]
     * @param mprime2 Parameter mprime of HOD2 [Msun]
     * @param beta1 Parameter beta of HOD
     * @param beta2 Parameter beta of HOD
     * @param zmin Minimal Redshift of binning
     * @param zmax Maximal Redshift of binning
     * @param mmin Minimal Halomass for binning [Msun]
     * @param mmax Maximal Halomass for binning [Msun]
     * @param kmin Minimal k for binning [1/Mpc]
     * @param kmax Maximal k for binning [1/Mpc]
     * @param Nbins Number of bins
     * @param g Array containing precomputed lensing efficiency
     * @param p_lens1 Array containing lens redshift distribution
     * @param p_lens2 Array containing lens redshift distribution
     * @param w Array containing comoving distance [Mpc]
     * @param dwdz Array containing derivative of comoving distance [Mpc]
     * @param hmf Array containing precomputed HMF [1/Mpc^3/Msun]
     * @param concentration Array containing concentration of NFW profiles
     * @param rho_bar Array containing matter density [Msun/Mpc^3]
     * @param n_bar1 Array containing galaxy number density [1/Mpc^3]
     * @param n_bar2 Array containing galaxy number density [1/Mpc^3]
     * @param H0 Hubble constant [km/s/Mpc]
     * @param OmM Omega_m
     * @param value Array which will contain results
     */
    __global__ void GPUkernel_3Halo(const double *params, double theta1, double theta2, double theta3, int npts, HOD *hod1, HOD *hod2, double *dev_params1, double *dev_params2,
                                    double zmin, double zmax,
                                    double mmin, double mmax, double kmin, double kmax, int Nbins, const double *g,
                                    const double *p_lens1, const double *p_lens2, const double *w, const double *dwdz, const double *hmf,
                                    const double *P_lin, const double *b_h, const double *concentration, const double *rho_bar,
                                    const double *n_bar1, const double *n_bar2, double *value);

#endif

    /**
     * G_g Function (helper for NNM)
     * @param k Wavevector [1/Mpc]
     * @param m Halo mass [Msun]
     * @param z redshift
     * @param f f parameter of HOD
     * @param alpha Alpha parameter
     * @param mth Mmin parameter [Msun]
     * @param sigma Sigma parameter
     * @param mprime Mprime parameter [Msun]
     * @param beta Beta parameter
     * @param zmin Minimal Redshift of binning
     * @param zmax Maximal Redshift of binning
     * @param mmin Minimal Halomass for binning [Msun]
     * @param mmax Maximal Halomass for binning [Msun]
     * @param Nbins Number of bins
     * @param Array contain matter density
     * @param Array containing concentration
     */
    __host__ __device__ double G_g(double k, double m, double z, HOD *hod, double *dev_params,
                                   double zmin, double zmax, double mmin, double mmax,
                                   int Nbins, const double *rho_bar, const double *concentration);

    /**
     * G_gg Function (helper for NNM)
     * @param k1 Wavevector 1 [1/Mpc]
     * @param k2 Wavevector 2 [1/Mpc]
     * @param m Halo mass [Msun]
     * @param z redshift
     * @param f1 Parameter f of HOD 1
     * @param f2 Parameter f of HOD2
     * @param alpha1 Parameter alpha of HOD1
     * @param alpha2 Parameter alpha of HOD2
     * @param mmin1 Parameter mmin of HOD1 [Msun]
     * @param mmin2 Parameter mmin of HOD2 [Msun]
     * @param sigma1 Parameter sigma of HOD1
     * @param sigma2 Parameter sigma of HOD2
     * @param mprime1 Parameter mprime of HOD1 [Msun]
     * @param mprime2 Parameter mprime of HOD2 [Msun]
     * @param beta1 Parameter beta of HOD
     * @param beta2 Parameter beta of HOD
     * @param A Parameter A of halo model
     * @param epsilon Parameter epsilon of halo model
     * @param zmin Minimal Redshift of binning
     * @param zmax Maximal Redshift of binning
     * @param mmin Minimal Halomass for binning [Msun]
     * @param mmax Maximal Halomass for binning [Msun]
     * @param Nbins Number of bins
     * @param Array contain matter density
     * @param Array containing concentration
     * @param sameType True if type1==type2
     * @param scale1 scaling weight for satellite variance (type a) (optional)
     * @param scale2 scaling weight for satellite variance (type b) (optional)
     */
    __host__ __device__ double G_gg(double k1, double k2, double m, double z, HOD *hod1, HOD *hod2, double *dev_params1, double *dev_params2, double A, double epsilon,
                                    double zmin, double zmax, double mmin, double mmax,
                                    int Nbins, const double *rho_bar, const double *concentration, double scale1 = 1, double scale2 = 1);
  }
  struct nnmap_container
  {
    NNMap_Model *nnmap;
    double theta1, theta2, theta3;
    int type1, type2;
  };

}
#endif // G3LHALO_NNMAP_MODEL_H
