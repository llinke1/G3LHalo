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
    double *dev_g;             //< Lensing efficiency
    double *dev_p_lens;        //< Lens redshift distribution
    double *dev_w;             //< Comoving distance [Mpc]
    double *dev_dwdz;          //< Derivative of comoving distance wrt redshift [Mpc]
    double *dev_hmf;           //< Halo mass function [1/Mpc^3/Msun]
    double *dev_P_lin;         //< Linear Power spectrum [1/Mpc^3]
    double *dev_b_h;           //< Linear Halo bias
    double *dev_concentration; //< Concentration of NFW profiles

    /// Lookup tables that are updated in initialization
    double *dev_rho_bar; //< Matter density [Msun /Mpc^3]
    double *dev_n_bar;   //< Galaxy number density [1/Mpc^3]
#endif

    HOD *hod; // HOD Model

    // Empty constructor
    NMapMap_Model(){};

    /**
     * Constructor from values
     * @param cosmology_ Cosmology object, contain LCDM parameters
     * @param zmin_ Minimal redshift for binning
     * @param zmax_ Maximal redshift for binning
     * @param kmin_ Minimal k for binning [1/Mpc]
     * @param kmax_ Maximal k for binning [1/Mpc]
     * @param mmin_ Minimal halo mass for binning [Msun]
     * @param mmax_ Maximal halo mass for binning [Msun]
     * @param Nbins_ Number of bins
     * @param g_ Lensing efficiency
     * @param p_lens_ Lens redshift distribution
     * @param w_ Comoving distance
     * @param dwdz_ Derivative of comoving distance
     * @param hmf_ Precalculated halo mass function
     * @param P_lin_ Linear matter powerspectrum
     * @param b_h_ halo bias
     * @param concentration_ concentration parameter
     * @param hod_ HOD object
     */
    NMapMap_Model(Cosmology *cosmology_, const double &zmin_, const double &zmax_, const double &kmin_, const double &kmax_,
                  const double &mmin_, const double &mmax_, const int &Nbins_,
                  double *g_, double *p_lens_, double *w_,
                  double *dwdz_, double *hmf_, double *P_lin_, double *b_h_, double *concentration_,
                  HOD *hod_);

    // Destructor
    ~NMapMap_Model();

    /**
     * Changes HOD params and updates galaxy and matter density
     * @params hod_ new HOD
     */
    void updateHOD(HOD *hod_);

    /**
     * Calculates matter and galaxy densities for current HOD parameters
     * Uses logarithmic integral over halo masses
     */
    void updateDensities();

    /**
     * Calculates 1 Halo term of NMapMap
     * @param theta1 First aperture radius [rad]
     * @param theta2 Second aperture radius [rad]
     * @param theta3 Third aperture radius [rad]
     * @return 1 Halo term of NNMap
     */
    double NMapMap_1h(const double &theta1, const double &theta2, const double &theta3);

    /**
     * Calculates 2 Halo term of NMapMap
     * @param theta1 First aperture radius [rad]
     * @param theta2 Second aperture radius [rad]
     * @param theta3 Third aperture radius [rad]
     * @return 2 Halo term of NNMap
     */
    double NMapMap_2h(const double &theta1, const double &theta2, const double &theta3);

    /**
     * Calculates 3 Halo term of NMapMap
     * @param theta1 First aperture radius [rad]
     * @param theta2 Second aperture radius [rad]
     * @param theta3 Third aperture radius [rad]
     * @return 3 Halo term of NNMap
     */
    double NMapMap_3h(const double &theta1, const double &theta2, const double &theta3);

    /**
     * Calculates complete NMapMap
     * @param theta1 First aperture radius [rad]
     * @param theta2 Second aperture radius [rad]
     * @param theta3 Third aperture radius [rad]
     * @return Complete NNMap
     */
    double NMapMap(const double &theta1, const double &theta2, const double &theta3);

    /**
     * Calculates complete NMapMap for range of aperture radii and specified params
     * First updates HOD params and densities, then does calculation and writes it into results-vector
     * @param thetas1 First aperture radii [rad]
     * @param thetas2 Second aperture radii [rad]
     * @param thetas3 Third aperture radii [rad]
     * @param hod HOD modell
     * @param results COntainer for results
     */
    void calculateAll(const std::vector<double> &thetas1, const std::vector<double> &thetas2, const std::vector<double> &thetas3,
                      HOD *hod, std::vector<double> &results);
  };

  namespace NMM
  {
    /**
     * Kernel function for 1-halo term
     * @param theta1 First aperture radius [rad]
     * @param theta2 Second aperture radius [rad]
     * @param theta3 Third aperture radius [rad]
     * @param l1 First ell-vector
     * @param l2 Second ell-vector
     * @param phi Angle between ells [rad]
     * @param m Halo mass [Msun]
     * @param z redshift
     * @param zmin Minimal redshift for binning
     * @param zmax Maximal redshift for binning
     * @param mmin Minimal halomass for binning [Msun]
     * @param mmax Maximal halomass for binning [Msun]
     * @param Nbins number of bins
     * @param hod HOD object
     * @param dev_params Pointer to HOD params on device, should match hod
     * @param g lensing efficiency
     * @param p_lens Lens redshift distribution
     * @param w Comoving distance
     * @param dwdz Derivative of comoving distance
     * @param hmf tabularized halo mass function
     * @param concentration concentration parameter
     * @param rho_bar average matter density
     * @param n_bar average galaxy number density
     */
    __device__ __host__ double kernel_function_1halo(double theta1, double theta2, double theta3, double l1, double l2, double phi, double m, double z,
                                                     double zmin, double zmax, double mmin, double mmax, int Nbins,
                                                     HOD *hod, double *dev_params,
                                                     const double *g, const double *p_lens, const double *w, const double *dwdz,
                                                     const double *hmf, const double *concentration,
                                                     const double *rho_bar, const double *n_bar);

    /**
     * GPU kernel function for 1-halo term
     * @param params Integration parameters
     * @param theta1 First aperture radius [rad]
     * @param theta2 Second aperture radius [rad]
     * @param theta3 Third aperture radius [rad]
     * @param npts Number of integration points
     * @param hod HOD object containing parameters
     * @param dev_params Pointer to HOD parameters on device, should be the same as in hod object
     * @param zmin Minimal redshift of binning
     * @param zmax Maximal redshift of binning
     * @param mmin Minimal halomass of binning [Msun]
     * @param mmax Maximal halomass of binning [Msun]
     * @param Nbins Number of bins
     * @param g lensing efficiency
     * @param p_lens Lens redshift distribution
     * @param w Comoving distance
     * @param dwdz Derivative of comoving distance
     * @param hmf tabularized halo mass function
     * @param concentration concentration parameter
     * @param rho_bar average matter density
     * @param n_bar average galaxy number density
     * @param value Output parameter, will contain galaxy number density
     */
    __global__ void GPUkernel_1Halo(const double *params, double theta1, double theta2, double theta3, int npts,
                                    HOD *hod, double *dev_params,
                                    double zmin, double zmax, double mmin, double mmax,
                                    int Nbins, const double *g, const double *p_lens, const double *w,
                                    const double *dwdz, const double *hmf, const double *concentration, const double *rho_bar, const double *n_bar,
                                    double *value);

    /**
     * Integrand for 1-halo term for the use with cubature
     * @param ndim Dimensionality of integral (must be 5 here)
     * @param npts Number of integration points
     * @param m Array containing halo masses of integration points
     * @param thisPtr Pointer to integration container (of form n_z_container)
     * @param fdim Dimension of function (must be 1 here)
     * @param value Will contain integrand values
     */
    int integrand_1halo(unsigned ndim, size_t npts, const double *params, void *thisPtr, unsigned fdim, double *value);

    /**
     * Kernel function for 2-halo term
     * @param theta1 First aperture radius [rad]
     * @param theta2 Second aperture radius [rad]
     * @param theta3 Third aperture radius [rad]
     * @param l1 First ell-vector
     * @param l2 Second ell-vector
     * @param phi Angle between ells [rad]
     * @param m Halo mass [Msun]
     * @param z redshift
     * @param zmin Minimal redshift for binning
     * @param zmax Maximal redshift for binning
     * @param mmin Minimal halomass for binning [Msun]
     * @param mmax Maximal halomass for binning [Msun]
     * @param Nbins number of bins
     * @param hod HOD object
     * @param dev_params Pointer to HOD params on device, should match hod
     * @param g lensing efficiency
     * @param p_lens Lens redshift distribution
     * @param w Comoving distance
     * @param dwdz Derivative of comoving distance
     * @param hmf tabularized halo mass function
     * @param P_lin linearized matter power spectrum
     * @param b_h halo bias
     * @param concentration concentration parameter
     * @param rho_bar average matter density
     * @param n_bar average galaxy number density
     */
    __device__ __host__ double kernel_function_2halo(double theta1, double theta2, double theta3, double l1, double l2, double phi,
                                                     double m1, double m2, double z, double zmin, double zmax, double mmin, double mmax,
                                                     double kmin, double kmax, int Nbins, HOD *hod, double *dev_params,
                                                     const double *g, const double *p_lens, const double *w, const double *dwdz, const double *hmf,
                                                     const double *P_lin, const double *b_h, const double *concentration,
                                                     const double *rho_bar, const double *n_bar);

    /**
     * GPU kernel function for 2-halo term
     * @param params Integration parameters
     * @param theta1 First aperture radius [rad]
     * @param theta2 Second aperture radius [rad]
     * @param theta3 Third aperture radius [rad]
     * @param npts Number of integration points
     * @param hod HOD object containing parameters
     * @param dev_params Pointer to HOD parameters on device, should be the same as in hod object
     * @param zmin Minimal redshift of binning
     * @param zmax Maximal redshift of binning
     * @param mmin Minimal halomass of binning [Msun]
     * @param mmax Maximal halomass of binning [Msun]
     * @param Nbins Number of bins
     * @param g lensing efficiency
     * @param p_lens Lens redshift distribution
     * @param w Comoving distance
     * @param dwdz Derivative of comoving distance
     * @param hmf tabularized halo mass function
     * @param P_lin linearized matter power spectrum
     * @param b_h halo bias
     * @param concentration concentration parameter
     * @param rho_bar average matter density
     * @param n_bar average galaxy number density
     * @param value Output parameter, will contain galaxy number density
     */
    __global__ void GPUkernel_2Halo(const double *params, double theta1, double theta2, double theta3, int npts, HOD *hod, double *dev_params,
                                    double zmin, double zmax, double mmin, double mmax, double kmin, double kmax, int Nbins,
                                    const double *g, const double *p_lens, const double *w, const double *dwdz,
                                    const double *hmf, const double *P_lin, const double *b_h, const double *concentration,
                                    const double *rho_bar, const double *n_bar,
                                    double *value);


    /**
     * Integrand for 2-halo term for the use with cubature
     * @param ndim Dimensionality of integral (must be 5 here)
     * @param npts Number of integration points
     * @param m Array containing halo masses of integration points
     * @param thisPtr Pointer to integration container (of form n_z_container)
     * @param fdim Dimension of function (must be 1 here)
     * @param value Will contain integrand values
     */
    int integrand_2halo(unsigned ndim, size_t npts, const double *params, void *thisPtr, unsigned fdim, double *value);



    /**
     * Kernel function for 3-halo term
     * @param theta1 First aperture radius [rad]
     * @param theta2 Second aperture radius [rad]
     * @param theta3 Third aperture radius [rad]
     * @param l1 First ell-vector
     * @param l2 Second ell-vector
     * @param phi Angle between ells [rad]
     * @param m Halo mass [Msun]
     * @param z redshift
     * @param zmin Minimal redshift for binning
     * @param zmax Maximal redshift for binning
     * @param mmin Minimal halomass for binning [Msun]
     * @param mmax Maximal halomass for binning [Msun]
     * @param Nbins number of bins
     * @param hod HOD object
     * @param dev_params Pointer to HOD params on device, should match hod
     * @param g lensing efficiency
     * @param p_lens Lens redshift distribution
     * @param w Comoving distance
     * @param dwdz Derivative of comoving distance
     * @param hmf tabularized halo mass function
     * @param P_lin linearized matter power spectrum
     * @param b_h halo bias
     * @param concentration concentration parameter
     * @param rho_bar average matter density
     * @param n_bar average galaxy number density
     */
    __device__ __host__ double kernel_function_3halo(double theta1, double theta2, double theta3, double l1, double l2, double phi,
                                                     double m1, double m2, double m3, double z, double zmin, double zmax, double mmin,
                                                     double mmax, double kmin, double kmax, int Nbins, HOD *hod, double *dev_params,
                                                     const double *g, const double *p_lens, const double *w,
                                                     const double *dwdz, const double *hmf, const double *P_lin, const double *b_h,
                                                     const double *concentration, const double *rho_bar, const double *n_bar);


    /**
     * GPU kernel function for 3-halo term
     * @param params Integration parameters
     * @param theta1 First aperture radius [rad]
     * @param theta2 Second aperture radius [rad]
     * @param theta3 Third aperture radius [rad]
     * @param npts Number of integration points
     * @param hod HOD object containing parameters
     * @param dev_params Pointer to HOD parameters on device, should be the same as in hod object
     * @param zmin Minimal redshift of binning
     * @param zmax Maximal redshift of binning
     * @param mmin Minimal halomass of binning [Msun]
     * @param mmax Maximal halomass of binning [Msun]
     * @param Nbins Number of bins
     * @param g lensing efficiency
     * @param p_lens Lens redshift distribution
     * @param w Comoving distance
     * @param dwdz Derivative of comoving distance
     * @param hmf tabularized halo mass function
     * @param P_lin linearized matter power spectrum
     * @param b_h halo bias
     * @param concentration concentration parameter
     * @param rho_bar average matter density
     * @param n_bar average galaxy number density
     * @param value Output parameter, will contain galaxy number density
     */
    __global__ void GPUkernel_3Halo(const double *params, double theta1, double theta2, double theta3, int npts, HOD *hod, double *dev_params,
                                    double zmin, double zmax,
                                    double mmin, double mmax, double kmin, double kmax, int Nbins, const double *g,
                                    const double *p_lens, const double *w, const double *dwdz, const double *hmf,
                                    const double *P_lin, const double *b_h, const double *concentration, const double *rho_bar,
                                    const double *n_bar, double *value);



    /**
     * Integrand for 3-halo term for the use with cubature
     * @param ndim Dimensionality of integral (must be 5 here)
     * @param npts Number of integration points
     * @param m Array containing halo masses of integration points
     * @param thisPtr Pointer to integration container (of form n_z_container)
     * @param fdim Dimension of function (must be 1 here)
     * @param value Will contain integrand values
     */
    int integrand_3halo(unsigned ndim, size_t npts, const double *params, void *thisPtr, unsigned fdim, double *value);

  }

// Structure for NMM integration
  struct nmapmap_container
  {
    NMapMap_Model *nmapmap;
    double theta1, theta2, theta3;
  };

}

#endif // G3LHALO_NMAPMAP_MODEL_H