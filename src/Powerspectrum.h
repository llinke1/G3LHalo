#ifndef G3LHALO_POWERSPECTRUM_H
#define G3LHALO_POWERSPECTRUM_H

#include "Cosmology.h"

namespace g3lhalo
{
  /**
   * Class for linear power spectrum
   * Implements Eisenstein & Hu 1998 Transfer function
   * @author Laila Linke llinke@astro.uni-bonn.de
   */
  class Powerspectrum
  {
  public:
    ///Cosmology used
    Cosmology* cosmo;
    ///Amplitude of Power spectrum [Mpc^3]
    double amplitude;
    
    ///Parameters for Transfer function
    double omhh;		// Omega_matter*h^2
    double obhh;		// Omega_baryon*h^2 
    double theta_cmb;		// Tcmb in units of 2.7 K 
    double z_equality;		// Redshift of matter-radiation equality, really 1+z 
    double k_equality;		// Scale of equality, in Mpc^-1 
    double z_drag;		// Redshift of drag epoch 
    double R_drag;		// Photon-baryon ratio at drag epoch 
    double R_equality;		// Photon-baryon ratio at equality epoch 
    double sound_horizon;	// Sound horizon at drag epoch, in Mpc 
    double k_silk;		// Silk damping scale, in Mpc^-1 
    double alpha_c;		// CDM suppression 
    double beta_c;		// CDM log shift 
    double alpha_b;		// Baryon suppression 
    double beta_b;		// Baryon envelope shift 
    double beta_node;		// Sound horizon shift 
    double k_peak;		// Fit to wavenumber of first peak, in Mpc^-1 
    double sound_horizon_fit;	// Fit to sound horizon, in Mpc 
    double alpha_gamma;		// Gamma suppression in approximate TF



    //Empty constructor
    Powerspectrum(){};
    
    //Constructor from Cosmology
    Powerspectrum(Cosmology* cosmo_);

     /**
      * Sets a lot of internal parameters for Transfer function
      */
    void set_transfer_parameters();


    /**
     * Sets amplitude of power spectrum
     * \f$A=\frac{(2\pi)^{2}\, \sigma_8^2}{\int {\rm d}k\;k^2\,W_{8h^{-1}}^2(k)\, T^2(k)\, k^{n_s} }$\f
     */
    void set_amplitude();


  /**
   * Fourier Transform of Tophat Filter
   * \f$W_R(k)=\frac{3}{(kR)^3}[\sin(kR)-kR\cos(kR)]$\f
   * @param k Wavenumber [1/Mpc]
   * @param R Scale [Mpc]
   */
  double tophat(const double& k,const double& R);


  /**
   * Derivative of Fourier Transform of Tophat Filter wr to scale [1/Mpc]
   * \f$\frac{{\rm d}W_R(k)}{{\rm d}R} = \frac{3}{R}[\frac{\sin(kR)/{kR} - W_R(k)]$\f
   * @param k Wavenumber [1/Mpc]
   * @param R Scale [Mpc]
   */
  double d_tophat(const double& k, const double& R);

    /**
     * Calculates mstar for Bullock Concentration
     * @param z redshift
     */
    double mStar(const double& z);


    /**
     * Calculates concentration 
     * Uses Bullock 2001 relation
     * @param m halo mass [Msun]
     * @param mstar [Msun]
     * @param z redshift
     */
    double concentration(const double& m, const double& mstar, const double& z);

  
    /**
     * Transferfunction (Eisenstein & Hu 1998)
     * Based on Libastro implementation from C.Angrick
     * @param k Wavenumber [1/Mpc]
     */
    double transfer(const double& k);



    
    /**
     * Returns Powerspectrum at wavenumber k and redshift z [1/Mpc^3]
     * \f$P(k,z)=A\, k^{n_s}\, T^2(k)\, D^2(z)$\f
     * @param k wavenumber [1/Mpc]
     * @param z redshift
     */
    double spectrum(const double& k, const double& z);

    /**
     * Returns power spectrum variance at mass scale m
     * \f$\sigma^2(m)=\frac{1}{2\pi^2}\int {\rm d}k \; k^2\; P(k, z)\; |W_R(k)|^2
     * @param m Mass scale [Msun]
     */
    double sigma2(const double& m, const double& z);
    
    /**
     * Returns derivative power spectrum variance w.r.t. mass scale m
     * \f$\frac{{\rm d}\sigma^2}{{\rm d}m} = \frac{1}{\pi}\int {\rm d}k\; k^2\, P(k, z)\, W_R(k)\, \frac{{\rm d}W_R(k)}{{\rm d}R}  \frac{{\rm d}R}{{\rm d}m}$\f
     * @param m Mass scale [Msun]
     
     */
    double dsigma2dm(const double& m, const double& z);


    
    /**
     * Calculates concentration for range of redshifts
     * Calls double concentration(const double&, const double&)
     * @param ms halo masses [Msun]
     * @param zs redshifts for which distance is calculated
     * @param result vector which will contain concentration
     */
    void getConcentration(const std::vector<double>& zs,
			  const std::vector<double>& ms,
			  std::vector<double>& result);

    void getSigma2(const std::vector<double>& zs,
		   const std::vector<double>& ms,
		   std::vector<double>& result);

    /**
     * Calculates Powerspectrum for range of redshifts
     * Calls double spectrum(const double&, const double&)
     * @param ks halo masses [Msun]
     * @param zs redshifts for which distance is calculated
     * @param result vector which will contain concentration
     */
    void getSpectrum(const std::vector<double>& zs,
		     const std::vector<double>& ks,
		     std::vector<double>& result);

    void getTransfer(const std::vector<double>& ks,
		     std::vector<double>& result);


    
  };

  /**
   * Integrand for powerspectrum amplitude, for use with cubature
   * @param ndim Dimension of Integral
   * @param npts Number of simultaneous integration points
   * @param z Integration points
   * @param thisPtr Pointer to powerspectrum object
   * @param value Will contain integration values
   */
  int integrand_amplitude(unsigned ndim, size_t npts, const double* z, void* thisPtr, unsigned fdim, double* value);

  /**
   * Integrand for sigma2, for use with cubature
   * @param ndim Dimension of Integral
   * @param npts Number of simultaneous integration points
   * @param z Integration points
   * @param thisPtr Pointer to powerspectrum object
   * @param value Will contain integration values
   */
  int integrand_sigma2(unsigned ndim, size_t npts, const double* z, void* thisPtr, unsigned fdim, double* value);

    /**
   * Integrand for dsigma2/dM, for use with cubature
   * @param ndim Dimension of Integral
   * @param npts Number of simultaneous integration points
   * @param z Integration points
   * @param thisPtr Pointer to powerspectrum object
   * @param value Will contain integration values
   */
  int integrand_dsigma2dM(unsigned ndim, size_t npts, const double* z, void* thisPtr, unsigned fdim, double* value);



  class containerSigma2
  {
  public:
    Powerspectrum* powerspectrum;
    double R;
    double dR;
    double z;
  };
  
}



#endif //G3LHALO_POWERSPECTRUM_H
