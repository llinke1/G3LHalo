#ifndef G3LHALO_COSMOLOGY_H
#define G3LHALO_COSMOLOGY_H

#include <vector>
#include <string>
#include <iostream>
#include "constants.h"

namespace g3lhalo
{
  /**
   * Class for flat LCDM cosmology 
   * @author Laila Linke llinke@astro.uni-bonn.de
   */
  class Cosmology
  {
  public:
    double Omega_m;    //< \f $\Omega_m$ \f today
    double Omega_Lambda;     //< \f $\Omega_\Lambda$\f today
    double Omega_b; //< \f $\Omega_b$\f today
    double sigma8; //< \f $\sigma_8$\f
    double H0; //< Hubble constant [km/s/Mpc]
    double ns; //< Spectral index


    double c=299792;  //Speed of light [km/s]
    double G=1.327e11; //Newtonian gravitational constant [km^3/Msun/s^2]
    double Mpc_in_km=3.086e19; //Megaparsec in km
    double rho_crit;

    Cosmology(){}; ///<Empty Constructor
    
    /**
     * Constructor from values
     * @param omega_m new matter density
     * @param omega_l new Lambda density
     * @param sigma_8 new power spectrum normalization
     * @param H0 new Hubble constant [km/s/Mpc]
     * @param ns new Spectral index
     */
  Cosmology(const double& Omega_m_, const double& Omega_Lambda_,
	    const double& Omega_b_, const double& sigma8_,
	    const double& H0_, const double& ns_)
    : Omega_m{Omega_m_}, Omega_Lambda{Omega_Lambda_}, Omega_b{Omega_b_}, sigma8{sigma8_}, H0{H0_}, ns{ns_}
    {
      rho_crit=3*H0*H0*Mpc_in_km/8/g3lhalo::pi/G;
    };

      /**
       * Constructor from parameter file
       * @param fn_parameters Filename of parameter file
       */
    Cosmology(const std::string& fn_parameters);
  
    
    /**
     * Gives out critical density for spherical collapse
     * \f$\delta_c=\frac{3(12\pi)^{2/3}}{20}[1+0.0123\log_{10}(\Omega_m(z))]$\f
     * Based on Nakamuro & Suto (1997; Eq. C28 in Appendix)
     * @param z redshift
     */
    double delta_c(const double& z);

    /**
     * Gives out matter density parameter at redshift
     * \f$\Omega_m(z)=\frac{\Omega_m\, (1+z)^3}{E(z)}$\f
     * @param z redshift
     */
    double Omega_m_(const double& z);

    /**
     * Gives out expansion function
     * \f$E(z)=\Omega_m\,(1+z)^3 + \Omega_\Lamda$\f
     * @param z redshift
     */
    double E(const double& z);

    /**
     * Gives out comoving distance at redshift z [Mpc]
     * \f$\chi=\frac{c}{H_0}\int_0^z\frac{{\rm d}z'}{\sqrt{E(z')}}$\f
     * @param z redshift
     */
    double comovingDistance(const double& z);

    /**
     * Gives out derivative of comoving distance w.r.t z [Mpc]
     * \f$\frac{{\rm d}\chi}{{\rm d}z}=\frac{c}{H_0}\frac{1}{\sqrt{E(z)}}$\f
     * @param z redshift
     */
    double derivative_comovingDistance(const double& z);

    /**
     * Gives out Linear Growth Function
     * Currently uses approximation by Caroll et al (1992)
     * \f $D(a)=2.5a\,\frac{\Omega_m(a)}{\Omega_m(a)^{4/7}-\Omega_\Lamda+(1+\Omega_m/2)(1+\Omega_\Lambda/70)}$\f
     * Plan: move to exact Formula by Eisenstein(1997)
     * @warning Is not normalized, needs to be divided by D(0.0)!
     * @param z redshift
     */
    double growthFunction(double z);


    /**
     * Gives out projection function for Limber equation
     * \f$g(z)=\int_z^\infty \mathrm{d}z^\prime p_s(z^\prime) \frac{w(z^\prime)-w(z)}{w(z^\prime)}$\f
     * @warning Assumes flat Universe!
     * @param z redshift for which g is calculates
     * @param nz source redshift distribution
     * @param zs redshifts for which nz is given
     */
    double lensingEfficiency(const double& z, const std::vector<double>& nz,
			     const std::vector<double>& zs);
    

    /**
     * Calculates lensing Efficiency for range of lens redshifts
     * Calls double lensingEfficiency(const double&, const std::vector<double>&, const std::vector<double>&)
     * @param zs lens redshifts for which g is calculated
     * @param nz source redshift distribution
     * @param zsources source redshifts
     * @param result vector which will contain lensing efficiency
     */
    void getLensingEfficiency(const std::vector<double>& zs,
			      const std::vector<double>& nz,
			      const std::vector<double>& zsources,
			      std::vector<double>& result);


    /**
     * Calculates comoving distance for range of redshifts
     * Calls double comovingDistance(const double&)
     * @param zs redshifts for which distance is calculated
     * @param result vector which will contain comoving distance
     */
    void getComovingDistance(const std::vector<double>& zs,
			     std::vector<double>& result);


    /**
     * Calculates Derivative of comoving distance for range of redshifts
     * Calls double kernel_comovingDistance(const double&)
     * @param zs redshifts for which distance is calculated
     * @param result vector which will contain comoving distance
     */
    void getDerivativeComovingDistance(const std::vector<double>& zs,
				       std::vector<double>& result);
  };


  /**
   * Integrand for comoving distance, for use with cubature
   * @param ndim Dimension of Integral
   * @param npts Number of simultaneous integration points
   * @param z Integration points
   * @param thisPtr Pointer to cosmology object
   * @param value Will contain integration values
   */
  int integrand_comovingDistance(unsigned ndim, size_t npts, const double* z, void* thisPtr, unsigned fdim, double* value);
}



#endif //G3LHALO_COSMOLOGY_H
