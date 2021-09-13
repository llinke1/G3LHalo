#ifndef G3LHALO_HMF_H
#define G3LHALO_HMF_H

#include "Cosmology.h"
#include "Powerspectrum.h"
#include <iostream>
namespace g3lhalo
{
  /**
   * Class for Halo Mass Function
   * At the moment: Only uses Sheth-Tormen
   * @author Laila Linke llinke@astro.uni-bonn.de
   */
  class Halomassfunction
  {
  public:

    ///LCDM Cosmology
    Cosmology* cosmo;

    /// Powerspectrum
    Powerspectrum* powerspectrum;
    
    /// Empty constructor
    Halomassfunction(){};

    /// Constructor from parameters
    Halomassfunction(Cosmology* cosmo_, Powerspectrum* powerspectrum_): cosmo(cosmo_), powerspectrum(powerspectrum_){};



    
    /**
     * HMF, right now only Sheth-Tormen
     * returns Halo number density [1/Mpc^3]
     * @param m Mass [Msun]
     * @param z Redshift
     */
    double n(const double& m, const double& z);


    /**
     * Linear halo bias parameter
     * Right now only Sheth Tormen from Scoccimarro(2001)
     * @param m halo mass [Msun]
     * @param z redshift
     */
    double b_h(const double& m, const double& z);
  

      /**
       * Calculates n(m,z)
       * Calls double n(const double&, const double& )
       * @param zs redshifts for which n(m,z) is calculated
       * @param ms halo masses for which n(m,z) is calculated [Msun]
       * @param result vector which will contain HMF
       */
    void getHMF(const std::vector<double>& zs,
		const std::vector<double> &ms,
		std::vector<double>& result);

    /**
     * Calculates halo bias
     * Calls double b_h(const double&, const double& )
     * @param zs redshifts for which b_h(m,z) is calculated
     * @param ms halo masses for which b_h(m,z) is calculated [Msun]
     * @param result vector which will contain Halo bias
     */
    void getHaloBias(const std::vector<double>& zs,
		   const std::vector<double> &ms,
		   std::vector<double>& result);

    
  };
}



#endif //G3LHALO_HMF_H
