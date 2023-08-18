#ifndef G3LHALO_FITHELPERS_H
#define G3LHALO_FITHELPERS_H

#include "NNMap_Model.h"
#include "Params.h"
#include "Priors.h"

#include <vector>
#include <string>

//GSL headers
#include <gsl/gsl_multimin.h>

namespace g3lhalo
{
  /**
   * Class containing all data needed for fit
   */
  class fitData
  {
  private:

    /**
     * Reads in N1N2M, N1N1M, N2N2M and Covariance
     * @param fn1 Filename for N1N2M
     * @param fn2 Filename for N1N1M
     * @param fn3 Filename for N2N2M
     * @param fncov Filename for inverse Covariance
     */
    void readInData(const std::string& fn1, const std::string& fn2, const std::string& fn3, const std::string& fncov);

 

  public:

    /// Vectors for aperture scale radii [arcmin]
    std::vector<double> theta11, theta12, theta13;
    std::vector<double> theta21, theta22, theta23;
    std::vector<double> theta31, theta32, theta33;
    std::vector<double> cov; //< Inverse Covariance

    std::vector<double> N1N2Map, N1N1Map, N2N2Map; //< Measured Data

    NNMap_Model* model; // Class for Model

    Priors* priors; // Priors

    // Empty constructor
    fitData(){};

    // Constructor from Values
    fitData(const std::string& fn1, const std::string& fn2, const std::string& fn3, const std::string& fncov, Priors* priors_, NNMap_Model* model_);
    
    
  };


  /**
   * Calculates Chi Square
   * @param params HOD Parameters as gsl_vector
   * @param data fitData
   */
  double getChiSquared(const gsl_vector* params, void* data);

  /**
   * Converts from gsl vector (Normalized to 0-1 within prior range) to Params
   * @param params_gsl GSL parameters
   * @param priors Priors
   * @param params Will contain Parameters
   */
  void getParamsFromGSL(const gsl_vector* params_gsl, const Priors* priors, Params* params);

  /**
   * Converts Params to GSL vector (Normalized to 0-1 within prior range)
   * @param params Parameters
   * @param priors Priors
   * @param params_gsl GSL parameters
   */
  void getGSLFromParams(const Params* params, const Priors* priors, gsl_vector* params_gsl);

  /**
   * Reads in sampling points and writes them in params vector
   * @param filename File with sampling points
   * @param params Vector which will contain Parameters
   */
  void readInSamplings(const std::string& filename, std::vector<Params>& params);
  
}


#endif //G3LHALO_FITHELPERS_H
