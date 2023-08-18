#ifndef G3LHALO_PARAMS_H
#define G3LHALO_PARAMS_H

#include <string>
#include <vector>

namespace g3lhalo
{

  /**
   * Class containing parameters of G3L Halomodel
   *
   * @author Laila Linke llinke@astro.uni-bonn.de
   */
  class Params
  {
  public:
    double f; //f parameter
    double alpha; // alpha Parameter
    double mmin; // MMin [Msun]
    double sigma; // Sigma Parameter
    double mprime; //Mprime [Msun]
    double beta; //beta

   
    Params(){}; //Empty constructor

    /**
     * Constructor from values
     */
    Params(double f_, double alpha_, 
	   double mmin_, double sigma_, 
	   double mprime_, double beta_)
      : f{f_}, alpha{alpha_},  mmin{mmin_}, sigma{sigma_}, mprime{mprime_},  beta{beta_}{
      };


    /**
     * Constructor from file
     * @param filename FIlename from which params are read
     */
    Params(std::string filename);

    
    
  };
}
  /**
   * Override of << for output
   */
std::ostream& operator<< (std::ostream& out, g3lhalo::Params const& params);

#endif // G3LHALO_PARAMS_H
