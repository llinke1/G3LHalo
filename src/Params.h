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
    // /// Number of parameters
    // int N=14;

    double f1; //f parameter
    double alpha1; // alpha Parameter
    double mmin1; // MMin [Msun]
    double sigma1; // Sigma Parameter
    double mprime1; //Mprime [Msun]
    double beta1; //beta
    // double A; // A parameter [1/Msun]
    // double epsilon; // epsilon parameter
   
    Params(){}; //Empty constructor

    /**
     * Constructor from values
     */
    Params(double f1_, double alpha1_, 
	   double mmin1_, double sigma1_, 
	   double mprime1_, double beta1_)
      : f1{f1_}, alpha1{alpha1_},  mmin1{mmin1_}, sigma1{sigma1_}, mprime1{mprime1_},  beta1{beta1_}{
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
