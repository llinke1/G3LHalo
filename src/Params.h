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
    /// Number of parameters
    int N=14;

    double f1, f2; //f parameter
    double alpha1, alpha2; // alpha Parameter
    double mmin1, mmin2; // MMin [Msun]
    double sigma1, sigma2; // Sigma Parameter
    double mprime1, mprime2; //Mprime [Msun]
    double beta1, beta2; //beta
    double A; // A parameter [1/Msun]
    double epsilon; // epsilon parameter
   
    Params(){}; //Empty constructor

    /**
     * Constructor from values
     */
    Params(double f1_, double f2_, double alpha1_, double alpha2_,
	   double mmin1_, double mmin2_, double sigma1_, double sigma2_,
	   double mprime1_, double mprime2_, double beta1_, double beta2_,
	   double A_, double epsilon_)
      : f1{f1_}, f2{f2_}, alpha1{alpha1_}, alpha2{alpha2_}, mmin1{mmin1_}, mmin2{mmin2_}, sigma1{sigma1_}, sigma2{sigma2_}, mprime1{mprime1_}, mprime2{mprime2_}, beta1{beta1_}, beta2{beta2_}, A{A_}, epsilon{epsilon_}{
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
