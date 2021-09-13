#ifndef G3LHALO_PRIORS_H
#define G3LHALO_PRIORS_H

#include <string>

namespace g3lhalo
{
  class Priors
  {
  public:
    /// Number of parameters
    int N=14;

    /// Borders of Prior range
    double fmin, fmax; //f parameter
    double alphamin, alphamax; //alpha parameter
    double mminmin, mminmax; // Mmin parameter [Msun]
    double sigmamin, sigmamax; //Sigma parameter
    double mprimemin, mprimemax; //Mprime parameter [Msun]
    double betamin, betamax; //Beta parameter
    double Amin, Amax; // A parameter
    double epsilonmin, epsilonmax; // Epsilon parameter

    /// Empty Constructor
    Priors(){};

    /**
     * Constructor from filename
     * @param filename Filename from which priors are readed
     */
    Priors(std::string& filename);
  };
}

#endif //G3LHALO_PRIORS_H
