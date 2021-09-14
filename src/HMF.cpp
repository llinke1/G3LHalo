#include "HMF.h"

#include <cmath>
#include <iostream>
#include "constants.h"

double g3lhalo::Halomassfunction::n(const double& m, const double& z)
{

  double A=0.322;
  double q=0.707;
  double p=0.3;
  
  // Get sigma^2(m, z)
  double sigma2=powerspectrum->sigma2(m, z);


  // Get dsigma^2/dm
  double dsigma2=powerspectrum->dsigma2dm(m, z);

  // Get critical density contrast
  double delta_c=cosmo->delta_c(z);

  // Get mean density
  double rho_mean=cosmo->Omega_m_(z)*cosmo->rho_crit;

  double nuSq=delta_c*delta_c/sigma2;
  
  /*
   * \f $n(m,z)=-\frac{\bar{\rho}}{m \sigma} \dv{\sigma}{m} A \sqrt{2q/pi} (1+(\frac{\sigma^2}{q\delta_c^2})^p) \frac{\delta_c}{\sigma} \exp(-\frac{q\delta_c^2}{2\sigma^2})$ \f
   */
  return -rho_mean/m/sigma2*dsigma2*A*(1+pow(q*nuSq, -p))*sqrt(q*nuSq/2/g3lhalo::pi)*exp(-0.5*q*nuSq);
}


double g3lhalo::Halomassfunction::b_h(const double& m, const double& z)
{
  double q=0.707;
  double p=0.3;
  
  // Get sigma^2(m, z)
  double sigma2=powerspectrum->sigma2(m, z);

  // Get critical density contrast
  double delta_c=cosmo->delta_c(z);

  return 1+1./delta_c*(q*delta_c*delta_c/sigma2 - 1 + 2*p/(1+pow(q*delta_c*delta_c/sigma2, p)));
  
}


void g3lhalo::Halomassfunction::getHMF(const std::vector<double>& zs,
				    const std::vector<double> &ms,
				    std::vector<double>& result)
{
  int N=zs.size();

  for(int i=0; i<N; i++)
    {
      double z=zs[i];
      for(int j=0; j<N; j++)
	{
	  double m=ms[j];
	  result.push_back(n(m, z));
	}
    }
}

void g3lhalo::Halomassfunction::getHaloBias(const std::vector<double>& zs,
					 const std::vector<double> &ms,
					 std::vector<double>& result)
{
  int N=zs.size();

  for(int i=0; i<N; i++)
    {
      double z=zs[i];
      for(int j=0; j<N; j++)
	{
	  double m=ms[j];
	  result.push_back(b_h(m, z));
	}
    }
}
