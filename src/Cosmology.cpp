#include "Cosmology.h"

#include "cubature.h"

#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include "constants.h"

g3lhalo::Cosmology::Cosmology(const std::string& fn_parameters)
{
  std::ifstream input(fn_parameters.c_str());
  if(input.fail())
    {
      std::cout<<"Cosmology: Could not open "<<fn_parameters<<std::endl;
      return;
    };
  std::vector<std::string> parameterNames;
  std::vector<double> parameterValues;
  
  if(input.is_open())
    {
      std::string line;
      while(std::getline(input, line))
	{
	  if(line[0]=='#' || line.empty()) continue;
	  std::string name;
	  double value;
	  std::istringstream iss(line);
	  iss>>name>>value;
	  parameterNames.push_back(name);
	  parameterValues.push_back(value);
	};
    };

  for(unsigned int i=0; i<parameterNames.size(); i++)
    {
      if(parameterNames.at(i)=="Omega_m")
	{
	  Omega_m=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="Omega_Lambda")
	{
	  Omega_Lambda=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="Omega_b")
	{
	  Omega_b=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="sigma8")
	{
	  sigma8=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="H0")
	{
	  H0=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="ns")
	{
	  ns=parameterValues.at(i);
	}
      else
	{
	  std::cout<<"Cosmology::Parameter file is not in the right format"
		   <<std::endl;
	  return;
	};
    };

  rho_crit=3*H0*H0*Mpc_in_km/8/g3lhalo::pi/G;

}

double g3lhalo::Cosmology::delta_c(const double& z)
{
  return 1.686*(1+0.0123*log10(Omega_m_(z)));
}

double g3lhalo::Cosmology::Omega_m_(const double& z)
{
  return Omega_m*(1+z)*(1+z)*(1+z)/E(z);
}

double g3lhalo::Cosmology::E(const double& z)
{
  return Omega_m*(1+z)*(1+z)*(1+z)+Omega_Lambda;
}

double g3lhalo::Cosmology::comovingDistance(const double& z)
{
  double z_min[1]={0};
  double z_max[1]={z};
  double result, error;

  hcubature_v(1, integrand_comovingDistance, this, 1, z_min, z_max, 0, 0, 1e-4, ERROR_L1, &result, &error);
  
  return result;  
}


double g3lhalo::Cosmology::derivative_comovingDistance(const double& z)
{
  return c/H0/sqrt(E(z));
}

int g3lhalo::integrand_comovingDistance(unsigned ndim, size_t npts, const double* z, void* thisPtr, unsigned fdim, double* value)
{
  if(fdim!=1)
    {
      std::cerr<<"Cosmology::integrand_comovingDistance: Wrong fdim"<<std::endl;
      exit(1);
    };
  Cosmology* cosmo = (Cosmology*) thisPtr;

  #pragma omp parallel for
  for(unsigned int i=0; i<npts; i++)
    {
      double z_=z[i*ndim];
      value[i]=cosmo->derivative_comovingDistance(z_);
    };
  
  return 0;
}

double g3lhalo::Cosmology::growthFunction(double z)
{
  double om_m=Omega_m_(z);
  return 2.5/(z+1)*om_m/(pow(om_m, 4./7.)-Omega_Lambda+(1.+0.5*om_m)*(1+Omega_Lambda/70));
}


double g3lhalo::Cosmology::lensingEfficiency(const double& z, 
					     const std::vector<double>& nz,
					     const std::vector<double>& zs)
{
  double w=comovingDistance(z);
  double result=0;
  int N=zs.size();
  double deltaZ=(zs.at(N-1)-zs.at(0))/N;
  
  for(int i=0; i<N; i++)
    {
      double zprime=zs.at(i);
      double wprime=comovingDistance(zprime);
      if(zprime>=z)
	{
	  result+=nz.at(i)*(wprime-w)/wprime;
	};
    };
  return result*deltaZ;
}


void g3lhalo::Cosmology::getLensingEfficiency(const std::vector<double>& zs,
					      const std::vector<double>& nz,
					      const std::vector<double>& zsources,
					      std::vector<double>& result)
{
  int N=zs.size();
  
  for(int i=0;i<N;i++)
    {
      double z=zs[i];
      result.push_back(lensingEfficiency(z, nz, zsources));
    };
  return;
}

void g3lhalo::Cosmology::getComovingDistance(const std::vector<double>& zs,
			 std::vector<double>& result)
{
  int N=zs.size();

  for(int i=0;i<N;i++)
    {
      double z=zs[i];
      result.push_back(comovingDistance(z));
    };
  return;
}

void g3lhalo::Cosmology::getDerivativeComovingDistance(const std::vector<double>& zs,
				   std::vector<double>& result)
{
  int N=zs.size();

  for(int i=0;i<N;i++)
    {
      double z=zs[i];
      result.push_back(derivative_comovingDistance(z));
    };
  return;
}
