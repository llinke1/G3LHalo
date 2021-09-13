#include "Priors.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>


g3lhalo::Priors::Priors(std::string& filename)
{
  // Open file
  std::ifstream input(filename.c_str());
  if(input.fail())
    {
      std::cerr<<"Priors: Could not open "<<filename<<std::endl;
      std::cerr<<"Exiting"<<std::endl;
      exit(1);
    };

  // Set up read in
  std::vector<std::string> parameterNames;
  std::vector<double> parameterValuesMin;
  std::vector<double> parameterValuesMax;
  
  if(input.is_open())
    {
      std::string line;
      while(std::getline(input, line))
	{
	  if(line[0]=='#' || line.empty()) continue;
	  std::string name;
	  double valmin, valmax;
	  std::istringstream iss(line);
	  iss>>name>>valmin>>valmax;
	  parameterNames.push_back(name);
	  parameterValuesMin.push_back(valmin);
	  parameterValuesMax.push_back(valmax);
	};
    };

  // Assigning to Params
  for(unsigned int i=0; i<parameterNames.size(); i++)
    {
      if(parameterNames.at(i)=="M_min")
	{
	  mminmin=parameterValuesMin.at(i);
	  mminmax=parameterValuesMax.at(i);
	}
      else if(parameterNames.at(i)=="sigma")
	{
	  sigmamin=parameterValuesMin.at(i);
	  sigmamax=parameterValuesMax.at(i);
	}
      else if(parameterNames.at(i)=="M'")
	{
	  mprimemin=parameterValuesMin.at(i);
	  mprimemax=parameterValuesMax.at(i);
	}
      else if(parameterNames.at(i)=="beta")
	{
	  betamin=parameterValuesMin.at(i);
	  betamax=parameterValuesMax.at(i);
	}
      else if(parameterNames.at(i)=="A")
	{
	  Amin=parameterValuesMin.at(i);
	  Amax=parameterValuesMax.at(i);
	}
      else if(parameterNames.at(i)=="epsilon")
	{
	  epsilonmin=parameterValuesMin.at(i);
	  epsilonmax=parameterValuesMax.at(i);
	}
      else if(parameterNames.at(i)=="alpha")
	{
	  alphamin=parameterValuesMin.at(i);
	  alphamax=parameterValuesMax.at(i);
	}
      else if(parameterNames.at(i)=="f")
	{
	  fmin=parameterValuesMin.at(i);
	  fmax=parameterValuesMax.at(i);
	}     
      else
	{
	  std::cerr<<"Priors: Prior file is not in the right format"
		   <<std::endl;
	  std::cerr<<"Exiting"<<std::endl;
	  exit(1);
	};
    };
}
