#include "Params.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

g3lhalo::Params::Params(std::string filename)
{

  // Open file
  std::ifstream input(filename.c_str());
  if(input.fail())
    {
      std::cerr<<"Params: Could not open "<<filename<<std::endl;
      std::cerr<<"Exiting"<<std::endl;
      exit(1);
    };

  // Set up read in
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

  // Assigning to Params
  for(unsigned int i=0; i<parameterNames.size(); i++)
    {
      if(parameterNames.at(i)=="M_min1")
	{
	  mmin1=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="sigma_1")
	{
	  sigma1=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="M'_1")
	{
	  mprime1=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="beta_1")
	{
	  beta1=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="alpha_1")
	{
	  alpha1=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="f_1")
	{
	  f1=parameterValues.at(i);
	}    
      else
	{
	  std::cerr<<"Params: Parameter file is not in the right format"
		   <<std::endl;
	std::cerr<<"Check "<<filename<<std::endl;
	  std::cerr<<"Exiting"<<std::endl;
	  exit(1);
	};
    };
}


std::ostream& operator<< (std::ostream& out, g3lhalo::Params const& params)
{
  out<<"M_min1 "<<params.mmin1<<std::endl;
  out<<"sigma_1 "<<params.sigma1<<std::endl;
  out<<"M'_1 "<<params.mprime1<<std::endl;
  out<<"beta_1 "<<params.beta1<<std::endl;
  out<<"alpha_1 "<<params.alpha1<<std::endl;
  out<<"f_1 "<<params.f1<<std::endl;
  return(out);
}
