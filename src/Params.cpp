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
      if(parameterNames.at(i)=="M_min")
	{
	  mmin=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="sigma")
	{
	  sigma=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="M'")
	{
	  mprime=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="beta")
	{
	  beta=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="alpha")
	{
	  alpha=parameterValues.at(i);
	}
      else if(parameterNames.at(i)=="f")
	{
	  f=parameterValues.at(i);
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
  out<<"M_min "<<params.mmin<<std::endl;
  out<<"sigma "<<params.sigma<<std::endl;
  out<<"M' "<<params.mprime<<std::endl;
  out<<"beta "<<params.beta<<std::endl;
  out<<"alpha "<<params.alpha<<std::endl;
  out<<"f "<<params.f<<std::endl;
  return(out);
}
