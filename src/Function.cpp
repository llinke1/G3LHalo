#include "Function.h"

#include <string>
#include <fstream>

g3lhalo::Function::Function(std::string filename, const double& default_value)
{
  //Reading in from file
  if(filename != "none") //Only read in if filename is not "none"
    {
      std::ifstream input(filename.c_str());
      if(!input.is_open()) //checking if file can be opened
	{
	  std::cerr << "Function: Could not open input file:"<<filename<<" Exiting. \n";
	  exit(1);
	};
      double x,y;
      while(input>>x>>y)
	{ 
	  x_values_.push_back(x);
	  y_values_.push_back(y);
	};

      if(input.bad())
	{
	  std::cerr << "Function: Invalid values in "<<filename<<" Exiting. \n";
	  exit(1);
	};
    };
  //Setting default value
  default_value_=default_value;
}

double g3lhalo::Function::interpolate(const int& index1, const int& index2,
				   const double& x)
{
  double a = (y_values_.at(index1) - y_values_.at(index2)) /
    (x_values_.at(index1) - x_values_.at(index2));

  double b = y_values_.at(index1) - a * x_values_.at(index1);

  return a * x + b;  
}

double g3lhalo::Function::at(const double& x)
{
  //If Function is empty: Return Default Value
  if(x_values_.size() == 0) return default_value_;

  if(x>x_values_.back()) return default_value_;	

  if(x<x_values_.front()) return default_value_;

  //Return Functionvalue
  for (unsigned int i=0; i<x_values_.size()-1; i++)
    {
      if(x_values_.at(i)<=x && x_values_.at(i+1)>x)
	{
	  double a=(y_values_.at(i+1)-y_values_.at(i))/(x_values_.at(i+1)-x_values_.at(i));
	  return a*(x-x_values_.at(i))+y_values_.at(i);
	};
    };

 
  return default_value_;
}

double g3lhalo::Function::x_lower(const double& y)
{
  for(unsigned int i=0; i<x_values_.size()-1; i++)
    {
      if(y_values_.at(i)<=y && y_values_.at(i+1)>y)
	{
	  return x_values_.at(i);
	}
    }

  return x_values_.back();
}


int g3lhalo::Function::read(const int& Nbins, const double& min, const double& max, std::vector<double>& values)
{
  double bin=(max-min)/Nbins;
  for(int i=0; i<Nbins; i++)
    {
      double x=min+i*bin;
      double val=at(x);
      values.push_back(val);
    };
  int Nvalues=values.size();
  if(Nvalues!=Nbins)
    {
      std::cerr<<"Function::read: Couldn't read function! Exiting."<<std::endl;
      exit(1);
    };
  return 0;
}


int g3lhalo::Function::readLog(const int& Nbins, const double& min, const double& max, std::vector<double>& values)
{
  double bin=log(max/min)/Nbins;
  for(int i=0; i<Nbins; i++)
    {
      double x=exp(log(min)+i*bin);
      double val=at(x);
      values.push_back(val);
    };
  if(values.size()!=Nbins)
    {
      std::cerr<<"Function::readLog: Couldn't read function! Exiting."<<std::endl;
      exit(1);
    };
  return 0;
}
