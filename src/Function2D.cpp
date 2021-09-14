#include "Function2D.h"

#include <string>
#include <fstream>
#include <algorithm> // std::min_element
#include <iterator>  // std::begin, std::end

g3lhalo::Function2D::Function2D(std::string filename, const double& default_value)
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
      double x1, x2, y;
      while(input>>x1>>x2>>y)
	{ 
	  x1_values_.push_back(x1);
	  x2_values_.push_back(x2);
	  y_values_.push_back(y);
	};

      if(input.bad())
	{
	  std::cerr << "Function2D: Invalid values in "<<filename<<" Exiting. \n";
	  exit(1);
	};
    };
  //Setting default value
  default_value_=default_value;

  x1_min=*std::min_element( std::begin(x1_values_), std::end(x1_values_) );
  x1_max=*std::max_element( std::begin(x1_values_), std::end(x1_values_) );
  x2_min=*std::min_element( std::begin(x2_values_), std::end(x2_values_) );
  x2_max=*std::max_element( std::begin(x2_values_), std::end(x2_values_) );
  
  N1 = sqrt(x1_values_.size());
  N2 = sqrt(x2_values_.size());
}


double g3lhalo::Function2D::at(const double& x1, const double& x2)
{
  //If Function is empty: Return Default Value
  if(x1_values_.size() == 0 || x2_values_.size() == 0) return default_value_;


  
  // Get index x1 (lin)
  int ix1=std::round((x1-x1_min)*N1/(x1_max-x1_min));
  // Get index x2 (log)
  int ix2=std::round(log(x2/x2_min)*N2/log(x2_max/x2_min));

  if(ix1 > N1 || ix2 > N2) return default_value_;
  if(ix1 < 0 || ix2 < 0) return default_value_;
  
  //Return Functionvalue
  int index=ix1*N2+ix2;
  return y_values_.at(index);
}

int g3lhalo::Function2D::read(const int& Nbins, const double& min1, const double& max1, const double& min2, const double& max2, std::vector<double>& values)
{
  x1_min=min1;
  x1_max=max1;
  x2_min=min2;
  x2_max=max2;
  N1=Nbins;
  N2=Nbins;
  
  double bin1=(max1-min1)/Nbins;
  double bin2=log(max2/min2)/Nbins;
  
  for(int i=0; i<Nbins; i++)
    {
      double x1=min1+i*bin1;
      for (int j=0; j<Nbins; j++)
	{
	  double x2=exp(log(min2)+j*bin2);
	  double val=at(x1, x2);
	  values.push_back(val);
	
	};
    };
  if(values.size()!=Nbins*Nbins)
    {
      std::cerr<<"Function2D::read: Couldn't read function! Exiting."<<std::endl;
      exit(1);
    };
  return 0;
}
