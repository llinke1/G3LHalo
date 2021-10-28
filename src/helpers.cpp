#include "helpers.h"

#include <fstream>


void g3lhalo::checkCmdLine(int argc, int n_params, std::string usage, std::string example)
  {
    if(argc != n_params+1)
      {
	std::cerr<<"Wrong number of CMD Line arguments"<<std::endl;
  std::cerr<<"Expected:"<<n_params+1<<std::endl;
  std::cerr<<"Got:"<<argc<<std::endl;
	std::cerr<<"Usage:"<<usage<<std::endl;
	std::cerr<<"Example:"<<example<<std::endl;
	exit(1);
      };
  }


void g3lhalo::writeToFile(const std::vector<double>& x, const std::vector<double>& y, const std::string& filename)
{
  std::ofstream file;
  file.open(filename);

  int N=x.size();
  for(int i=0; i<N; i++)
    {
      file<<x.at(i)<<" "<<y.at(i)<<std::endl;
    };
}

void g3lhalo::writeToFile(const std::vector<double>& x1, const std::vector<double>& x2, const std::vector<double>& y, const std::string& filename)
{
  std::ofstream file;
  file.open(filename);

  int N=x1.size();
  for(int i=0; i<N; i++)
    {
      for(int j=0; j<N; j++)
	{
	  file<<x1.at(i)<<" "<<x2.at(j)<<" "<<y.at(i*N+j)<<std::endl;
	};
    };
}
