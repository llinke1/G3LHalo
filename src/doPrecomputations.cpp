#include "Cosmology.h"
#include "Function.h"
#include "helpers.h"
#include "Powerspectrum.h"
#include "HMF.h"

int main(int argc, char* argv[])
{
  int n_params=16;
  std::string usage="./doPrecomputations.x \n Cosmology File \n  File with source redshift distribution \n minimal redshift \n maximal redshift \n minimal halo mass [Msun] \n maximal halo mass [Msun] \n minimal k [1/Mpc] \n maximal k [1/Mpc] Number of bins \n Filename lensing efficiency \n Filename angular diameter distance \n Filename derivative angular diameter distance \n Filename HMF \n Filename linear power spectrum \n Filename halo bias \n Filename concentration \n";
  std::string example="./doPrecomputations.x cosmo.param ../results/ n_z_sources.dat 0.1 2 1e10 1e17 0.01 1000 256";

  g3lhalo::checkCmdLine(argc, n_params, usage, example);

  std::string fn_cosmo=argv[1];
  std::string fn_nz=argv[2];
  double z_min=std::stod(argv[3]);
  double z_max=std::stod(argv[4]);
  double m_min=std::stod(argv[5]);
  double m_max=std::stod(argv[6]);
  double k_min=std::stod(argv[7]);
  double k_max=std::stod(argv[8]);
  int Nbins=std::stoi(argv[9]);

  std::string fn_g=argv[10];
  std::string fn_w=argv[11];
  std::string fn_dwdz=argv[12];
  std::string fn_hmf=argv[13];
  std::string fn_p=argv[14];
  std::string fn_bh=argv[15];
  std::string fn_conc=argv[16];
  
#if VERBOSE
  std::cerr<<"Finished reading CLI"<<std::endl;
#endif
  
  // Set up binning

  double z_bin=(z_max-z_min)/Nbins;
  double m_bin=log(m_max/m_min)/Nbins;
  double k_bin=log(k_max/k_min)/Nbins;

  std::vector<double> zs, ms, ks;

  for(int i=0; i<Nbins; i++)
    {
      zs.push_back(z_min+i*z_bin);
      ms.push_back(exp(log(m_min)+m_bin*i));
      ks.push_back(exp(log(k_min)+k_bin*i));
    };
  
  // Read n(z)
  g3lhalo::Function nz(fn_nz, 0);
  
  
  // Set up cosmology
  g3lhalo::Cosmology cosmo(fn_cosmo);

  // Calculate lensing efficiency
  std::vector<double> lensingEfficiency;
  cosmo.getLensingEfficiency(zs, nz.y_values_, nz.x_values_, lensingEfficiency);

  g3lhalo::writeToFile(zs, lensingEfficiency, fn_g);

#if VERBOSE
  std::cerr<<"Finished calculating Lensing Efficiency"<<std::endl;
#endif
  
  // Calculate comoving distance
  std::vector<double> comovingDistance;
  cosmo.getComovingDistance(zs, comovingDistance);

  g3lhalo::writeToFile(zs, comovingDistance, fn_w);

#if VERBOSE
  std::cerr<<"Finished calculating comoving distance"<<std::endl;
#endif
  
  // Calculate Derivative of Comoving Distance
  std::vector<double> derivativeComovingDistance;
  cosmo.getDerivativeComovingDistance(zs, derivativeComovingDistance);

#if VERBOSE
  std::cerr<<"Finished calculating derivative comoving distance"<<std::endl;
#endif
  g3lhalo::writeToFile(zs, derivativeComovingDistance, fn_dwdz);


  // Set up Powerspectrum
  g3lhalo::Powerspectrum powerspectrum(&cosmo);

  // Calculate powerspectrum
  std::vector<double> spectrum;
  powerspectrum.getSpectrum(zs, ks, spectrum);
  g3lhalo::writeToFile(zs, ks, spectrum, fn_p);
#if VERBOSE
  std::cerr<<"Finished calculating power spectrum"<<std::endl;
#endif
   
  // Calculate concentration
  std::vector<double> concentration;
  powerspectrum.getConcentration(zs, ms, concentration);
  g3lhalo::writeToFile(zs, ms, concentration, fn_conc);
#if VERBOSE
  std::cerr<<"Finished calculating concentration"<<std::endl;
#endif
  // Set up Halo Mass Function
  g3lhalo::Halomassfunction hmf(&cosmo, &powerspectrum);

#if VERBOSE
  std::cerr<<"Finished initializing hmf"<<std::endl;
#endif
  // Calculate Halomassfunction
  std::vector<double> n;
  hmf.getHMF(zs, ms, n);
  g3lhalo::writeToFile(zs, ms, n, fn_hmf);

#if VERBOSE
  std::cerr<<"Finished calculating hmf"<<std::endl;
#endif
  
  // Calculate Halo bias
  std::vector<double> b_h;
  hmf.getHaloBias(zs, ms, b_h);
  g3lhalo::writeToFile(zs, ms, b_h, fn_bh);

#if VERBOSE
  std::cerr<<"Finished calculating halo bias"<<std::endl;
  std::cerr<<"Done!"<<std::endl;
#endif
  
  return 0;
  

  
}
  
