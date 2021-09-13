#include "NNMap_Model.h"
#include "fitHelpers.h"
#include "helpers.h"
#include "Function.h"
#include "Function2D.h"
#include "Priors.h"

#include <fstream>

int main(int argc, char* argv[])
{
  int n_params=24;
  std::string usage="./calculateChiSquared.x \n Cosmology File \n minimal redshift \n maximal redshift \n minimal halo mass [Msun] \n maximal halo mass [Msun] \n minimal k [1/Mpc] \n maximal k [1/Mpc] Number of bins \n File for lensing efficiency \n File for lens redshift distribution \n File for lens redshift distribution \n File for comoving distance \n File for derivative of comoving distance \n File for Halo Mass function \n File for Linear power spectrum \n File for Halo bias \n File for concentration \n File for Halomodel Params \n File with N1N2M measurement \n File with N1N2M measurement \n File with N2N2M measurement \n File with covariance \n File with priors \n File with sampling points";
  std::string example="./calculateChiSquared.x cosmo.param 0.1 2 1e10 1e17 0.01 1000 256 g.dat w.dat dwdz.dat hmf.dat Plin.dat b_h.dat conc.dat hod.param n1n2m.dat n1n1m.dat n2n2m.dat cov.dat priors.param samplingpoints.dat";

  g3lhalo::checkCmdLine(argc, n_params, usage, example);
  
  std::string fn_cosmo=argv[1];
  double zmin=std::stod(argv[2]);
  double zmax=std::stod(argv[3]);
  double kmin=std::stod(argv[4]);
  double kmax=std::stod(argv[5]);
  double mmin=std::stod(argv[6]);
  double mmax=std::stod(argv[7]);
  int Nbins=std::stoi(argv[8]);

  std::string fn_g=argv[9];
  std::string fn_plens1=argv[10];
  std::string fn_plens2=argv[11];
  std::string fn_w=argv[12];
  std::string fn_dwdz=argv[13];
  std::string fn_hmf=argv[14];
  std::string fn_P_lin=argv[15];
  std::string fn_b_h=argv[16];
  std::string fn_concentration=argv[17];
  std::string fn_params=argv[18];

  std::string fn_n1n2map=argv[19];
  std::string fn_n1n1map=argv[20];
  std::string fn_n2n2map=argv[21];
  std::string fn_cov=argv[22];
  std::string fn_priors=argv[23];
  std::string fn_samplingpoints=argv[24];

#if VERBOSE
  std::cerr<<"Finished reading cli"<<std::endl;
#endif
  
  g3lhalo::Function g(fn_g, 1.0);
  g3lhalo::Function plens1(fn_plens1, 0.0);
  g3lhalo::Function plens2(fn_plens2, 0.0);
  g3lhalo::Function w(fn_w, 0.0);
  g3lhalo::Function dwdz(fn_dwdz, 0.0);

#if VERBOSE
  std::cerr<<"Finished assigning functions1D"<<std::endl;
#endif
  
  g3lhalo::Function2D hmf(fn_hmf, 0.0);
  g3lhalo::Function2D P_lin(fn_P_lin, 0.0);
  g3lhalo::Function2D b_h(fn_b_h, 0.0);
  g3lhalo::Function2D concentration(fn_concentration, 0.0);

#if VERBOSE
  std::cerr<<"Finished assigning functions2D"<<std::endl;
#endif
  
  g3lhalo::Params params(fn_params);

#if VERBOSE
  std::cerr<<"Finished assigning fucntions"<<std::endl;
#endif
  
  std::vector<double> g_val, plens1_val, plens2_val, w_val, dwdz_val, hmf_val, P_lin_val, b_h_val, concentration_val;

  if( g.read(Nbins, zmin, zmax, g_val)
      || plens1.read(Nbins, zmin, zmax, plens1_val)
      || plens2.read(Nbins, zmin, zmax, plens2_val)
      || w.read(Nbins, zmin, zmax, w_val)
      || dwdz.read(Nbins, zmin, zmax, dwdz_val)
      || hmf.read(Nbins, zmin, zmax, mmin, mmax, hmf_val)
      || P_lin.read(Nbins, zmin, zmax, kmin, kmax, P_lin_val)
      || b_h.read(Nbins, zmin, zmax, mmin, mmax, b_h_val)
      || concentration.read(Nbins, zmin, zmax, mmin, mmax, concentration_val))
    {
      std::cerr<<"Problem reading in files. Exiting."<<std::endl;
      exit(1);
    };

#if VERBOSE
  std::cerr<<"Finished reading in Functions"<<std::endl;
#endif
  // Set up cosmology
  g3lhalo::Cosmology cosmo(fn_cosmo);
  
  g3lhalo::NNMap_Model nnmap(&cosmo, zmin, zmax, kmin, kmax, mmin, mmax, Nbins, g_val.data(), plens1_val.data(), plens2_val.data(), w_val.data(), dwdz_val.data(), hmf_val.data(), P_lin_val.data(), b_h_val.data(), concentration_val.data(), &params);

#if VERBOSE
  std::cerr<<"Finished initializing NNMap"<<std::endl;
#endif
  
  g3lhalo::Priors priors(fn_priors);
  
  g3lhalo::fitData fitdata(fn_n1n2map, fn_n1n1map, fn_n2n2map, fn_cov, &priors, &nnmap);

  // Read in samplings
  std::vector<g3lhalo::Params> samplings;
  g3lhalo::readInSamplings(fn_samplingpoints, samplings);

  gsl_vector* params_gsl;
  params_gsl=gsl_vector_alloc(params.N);

  // Calculate chi2 for samplings
  for(int i=0; i<samplings.size(); i++)
    {
      getGSLFromParams(&samplings[i], &priors, params_gsl);

      getChiSquared(params_gsl, &fitdata);
    };

  gsl_vector_free(params_gsl);
  return 0;
}
