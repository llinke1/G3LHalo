#include "NNMap_Model.h"
#include "Params.h"
#include "Priors.h"
#include "Function.h"
#include "Function2D.h"
#include "constants.h"
#include "helpers.h"

#include <fstream>

int main(int argc, char* argv[])
  {
  int n_params=20;
  std::string usage="./calculateDerivativeNNMap_twoPop.x \n Cosmology File \n minimal redshift \n maximal redshift \n minimal halo mass [Msun] \n maximal halo mass [Msun] \n minimal k [1/Mpc] \n maximal k [1/Mpc] Number of bins \n File for lensing efficiency \n File for lens redshift distribution \n File for lens redshift distribution \n File for comoving distance \n File for derivative of comoving distance \n File for Halo Mass function \n File for Linear power spectrum \n File for Halo bias \n File for concentration \n File for Halomodel Params \n File with thetas \n File for Halomodel Priors";
  std::string example="./calculateNNMap_twoPop.x cosmo.param 0.1 2 1e10 1e17 0.01 1000 256 g.dat w.dat dwdz.dat hmf.dat Plin.dat b_h.dat conc.dat hod.param thetas.dat priors.param";

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

  std::string fn_thetas=argv[19];
  std::string fn_priors=argv[20];

#if VERBOSE
  std::cerr<<"Finished reading cli"<<std::endl;
#endif
  
  g3lhalo::Function g(fn_g, 1.0);
  g3lhalo::Function plens1(fn_plens1, 0.0);
  g3lhalo::Function plens2(fn_plens2, 0.0);
  g3lhalo::Function w(fn_w, 0.0);
  g3lhalo::Function dwdz(fn_dwdz, 0.0);

#if VERBOSE
   std::cerr<<"Finished assigning fucntions1D"<<std::endl;
#endif
   
  g3lhalo::Function2D hmf(fn_hmf, 0.0);
  g3lhalo::Function2D P_lin(fn_P_lin, 0.0);
  g3lhalo::Function2D b_h(fn_b_h, 0.0);
  g3lhalo::Function2D concentration(fn_concentration, 0.0);

#if VERBOSE
  std::cerr<<"Finished assigning fucntions2D"<<std::endl;
#endif
  
  g3lhalo::Params params(fn_params);
  g3lhalo::Priors priors(fn_priors);
  double h=0.001;

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
  
  /** READ IN THETA VALUES **/
  std::ifstream input(fn_thetas);
  std::vector<double> thetas1, thetas2, thetas3;
  double theta1, theta2, theta3;
  while(input>>theta1>>theta2>>theta3)
    {
      thetas1.push_back(theta1*g3lhalo::arcmin_in_rad);
      thetas2.push_back(theta2*g3lhalo::arcmin_in_rad);
      thetas3.push_back(theta3*g3lhalo::arcmin_in_rad);
    };
  int n=thetas1.size();

#if VERBOSE
  std::cerr<<"Finished reading in Thetas"<<std::endl;
#endif
  
  /** Get param differentials **/
  double df=h*(priors.fmax-priors.fmin);
  double dalpha=h*(priors.alphamax-priors.alphamin);
  double dmmin=h*(priors.mminmax-priors.mminmin);
  double dsigma=h*(priors.sigmamax-priors.sigmamin);
  double dmprime=h*(priors.mprimemax-priors.mprimemin);
  double dbeta=h*(priors.betamax-priors.betamin);
  double dA=h*(priors.Amax-priors.Amin);
  double depsilon=h*(priors.epsilonmax-priors.epsilonmin);

  std::vector<double> nnm_values;
  //f1+df
   g3lhalo::Params newParams=params;
   newParams.f1=params.f1+df;
   nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for f1+df"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //f1-df
  newParams=params;
  newParams.f1=std::max(params.f1-df, 0.01);
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for f1-df"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //f2+df
  newParams=params;
  newParams.f2=params.f2+df;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for f2+df"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //f2-df
  newParams=params;
  newParams.f2=std::max(params.f2-df, 0.01);
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for f2-df"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();

  //alpha1+dalpha
  newParams=params;
  newParams.alpha1=params.alpha1+dalpha;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for alpha1+dalpha"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //alpha1-dalpha
  newParams=params;
  newParams.alpha1=params.alpha1-dalpha;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for alpha1-dalpha"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
    //alpha2+dalpha
  newParams=params;
  newParams.alpha2=params.alpha2+dalpha;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for alpha2+dalpha"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //alpha2-dalpha
  newParams=params;
  newParams.alpha2=params.alpha2-dalpha;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for alpha2-dalpha"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;


  //mmin1+dmmin
  newParams=params;
  newParams.mmin1=params.mmin1+dmmin;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for mmin1+mmin"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //mmin1-dmmin
  newParams=params;
  newParams.mmin1=params.mmin1-dmmin;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for mmin1-mmin"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //mmin2+dmmin
  newParams=params;
  newParams.mmin2=params.mmin2+dmmin;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for mmin2+mmin"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //mmin2-dmmin
  newParams=params;
  newParams.mmin2=params.mmin2-dmmin;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for mmin2-mmin"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //sigma1+dsigma
  newParams=params;
  newParams.sigma1=params.sigma1+dsigma;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for sigma1+dsigma"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //sigma1-dsigma
  newParams=params;
  newParams.sigma1=params.sigma1-dsigma;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for sigma1-dsigma"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //sigma2+dsigma
  newParams=params;
  newParams.sigma2=params.sigma2+dsigma;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for sigma2+dsigma"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //sigma2-dsigma
  newParams=params;
  newParams.sigma2=params.sigma2-dsigma;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for sigma2-dsigma"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //mprime1+dmprime
  newParams=params;
  newParams.mprime1=params.mprime1+dmprime;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for mprime1+dmprime"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //mprime1-dmprime
  newParams=params;
  newParams.mprime1=params.mprime1-dmprime;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for mprime1-dmprime"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //mprime2+dmprime
  newParams=params;
  newParams.mprime2=params.mprime2+dmprime;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for mprime2+dmprime"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //mprime2-dmprime
  newParams=params;
  newParams.mprime2=params.mprime2-dmprime;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for mprime2-dmprime"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //beta1+dbeta
  newParams=params;
  newParams.beta1=params.beta1+dbeta;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for beta1+dbeta"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //beta1-dbeta
  newParams=params;
  newParams.beta1=params.beta1-dbeta;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for beta1-dbeta"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
    //beta2+dbeta
  newParams=params;
  newParams.beta2=params.beta2+dbeta;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for beta2+dbeta"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //beta2-dbeta
  newParams=params;
  newParams.beta2=params.beta2-dbeta;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for beta2-dbeta"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //A+dA
  newParams=params;
  newParams.A=params.A+dA;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for A+dA"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //A-dA
  newParams=params;
  newParams.A=params.A-dA;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for A-dA"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
   //epsilon+depsilon
  newParams=params;
  newParams.epsilon=params.epsilon+depsilon;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for epsilon+depsilon"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  //epsilon-depsilon
  newParams=params;
  newParams.epsilon=params.epsilon-depsilon;
    nnmap.calculateAll(thetas1, thetas2, thetas3, &newParams, nnm_values);
#if VERBOSE
  std::cerr<<"Finished calculation for epsilon-depsilon"<<std::endl;
#endif
  for(int i=0; i<3*n; i++)
    {
      std::cout<<nnm_values.at(i)<<" ";
    };
  std::cout<<std::endl;
  nnm_values.clear();
  
  return 0;
}
