#include "Powerspectrum.h"

#include "cubature.h"

#include <iostream>
#include <numbers>
#include <cmath>

g3lhalo::Powerspectrum::Powerspectrum(Cosmology* cosmo_):cosmo(cosmo_)
  {
    set_transfer_parameters();
    set_amplitude();
  }

void g3lhalo::Powerspectrum::set_transfer_parameters()
{
  double z_drag_b1, z_drag_b2;
  double alpha_c_a1, alpha_c_a2, beta_c_b1, beta_c_b2, alpha_b_G, y;
  double f_baryon;
  double Tcmb;

  double h=cosmo->H0/100;
  omhh = cosmo->Omega_m*h*h;
  obhh = cosmo->Omega_b*h*h;
  
  f_baryon=obhh/omhh;
  Tcmb = 2.72548; // PLANCK
  theta_cmb = Tcmb/2.7;
  double theta_cmbSq=theta_cmb*theta_cmb;
  double theta_cmb4=theta_cmbSq*theta_cmbSq;

  z_equality = 2.50e4*omhh/theta_cmb4;  // Really 1+z
  k_equality = 0.0746*omhh/theta_cmbSq;

  z_drag_b1 = 0.313*pow(omhh,-0.419)*(1.0+0.607*pow(omhh,0.674));
  z_drag_b2 = 0.238*pow(omhh,0.223);
  z_drag = 1291.0*pow(omhh,0.251)/(1.0+0.659*pow(omhh,0.828))*(1.0+z_drag_b1*pow(obhh,z_drag_b2));
  
  R_drag = 31.5*obhh/(theta_cmb4)*(1000.0/(1.0+z_drag));
  R_equality = 31.5*obhh/(theta_cmb4)*(1000.0/z_equality);

  sound_horizon = 2.0/3.0/k_equality*sqrt(6.0/R_equality)*log((sqrt(1.0+R_drag)+sqrt(R_drag+R_equality))/(1.0+sqrt(R_equality)));

  k_silk = 1.6*pow(obhh,0.52)*pow(omhh,0.73)*(1.0+pow(10.4*omhh,-0.95));

  alpha_c_a1 = pow(46.9*omhh,0.670)*(1.0+pow(32.1*omhh,-0.532));
  alpha_c_a2 = pow(12.0*omhh,0.424)*(1.0+pow(45.0*omhh,-0.582));
  alpha_c = pow(alpha_c_a1,-f_baryon)*pow(alpha_c_a2,-pow(f_baryon, 3));
  
  beta_c_b1 = 0.944/(1.0+pow(458.0*omhh,-0.708));
  beta_c_b2 = pow(0.395*omhh, -0.0266);
  beta_c = 1.0/(1.0+beta_c_b1*(pow(1.0-f_baryon, beta_c_b2)-1.0));

  y = z_equality/(1.0+z_drag);
  alpha_b_G = y*(-6.0*sqrt(1.0+y)+(2.0+3.0*y)*log((sqrt(1.0+y)+1.0)/(sqrt(1.0+y)-1.0)));
  alpha_b = 2.07*k_equality*sound_horizon*pow(1.0+R_drag,-0.75)*alpha_b_G;

  beta_node = 8.41*pow(omhh, 0.435);
  beta_b = 0.5+f_baryon+(3.0-2.0*f_baryon)*sqrt(pow(17.2*omhh,2.0)+1.0);

  k_peak = 2.5*3.14159*(1.0+0.217*omhh)/sound_horizon;
  sound_horizon_fit = 44.5*log(9.83/omhh)/sqrt(1.0+10.0*pow(obhh,0.75));

  alpha_gamma = 1.0-0.328*log(431.0*omhh)*f_baryon + 0.38*log(22.3*omhh)*(f_baryon*f_baryon);
}

void g3lhalo::Powerspectrum::set_amplitude()
{
  double factor=std::numbers::pi*cosmo->sigma8;

  double k_min[1]={0};
  double k_max[1]={1e12};
  double result, error;
  
  hcubature_v(1, integrand_amplitude, this, 1, k_min, k_max, 0, 0, 1e-4, ERROR_L1, &result, &error);
  amplitude=2*factor*factor/result;
}

int g3lhalo::integrand_amplitude(unsigned ndim, size_t npts, const double* k, void* thisPtr, unsigned fdim, double* value)
{
  if(fdim!=1)
    {
      std::cerr<<"Powerspectrum::integrand_amplitude: Wrong fdim"<<std::endl;
      exit(1);
    };
  Powerspectrum* powerspectrum = (Powerspectrum*) thisPtr;

  double R=8./powerspectrum->cosmo->H0*100;
 
  
  #pragma omp parallel for
  for(unsigned int i=0; i<npts; i++)
    {
      double k_=k[i*ndim];
      double W=powerspectrum->tophat(k_, R);
      double T=powerspectrum->transfer(k_);
      
      value[i]=W*W*T*T*pow(k_,  powerspectrum->cosmo->ns+2);
    };
  
  return 0;
}

double g3lhalo::Powerspectrum::tophat(const double& k, const double& R)
{
  double x=k*R;
  return 3/x/x/x*(sin(x)-x*cos(x));
}

double g3lhalo::Powerspectrum::d_tophat(const double& k, const double& R)
{
  double x=k*R;
  return ((x*x-3.)*sin(x)+3.*x*cos(x))*3*k/x/x/x/x;
}


double g3lhalo::Powerspectrum::mStar(const double& z)
{
  double delta_c=cosmo->delta_c(z);

  double m_min=1e10;
  double m_max=1e18;
  int Nbins=256;

  double m_bin=log(m_max/m_min)/Nbins;

  std::vector<double> difference;
  std::vector<double> ms;
  for(int i=0; i<Nbins; i++)
    {
      double m= exp(log(m_min)+m_bin*i);
      double diff=delta_c - sqrt(sigma2(m, z));
      ms.push_back(m);
      difference.push_back(sqrt(diff*diff));
    };

  int index_minElement=std::distance(difference.begin(), std::min_element(difference.begin(), difference.end()));

  return ms.at(index_minElement);			      
  
}

double g3lhalo::Powerspectrum::concentration(const double& m, const double& mstar, const double& z)
{
  double c0=9;
  double alpha=0.13;

  double concentration=c0/(1.+z)*pow(m/mstar, -alpha);
  return concentration;
}


double g3lhalo::Powerspectrum::transfer(const double& k)
{
  double T_c_ln_beta, T_c_ln_nobeta, T_c_C_alpha, T_c_C_noalpha;
  double q, xx, xx_tilde;
  double T_c_f, T_c, s_tilde, T_b_T0, T_b, f_baryon, T_full;
  
   
  if (k==0.0)
    return 1.0;
  
  q = k/13.41/k_equality;
  xx = k*sound_horizon;
  
  T_c_ln_beta = log(2.718282+1.8*beta_c*q);
  T_c_ln_nobeta = log(2.718282+1.8*q);
  T_c_C_alpha = 14.2/alpha_c + 386.0/(1.0+69.9*pow(q,1.08));
  T_c_C_noalpha = 14.2 + 386.0/(1+69.9*pow(q,1.08));
  
  T_c_f = 1.0/(1.0+pow(xx/5.4, 4));
  T_c = T_c_f*T_c_ln_beta/(T_c_ln_beta+T_c_C_noalpha*q*q) +
    (1.-T_c_f)*T_c_ln_beta/(T_c_ln_beta+T_c_C_alpha*q*q);
  
  s_tilde = sound_horizon*pow(1.+pow(beta_node/xx, 3),-1./3.);
  xx_tilde = k*s_tilde;

  T_b_T0 = T_c_ln_nobeta/(T_c_ln_nobeta+T_c_C_noalpha*q*q);
  T_b = sin(xx_tilde)/(xx_tilde)*(T_b_T0/(1.+pow(xx/5.2, 2))+
				  alpha_b/(1.+pow(beta_b/xx, 3))*exp(-pow(k/k_silk,1.4)));
  
  f_baryon = obhh/omhh;
  T_full = f_baryon*T_b + (1.-f_baryon)*T_c;

  return T_full;
}


double g3lhalo::Powerspectrum::spectrum(const double& k,const double& z)
{
  double growth=cosmo->growthFunction(z)/cosmo->growthFunction(0.0);
 
  double trans=transfer(k);

  return amplitude*trans*trans*growth*growth*pow(k,cosmo->ns);
}

double g3lhalo::Powerspectrum::sigma2(const double& m, const double& z)
{
  double om=cosmo->Omega_m_(z);
  double R=pow(0.75/std::numbers::pi*m/cosmo->rho_crit/om, 1./3.);

 
  containerSigma2 container;
  container.powerspectrum=this;
  container.R=R;
  container.z=z;
  
  double k_min[1]={0};
  double k_max[1]={1e12};
  double result, error;
  
  hcubature_v(1, integrand_sigma2, &container, 1, k_min, k_max, 0, 0, 1e-4, ERROR_L1, &result, &error);
  
  return result/2/std::numbers::pi/std::numbers::pi;
}


int g3lhalo::integrand_sigma2(unsigned ndim, size_t npts, const double* k, void* thisPtr, unsigned fdim, double* value)
{
  if(fdim!=1)
    {
      std::cerr<<"Powerspectrum::integrand_sigma2: Wrong fdim"<<std::endl;
      exit(1);
    };
  containerSigma2* container = (containerSigma2*) thisPtr;

 
  double R=container->R;
  double z=container->z;
 
  
  #pragma omp parallel for
  for(unsigned int i=0; i<npts; i++)
    {
      double k_=k[i*ndim];
      double W=container->powerspectrum->tophat(k_, R);
      
      value[i]=k_*k_*W*W*container->powerspectrum->spectrum(k_, z);
    };
  
  return 0;
}


double g3lhalo::Powerspectrum::dsigma2dm(const double& m, const double& z)
{
  double rho_m=cosmo->rho_crit*cosmo->Omega_m_(z);
  double R=0.6204*pow(m/rho_m, 1./3.);
  double dR=0.2068*pow(rho_m*m*m, -1./3.);
  
  containerSigma2 container;
  container.powerspectrum=this;
  container.R=R;
  container.dR=dR;
  
  double k_min[1]={0};
  double k_max[1]={1e12};
  double result, error;
  
  hcubature_v(1, integrand_dsigma2dM, &container, 1, k_min, k_max, 0, 0, 1e-4, ERROR_L1, &result, &error);

  return result/std::numbers::pi/std::numbers::pi;
}


int g3lhalo::integrand_dsigma2dM(unsigned ndim, size_t npts, const double* k, void* thisPtr, unsigned fdim, double* value)
{
  if(fdim!=1)
    {
      std::cerr<<"Powerspectrum::integrand_dsigma2dM: Wrong fdim"<<std::endl;
      exit(1);
    };
  containerSigma2* container = (containerSigma2*) thisPtr;

 
  double R=container->R;
  double z=container->z;
  double dR=container->dR;
 
  
  #pragma omp parallel for
  for(unsigned int i=0; i<npts; i++)
    {
      double k_=k[i*ndim];
      double W=container->powerspectrum->tophat(k_, R);
      double Wprime=container->powerspectrum->d_tophat(k_, R);
      
      value[i]=k_*k_*W*Wprime*dR*container->powerspectrum->spectrum(k_, z);
    };
  
  return 0;
}


void g3lhalo::Powerspectrum::getConcentration(const std::vector<double>& zs,
					      const std::vector<double>& ms,
					      std::vector<double>& result)
{
  int N=zs.size();

  for(int i=0; i<N; i++)
    {
      double z=zs.at(i);
      double mstar=mStar(z);
      
      for(int j=0; j<N; j++)
	{
	  double m=ms.at(j);
	  double conc=concentration(m, mstar, z);
	  result.push_back(conc);
	};
    };
}

void g3lhalo::Powerspectrum::getSigma2(const std::vector<double>& zs,
					      const std::vector<double>& ms,
					      std::vector<double>& result)
{
  int N=zs.size();

  for(int i=0; i<N; i++)
    {
      double z=zs.at(i);
      
      for(int j=0; j<N; j++)
	{
	  double m=ms.at(j);
	  result.push_back(sigma2(m, z));
	};
    };
}

void g3lhalo::Powerspectrum::getSpectrum(const std::vector<double>& zs,
					 const std::vector<double>& ks,
					 std::vector<double>& result)
{
  int N=zs.size();

  for(int i=0; i<N; i++)
    {
      double z=zs[i];
      for(int j=0; j<N; j++)
	{
	  double k=ks[j];
	  result.push_back(spectrum(k, z));
	}
    }
}


void g3lhalo::Powerspectrum::getTransfer( const std::vector<double>& ks,
					 std::vector<double>& result)
{
  int N=ks.size();

  for(int i=0; i<N; i++)
    {
      double k=ks[i];
      result.push_back(transfer(k));
      
    }
}
