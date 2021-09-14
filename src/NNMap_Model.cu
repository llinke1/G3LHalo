#include "NNMap_Model.h"
#include "constants.h"
#include "cubature.h"
#include <iostream>
#include <fstream>


#if GPU
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

#include <algorithm>




g3lhalo::NNMap_Model::NNMap_Model(Cosmology* cosmology_, const double& zmin_, const double& zmax_, const double& kmin_, const double& kmax_, const double& mmin_, const double& mmax_, const int& Nbins_, double* g_, double* p_lens1_, double* p_lens2_, double* w_, double* dwdz_, double* hmf_, double* P_lin_, double* b_h_, double* concentration_, Params* params_)
{

  // Set parameters
  cosmology=cosmology_;
  
  zmin=zmin_;
  zmax=zmax_;
  kmin=kmin_;
  kmax=kmax_;
  mmin=mmin_;
  mmax=mmax_;
  Nbins=Nbins_;
  g=g_;
  p_lens1=p_lens1_;
  p_lens2=p_lens2_;
  w=w_;
  dwdz=dwdz_;
  hmf=hmf_;
  P_lin=P_lin_;
  b_h=b_h_;
  concentration=concentration_;
  params=params_;

  zmin_integral=zmin;
  zmax_integral=zmax-(zmax-zmin)/Nbins;
  mmin_integral=mmin;
  mmax_integral=exp(log(mmin)+log(mmax/mmin)*(Nbins-1)/Nbins);
  
  
#if GPU
  // Allocation of memory for precomputed functions on device
  CUDA_SAFE_CALL(cudaMalloc(&dev_g, Nbins*sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_p_lens1, Nbins*sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_p_lens2, Nbins*sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_w, Nbins*sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_dwdz, Nbins*sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_hmf, Nbins*Nbins*sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_P_lin, Nbins*Nbins*sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_b_h, Nbins*Nbins*sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_concentration, Nbins*Nbins*sizeof(double)));


  // Copying of precomputed functions to device
  CUDA_SAFE_CALL(cudaMemcpy(dev_g, g, Nbins*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_p_lens1, p_lens1, Nbins*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_p_lens2, p_lens2, Nbins*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_w, w, Nbins*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_dwdz, dwdz, Nbins*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_hmf, hmf, Nbins*Nbins*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_P_lin, P_lin, Nbins*Nbins*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_b_h, b_h, Nbins*Nbins*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_concentration, concentration, Nbins*Nbins*sizeof(double), cudaMemcpyHostToDevice));



  // Allocation of memory for densities on device
  CUDA_SAFE_CALL(cudaMalloc(&dev_rho_bar, Nbins*sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_n_bar1, Nbins*sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_n_bar2, Nbins*sizeof(double)));
  
#endif // GPU

  // Allocattion of memory for densities on Host
  rho_bar = (double*) malloc(Nbins*sizeof(double));
  n_bar1 = (double*) malloc(Nbins*sizeof(double));
  n_bar2 = (double*) malloc(Nbins*sizeof(double));

#if VERBOSE
  std::cerr<<"Finished memory setting"<<std::endl;
#endif //VERBOSE

  // Calculate densities (stored in rho_bar, n_bar1 and n_bar2)
  updateDensities();

#if GPU
  // Copying of densities to device
  CUDA_SAFE_CALL(cudaMemcpy(dev_rho_bar, rho_bar, Nbins*sizeof(double), cudaMemcpyHostToDevice));
#endif
#if VERBOSE
  std::cerr<<"Finished initalizing NNMap"<<std::endl;
#endif // VERBOSE
}

g3lhalo::NNMap_Model::~NNMap_Model()
{
#if GPU // Free device memory
  cudaFree(dev_g);
  cudaFree(dev_p_lens1);
  cudaFree(dev_p_lens2);
  cudaFree(dev_w);
  cudaFree(dev_dwdz);
  cudaFree(dev_hmf);
  cudaFree(dev_P_lin);
  cudaFree(dev_b_h);
  cudaFree(dev_concentration);
  cudaFree(dev_rho_bar);
  cudaFree(dev_n_bar1);
  cudaFree(dev_n_bar2);
#endif //GPU
  //Free host memory
  free(rho_bar);
  free(n_bar1);
  free(n_bar2);  
}

void g3lhalo::NNMap_Model::updateParams(Params* params_)
{
  params=params_; //< Change HOD Params
  updateDensities(); //< Change matter and galaxy densities
}




void g3lhalo::NNMap_Model::updateDensities()
{
  // Set integral borders (logarithmic)
  double m_min[1]={std::log10(mmin_integral)};
  double m_max[1]={std::log10(mmax_integral)};
  double error, result;

  for(int i=0; i<Nbins; i++) // Go over all redshift bins
    {
      double z=zmin+i*(zmax-zmin)/Nbins; // Get redshift bin

      // \f$\bar{\rho}(z) = \Omega_m(z) \rho_{crit}$\f
      rho_bar[i]=cosmology->Omega_m_(z)*cosmology->rho_crit;

      // Set container for integral
      n_z_container container;
      container.nnmap=this;
      container.z=z; //redshift 

      container.type=1; //Galaxy type
      // Calculate integral for type 1
      hcubature_v(1, integrand_nbar, &container, 1, m_min, m_max, 0, 0, 1e-4, ERROR_L1, &result, &error);
      // Set n_bar1 (ln10 is to account for logarithmic integral)
      n_bar1[i]=g3lhalo::ln10*result;
      
      container.type=2; //Galaxy type
      // Calculate integral for type 2
      hcubature_v(1, integrand_nbar, &container, 1, m_min, m_max, 0, 0, 1e-4, ERROR_L1, &result, &error);
      // Set n_bar2 (ln10 is to account for logarithmic integral)
      n_bar2[i]=g3lhalo::ln10*result;
    };
  
#if VERBOSE
  std::cerr<<"Finished calculating densities"<<std::endl;
  std::cerr<<n_bar1[0]<<std::endl;
#endif //VERBOSE

#if GPU //Copy to device
  CUDA_SAFE_CALL(cudaMemcpy(dev_n_bar1, n_bar1, Nbins*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_n_bar2, n_bar2, Nbins*sizeof(double), cudaMemcpyHostToDevice));
#endif //GPU

#if VERBOSE 
  std::cerr<<"Finished density update"<<std::endl;
#endif //VERBOSE
}

double g3lhalo::NNMap_Model::NNMap_1h(const double& theta1, const double& theta2, const double& theta3, const int& type1, const int& type2)
{

  // Set Container
  nnmap_container container;
  container.nnmap=this;
  container.theta1=theta1;
  container.theta2=theta2;
  container.theta3=theta3;
  container.type1=type1;
  container.type2=type2;

  // Set Integral borders
  // Logarithmic over l1, l2, and m
  double params_min[5]={std::log10(lmin_integral), std::log10(lmin_integral), phimin_integral, std::log10(mmin_integral), zmin_integral};
  double params_max[5]={std::log10(lmax_integral), std::log10(lmax_integral), phimax_integral, std::log10(mmax_integral), zmax_integral};

  // Do calculation
  double result, error;
  hcubature_v(1, integrand_1halo, &container, 5, params_min, params_max, 5000000, 0, 1e-3, ERROR_L1, &result, &error);

  // Return result (Prefactor includes three ln10 terms due to logarithmic integral)
  return result*3*cosmology->H0*cosmology->H0*cosmology->Omega_m/2/g3lhalo::c/g3lhalo::c*g3lhalo::ln10*g3lhalo::ln10*g3lhalo::ln10/(2*g3lhalo::pi)/(2*g3lhalo::pi)/(2*g3lhalo::pi);
  
}

double g3lhalo::NNMap_Model::NNMap_2h(const double& theta1, const double& theta2, const double& theta3, const int& type1, const int& type2)
{
  // Set Container
  nnmap_container container;
  container.nnmap=this;
  container.theta1=theta1;
  container.theta2=theta2;
  container.theta3=theta3;
  container.type1=type1;
  container.type2=type2;

  // Set integral borders
  // Logarithmic over l1, l2, m1, and m2
  double params_min[6]={std::log10(lmin_integral), std::log10(lmin_integral), phimin_integral, std::log10(mmin_integral), std::log10(mmin_integral), zmin_integral};
  double params_max[6]={std::log10(lmax_integral), std::log10(lmax_integral),  phimax_integral, std::log10(mmax_integral), std::log10(mmax_integral), zmax_integral};

  // Do calculation
  double result, error;
  hcubature_v(1, integrand_2halo, &container, 6, params_min, params_max, 5000000, 0, 1e-3, ERROR_L1, &result, &error);

  // Return result (Prefactor includes four ln10 terms due to logarithmic integral)
  return result*3*cosmology->H0*cosmology->H0*cosmology->Omega_m/2/g3lhalo::c/g3lhalo::c*g3lhalo::ln10*g3lhalo::ln10*g3lhalo::ln10*g3lhalo::ln10/(2*g3lhalo::pi)/(2*g3lhalo::pi)/(2*g3lhalo::pi);
  
}

double g3lhalo::NNMap_Model::NNMap_3h(const double& theta1, const double& theta2, const double& theta3, const int& type1, const int& type2)
{
  // Set container
  nnmap_container container;
  container.nnmap=this;
  container.theta1=theta1;
  container.theta2=theta2;
  container.theta3=theta3;
  container.type1=type1;
  container.type2=type2;

  // Set integral borders
  // Logarithmic over l1, l2, m1, m2, and m3
  double params_min[7]={std::log10(lmin_integral),std::log10(lmin_integral),phimin_integral, std::log10(mmin_integral), std::log10(mmin_integral), std::log10(mmin_integral), zmin_integral};
  double params_max[7]={std::log10(lmax_integral),std::log10(lmax_integral), phimax_integral, std::log10(mmax_integral), std::log10(mmax_integral), std::log10(mmax_integral), zmax_integral};

  // Do calculation
  double result, error;
  hcubature_v(1, integrand_3halo, &container, 7, params_min, params_max, 5000000, 0, 1e-4, ERROR_L1, &result, &error);

  // Return result (Prefactor includes five ln10 terms due to logarithmic integral)
  return result*3*cosmology->H0*cosmology->H0*cosmology->Omega_m/2/g3lhalo::c/g3lhalo::c*g3lhalo::ln10*g3lhalo::ln10*g3lhalo::ln10*g3lhalo::ln10*g3lhalo::ln10/(2*g3lhalo::pi)/(2*g3lhalo::pi)/(2*g3lhalo::pi);
  
}


double g3lhalo::NNMap_Model::NNMap(const double& theta1, const double& theta2, const double& theta3, const int& type1, const int& type2)
{
  return NNMap_1h(theta1, theta2, theta3, type1, type2) // 1-halo term
    +NNMap_2h(theta1, theta2, theta3, type1, type2) // 2-halo term
    +NNMap_3h(theta1, theta2, theta3, type1, type2); // 3-halo term
}



void g3lhalo::NNMap_Model::calculateAll(const std::vector<double>& thetas1, const std::vector<double>& thetas2, const std::vector<double>& thetas3, Params* params_, std::vector<double>& results)
{
  updateParams(params_);
  int n=thetas1.size();
  //N1N2
  for(int i=0; i<n; i++)
    {     
      double theta1=thetas1.at(i); //Read Thetas
      double theta2=thetas2.at(i);
      double theta3=thetas3.at(i);
      
      results.push_back(NNMap(theta1, theta2, theta3, 1, 2));
    }
  //N1N1
  for(int i=0; i<n; i++)
    {     
      double theta1=thetas1.at(i); //Read Thetas
      double theta2=thetas2.at(i);
      double theta3=thetas3.at(i);
      
      results.push_back(NNMap(theta1, theta2, theta3, 1, 1));
      }
  //N2N2
  for(int i=0; i<n; i++)
    {     
      double theta1=thetas1.at(i); //Read Thetas
      double theta2=thetas2.at(i);
      double theta3=thetas3.at(i);
      
      results.push_back(NNMap(theta1, theta2, theta3, 2, 2));
    }
  return;
}


void g3lhalo::NNMap_Model::pickParams(const int& type, double& f, double& alpha, double& mth, double& sigma, double& mprime, double& beta)
{
  if(type==1)
    {
      f=params->f1;
      alpha=params->alpha1;
      mth=params->mmin1;
      sigma=params->sigma1;
      mprime=params->mprime1;
      beta=params->beta1;
    }
  else if(type==2)
    {
      f=params->f2;
      alpha=params->alpha2;
      mth=params->mmin2;
      sigma=params->sigma2;
      mprime=params->mprime2;
      beta=params->beta2;
    }
  else
    {
      std::cerr<<"NNMap_Model:: wrong type "<<type<<std::endl;
      exit(1);
    };
}

/************************************* OUTSIDE OF CLASS *********************************************************/

/************************************ GALAXY NUMBER DENSITY *****************************************************/
int g3lhalo::integrand_nbar(unsigned ndim, size_t npts, const double* m, void* thisPtr, unsigned fdim, double* value)
{
  // Check if fdim is correct
  if(fdim!=1)
    {
      std::cerr<<"NNMap_Model::integrand_nbar: Wrong fdim"<<std::endl;
      exit(1);
    };
  // Read out Container
  n_z_container* container = (n_z_container*) thisPtr;
  NNMap_Model* nnmap = container-> nnmap;
  double z = container->z;
  int type = container->type;

  // Set HOD params
  double f, alpha, mth, sigma, mprime, beta;
  nnmap->pickParams(type, f, alpha, mth, sigma, mprime, beta);

#if GPU // Calculation with GPU

  // Allocate on device for integration values
  double* dev_value;
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_value, fdim*npts*sizeof(double)));
  
  // Allocate on device for masses
  double* dev_ms;
  CUDA_SAFE_CALL(cudaMalloc(&dev_ms, npts*ndim*sizeof(double)));

  // Copy masses to device  
  CUDA_SAFE_CALL(cudaMemcpy(dev_ms, m, npts*ndim*sizeof(double), cudaMemcpyHostToDevice));

  // Do calculation
  g3lhalo::GPUkernel_nbar<<<BLOCKS, THREADS>>>(dev_ms, z, npts, alpha, mth, sigma, mprime, beta, nnmap->zmin, nnmap->zmax,
					    nnmap->mmin, nnmap->mmax, nnmap->Nbins, nnmap->dev_hmf, dev_value);

  cudaFree(dev_ms); // Clean up

  // Copy results from device to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim*npts*sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); // Clean up

#else // Calculation on CPU
  
#pragma omp parallel for //Omp parallelization
  for (unsigned int i=0; i<npts; i++)
    {
      double m_ = pow(10, m[i*ndim]);
      value[i]= kernel_function_nbar(m_, z, alpha, mth, sigma, mprime, beta, nnmap->zmin, nnmap->zmax, nnmap->mmin, nnmap->mmax, nnmap->Nbins, nnmap->hmf);
    };
#endif //GPU
  
  return 0; //Success :)
}

__device__ __host__ double g3lhalo::kernel_function_nbar(double m, double z, double alpha,
							 double mth, double sigma, double mprime, double beta,
							 double zmin, double zmax,
							 double mmin, double mmax, int Nbins, const double* hmf)
{
  double Nc=g3lhalo::Ncen(m, alpha, mth, sigma); //<Number of central galaxies
  double Ns=g3lhalo::Nsat(m, mth, sigma, mprime, beta); //< Number of satellite galaxies

  // Read indices for mass and redshift
  int m_ix=std::round(std::log(m/mmin)*Nbins/std::log(mmax/mmin));
  int z_ix=std::round((z-zmin)*Nbins/(zmax-zmin));
  
  return  m*hmf[int(z_ix*Nbins+m_ix)]*(Nc+Ns);
}

#if GPU

__global__ void g3lhalo::GPUkernel_nbar(const double* ms, double z, int npts, double alpha,
					double mth, double sigma, double mprime, double beta,
					double zmin, double zmax,
					double mmin, double mmax, int Nbins, const double* hmf,
					double* value)
{
  ///Index of thread
  int thread_index=blockIdx.x * blockDim.x + threadIdx.x;
  
  //Grid-Stride Loop, so I get npts evaluations
  for(int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {
      double m=pow(10, ms[i]);
      
      value[i] = kernel_function_nbar(m, z, alpha, mth, sigma, mprime, beta, zmin, zmax, mmin, mmax, Nbins, hmf);
    };
}
#endif
/********************************************* 1 HALO TERM *****************************************************************/

int g3lhalo::integrand_1halo(unsigned ndim, size_t npts, const double* params, void* thisPtr, unsigned fdim, double* value)
{
  // Check if fdim is correct
  if(fdim!=1)
    {
      std::cerr<<"NNMap_Model::integrand_nbar: Wrong fdim"<<std::endl;
      exit(1);
    };

  // Read out container
  nnmap_container* container = (nnmap_container*) thisPtr;
  NNMap_Model* nnmap = container-> nnmap;
  double theta1 = container->theta1;
  double theta2 = container->theta2;
  double theta3 = container->theta3;
  int type1 = container->type1;
  int type2 = container->type2;

  // Set HOD parameters
  double f1, f2, alpha1, alpha2, mmin1, mmin2, sigma1, sigma2, mprime1, mprime2, beta1, beta2, A, epsilon;
  A=nnmap->params->A;
  epsilon=nnmap->params->epsilon;
  nnmap->pickParams(type1, f1, alpha1, mmin1, sigma1, mprime1, beta1);
  nnmap->pickParams(type2, f2, alpha2, mmin2, sigma2, mprime2, beta2);
  
  //std::cerr<<f1<<std::endl;
#if GPU // Calculation on GPU

  // Allocate for integration values on device
  double* dev_value;
  CUDA_SAFE_CALL(cudaMalloc(&dev_value, fdim*npts*sizeof(double)));

  // Allocate for integration variables on device
  double* dev_params;
  CUDA_SAFE_CALL(cudaMalloc(&dev_params, npts*ndim*sizeof(double)));

  // Copy integration variables to device
  CUDA_SAFE_CALL(cudaMemcpy(dev_params, params, npts*ndim*sizeof(double), cudaMemcpyHostToDevice));

  // Do calculation
  GPUkernel_1Halo<<<BLOCKS, THREADS>>>(dev_params, theta1, theta2, theta3, npts, type1, type2, f1, f2, alpha1, alpha2,
				       mmin1, mmin2, sigma1, sigma2, mprime1, mprime2, beta1, beta2, A, epsilon, nnmap->zmin,
				       nnmap->zmax,
				       nnmap->mmin, nnmap->mmax, nnmap->Nbins, nnmap->dev_g, nnmap->dev_p_lens1, nnmap->dev_p_lens2,
				       nnmap->dev_w, nnmap->dev_dwdz, nnmap->dev_hmf, nnmap->dev_concentration, nnmap->dev_rho_bar,
				       nnmap->dev_n_bar1, nnmap->dev_n_bar2,
				       dev_value);
  
  cudaFree(dev_params); // Clean up

  // Copy values from device to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim*npts*sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); // Clean up
  
#else  // Calculation on CPU
  
  #pragma omp parallel for
  for (unsigned int i=0; i<npts; i++)
    {
      double l1 = pow(10, params[i*ndim]);
      double l2 = pow(10, params[i*ndim+1]);
      double phi = params[i*ndim+2];
      double m = pow(10, params[i*ndim+3]);
      double z = params[i*ndim+4];
      value[i]= kernel_function_1halo(theta1, theta2, theta3, l1, l2, phi, m, z, nnmap->zmin, nnmap->zmax, nnmap->mmin, nnmap->mmax,
				      nnmap->Nbins, type1, type2, f1, f2, alpha1, alpha2,
				      mmin1, mmin2, sigma1, sigma2, mprime1, mprime2, beta1, beta2, A, epsilon, nnmap->g,
				      nnmap->p_lens1, nnmap->p_lens2, nnmap->w, nnmap->dwdz, nnmap->hmf, nnmap->concentration,
				      nnmap->rho_bar, nnmap->n_bar1, nnmap->n_bar2);
    };

#endif
  return 0; //Success :)
}

__device__ __host__ double g3lhalo::kernel_function_1halo( double theta1, double theta2, double theta3, double l1, double l2, double phi, double m, double z,
							  double zmin, double zmax, double mmin, double mmax, int Nbins,
							  int type1, int type2,
							  double f1,  double f2, double alpha1, double alpha2, double mmin1,
							  double mmin2, double sigma1, double sigma2, double mprime1,
							  double mprime2, double beta1, double beta2, double A, double epsilon,
							  const double* g, const double* p_lens1,
							  const double* p_lens2, const double* w, const double* dwdz,
							  const double* hmf, const double* concentration,
							  const double* rho_bar, const double* n_bar1, const double* n_bar2)
{
  // Get inidices of z and m
  int z_ix=std::round(((z-zmin)*Nbins/(zmax-zmin)));
  int m_ix=std::round((std::log(m/mmin)*Nbins/std::log(mmax/mmin)));
  
  // Get lens galaxy densities
  double nbar1, nbar2;
  if(type1==1)
    {
      nbar1=n_bar1[z_ix];
    }
  else
    {
      nbar1=n_bar2[z_ix];
    };
  if(type2==1)
    {
      nbar2=n_bar1[z_ix];
    }
  else
    {
      nbar2=n_bar2[z_ix];
    };
  

  double w_=w[z_ix]; //< Comoving distance [Mpc]
  
  double l3=sqrt(l1*l1+l2*l2+2*l1*l2*cos(phi));
  double k1=l1/w_;
  double k2=l2/w_;
  double k3=l3/w_;
  
  // 3D Galaxy-Galaxy-Matter Bispectrum
  double Bggd=1./nbar1/nbar2/rho_bar[z_ix]*hmf[z_ix*Nbins+m_ix]*m*u_NFW(k3, m, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar,
									concentration)
    *G_gg(k1, k2, m, z, f1, f2, alpha1, alpha2, mmin1, mmin2, sigma1, sigma2, mprime1, mprime2, beta1, beta2, A, epsilon, zmin, zmax,
	  mmin, mmax, Nbins, rho_bar, concentration, (type1==type2));

  // 2D Galaxy-Galaxy-Convergence Bispectrum
  double bggk=Bggd/dwdz[z_ix]*g[z_ix]*p_lens1[z_ix]*p_lens2[z_ix]/w_/w_/w_*(1.+z);
      
  return l1*l2*m*l1*l2*apertureFilter(theta1*l1)*apertureFilter(theta2*l2)*apertureFilter(theta3*l3)*bggk;
}

#if GPU
__global__ void g3lhalo::GPUkernel_1Halo(const double* params, double theta1, double theta2, double theta3,
					 int npts, int type1, int type2, double f1,  double f2,
					 double alpha1, double alpha2, double mmin1,
					 double mmin2, double sigma1, double sigma2, double mprime1,
					 double mprime2, double beta1, double beta2, double A, double epsilon,
					 double zmin, double zmax, double mmin, double mmax,
					 int Nbins, const double* g, const double* p_lens1, const double* p_lens2,
					 const double* w, const double* dwdz, const double* hmf, const double* concentration,
					 const double* rho_bar, const double* n_bar1, const double* n_bar2,
					 double* value)
{
  ///Index of thread
  int thread_index=blockIdx.x * blockDim.x + threadIdx.x;
  
  
  //Grid-Stride Loop, so I get npts evaluations
  for(int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {

      double l1=pow(10, params[i*5]);
      double l2=pow(10, params[i*5+1]);
      double phi=params[i*5+2];
      double m=pow(10, params[i*5+3]);
      double z=params[i*5+4];

      
      value[i]=kernel_function_1halo(theta1, theta2, theta3, l1, l2, phi, m, z, zmin, zmax, mmin, mmax, Nbins, type1, type2, f1, f2,
				     alpha1, alpha2, mmin1, mmin2, sigma1, sigma2, mprime1, mprime2, beta1, beta2, A, epsilon, g,
				     p_lens1, p_lens2, w, dwdz, hmf, concentration, rho_bar, n_bar1, n_bar2);
    };
}
#endif
/********************************************* 2 HALO TERM *****************************************************************/

int g3lhalo::integrand_2halo(unsigned ndim, size_t npts, const double* params, void* thisPtr, unsigned fdim, double* value)
{
  if(fdim!=1) // Check fdim
    {
      std::cerr<<"NNMap_Model::integrand_nbar: Wrong fdim"<<std::endl;
      exit(1);
    };

  // Read out Container
  nnmap_container* container = (nnmap_container*) thisPtr;
  NNMap_Model* nnmap = container-> nnmap;
  double theta1 = container->theta1;
  double theta2 = container->theta2;
  double theta3 = container->theta3;
  int type1 = container->type1;
  int type2 = container->type2;

  // Pick HOD Parameters
  double f1, f2, alpha1, alpha2, mmin1, mmin2, sigma1, sigma2, mprime1, mprime2, beta1, beta2;
  nnmap->pickParams(type1, f1, alpha1, mmin1, sigma1, mprime1, beta1);
  nnmap->pickParams(type2, f2, alpha2, mmin2, sigma2, mprime2, beta2);  
  double A=nnmap->params->A;
  double epsilon=nnmap->params->epsilon;
  
#if GPU // Do calculation on GPU

  int npts_it=100000; //< Maximal number of integration points simultaneously executed (limited by device RAM!)
  int number_it=int(npts/npts_it)+1; //< Number of iterations necessary to get npts executions
  
  for(int i=0; i<number_it; i++)
    {
      // Set points for which this iteration goes through
      int start=i*npts_it;
      int end=std::min<int>((i+1)*npts_it, npts);
      int n_it=end-start;

      //Set parameters for which this iteration goes through
      double params_it[n_it*ndim];
#pragma omp parallel for
      for(int j=0; j<n_it*ndim; j++)
	{
	  params_it[j]=params[start*ndim+j];
	};
      
      
      // Allocate device memory for parameters
      double* dev_params;
      CUDA_SAFE_CALL(cudaMalloc(&dev_params, n_it*ndim*sizeof(double)));

      // Copy parameters to device memory
      CUDA_SAFE_CALL(cudaMemcpy(dev_params, &params_it, n_it*ndim*sizeof(double), cudaMemcpyHostToDevice));

      // Allocate memory for results on device
      double* dev_value;
      CUDA_SAFE_CALL(cudaMalloc(&dev_value, fdim*n_it*sizeof(double)));

      // Do calculation
      GPUkernel_2Halo<<<BLOCKS, THREADS>>>(dev_params, theta1, theta2, theta3, n_it, type1, type2, f1, f2, alpha1, alpha2,
					   mmin1, mmin2, sigma1, sigma2, mprime1, mprime2, beta1, beta2, A, epsilon, nnmap->zmin,
					   nnmap->zmax, nnmap->mmin, nnmap->mmax, nnmap->kmin, nnmap->kmax, nnmap->Nbins,
					   nnmap->dev_g, nnmap->dev_p_lens1, nnmap->dev_p_lens2, nnmap->dev_w, nnmap->dev_dwdz,
					   nnmap->dev_hmf, nnmap->dev_P_lin, nnmap->dev_b_h, nnmap->dev_concentration,
					   nnmap->dev_rho_bar, nnmap->dev_n_bar1, nnmap->dev_n_bar2, dev_value);

      cudaFree(dev_params); //< Clean up

      // Copy of results to host
      double value_it[fdim*n_it];
      CUDA_SAFE_CALL(cudaMemcpy(&value_it, dev_value, fdim*n_it*sizeof(double), cudaMemcpyDeviceToHost));
      
      cudaFree(dev_value); //< Clean up

      // Add result to overall vector
#pragma omp parallel for
      for(int j=0; j<fdim*n_it; j++)
	{
	  value[start*fdim+j]=value_it[j];
	};
    };
  
#else  // Do calculation on CPU

  #pragma omp parallel for
  for (unsigned int i=0; i<npts; i++)
    {
      double l1 = pow(10, params[i*ndim]);
      double l2 = pow(10, params[i*ndim+1]);
      double phi = params[i*ndim+2];
      double m1 = pow(10, params[i*ndim+3]);
      double m2 = pow(10, params[i*ndim+4]);
      double z = params[i*ndim+5];
      value[i]= kernel_function_2halo(theta1, theta2, theta3, l1, l2, phi, m1, m2, z, nnmap->zmin, nnmap->zmax, nnmap->mmin,
				      nnmap->mmax, nnmap->kmin, nnmap->kmax, nnmap->Nbins, type1, type2, f1,  f2, alpha1, alpha2,
				      mmin1, mmin2, sigma1, sigma2, mprime1, mprime2, beta1, beta2, A,  epsilon, nnmap->g,
				      nnmap->p_lens1, nnmap->p_lens2, nnmap->w,  nnmap->dwdz, nnmap->hmf, nnmap->P_lin, nnmap->b_h,
				      nnmap->concentration, nnmap->rho_bar, nnmap->n_bar1, nnmap->n_bar2);
    };

#endif
  return 0; //Success :)
}


__device__ __host__ double g3lhalo::kernel_function_2halo(double theta1, double theta2, double theta3, double l1, double l2,
							  double phi, double m1, double m2, double z, double zmin, double zmax,
							  double mmin, double mmax, double kmin, double kmax, int Nbins, int type1,
							  int type2, double f1, double f2, double alpha1, double alpha2, double mmin1,
							  double mmin2, double sigma1, double sigma2, double mprime1,
							  double mprime2, double beta1, double beta2, double A, double epsilon,
							  const double* g, const double* p_lens1, const double* p_lens2,
							  const double* w, const double* dwdz, const double* hmf, const double* P_lin,
							  const double* b_h,  const double* concentration, const double* rho_bar,
							  const double* n_bar1, const double* n_bar2)
  {
    
    // Get Indices of z, m1, and m2
    int z_ix=std::round((z-zmin)*Nbins/(zmax-zmin));
    int m1_ix=std::round(std::log(m1/mmin)*Nbins/std::log(mmax/mmin));
    int m2_ix=std::round(std::log(m2/mmin)*Nbins/std::log(mmax/mmin));

    // Get galaxy number densities
    double nbar1, nbar2;
    if(type1==1) nbar1=n_bar1[z_ix];
    else nbar1=n_bar2[z_ix];
    
    if(type2==1) nbar2=n_bar1[z_ix];
    else nbar2=n_bar2[z_ix];

    double w_=w[z_ix]; //Comoving distance [Mpc]
    double l3=sqrt(l1*l1+l2*l2+2*l1*l2*cos(phi));
    double k1=l1/w_;
    double k2=l2/w_;
    double k3=l3/w_;

    int k1_ix=std::round(std::log(k1/kmin)*Nbins/std::log(kmax/kmin));
    int k2_ix=std::round(std::log(k2/kmin)*Nbins/std::log(kmax/kmin));
    int k3_ix=std::round(std::log(k3/kmin)*Nbins/std::log(kmax/kmin));

    // Set Powerspectrum
    //Use linear approx for very small ks (should work reasonably well)
    double Pk1, Pk2, Pk3;
    if(k1_ix>=0) Pk1=P_lin[z_ix*Nbins+k1_ix];
    else Pk1=P_lin[z_ix*Nbins]/kmin*k1;

    if(k2_ix>=0) Pk2=P_lin[z_ix*Nbins+k2_ix];
    else Pk2=P_lin[z_ix*Nbins]/kmin*k2;
    
    if(k3_ix>=0) Pk3=P_lin[z_ix*Nbins+k3_ix];
    else Pk3=P_lin[z_ix*Nbins]/kmin*k3;

    // 3D Bispectrum
    double Bggd=1./nbar1/nbar2/rho_bar[z_ix]*hmf[z_ix*Nbins+m1_ix]*hmf[z_ix*Nbins+m2_ix]*b_h[z_ix*Nbins+m1_ix]*b_h[z_ix*Nbins+m2_ix]
      *(m1*u_NFW(k3, m1, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration)
	*G_gg(k1, k2, m2, z, f1, f2, alpha1, alpha2, mmin1, mmin2, sigma1, sigma2, mprime1, mprime2, beta1, beta2, A, epsilon, zmin,
	      zmax, mmin, mmax, Nbins, rho_bar, concentration, (type1==type2))*Pk3
	+m2*u_NFW(k3, m2, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration)
	*G_g(k1, m1, z, f1, alpha1, mmin1, sigma1, mprime1, beta1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration)
	*G_g(k2, m2, z, f2, alpha2, mmin2, sigma2, mprime2, beta2, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration)
	*Pk1
	+m2*u_NFW(k3, m2, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration)
	*G_g(k2, m1, z, f2, alpha2, mmin2, sigma2, mprime2, beta2, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration)
	*G_g(k1, m2, z, f1, alpha1, mmin1, sigma1, mprime1, beta1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration)
	*Pk2);
    
    // 2D Bispectrum
    double bggk=1./dwdz[z_ix]*g[z_ix]*p_lens1[z_ix]*p_lens2[z_ix]/w_/w_/w_*(1.+z)*Bggd;

    return l1*l2*m1*m2*l1*l2*apertureFilter(theta1*l1)*apertureFilter(theta2*l2)*apertureFilter(theta3*l3)*bggk;
  }


#if GPU

__global__ void g3lhalo::GPUkernel_2Halo(const double* params, double theta1, double theta2, double theta3, int npts, int type1,
					 int type2, double f1,  double f2, double alpha1, double alpha2, double mmin1, double mmin2,
					 double sigma1, double sigma2, double mprime1, double mprime2, double beta1, double beta2,
					 double A, double epsilon, double zmin, double zmax, double mmin, double mmax, double kmin,
					 double kmax, int Nbins, const double* g, const double* p_lens1, const double* p_lens2,
					 const double* w, const double* dwdz, const double* hmf, const double* P_lin,
					 const double* b_h, const double* concentration, const double* rho_bar, const double* n_bar1,
					 const double* n_bar2, double* value)
{
  ///Index of thread
  int thread_index=blockIdx.x * blockDim.x + threadIdx.x;

  
  //Grid-Stride Loop, so I get npts evaluations
  for(int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {
      double l1=pow(10, params[i*6]);
      double l2=pow(10, params[i*6+1]);
      double phi=params[i*6+2];
      double m1=pow(10, params[i*6+3]);
      double m2=pow(10, params[i*6+4]);
      double z=params[i*6+5];


      value[i]=kernel_function_2halo(theta1, theta2, theta3, l1, l2, phi, m1, m2, z, zmin, zmax, mmin, mmax, kmin, kmax, Nbins, type1,
				     type2, f1, f2, alpha1, alpha2, mmin1, mmin2, sigma1, sigma2, mprime1, mprime2, beta1, beta2, A,
				     epsilon, g, p_lens1, p_lens2, w, dwdz, hmf, P_lin, b_h, concentration, rho_bar, n_bar1, n_bar2);
    };
}

#endif
/********************************************* 3 HALO TERM *****************************************************************/

int g3lhalo::integrand_3halo(unsigned ndim, size_t npts, const double* params, void* thisPtr, unsigned fdim, double* value)
{
  if(fdim!=1) //< Check fdim
    {
      std::cerr<<"NNMap_Model::integrand_3Halo: Wrong fdim"<<std::endl;
      exit(1);
    };
  //Read out Container
  nnmap_container* container = (nnmap_container*) thisPtr;
  NNMap_Model* nnmap = container-> nnmap;
  double theta1 = container->theta1;
  double theta2 = container->theta2;
  double theta3 = container->theta3;
  int type1 = container->type1;
  int type2 = container->type2;

  // Set HOD Params
  double f1, f2, alpha1, alpha2, mmin1, mmin2, sigma1, sigma2, mprime1, mprime2, beta1, beta2;
  nnmap->pickParams(type1, f1, alpha1, mmin1, sigma1, mprime1, beta1);
  nnmap->pickParams(type2, f2, alpha2, mmin2, sigma2, mprime2, beta2);
   
#if GPU // Do calculation on GPU

  // Allocate for integration values on device
  double* dev_value;
  CUDA_SAFE_CALL(cudaMalloc(&dev_value, fdim*npts*sizeof(double)));

  // Allocate for integration variables on device
  double* dev_params;
  CUDA_SAFE_CALL(cudaMalloc(&dev_params, npts*ndim*sizeof(double)));

  // Copy integration variables to device
  CUDA_SAFE_CALL(cudaMemcpy(dev_params, params, npts*ndim*sizeof(double), cudaMemcpyHostToDevice));

  // Do calculation
  GPUkernel_3Halo<<<BLOCKS, THREADS>>>(dev_params, theta1, theta2, theta3, npts, type1, type2, f1, f2, alpha1, alpha2, mmin1, mmin2,
				       sigma1, sigma2, mprime1, mprime2, beta1, beta2, nnmap->zmin, nnmap->zmax, nnmap->mmin,
				       nnmap->mmax, nnmap->kmin, nnmap->kmax, nnmap->Nbins, nnmap->dev_g, nnmap->dev_p_lens1,
				       nnmap->dev_p_lens2, nnmap->dev_w, nnmap->dev_dwdz, nnmap->dev_hmf, nnmap->dev_P_lin,
				       nnmap->dev_b_h, nnmap->dev_concentration, nnmap->dev_rho_bar, nnmap->dev_n_bar1,
				       nnmap->dev_n_bar2, dev_value);

  cudaFree(dev_params); //< Clean up

  // Copy values from device to host
  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim*npts*sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(dev_value); //< Clean up 
  
#else  // Calculation on CPU

  #pragma omp parallel for
  for (unsigned int i=0; i<npts; i++)
    {
      double l1 = pow(10, params[i*ndim]);
      double l2 = pow(10, params[i*ndim+1]);
      double phi = params[i*ndim+2];
      double m1 = pow(10, params[i*ndim+3]);
      double m2 = pow(10, params[i*ndim+4]);
      double m3 = pow(10, params[i*ndim+5]);
      double z = params[i*ndim+6];
      value[i]= kernel_function_3halo(theta1, theta2, theta3, l1, l2, phi, m1, m2, m3, z, nnmap->zmin, nnmap->zmax, nnmap->mmin,
				      nnmap->mmax, nnmap->kmin, nnmap->kmax, nnmap->Nbins, type1, type2, f1, f2, alpha1, alpha2,
				      mmin1, mmin2, sigma1, sigma2, mprime1, mprime2, beta1, beta2, nnmap->g, nnmap->p_lens1,
				      nnmap->p_lens2,  nnmap->w,  nnmap->dwdz, nnmap->hmf, nnmap->P_lin, nnmap->b_h,
				      nnmap->concentration, nnmap->rho_bar, nnmap->n_bar1, nnmap->n_bar2);
    };

#endif
  return 0; //Success :)
}



__device__ __host__ double g3lhalo::kernel_function_3halo(double theta1, double theta2, double theta3, double l1, double l2,
							  double phi, double m1, double m2, double m3, double z, double zmin,
							  double zmax, double mmin, double mmax, double kmin, double kmax, int Nbins,
							  int type1, int type2, double f1,  double f2, double alpha1, double alpha2,
							  double mmin1, double mmin2, double sigma1, double sigma2, double mprime1,
							  double mprime2, double beta1, double beta2, const double* g,
							  const double* p_lens1, const double* p_lens2, const double* w,
							  const double* dwdz, const double* hmf, const double* P_lin,
							  const double* b_h,  const double* concentration, const double* rho_bar,
							  const double* n_bar1, const double* n_bar2)
{
  // Index of z, m1, m2, and m3
  int z_ix=std::round((z-zmin)*Nbins/(zmax-zmin));
  int m1_ix=std::round(std::log(m1/mmin)*Nbins/std::log(mmax/mmin));
  int m2_ix=std::round(std::log(m2/mmin)*Nbins/std::log(mmax/mmin));
  int m3_ix=std::round(std::log(m3/mmin)*Nbins/std::log(mmax/mmin));

  double nbar1, nbar2;
  if(type1==1)  nbar1=n_bar1[z_ix];
  else  nbar1=n_bar2[z_ix];

  if(type2==1)	  nbar2=n_bar1[z_ix];
  else	  nbar2=n_bar2[z_ix];

  double w_=w[z_ix]; // Comoving distance [Mpc]
  double l3=sqrt(l1*l1+l2*l2+2*l1*l2*cos(phi));
  double k1=l1/w_;
  double k2=l2/w_;
  double k3=l3/w_;
  
 
  // 3D Bispectrum
  double Bggd=1./nbar1/nbar2/rho_bar[z_ix]*hmf[z_ix*Nbins+m1_ix]
    *hmf[z_ix*Nbins+m2_ix]*hmf[z_ix*Nbins+m3_ix]
    *m3*u_NFW(k3, m3, z, 1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration)
    *G_g(k1, m1, z, f1, alpha1, mmin1, sigma1, mprime1, beta1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration)
    *G_g(k2, m2, z, f2, alpha2, mmin2, sigma2, mprime2, beta2, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration)
    *Bi_lin(k1, k2, cos(phi), z, kmin, kmax, zmin, zmax, Nbins, P_lin)*b_h[z_ix*Nbins+m1_ix]*b_h[z_ix*Nbins+m2_ix]*b_h[z_ix*Nbins+m3_ix];

  // 2D bispcetrum
  double bggk=Bggd/dwdz[z_ix]*g[z_ix]*p_lens1[z_ix]*p_lens2[z_ix]/w_/w_/w_*(1.+z);

  return l1*l2*m1*m2*m3*l1*l2*apertureFilter(theta1*l1)*apertureFilter(theta2*l2)*apertureFilter(theta3*l3)*bggk;
}

#if GPU
__global__ void g3lhalo::GPUkernel_3Halo(const double* params, double theta1, double theta2, double theta3, int npts, int type1,
					 int type2,  double f1,  double f2, double alpha1, double alpha2, double mmin1, double mmin2,
					 double sigma1, double sigma2, double mprime1, double mprime2, double beta1, double beta2,
					 double zmin, double zmax, double mmin, double mmax, double kmin, double kmax, int Nbins,
					 const double* g, const double* p_lens1, const double* p_lens2, const double* w,
					 const double* dwdz, const double* hmf, const double* P_lin, const double* b_h,
					 const double* concentration, const double* rho_bar, const double* n_bar1,
					 const double* n_bar2, double* value)

{
  ///Index of thread
  int thread_index=blockIdx.x * blockDim.x + threadIdx.x;

  
  //Grid-Stride Loop, so I get npts evaluations
  for(int i=thread_index; i<npts; i+=blockDim.x*gridDim.x)
    {
      double l1=pow(10, params[i*7]);
      double l2=pow(10, params[i*7+1]);
      double phi=params[i*7+2];
      double m1=pow(10, params[i*7+3]);
      double m2=pow(10, params[i*7+4]);
      double m3=pow(10, params[i*7+5]);
      double z=params[i*7+6];

      value[i]=kernel_function_3halo(theta1, theta2, theta3, l1, l2, phi, m1, m2, m3, z, zmin, zmax, mmin, mmax, kmin, kmax, Nbins,
				     type1, type2, f1, f2, alpha1, alpha2, mmin1, mmin2, sigma1, sigma2, mprime1, mprime2, beta1,
				     beta2, g,  p_lens1, p_lens2,  w,  dwdz, hmf, P_lin, b_h, concentration, rho_bar, n_bar1,
				     n_bar2);

    };
 
}
#endif
/********************************************* OTHER FUNCTIONS *****************************************************************/



__host__ __device__ double g3lhalo::F(double k1, double k2, double cosphi)
{
  return 0.7143+0.2857*cosphi*cosphi+0.5*cosphi*(k1/k2+k2/k2);
}


double g3lhalo::Bi_lin(double k1, double k2, double cosphi, double z,
		       double kmin, double kmax, double zmin, double zmax,
		       int Nbins, const double* P_lin)
{
  // Get k3 = |\vec{k1}+\vec{k2}|
  double k3=sqrt(k1*k1+k2*k2+2*k1*k2*cosphi);

  //Get Cosine of angle between k1 and k2
  double c_phi12=cosphi;//(k3*k3-k1*k1-k2*k2)/(2*k1*k2);

  //Get cosine of angle between k1 and k3
   double c_phi13=(k2*k2-k3*k3-k1*k1)/(2*k1*k3);

  //Get cosine of angle between k2 and k3
   double c_phi23=(k1*k1-k2*k2-k3*k3)/(2*k3*k2);

   int z_ix=std::round((z-zmin)*Nbins/(zmax-zmin));

   int k1_ix=std::round(std::log(k1/kmin)*Nbins/std::log(kmax/kmin));
   int k2_ix=std::round(std::log(k2/kmin)*Nbins/std::log(kmax/kmin));
   int k3_ix=std::round(std::log(k3/kmin)*Nbins/std::log(kmax/kmin));



      // Set Powerspectrum
      double Pk1, Pk2, Pk3;
      if(k1_ix>=0)
	{
	  Pk1=P_lin[z_ix*Nbins+k1_ix];
	}
      else //Use linear approx for very small ks (should work reasonably well)
	{
	  Pk1=P_lin[z_ix*Nbins]/kmin*k1;
	};

      if(k2_ix>=0)
	{
	  Pk2=P_lin[z_ix*Nbins+k2_ix];
	}
      else //Use linear approx for very small ks (should work reasonably well)
	{
	  Pk2=P_lin[z_ix*Nbins]/kmin*k2;
	};

      if(k3_ix>=0)
	{
	  Pk3=P_lin[z_ix*Nbins+k3_ix];
	}
      else //Use linear approx for very small ks (should work reasonably well)
	{
	  Pk3=P_lin[z_ix*Nbins]/kmin*k3;
	};
   
 
  //Get B(k1, k2)
  return 2*(F(k1, k2, c_phi12)*Pk1*Pk2
	    +F(k1, k3, c_phi13)*Pk1*Pk3
	    +F(k2, k3, c_phi23)*Pk2*Pk3);
}


__host__ __device__ double g3lhalo::apertureFilter(double eta)
{
  return 0.5*eta*eta*exp(-0.5*eta*eta);
}


__host__ __device__ void g3lhalo::SiCi(double x, double& si, double& ci)
  {
  double x2=x*x;
  double x4=x2*x2;
  double x6=x2*x4;
  double x8=x4*x4;
  double x10=x8*x2;
  double x12=x6*x6;
  double x14=x12*x2;

  if(x<4)
    { 
      
      double a=1-4.54393409816329991e-2*x2+1.15457225751016682e-3*x4
	-1.41018536821330254e-5*x6+9.43280809438713025e-8*x8
	-3.53201978997168357e-10*x10+7.08240282274875911e-13*x12
	-6.05338212010422477e-16*x14;

      double b=1+1.01162145739225565e-2*x2+4.99175116169755106e-5*x4
	+1.55654986308745614e-7*x6+3.28067571055789734e-10*x8
	+4.5049097575386581e-13*x10+3.21107051193712168e-16*x12;

      si=x*a/b;

      double gamma=0.5772156649;
      a=-0.25+7.51851524438898291e-3*x2-1.27528342240267686e-4*x4
	+1.05297363846239184e-6*x6-4.68889508144848019e-9*x8
	+1.06480802891189243e-11*x10-9.93728488857585407e-15*x12;
      
      b=1+1.1592605689110735e-2*x2+6.72126800814254432e-5*x4
	+2.55533277086129636e-7*x6+6.97071295760958946e-10*x8
	+1.38536352772778619e-12*x10+1.89106054713059759e-15*x12
	+1.39759616731376855e-18*x14;

      ci=gamma+std::log(x)+x2*a/b;
    }
  else
    {
      double x16=x8*x8;
      double x18=x16*x2;
      double x20=x10*x10;
      double cos_x=cos(x);
      double sin_x=sin(x);

      double f=(1+7.44437068161936700618e2/x2+1.96396372895146869801e5/x4
		+2.37750310125431834034e7/x6+1.43073403821274636888e9/x8
		+4.33736238870432522765e10/x10+6.40533830574022022911e11/x12
		+4.20968180571076940208e12/x14+1.00795182980368574617e13/x16
		+4.94816688199951963482e12/x18-4.94701168645415959931e11/x20)
	/(1+7.46437068161927678031e2/x2+1.97865247031583951450e5/x4
	  +2.41535670165126845144e7/x6+1.47478952192985464958e9/x8
	  +4.58595115847765779830e10/x10+7.08501308149515401563e11/x12
	  +5.06084464593475076774e12/x14+1.43468549171581016479e13/x16
	  +1.11535493509914254097e13/x18)/x;
      
      double g=(1+8.1359520115168615e2/x2+2.35239181626478200e5/x4
		+3.12557570795778731e7/x6+2.06297595146763354e9/x8
		+6.83052205423625007e10/x10+1.09049528450362786e12/x12
		+7.57664583257834349e12/x14+1.81004487464664575e13/x16
		+6.43291613143049485e12/x18-1.36517137670871689e12/x20)/
	(1+8.19595201151451564e2/x2+2.40036752835578777e5/x4
	 +3.26026661647090822e7/x6+2.23355543278099360e9/x8
	 +7.87465017341829930e10/x10+1.39866710696414565e12/x12
	 +1.17164723371736605e13/x14+4.01839087307656620e13/x16
	 +3.99653257887490811e13/x18)/x2;

      si=0.5*pi-f*cos_x-g*sin_x;
      ci=f*sin_x-g*cos_x;
    };
  return;
}


__host__ __device__  double g3lhalo::r_200(double m, double z, double zmin, double zmax, int Nbins,
				  const double* rho_bar)
{
  int z_ix=int((z-zmin)*Nbins/(zmax-zmin));
  return pow(0.239*m/rho_bar[z_ix]/200, 1./3.);
}


__host__ __device__ double g3lhalo::u_NFW(double k, double m, double z, double f,
				 double zmin, double zmax, double mmin, double mmax,
				 int Nbins, const double* rho_bar, const double* concentration)
{
  // Get indices of z and m
  int z_ix=int((z-zmin)*Nbins/(zmax-zmin));
  int m_ix=int(std::log(m/mmin)*Nbins/std::log(mmax/mmin));

  // Get concentration
  double c=f*concentration[z_ix*Nbins+m_ix];

  double arg1=k*r_200(m,z, zmin, zmax, Nbins, rho_bar)/c;
  double arg2=arg1*(1+c);
    
  double si1, ci1, si2, ci2;
  SiCi(arg1, si1, ci1);
  SiCi(arg2, si2, ci2);
  
  double term1=sin(arg1)*(si2-si1);
  double term2=cos(arg1)*(ci2-ci1);
  double term3=-sin(arg1*c)/arg2;
  double F=std::log(1.+c)-c/(1.+c);
   
  return (term1+term2+term3)/F;
}


__host__ __device__ double g3lhalo::Nsat(double m, double mth, double sigma, double mprime, double beta)
{
  return 0.5*(1+erf(log(m/mth)/sigma/1.414213562))*pow(m/mprime, beta); 
}


 __host__ __device__ double g3lhalo::Ncen(double m, double alpha, double mth, double sigma)
 {
     return 0.5*alpha*(1+erf(log(m/mth)/sigma/1.414213562));
 }

__host__ __device__ double g3lhalo::NsatNsat(double m, double mmin1, double mmin2, double sigma1, double sigma2, double mprime1, double mprime2, double beta1, double beta2, double A, double epsilon, bool sameType)
{
  double Ns1=Nsat(m, mmin1, sigma1, mprime1, beta1);

  if(sameType) return Ns1*Ns1; 
  
  double Ns2=Nsat(m, mmin2, sigma2, mprime2, beta2);
 
  double results=Ns1*Ns2+A*pow(m, epsilon)*sqrt(Ns1*Ns2);
  if(results<0) return 0;
  return results;
}


__host__ __device__ double g3lhalo::G_g(double k, double m, double z, double f, double alpha,
			       double mth,
			       double sigma, const double mprime, double beta, 
			       double zmin, double zmax, double mmin, double mmax,
			       int Nbins, const double* rho_bar, const double* concentration )
{
  return Ncen(m, alpha, mth, sigma)+Nsat(m, mth, sigma, mprime, beta)*u_NFW(k, m, z, f, zmin, zmax, mmin, mmax, Nbins, rho_bar,
									    concentration);
}

__host__ __device__ double g3lhalo::G_gg(double k1, double k2, double m, double z, double f1,
					 double f2, double alpha1, double alpha2, double mmin1,
					 double mmin2, double sigma1, double sigma2, double mprime1,
					 double mprime2, double beta1, double beta2, double A, double epsilon,
					 double zmin, double zmax, double mmin, double mmax,
					 int Nbins, const double* rho_bar, const double* concentration, bool sameType)
{
  double Nc1=Ncen(m, alpha1, mmin1, sigma1);
  double Nc2=Ncen(m, alpha2, mmin2, sigma2);
  double Ns1=Nsat(m, mmin1, sigma1, mprime1, beta1);
  double Ns2=Nsat(m, mmin2, sigma2, mprime2, beta2);
  
  return Nc1*Ns2*u_NFW(k2, m, z, f2, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration)
    + Nc2*Ns1*u_NFW(k1, m, z, f1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration)
    + NsatNsat(m, mmin1, mmin2, sigma1, sigma2, mprime1, mprime2, beta1, beta2, A, epsilon, sameType)
    *u_NFW(k1, m, z, f1, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration)
    *u_NFW(k2, m, z, f2, zmin, zmax, mmin, mmax, Nbins, rho_bar, concentration);
}













