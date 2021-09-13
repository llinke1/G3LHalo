#include "fitHelpers.h"
#include "constants.h"


#include <fstream>
#include <sstream>


g3lhalo::fitData::fitData(const std::string& fn1, const std::string& fn2, const std::string& fn3, const std::string& fncov, Priors* priors_, NNMap_Model* model_)
{
  priors=priors_;
  model=model_;

  readInData(fn1, fn2, fn3, fncov);
}

void g3lhalo::fitData::readInData(const std::string& fn1, const std::string& fn2, const std::string& fn3, const std::string& fncov)
{
  double x1tmp, x2tmp, x3tmp, ytmp, tmp;
  std::ifstream input;
  
  //Read in N1N2Map  
  input.open(fn1.c_str());
  if(!input.is_open()) //checking if file can be opened
    {
      std::cerr << "FitData: Could not open input file:"<<fn1<<" Exiting. \n";
      exit(1);
    };

  while(input>>x1tmp>>x2tmp>>x3tmp>>ytmp>>tmp)
    {
      theta11.push_back(x1tmp);
      theta12.push_back(x2tmp);
      theta13.push_back(x3tmp);
      N1N2Map.push_back(ytmp);
    };
  
  if(input.bad())
    {
      std::cout<<"Error: Couldn't read "<<fn1<<std::endl;
      exit(1);
    };
  input.close();
  //Read in N1N1Map
  input.open(fn2.c_str());
  if(!input.is_open()) //checking if file can be opened
    {
      std::cerr << "FitData: Could not open input file:"<<fn2<<" Exiting. \n";
      exit(1);
    };
  while(input>>x1tmp>>x2tmp>>x3tmp>>ytmp>>tmp)
    {
      theta21.push_back(x1tmp);
      theta22.push_back(x2tmp);
      theta23.push_back(x3tmp);
      N1N1Map.push_back(ytmp);
    };
  
  if(input.bad())
    {
      std::cout<<"Error: Couldn't read "<<fn2<<std::endl;
      exit(1);
    };
  input.close();
  //Read in N2N2Map
  input.open(fn3.c_str());
  if(!input.is_open()) //checking if file can be opened
    {
      std::cerr << "FitData: Could not open input file:"<<fn3<<" Exiting. \n";
      exit(1);
    };
  while(input>>x1tmp>>x2tmp>>x3tmp>>ytmp>>tmp)
    {
      theta31.push_back(x1tmp);
      theta32.push_back(x2tmp);
      theta33.push_back(x3tmp);
      N2N2Map.push_back(ytmp);
    };
  
  if(input.bad())
    {
      std::cout<<"Error: Couldn't read "<<fn3<<std::endl;
      exit(1);
    };
  
  input.close();
  //Read in covariance
  input.open(fncov.c_str());
  if(!input.is_open()) //checking if file can be opened
    {
      std::cerr << "FitData: Could not open input file:"<<fncov<<" Exiting. \n";
      exit(1);
    };
  
  double covtmp;
  
  while(input>>covtmp)
    {
      cov.push_back(covtmp);
    };
  
  if(input.bad())
    {
      std::cout<<"Error: Couldn't read "<<fncov<<std::endl;
    };
  input.close();
}



void g3lhalo::getGSLFromParams(const Params* params, const Priors* priors, gsl_vector* params_gsl)
{
  double f1n=(params->f1-priors->fmin)/(priors->fmax-priors->fmin);
  double f2n=(params->f2-priors->fmin)/(priors->fmax-priors->fmin);
  
  double alpha1n=(params->alpha1-priors->alphamin)/(priors->alphamax-priors->alphamin);
  double alpha2n=(params->alpha2-priors->alphamin)/(priors->alphamax-priors->alphamin);

  double mmin1n=log10(params->mmin1/priors->mminmin)/log10(priors->mminmax/priors->mminmin);
  double mmin2n=log10(params->mmin2/priors->mminmin)/log10(priors->mminmax/priors->mminmin);
  
  
  double sigma1n=(params->sigma1-priors->sigmamin)/(priors->sigmamax-priors->sigmamin);
  double sigma2n=(params->sigma2-priors->sigmamin)/(priors->sigmamax-priors->sigmamin);

  double mprime1n=log10(params->mprime1/priors->mprimemin)/log10(priors->mprimemax/priors->mprimemin);
  double mprime2n=log10(params->mprime2/priors->mprimemin)/log10(priors->mprimemax/priors->mprimemin);
  
  
  double beta1n=(params->beta1-priors->betamin)/(priors->betamax-priors->betamin);
  double beta2n=(params->beta2-priors->betamin)/(priors->betamax-priors->betamin);
  
  double An=(params->A-priors->Amin)/(priors->Amax-priors->Amin);
  double epsilonn=(params->epsilon-priors->epsilonmin)/(priors->epsilonmax-priors->epsilonmin);
  
  gsl_vector_set(params_gsl, 0, f1n);
  gsl_vector_set(params_gsl, 1, f2n);
  gsl_vector_set(params_gsl, 2, alpha1n);
  gsl_vector_set(params_gsl, 3, alpha2n);
  gsl_vector_set(params_gsl, 4, mmin1n);
  gsl_vector_set(params_gsl, 5, mmin2n);
  gsl_vector_set(params_gsl, 6, sigma1n);
  gsl_vector_set(params_gsl, 7, sigma2n);
  gsl_vector_set(params_gsl, 8, mprime1n);
  gsl_vector_set(params_gsl, 9, mprime2n);
  gsl_vector_set(params_gsl, 10, beta1n);
  gsl_vector_set(params_gsl, 11, beta2n);
  gsl_vector_set(params_gsl, 12, An);
  gsl_vector_set(params_gsl, 13, epsilonn);
  
}

void g3lhalo::getParamsFromGSL(const gsl_vector* params_gsl, const Priors* priors, Params* params)
{
  double f1=gsl_vector_get(params_gsl, 0)*(priors->fmax-priors->fmin)+priors->fmin;
  double f2=gsl_vector_get(params_gsl, 1)*(priors->fmax-priors->fmin)+priors->fmin;
  double alpha1=gsl_vector_get(params_gsl, 2)*(priors->alphamax-priors->alphamin)+priors->alphamin;
  double alpha2=gsl_vector_get(params_gsl, 3)*(priors->alphamax-priors->alphamin)+priors->alphamin;

  double mmin1=pow(10, log10(priors->mminmax/priors->mminmin)*gsl_vector_get(params_gsl, 4)+log10(priors->mminmin));
  double mmin2=pow(10, log10(priors->mminmax/priors->mminmin)*gsl_vector_get(params_gsl, 5)+log10(priors->mminmin));


  double sigma1=gsl_vector_get(params_gsl, 6)*(priors->sigmamax-priors->sigmamin)+priors->sigmamin;
  double sigma2=gsl_vector_get(params_gsl, 7)*(priors->sigmamax-priors->sigmamin)+priors->sigmamin;

  double mprime1=pow(10, log10(priors->mprimemax/priors->mprimemin)*gsl_vector_get(params_gsl, 8)+log10(priors->mprimemin));
  double mprime2=pow(10, log10(priors->mprimemax/priors->mprimemin)*gsl_vector_get(params_gsl, 9)+log10(priors->mprimemin));


  double beta1=gsl_vector_get(params_gsl, 10)*(priors->betamax-priors->betamin)+priors->betamin;
  double beta2=gsl_vector_get(params_gsl, 11)*(priors->betamax-priors->betamin)+priors->betamin;
  double A=gsl_vector_get(params_gsl, 12)*(priors->Amax-priors->Amin)+priors->Amin;
  double epsilon=gsl_vector_get(params_gsl, 13)*(priors->epsilonmax-priors->epsilonmin)+priors->epsilonmin;
  
  params->f1=f1;
  params->f2=f2;
  params->alpha1=alpha1;
  params->alpha2=alpha2;
  params->mmin1=mmin1;
  params->mmin2=mmin2;
  params->sigma1=sigma1;
  params->sigma2=sigma2;
  params->mprime1=mprime1;
  params->mprime2=mprime2;
  params->beta1=beta1;
  params->beta2=beta2;
  params->A=A;
  params->epsilon=epsilon;
  
}


double g3lhalo::getChiSquared(const gsl_vector* params_gsl, void* dataptr)
{
  fitData * data = (fitData *)dataptr;

  
  // Set Params
  Params params;
  getParamsFromGSL(params_gsl, data->priors, &params);

  std::cerr<<"Calculated GSL Params"<<std::endl;
 
  data->model->updateParams(&params);

  std::cerr<<"Updated Params"<<std::endl;
  
  int Nx = data->theta11.size();

  double diff[3*Nx];

  
  /**** Calculating differences between predictions and measurements */

  // For <N1N2Map>
  for(int i=0; i<Nx; i++)
    {
      //      std::cerr<<i<<std::endl;
      double x1=data->theta11.at(i)*arcmin_in_rad;
      double x2=data->theta12.at(i)*arcmin_in_rad;
      double x3=data->theta13.at(i)*arcmin_in_rad;
      double y=data->N1N2Map.at(i);
      double y_hat=data->model->NNMap(x1, x2, x3, 1, 2);

      diff[i]=(y-y_hat);///0.3/y;
    };

    // For <N1N1Map>
  for(int i=0; i<Nx; i++)
    {
      //      std::cerr<<i<<std::endl;
      double x1=data->theta21.at(i)*arcmin_in_rad;
      double x2=data->theta22.at(i)*arcmin_in_rad;
      double x3=data->theta23.at(i)*arcmin_in_rad;
      double y=data->N1N1Map.at(i);
      double y_hat=data->model->NNMap(x1, x2, x3, 1, 1);

      diff[i+Nx]=(y-y_hat);///0.3/y;
    };

      // For <N2N2Map>
  for(int i=0; i<Nx; i++)
    {
      //      std::cerr<<i<<std::endl;
      double x1=data->theta31.at(i)*arcmin_in_rad;
      double x2=data->theta32.at(i)*arcmin_in_rad;
      double x3=data->theta33.at(i)*arcmin_in_rad;
      double y=data->N2N2Map.at(i);
      double y_hat=data->model->NNMap(x1, x2, x3, 2, 2);

      diff[i+2*Nx]=(y-y_hat);///0.3/y;
    };

  /***** Calculate chi2 *****************************************/
  double chi2=0;
  for(int i=0; i<3*Nx; i++)
    {
      for(int j=0; j<3*Nx; j++)
	{
	  chi2+=diff[i]*diff[j]*data->cov[i*3*Nx+j];
	  
	};
      
      //chi2+=diff[i]*diff[i];
    };

 
  std::cout<<chi2<<" "<<data->model->params->f1<<" "<<data->model->params->f2<<" "<<data->model->params->alpha1<<" "<<data->model->params->alpha2<<" "<<data->model->params->mmin1<<" "<<data->model->params->mmin2<<" "<<data->model->params->sigma1<<" "<<data->model->params->sigma2<<" "<<data->model->params->mprime1<<" "<<data->model->params->mprime2<<" "<<data->model->params->beta1<<" "<<data->model->params->beta2<<" "<<data->model->params->A<<" "<<data->model->params->epsilon<<std::endl;  
  return chi2;
}


void g3lhalo::readInSamplings(const std::string& filename, std::vector<Params>& params)
{
  double f1, f2, alpha1, alpha2, mmin1, mmin2, sigma1, sigma2, mprime1, mprime2, beta1, beta2, A, epsilon;
  std::ifstream input;
  

  input.open(filename.c_str());
  if(!input.is_open()) //checking if file can be opened
    {
      std::cerr << "readInSampling: Could not open input file:"<<filename<<" Exiting. \n";
      exit(1);
    };

  while(input>>f1>>f2>>alpha1>>alpha2>>mmin1>>mmin2>>sigma1>>sigma2>>mprime1>>mprime2>>beta1>>beta2>>A>>epsilon)
    {
      Params pars( f1, f2, alpha1, alpha2, mmin1, mmin2, sigma1, sigma2, mprime1, mprime2, beta1, beta2, A, epsilon);
      params.push_back(pars);
    };
  
  if(input.bad())
    {
      std::cout<<"Error: Couldn't read "<<filename<<std::endl;
      exit(1);
    };
  input.close();
}
