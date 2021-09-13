#ifndef G3LHALO_CONSTANTS_H
#define G3LHALO_CONSTANTS_H

#include <complex>



namespace g3lhalo
{
 /*****************************************/
  /* USEFUL NUMBERS ************************/
  /*****************************************/

  ///Pi
  const double pi=3.14159;

  ///Sqrt(2)
  const double sq2=sqrt(2);

  ///ln(10)
  const double ln10=log(10);

  ///Imaginary unit
  const std::complex<double> i(0.,1.);
  const std::complex<float> i_fl(0.,1.);

  
  ///1 Solarmass in kg
  const double solarmass_in_kg=1.98845e30;

  ///1 Mpc in m
  const double Mpc_in_m=3.086e22;

  ///1 rad in arcmin
  const double rad_in_arcmin=pi/180.0/60.0;
  const double arcmin_in_rad=2.909e-4; //Arcmin in Radians
  
  ///Speed of light in km/s
  const double c=2.99792e5;

  ///Newtonion Gravitational Constant in m^3/s^2/kg
  const double G=6.674e-11; 


}

#endif
