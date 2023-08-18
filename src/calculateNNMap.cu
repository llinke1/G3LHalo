#include "NNMap_Model.h"
#include "Params.h"
#include "Function.h"
#include "Function2D.h"
#include "constants.h"
#include "helpers.h"
#include "Cosmology.h"
#include "HOD.h"

#include <fstream>

int main(int argc, char *argv[])
{
  int n_params = 22;
  std::string usage =
      "./calculateNNMap.x \n \
      Cosmology File \n \
      minimal redshift \n \
      maximal redshift \n \
      minimal halo mass [Msun] \n \
      maximal halo mass [Msun] \n \
      minimal k [1/Mpc] \n \
      maximal k [1/Mpc] \n \
      Number of bins \n \
      File for lensing efficiency \n \
      File for lens redshift distribution \n \
      File for lens redshift distribution \n \
      File for comoving distance \n \
      File for derivative of comoving distance \n \
      File for Halo Mass function \n File for Linear power spectrum \n \
      File for Halo bias \n \
      File for concentration \n \
      File for Halomodel Params \n \
      File for Halomodel Params \n \
      Parameter A for HOD correlation \n \
      Parameter epsilon for HOD correlation \n \
      File with thetas";
  std::string example = "./calculateNNMap.x cosmo.param 0.1 2 1e10 1e17 0.01 1000 256 g.dat w.dat dwdz.dat hmf.dat Plin.dat b_h.dat conc.dat hod1.param hod2.param 0 0 thetas.dat";

  g3lhalo::checkCmdLine(argc, n_params, usage, example);

  std::string fn_cosmo = argv[1];
  double zmin = std::stod(argv[2]);
  double zmax = std::stod(argv[3]);
  double kmin = std::stod(argv[4]);
  double kmax = std::stod(argv[5]);
  double mmin = std::stod(argv[6]);
  double mmax = std::stod(argv[7]);
  int Nbins = std::stoi(argv[8]);

  std::string fn_g = argv[9];
  std::string fn_plens1 = argv[10];
  std::string fn_plens2 = argv[11];
  std::string fn_w = argv[12];
  std::string fn_dwdz = argv[13];
  std::string fn_hmf = argv[14];
  std::string fn_P_lin = argv[15];
  std::string fn_b_h = argv[16];
  std::string fn_concentration = argv[17];
  std::string fn_params1 = argv[18];
  std::string fn_params2 = argv[19];
  double A = std::stod(argv[20]);
  double epsilon = std::stod(argv[21]);

  std::string fn_thetas = argv[22];

#if VERBOSE
  std::cerr << "Finished reading cli" << std::endl;
#endif

  g3lhalo::Function g(fn_g, 0.0);
  g3lhalo::Function plens1(fn_plens1, 0.0);
  g3lhalo::Function plens2(fn_plens2, 0.0);
  g3lhalo::Function w(fn_w, 0.0);
  g3lhalo::Function dwdz(fn_dwdz, 0.0);

#if VERBOSE
  std::cerr << "Finished assigning functions1D" << std::endl;
#endif

  g3lhalo::Function2D hmf(fn_hmf, 0.0);
  g3lhalo::Function2D P_lin(fn_P_lin, 0.0);
  g3lhalo::Function2D b_h(fn_b_h, 0.0);
  g3lhalo::Function2D concentration(fn_concentration, 0.0);

#if VERBOSE
  std::cerr << "Finished assigning fucntions2D" << std::endl;
#endif

  g3lhalo::Params params1(fn_params1);
  g3lhalo::Params params2(fn_params2);

  g3lhalo::HOD hod1(&params1);
  g3lhalo::HOD hod2(&params2);

  std::vector<double> g_val, plens1_val, plens2_val, w_val, dwdz_val, hmf_val, P_lin_val, b_h_val, concentration_val;

  if (g.read(Nbins, zmin, zmax, g_val) || plens1.read(Nbins, zmin, zmax, plens1_val) || plens2.read(Nbins, zmin, zmax, plens2_val) || w.read(Nbins, zmin, zmax, w_val) || dwdz.read(Nbins, zmin, zmax, dwdz_val) || hmf.read(Nbins, zmin, zmax, mmin, mmax, hmf_val) || P_lin.read(Nbins, zmin, zmax, kmin, kmax, P_lin_val) || b_h.read(Nbins, zmin, zmax, mmin, mmax, b_h_val) || concentration.read(Nbins, zmin, zmax, mmin, mmax, concentration_val))
  {
    std::cerr << "Problem reading in files. Exiting." << std::endl;
    exit(1);
  };

#if VERBOSE
  std::cerr << "Finished reading in Functions" << std::endl;
#endif

  // Set up cosmology
  g3lhalo::Cosmology cosmo(fn_cosmo);

  g3lhalo::NNMap_Model nnmap(&cosmo, zmin, zmax, kmin, kmax, mmin, mmax, Nbins,
                             g_val.data(), plens1_val.data(), plens2_val.data(), w_val.data(), dwdz_val.data(),
                             hmf_val.data(), P_lin_val.data(), b_h_val.data(), concentration_val.data(),
                             &hod1, &hod2, A, epsilon);

#if VERBOSE
  std::cerr << "Finished initializing NNMap" << std::endl;
#endif

  // READ IN THETA VALUES
  std::ifstream input(fn_thetas);
  std::vector<double> thetas1, thetas2, thetas3;
  double theta1, theta2, theta3;
  while (input >> theta1 >> theta2 >> theta3)
  {
    thetas1.push_back(theta1);
    thetas2.push_back(theta2);
    thetas3.push_back(theta3);
  };
  int n = thetas1.size();

#if VERBOSE
  std::cerr << "Finished reading in Thetas" << std::endl;
#endif

  //<N1N2Map>
  for (int i = 0; i < n; i++)
  {
    double theta1 = thetas1.at(i); // Read Thetas
    double theta2 = thetas2.at(i);
    double theta3 = thetas3.at(i);
    double theta_rad1 = theta1 * g3lhalo::arcmin_in_rad; // Convert Arcmin to Rad
    double theta_rad2 = theta2 * g3lhalo::arcmin_in_rad; // Convert Arcmin to Rad
    double theta_rad3 = theta3 * g3lhalo::arcmin_in_rad; // Convert Arcmin to Rad
#if VERBOSE
    std::cerr << "Calculation for " << theta1 << " " << theta2 << " " << theta3 << std::endl;
#endif
    std::cout << theta1 << " "
              << theta2 << " "
              << theta3 << " "
              << nnmap.NNMap_1h(theta_rad1, theta_rad2, theta_rad3, 1, 2) << " "
              << nnmap.NNMap_2h(theta_rad1, theta_rad2, theta_rad3, 1, 2) << " "
              << nnmap.NNMap_3h(theta_rad1, theta_rad2, theta_rad3, 1, 2) << " "
              << std::endl;
  }

  return 0;
}
