#include "NMapMap_Model.h"
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
    int n_params = 18;
    std::string usage =
        "./calculateNMapMap.x \n\ 
    Cosmology File \n \
    minimal redshift \n\ 
    maximal redshift \n \
    minimal halo mass [Msun] \n \
    maximal halo mass [Msun] \n \
    minimal k [1/Mpc] \n \
    maximal k [1/Mpc] \n \
    Number of bins \n \
    File for lensing efficiency \n \
    File for lens redshift distribution \n \
    File for comoving distance \n \
    File for derivative of comoving distance \n \
    File for Halo Mass function \n \
    File for Linear power spectrum \n \ 
    File for Halo bias \n \
    File for concentration \n \ 
    File for Halomodel Params \n \ 
    File with thetas";
    std::string example =
        "./calculateNMapMap.x cosmo.param 0.1 2 1e10 1e17 0.01 1000 256 g.dat w.dat dwdz.dat hmf.dat Plin.dat b_h.dat conc.dat hod.param thetas.dat";

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
    std::string fn_plens = argv[10];
    std::string fn_w = argv[11];
    std::string fn_dwdz = argv[12];
    std::string fn_hmf = argv[13];
    std::string fn_P_lin = argv[14];
    std::string fn_b_h = argv[15];
    std::string fn_concentration = argv[16];
    std::string fn_params = argv[17];
    std::string fn_thetas = argv[18];

#if VERBOSE
    std::cerr << "Finished reading cli" << std::endl;
#endif

    g3lhalo::Function g(fn_g, 0.0);
    g3lhalo::Function plens(fn_plens, 0.0);
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

    g3lhalo::Params params(fn_params);
    g3lhalo::HOD hod(&params);

    std::vector<double> g_val, plens_val, w_val, dwdz_val, hmf_val, P_lin_val, b_h_val, concentration_val;

    if (g.read(Nbins, zmin, zmax, g_val) || plens.read(Nbins, zmin, zmax, plens_val) || w.read(Nbins, zmin, zmax, w_val) || dwdz.read(Nbins, zmin, zmax, dwdz_val) || hmf.read(Nbins, zmin, zmax, mmin, mmax, hmf_val) || P_lin.read(Nbins, zmin, zmax, kmin, kmax, P_lin_val) || b_h.read(Nbins, zmin, zmax, mmin, mmax, b_h_val) || concentration.read(Nbins, zmin, zmax, mmin, mmax, concentration_val))
    {
        std::cerr << "Problem reading in files. Exiting." << std::endl;
        exit(1);
    };

#if VERBOSE
    std::cerr << "Finished reading in Functions" << std::endl;
#endif

    // Set up cosmology
    g3lhalo::Cosmology cosmo(fn_cosmo);

    g3lhalo::NMapMap_Model nmapmap(&cosmo, zmin, zmax, kmin, kmax, mmin, mmax, Nbins,
                                   g_val.data(), plens_val.data(), w_val.data(), dwdz_val.data(),
                                   hmf_val.data(), P_lin_val.data(), b_h_val.data(), concentration_val.data(),
                                   &hod);

#if VERBOSE
    std::cerr << "Finished initializing NMapMap" << std::endl;
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

    //<NMapMap>
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
                  << nmapmap.NMapMap_1h(theta_rad1, theta_rad2, theta_rad3) << " "
                  << nmapmap.NMapMap_2h(theta_rad1, theta_rad2, theta_rad3) << " "
                  << nmapmap.NMapMap_3h(theta_rad1, theta_rad2, theta_rad3) << " "
                  << std::endl;
    }

    return 0;
}
