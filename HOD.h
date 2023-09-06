#ifndef G3LHALO_HOD_H
#define G3LHALO_HOD_H

#include "Params.h"

// #if GPU
// #include <cuda_runtime_api.h>
// #include <cuda.h>
// #include "cuda_helpers.h"
// #endif

namespace g3lhalo
{
    /**
     * Class calculating the HOD
     * Uses the Model by Zheng 2005 for centrals and Zehavi 2005 for satellites
     * In the future: Other HOD models could be inherited from this class
     * @author Laila Linke laila.linke@uibk.ac.at
     */
    class HOD
    {
    public:
        /// Model parameters
        Params *params;
        Params *dev_params;

        /// Empty Constructor
        HOD(){};

        /// Constructor from values
        HOD(Params *params_);

        /// Destructor
        ~HOD();

        /**
         * Average number of satellite galaxies per halo
         * @param m Halo mass [msun]
         */
        __host__ __device__ double Nsat(double m);

        /**
         * Average number of central galaxies per halo
         * @param m Halo mass [msun]
         */
        __host__ __device__ double Ncen(double m);
    };
    /**
     * Average number of satellite galaxies pairs
     * @param m Halo mass [Msun]
     * @param hod1 HOD for first galaxy type
     * @param hod2 HOD for second galaxy type (can be the same as hod1)
     * @param scale1 scaling weight for satellite variance (type a) (optional)
     * @param scale2 scaling weight for satellite variance (type b) (optional)
     */
    __host__ __device__ double NsatNsat(double m, HOD hod1, HOD hod2, double A = 0, double epsilon = 0,  double scale1 = 1, double scale2 = 1);
}

#endif // G3LHALO_HOD_H