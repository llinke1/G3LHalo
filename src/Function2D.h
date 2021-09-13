#ifndef G3LHALO_FUNCTION2D_H
#define G3LHALO_FUNCTION2D_H

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
namespace g3lhalo
{
  /**
   * Class for tabularized functions
   *
   * @author Laila Linke llinke@astro.uni-bonn.de
   * @date August 2018
   */
  
  class Function2D
  {
  private:

    /**
     * @brief Returns the y value between two indices by linear interpolation
     * @param index1 First index
     * @param index2 Second index
     * @param x x-value corresponding to y
     */
    double interpolate(const int& index1, const int& index2, const double& x);

    ///Default Value of the Function
    double default_value_;
    
  public:

    double x1_min, x1_max, x2_min, x2_max;
    int N1, N2;
    
    ///X1-Values (lin binned)
    std::vector<double> x1_values_;

    ///X2-Values (log binned)
    std::vector<double> x2_values_;
    
    ///Y-Values
    std::vector<double> y_values_;

    ///Empty Constructor
    Function2D(){};

    /**
     * @brief Constructor from file
     * @detail Reads in an ascii file and stores first column as x and second 
     * column as y. If filename="none", the default value is used for all x
     *
     * @warning Requires a two-column ascii file
     * @param filename File with tabularized function or "none"
     * @param default_value Default Value of the function
     */
    Function2D(std::string filename, const double& default_value);

    /**
     * @brief Returns y-value corresponding to (x1,x2)
     * @param x1 x1-value
     * @param x2 x2-value
     */
    double at(const double& x1, const double& x2);


    int read(const int& Nbins, const double& min1, const double& max1, const double& min2, const double& max2, std::vector<double>& values);


  };
}





#endif //G3LCONG_FUNCTION_H
