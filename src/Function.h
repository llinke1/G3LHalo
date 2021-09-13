#ifndef G3LHALO_FUNCTION_H
#define G3LHALO_FUNCTION_H

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
  
  class Function
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

    ///X-Values
    std::vector<double> x_values_;

    ///Y-Values
    std::vector<double> y_values_;

    ///Empty Constructor
    Function(){};

    /**
     * @brief Constructor from file
     * @detail Reads in an ascii file and stores first column as x and second 
     * column as y. If filename="none", the default value is used for all x
     *
     * @warning Requires a two-column ascii file
     * @param filename File with tabularized function or "none"
     * @param default_value Default Value of the function
     */
    Function(std::string filename, const double& default_value);

    /**
     * @brief Returns y-value corresponding to x
     * @param x x-value
     */
    double at(const double& x);

    /**
     * @brief Returns the lower border of the x-bin corresponding to y
     * @param y y-Value
     */
    double x_lower(const double& y);

    int read(const int& Nbins, const double& min, const double& max, std::vector<double>& values);
    int readLog(const int& Nbins, const double& min, const double& max, std::vector<double>& values);
  };

}






#endif //G3LCONG_FUNCTION_H
