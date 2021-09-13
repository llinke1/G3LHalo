#ifndef G3LHALO_HELPERS_H
#define G3LHALO_HELPERS_H

#include <iostream>
#include <string>
#include <cstddef>
#include <vector>


namespace g3lhalo
{
  /**
   * @brief Checks if number of CMD line parameters is correct
   * If number of CMD line parameters is not correct, process exits with error
   * @param argc Number of CMD line parameters
   * @param n_params Number of expected parameters (aside from exec name)
   * @param usage Usage description of exec
   * @param example Example usage description
   */
  void checkCmdLine(int argc, int n_params, std::string usage, std::string example);


  /**
   * @brief Writes x and y values to an ASCII file
   * @param x x-values
   * @param y y-values
   * @param filename Filename to which stuff is written
   */
  void writeToFile(const std::vector<double>& x, const std::vector<double>& y,
		   const std::string& filename);

  /**
   * @brief Writes x1, x2, and y values to an ASCII file
   * @param x1 x1-values
   * @param x2 x2-values
   * @param y y-values
   * @param filename Filename to which stuff is written
   */
    void writeToFile(const std::vector<double>& x1, const std::vector<double>& x2,
		     const std::vector<double>& y, const std::string& filename);
};


#endif // G3LCONG_HELPERS_H
