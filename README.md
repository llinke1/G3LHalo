<!-- PROJECT LOGO -->
<br />

  <h3 align="center">G3LHalo</h3>

  <p align="center">
    Code for modelling the Galaxy-Galaxy-Galaxy-lensing aperture statistics with the Halomodel and for fitting the model to measured G3L signals
    <br />
    <a href="https://github.com/llinke1/G3LConGPU"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/llinke1/G3LConGPU">View Demo</a>
    ·
    <a href="https://github.com/llinke1/G3LConGPU/issues">Report Bug</a>
    ·
    <a href="https://github.com/llinke1/G3LConGPU/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This code can give halo model predictions for the Galaxy-Galaxy-Galaxy lensing (G3L) aperture statistics, and fit the free halo model parameters to measurements of the G3L aperture statistics. Integrations can be performed GPU accelerated, for which an NVIDIA GPU + CUDA are necessary (although a pure CPU version is also available). The modelling code is complementary to the G3L measurement code in <a href="https://github.com/llinke1/G3LConGPU">G3LConGPU</a>. To learn more about G3L, you can check out <a href="https://ui.adsabs.harvard.edu/abs/2020A%26A...634A..13L/abstract">Linke et al.(2020a)</a>, <a href="https://ui.adsabs.harvard.edu/abs/2020A%26A...640A..59L/abstract"> Linke et al.(2020b)</a>, and <a href="https://ui.adsabs.harvard.edu/abs/2019A%26A...622A.104S/abstract"> Simon et al.(2019)</a>. 


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these steps.

### Prerequisites
To use this code solely with the CPU version, these simple requirements are all that is needed:
* **g++** (Tested for version 9.3.0). 
Under Ubuntu this can be installed with
  ```sh
  sudo apt install build-essential
  ```
* **bash**. Should be available under Linux distributions, for use under Windows, consult how to create a Windows Subsytem for Linux (e.g. here [https://docs.microsoft.com/de-de/windows/wsl/install-win10].
* **openMP** (Tested for version 4.5). Under Ubuntu this can be installed with
```sh
sudo apt-get install libomp-dev
```
* **GNU Scientific Library** (Tested for version 2.6). Check here for how to install it: [https://www.gnu.org/software/gsl/]
* **python** (Tested for version 3.8.5, at least 3 is needed!). On Ubuntu this can be installed with
```sh
sudo apt-get install python3.8
```
To use the GPU accelerated version, additionally the following is needed:

* **NVIDIA graphics card with CUDA capability of at least 2**. Check here to see, if your card works: [https://en.wikipedia.org/wiki/CUDA#GPUs_supported].
* **CUDA SDK Toolkit** (Tested for version 10.1, at least 7 needed!)
Can be downloaded here [https://developer.nvidia.com/accelerated-computing-toolkit]

In general, some knowledge on CUDA and how GPUs work is useful to understand the code!


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/llinke1/G3LHalo.git
   ```
2. Install missing prerequisites
3. Go into source directory and open Makefile
```sh
cd src
xdg-open Makefile
```
4. If you want to use the CPU-only version, set the `GPU` parameter to `false`, if you want to use GPU acceleration set the `GPU` parameter to `true` and adapt the `-arch=` parameter to the architecture of your graphics card.
5. run `make`
6. Now, check if the folder `bin` was created and contains the necessary executables. Optionally, you could run `test_gpu.x` to check if the CUDA compilation worked.


<!-- USAGE EXAMPLES -->
## Usage

### Precomputations
The halo model depends on a range of cosmological functions, which are calculated as look-up tables by the code (but can also be user-provided). These are 
* the *lensing efficiency* g
* the *angular diameter distance* w
* the *derivative of the angular diameter distance w.r.t. redshift* dw/dz
* the *halo mass function* n(m,z)dz
* the *linear power spectrum* P(k)
* the *halo bias* b(h)
* and the *concentration of the NFW profile* c(m, z).
The executable `doPrecomputations.x` calculates these look-up tables as ASCII files. This can be turned off in the bash-scripts.

### Calculation of Halo Model G3L Aperture statistics
The complete calculation of the G3L aperture statistics prediction of the halo model is contained in the script `calculateNNMap.sh`. 

#### Required Input
The calculation requires the following input:
* A parameter file containing the cosmological parameters (see `cosmo.param` for an example)
* A parameter file containig the free parameters of the halomodel (see `hod.param` for an example)
* The redshift distribution of source galaxies (see `pz_sources.dat` for an example)
* The redshift distribution of lens galaxies (see `pz_lenses.dat`for an example)
* A file containing the aperture scale radii in arcmin for which NNMap should be calculated (see `thetas.dat`for an example)

#### Output
Aside from the tabulated precomputated functions, the main output of `calculateNNMap.sh` is the file `NNMap.dat`, whose columns contain:
1. Aperture scale radius 1 [arcmin]
2. Aperture scale radius 2 [arcmin]
3. Aperture scale radius 3 [arcmin]
4. 1-Halo term
5. 2-Halo term
6. 3-Halo term

### Fitting of Halo Model to G3L measurements
The complete fitting of the G3L model to aperture statistics measurements is contained in the script `fit.sh`.

#### Required Input
The calculation requires the following input:
* Measurements of N1N1M, N1N2M, and N2N2M and the related covariance matrix
* A parameter file containing the cosmological parameters (see `cosmo.param` for an example)
* A parameter file containig the initial parameters of the halomodel (see `hod.param` for an example)
* A parameter file containing the borders of the prior range of the halomodel (see `priors.param`)
* The redshift distribution of source galaxies (see `pz_sources.dat` for an example)
* The redshift distribution of lens galaxies (see `pz_lenses.dat`for an example)
* A file containing the aperture scale radii in arcmin for which NNMap should be calculated (see `thetas.dat`for an example)

#### Output
The output of the fit are
* The best-fitting parameters (`best_fit.param`)
* The Model prediction for the best-fitting parameters (`NNMap.dat`)
* The fisher matrix for the model (`fisher.dat`)
* The confidence bounds for the model parameters (`confidencevolume.dat`)


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Laila Linke - [@astro_laila](https://twitter.com/astro_laila) - llinke@astro.uni-bonn.de

Project Link: [https://github.com/llinke1/G3LConGPU](https://github.com/llinke1/G3LConGPU)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* The integration routines use <a href="https://github.com/stevengj/cubature"> cubature </a> 
* This ReadMe is based on <a href="https://github.com/othneildrew/Best-README-Template"> Best-README-Template </a>




