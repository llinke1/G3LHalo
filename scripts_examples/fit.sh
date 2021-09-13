#!/bin/bash

###############################################################################
# Skript fitting Halomodel to measured G3L aperture statistics 
# Simulataneously fits N1N1M, N1N2M, and N2N2M
#
# Author: Laila Linke, llinke@astro.uni-bonn.de
###############################################################################

# Folder for Results, is created if it doesn't exist
DIR_RESULTS="../../fitresults/"

# Folder with C++ executables
DIR_BIN="../bin/"

# Folder with Python scripts
DIR_PYTHON="../python/"

# ASCII file containing source n_z
# Format: z n(z)
FILE_NZ="pz_sources.dat"

# ASCII file containing lens n_z for population 1
# Format: z n(z)
FILE_NZ_LENS1="pz_lenses.dat"

# ASCII file containing lens n_z for population 2
# Format: z n(z)
FILE_NZ_LENS2="pz_lenses.dat"

# Folder with measurements
DIR_DATA="../../fitdata/"
FILE_N1N2M="$DIR_DATA/n1n2m.dat" # File with <N1N2M> measurement
FILE_N1N1M="$DIR_DATA/n1n1m.dat" # File with <N1N1M> measurement
FILE_N2N2M="$DIR_DATA/n2n2m.dat" # File with <N2N2M> measurement
FILE_COV="$DIR_DATA/cov_inv_m1m2.dat" #INVERSE covariancematrix of measurement

# Parameter file containing Cosmology
# Format: see cosmo.param
COSMO="cosmo.param"

# Parameter file containing initial Halomodel Parameters
# Format: see hod.param
FILE_PARAMS="hod.param"

# Parameter file containint prior boundaries
# Format: see priors.param
FILE_PRIORS="priors.param"

# Output Filename for best-fitting Halomodel parameters
FILE_OUT="$DIR_RESULTS/best_fit.param"

# ASCII file containing scale radii [arcmin]
# Format: see thetas.dat
FILE_THETAS="thetas.dat"

# Binning of Tabulated functions
Z_MIN=0.001
Z_MAX=2.0
K_MIN=1e-4
K_MAX=1e7
M_MIN=1e10
M_MAX=1e17
NBINS=256

# Filenames of Tabulated functions
# If DO_PRECOMPUTING is 1, these are created, otherwise they must already exist!
FILE_G=$DIR_RESULTS/g.dat # Lensing efficiency
FILE_W=$DIR_RESULTS/w.dat # Angular diameter distance
FILE_DWDZ=$DIR_RESULTS/dwdz.dat # Derivative of angular diameter distance wrt z
FILE_HMF=$DIR_RESULTS/hmf.dat # Halo mass function
FILE_P=$DIR_RESULTS/Plin.dat # Linear Power spectrum
FILE_BH=$DIR_RESULTS/b_h.dat # Halo Bias
FILE_CONC=$DIR_RESULTS/concentration.dat # NFW concentration


# Create Outputfolder if it does not exist
mkdir -p $DIR_RESULTS

# Set to 1 if Tabulated Functions need to be performed
DO_PRECOMPUTING=0

################ Calculate Tabulated Functions #################################

if [ $DO_PRECOMPUTING -gt 0 ]
then
    echo "Started with calculating tabulated functions | $(date +%T)"
    
    $DIR_BIN/doPrecomputations.x $COSMO $FILE_NZ $Z_MIN $Z_MAX $M_MIN $M_MAX $K_MIN $K_MAX $NBINS $FILE_G $FILE_W $FILE_DWDZ $FILE_HMF $FILE_P $FILE_BH $FILE_CONC
fi

################ Do Fit  #######################################################
echo "Started with Fit | $(date +%T)"
$DIR_BIN/fit_twoPop.x $COSMO $Z_MIN $Z_MAX $K_MIN $K_MAX $M_MIN $M_MAX $NBINS $FILE_G $FILE_NZ_LENS1 $FILE_NZ_LENS2 $FILE_W $FILE_DWDZ $FILE_HMF $FILE_P $FILE_BH $FILE_CONC $FILE_PARAMS $FILE_N1N2M $FILE_N1N1M $FILE_N2N2M $FILE_COV $FILE_PRIORS $FILE_OUT > $DIR_RESULTS/Fit.log

################ Calculate Aperture Statistics for best-fitting parameters #####
echo "Started with calculating NNMap | $(date +%T)"
$DIR_BIN/calculateNNMap_twoPop.x $COSMO $Z_MIN $Z_MAX $K_MIN $K_MAX $M_MIN $M_MAX $NBINS $FILE_G $FILE_NZ_LENS $FILE_NZ_LENS $FILE_W $FILE_DWDZ $FILE_HMF $FILE_P $FILE_BH $FILE_CONC $FILE_OUT $FILE_THETAS > $DIR_RESULTS/NNMap.dat

################ Calculate Derivatives of aperture statistics ##################
echo "Started with Derivative Calculation | $(date +%T)"
$DIR_BIN/calculateDerivativeNNMap_twoPop.x $COSMO $Z_MIN $Z_MAX $K_MIN $K_MAX $M_MIN $M_MAX $NBINS $FILE_G $FILE_NZ_LENS $FILE_NZ_LENS $FILE_W $FILE_DWDZ $FILE_HMF $FILE_P $FILE_BH $FILE_CONC $FILE_PARAMS $FILE_THETAS $FILE_PRIORS > $DIR_RESULTS/deriv_NNMap.dat

############### Calculate Fisher matrix ########################################
echo "Started calculating Fisher Matrix | $(date +%T)"
python $DIR_PYTHON/calculateFisher.py $FILE_COV $DIR_RESULTS/deriv_NNMap.dat $FILE_PRIORS $DIR_RESULTS/fisher.dat

################ Draw Sampling Points #########################################
echo "Started drawing Sampling Points | $(date +%T)"
python $DIR_PYTHON/drawSampling.py $DIR_RESULTS/fisher.dat $FILE_PARAMS $FILE_PRIORS $S > $DIR_RESULTS/samplings.dat

################# Calculate Chi Square of Samplings ###########################
echo "Started calculating Chi Square for Samplings | $(date +%T)"
$DIR_BIN/calculateChiSquared.x $COSMO $Z_MIN $Z_MAX $K_MIN $K_MAX $M_MIN $M_MAX $NBINS $FILE_G $FILE_NZ_LENS $FILE_NZ_LENS $FILE_W $FILE_DWDZ $FILE_HMF $FILE_P $FILE_BH $FILE_CONC $FILE_PARAMS $FILE_N1N2M $FILE_N1N1M $FILE_N2N2M $FILE_COV $FILE_PRIORS $DIR_RESULTS/samplings.dat > $DIR_RESULTS/chisquared_samplings.dat

################# Calculate Confidence Volume ##################################
echo "Started calculating Confidence Volume | $(date +%T)"
python $DIR_PYTHON/calculateConfidenceVolume.py $DIR_RESULTS/fisher.dat $FILE_PARAMS $DIR_RESULTS/chisquared_samplings.dat $DIR_RESULTS/samplings.dat $S $DIR_RESULTS/weights.dat $CL $DIR_RESULTS/confidencevolume.dat



echo "Done | $(date +%T)"
