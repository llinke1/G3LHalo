#!/bin/bash

###############################################################################
# Skript calculating G3L aperture statistics from Halomodel
# Calculates auto-correlation for both populations (N1N1M and N2N2M) and cross-correlation (N1N2M)
#
# Author: Laila Linke, llinke@astro.uni-bonn.de
###############################################################################

# Folder for Results, is created if it doesn't exist
DIR_RESULTS="../../g3l_halo_results/"

# Folder with C++ executables
DIR_BIN="../bin/"

# ASCII file containing source n_z
# Format: z n(z)
FILE_NZ="pz_sources.dat"

# ASCII file containing lens n_z for population 1
# Format: z n(z)
FILE_NZ_LENS1="pz_lenses.dat"

# ASCII file containing lens n_z for population 2
# Format: z n(z)
FILE_NZ_LENS2="pz_lenses.dat"

# Parameter file containing Cosmology
# Format: see cosmo.param
COSMO="cosmo.param"

# Parameter file containing Halomodel Parameters
# Format: see hod.param
FILE_PARAMS="hod.param"

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

################ Calculate Aperture Statistics #################################
echo "Started with calculating NNMap | $(date +%T)"
$DIR_BIN/calculateNNMap_twoPop.x $COSMO $Z_MIN $Z_MAX $K_MIN $K_MAX $M_MIN $M_MAX $NBINS $FILE_G $FILE_NZ_LENS1 $FILE_NZ_LENS2 $FILE_W $FILE_DWDZ $FILE_HMF $FILE_P $FILE_BH $FILE_CONC $FILE_PARAMS $FILE_THETAS > $DIR_RESULTS/NNMap.dat

echo "Done | $(date +%T)"
