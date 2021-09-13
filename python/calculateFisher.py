import numpy as np
import sys
import matplotlib.pyplot as plt
fn_cov=sys.argv[1] #Filename of inverted covariance matrix

fn_nnmap=sys.argv[2] #Values of nnmap for derivatives

fn_priors=sys.argv[3] # Priors

fn_out=sys.argv[4] # Output filename

h=0.001

# Read in priors
priors_dict={}
with open(fn_priors) as f:
    for line in f:
        (key, val1, val2)=line.split()
        priors_dict[key]=[float(val1), float(val2)]

# Calculate differentials
df=h*(priors_dict["f"][1]-priors_dict["f"][0])
dalpha=h*(priors_dict["alpha"][1]-priors_dict["alpha"][0])
dMmin=h*(priors_dict["M_min"][1]-priors_dict["M_min"][0])
dsigma=h*(priors_dict["sigma"][1]-priors_dict["sigma"][0])
dMprime=h*(priors_dict["M'"][1]-priors_dict["M'"][0])
dbeta=h*(priors_dict["beta"][1]-priors_dict["beta"][0])
dA=h*(priors_dict["A"][1]-priors_dict["A"][0])
depsilon=h*(priors_dict["epsilon"][1]-priors_dict["epsilon"][0])

d=np.array([df, df, dalpha, dalpha, dMmin, dMmin, dsigma, dsigma, dMprime, dMprime, dbeta, dbeta, dA, depsilon])
d=d.reshape((14, 1))

# Calculate Derivative
nnmap_derivs=np.loadtxt(fn_nnmap)
nnmap_plus=nnmap_derivs[::2]
nnmap_minus=nnmap_derivs[1::2]

derivs=(nnmap_plus-nnmap_minus)/2./d

# Calculate Fisher
cov=np.loadtxt(fn_cov)

fisher=np.dot(derivs, cov).dot(derivs. T)

np.savetxt(fn_out, fisher)
