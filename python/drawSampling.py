import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


fn_fisher=sys.argv[1]
fn_optParams=sys.argv[2]
fn_priors=sys.argv[3]
S=int(sys.argv[4]) # Number of sampling points

# Read in optimal parameters
param_dict={}
with open(fn_optParams) as f:
    for line in f:
        (key, val)=line.split()
        param_dict[key]=float(val)

Nparams=14
optParams=np.zeros(Nparams).reshape(Nparams)

optParams[0]=param_dict["f_1"]
optParams[1]=param_dict["f_2"]
optParams[2]=param_dict["alpha_1"]
optParams[3]=param_dict["alpha_2"]
optParams[4]=np.log(param_dict["M_min1"])
optParams[5]=np.log(param_dict["M_min2"])
optParams[6]=param_dict["sigma_1"]
optParams[7]=param_dict["sigma_2"]
optParams[8]=np.log(param_dict["M'_1"])
optParams[9]=np.log(param_dict["M'_2"])
optParams[10]=param_dict["beta_1"]
optParams[11]=param_dict["beta_2"]
optParams[12]=(param_dict["A"])
optParams[13]=param_dict["epsilon"]

#Read in prior bounds
priors_dict={}
with open(fn_priors) as f:
    for line in f:
        (key, val1, val2)=line.split()
        priors_dict[key]=[float(val1), float(val2)]
        
#Read in Fisher
fisher=np.loadtxt(fn_fisher, dtype=np.float64)
fisher[4]*=np.exp(optParams[4])
fisher[:,4]*=np.exp(optParams[4])
fisher[5]*=np.exp(optParams[5])
fisher[:,5]*=np.exp(optParams[5])
fisher[8]*=np.exp(optParams[8])
fisher[:,8]*=np.exp(optParams[8])
fisher[9]*=np.exp(optParams[9])
fisher[:,9]*=np.exp(optParams[9])



#Invert fisher
inv_fisher=np.linalg.inv(fisher)

# Draw Points
Sdrawn=0
while(Sdrawn<S):
    draw=np.random.multivariate_normal(optParams, inv_fisher)
    if((draw[0]>priors_dict["f"][0])
       & (draw[0]<priors_dict["f"][1])
       & (draw[1]>priors_dict["f"][0])
       & (draw[1]<priors_dict["f"][1])
       & (draw[2]>priors_dict["alpha"][0])
       & (draw[2]<priors_dict["alpha"][1])
       & (draw[3]>priors_dict["alpha"][0])
       & (draw[3]<priors_dict["alpha"][1])
       & (draw[4]>np.log(priors_dict["M_min"][0]))
       & (draw[4]<np.log(priors_dict["M_min"][1]))
       & (draw[5]>np.log(priors_dict["M_min"][0]))
       & (draw[5]<np.log(priors_dict["M_min"][1]))
       & (draw[6]>priors_dict["sigma"][0])
       & (draw[6]<priors_dict["sigma"][1])
       & (draw[7]>priors_dict["sigma"][0])
       & (draw[7]<priors_dict["sigma"][1])
       & (draw[8]>np.log(priors_dict["M'"][0]))
       & (draw[8]<np.log(priors_dict["M'"][1]))
       & (draw[9]>np.log(priors_dict["M'"][0]))
       & (draw[9]<np.log(priors_dict["M'"][1]))
       & (draw[10]>priors_dict["beta"][0])
       & (draw[10]<priors_dict["beta"][1])
       & (draw[11]>priors_dict["beta"][0])
       & (draw[11]<priors_dict["beta"][1])
       & (draw[12]>(priors_dict["A"][0]))
       & (draw[12]<(priors_dict["A"][1]))
       & (draw[12]>priors_dict["epsilon"][0])
       & (draw[12]<priors_dict["epsilon"][1])):
        print(draw[0], draw[1], (draw[2]), draw[3], np.exp(draw[4]),
              np.exp(draw[5]), draw[6], draw[7], np.exp(draw[8]), np.exp(draw[9]),
              (draw[10]), draw[11], (draw[12]), draw[13])
        Sdrawn+=1

