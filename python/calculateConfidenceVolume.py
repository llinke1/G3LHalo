import numpy as np
import sys

fn_fisher=sys.argv[1]
fn_optParams=sys.argv[2]
fn_chiSq=sys.argv[3]
fn_samplings=sys.argv[4]
S=int(sys.argv[5]) #Number sampling points
fn_weights=sys.argv[6]
cl=float(sys.argv[7])
fn_cl=sys.argv[8]

# Read in optimal parameters
Nparams=14
param_dict={}
with open(fn_optParams) as f:
    for line in f:
        (key, val)=line.split()
        param_dict[key]=float(val)

optParams=np.zeros(Nparams).reshape(Nparams)

optParams[0]=param_dict["f_1"]
optParams[1]=param_dict["f_2"]
optParams[2]=param_dict["alpha_1"]
optParams[3]=param_dict["alpha_2"]
optParams[4]=(param_dict["M_min1"])
optParams[5]=(param_dict["M_min2"])
optParams[6]=param_dict["sigma_1"]
optParams[7]=param_dict["sigma_2"]
optParams[8]=(param_dict["M'_1"])
optParams[9]=(param_dict["M'_2"])
optParams[10]=param_dict["beta_1"]
optParams[11]=param_dict["beta_2"]
optParams[12]=param_dict["A"]
optParams[13]=param_dict["epsilon"]

optParams.reshape(1, Nparams)


#Read in Fisher
fisher=np.loadtxt(fn_fisher)

#Read in chiSq
chiSq=np.loadtxt(fn_chiSq)[:,0]*1e-8

#Read in samplings
samplings=np.loadtxt(fn_samplings).reshape(S, Nparams)


# Calculate difference
difference=samplings-optParams

# Calculate weights
weights=np.zeros(S)

for i, w in enumerate(weights):
    w1=-0.5*chiSq[i]
    w2=0.5*np.dot(difference[i].dot(fisher), (difference.T)[:,i])
    weights[i]=np.exp(w1+w2)

weights=weights/np.sum(weights)
np.savetxt(fn_weights, weights)

# Calculate Confidence Volume
#Combine weights and points
weights=weights.reshape(S, 1)

points=np.hstack((samplings, weights))

#Sort points
points_sorted=np.array(sorted(points, key=lambda x: np.linalg.norm(x[:-1] - optParams)))

#Get all points inside Confidence Interval
i=0
summe=0
#cl_points=np.array([])
while((summe<cl) & (i<S)):
#    cl_points=np.append(cl_points, points_sorted[i])
    summe+=points_sorted[i][-1]
    print(*points_sorted[i], sep=" ")
    i+=1

cl_points=points_sorted[:i,:-1]

# Get confidence intervals
cl_mins=np.amin(cl_points, axis=0)
cl_maxs=np.amax(cl_points, axis=0)

cl=np.column_stack((cl_mins, cl_maxs))
np.savetxt(fn_cl, cl)
