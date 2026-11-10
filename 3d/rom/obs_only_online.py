__Author__ = "Amina Benaceur(*^)"
__Copyright__ = "Copyright December 2025\
                   (*)UM6P CC"

import os
import time
import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
#import scipy.sparse.linalg as scLA
from utils import gen_observables, gen_off_data, import_data,\
    build_observable_space, read_solutions, relative_POD, offline_pbdw_matrices,\
    solve_pbdw_statement, norme, \
    compute_inf_sup_constant

from matplotlib.ticker import MaxNLocator
#==============================================================================
# Meta-data 
#==============================================================================
#modifier : False si les resultats FEM sont deja disponibles
generate_data = False
pl.ion()
#modifier
PRE_DIR = "/home/YOUR-PATH/3d/fem"
OFF_DIR = PRE_DIR + "/Offline"
MAT_DIR = OFF_DIR + "/matrices"
BK_DIR = OFF_DIR + "/BK_SOL"
OBS_DIR = OFF_DIR + "/OBSERVATIONS"
TRUE_DIR = OFF_DIR + "/TRUE_SOLS"

obs_densities = ["VERY_COARSE_49", "COARSE_121", "LESS_225", #"NORMAL_361", 361 produces same nb patches as 676
                 "MORE_676", "DENSE_1521"]

for d in obs_densities:
    if (os.path.exists(OBS_DIR+ "/"+ d)==False):
        os.makedirs(OBS_DIR+ "/"+ d)

if generate_data == True:
    gen_observables(PRE_DIR, OFF_DIR, OBS_DIR, obs_densities)
    gen_off_data(PRE_DIR, OFF_DIR, BK_DIR, MAT_DIR, TRUE_DIR)

ref_density = 4

with open(PRE_DIR+"/observ.txt", "w") as f:
    f.writelines(obs_densities[ref_density])

observations = obs_densities[ref_density] # VERY_COARSE_49, "COARSE_121", "LESS_225", "NORMAL_361", "MORE_676", "DENSE_1521"
RIESZ_OBS_DIR = OFF_DIR + "/OBSERVATIONS/" + observations
MASS, STIFF, nb_dof, Mref = import_data(OFF_DIR, MAT_DIR, RIESZ_OBS_DIR)
# =============================================================================
# FEM simulation input
# =============================================================================
simu_time = 4
step = 0.1
Ll = 4 #length of the square side
nb_par = 20#len(os.listdir(TRUE_DIR))
true_set = range(0,nb_par+1,1)
BK_set = range(1,nb_par+1,2)
ref_SPM = MASS+STIFF
#invref_SPM = scLA.inv(ref_SPM)
eps_pod = 1e-4
podSPM = MASS+STIFF
ERR_NORM = podSPM
# =============================================================================
# Offline PBDW
# =============================================================================
observation_set = range(1,Mref+1,1)
observation_functionals = build_observable_space(observation_set, RIESZ_OBS_DIR, nb_dof, ref_SPM)
TRUE_solutions = read_solutions(TRUE_DIR, true_set, nb_dof)
BK_solutions = read_solutions(BK_DIR, BK_set, nb_dof).T
ref_background_space, sg, Nref = relative_POD(BK_solutions, eps_pod, nb_par, podSPM)

Zn_projection = np.matrix(np.zeros((1,nb_dof)))
Um_projection = np.matrix(np.zeros((1,nb_dof)))
projection = np.matrix(np.zeros((1,nb_dof)))
_,_,ref_GloMatrix = offline_pbdw_matrices(ref_SPM, ref_background_space.T, observation_functionals)
ref_beta = compute_inf_sup_constant(ref_SPM, ref_background_space, observation_functionals)
print("inf-sup = ", ref_beta)

# =============================================================================
# Online standard PBDW
# =============================================================================
markers = ['.','+','o','1','.', '<', '2','3','4',',','^', '*', 0,1,2,3,4,5,6,7]
min_err = 10
offline_train_set = list(range(0,nb_par,4)) + [nb_par-1]
offline_nb_par = len(offline_train_set)
offline_solutions = np.ndarray((offline_nb_par, nb_dof))
ref_H1_errors = []

for idid, idpar in enumerate(offline_train_set):
    true = np.matrix(TRUE_solutions[idpar,:]).T
    secMbr = np.matrix(np.zeros((1, Mref+Nref)))
    secMbr[0,0:Mref] = true.T @ ref_SPM @ observation_functionals
    ref_reduced_state_estimate = solve_pbdw_statement(ref_GloMatrix, secMbr.T)
    ref_eta_n = observation_functionals @ ref_reduced_state_estimate[0:Mref:,0]
    ref_z_n = ref_background_space @ ref_reduced_state_estimate[Mref:Mref+Nref:,0]
    ref_state_estimate = ref_z_n + ref_eta_n
    ref_err = ref_state_estimate-true
    ref_H1_errors.append(norme(ref_err, ERR_NORM)/norme(true, ERR_NORM))
    offline_solutions[idid,:] = ref_state_estimate.reshape((nb_dof))
        
xis, sg2, Nref2 = relative_POD(offline_solutions.T, eps_pod, None, podSPM)

Xi = xis
C = observation_functionals.T @ ref_SPM @ Xi # This is C
CT = C.T
CT_C = C.T @ C # This is C^T C

val, vec = np.linalg.eigh(CT_C)
print("eigvals : ", val)

obsOnly_H1_errors = []
online_solutions = np.ndarray((nb_par, nb_dof))
online_test_set = range(0,nb_par,1)
reduced_state_estimate = []

times_dairom = []

for idid, idpar in enumerate(online_test_set):
    true = np.matrix(TRUE_solutions[idpar,:]).T
    secMbr = (true.T @ ref_SPM @ observation_functionals).T
    start_time = time.time()
    #reduced_state_estimate = np.linalg.solve(CT_C, CT @ secMbr)
    #state_estimate = Xi @ reduced_state_estimate
    state_estimate = Xi @ np.linalg.solve(CT_C, CT @ secMbr)
    times_dairom.append(time.time()-start_time)
    err = state_estimate - true
    obsOnly_H1_errors.append(norme(err, ERR_NORM)/norme(true, ERR_NORM))
    online_solutions[idid,:] = state_estimate.reshape((nb_dof))

#diff = online_solutions[11,:] - TRUE_solutions[11,:]
#np.savetxt(PRE_DIR+"/../rom/sol_test.txt", abs(diff).T)

"""
plt.figure(1)
plt.plot(offline_train_set, ref_H1_errors, marker = markers[0], label = "PBDW")
plt.plot(online_test_set, obsOnly_H1_errors, marker = markers[1], label = "ObsOnly")
plt.title("Std PBDW: M="+str(Mref)+", eps_pod = "+str(eps_pod)+", N = "+str(Nref))
plt.ylabel('||u*(k)-true*(k)||/||true*(k)||')
plt.xlabel("Parameter value")
plt.yscale('log')
plt.legend()
"""

pbdw_H1_errors = []
pbdw_solutions = np.ndarray((nb_par, nb_dof))

times_pbdw = []

for idid, idpar in enumerate(online_test_set):
    true = np.matrix(TRUE_solutions[idpar,:]).T
    secMbr = np.matrix(np.zeros((1, Mref+Nref))).T
    secMbr[0:Mref,0] = (true.T @ ref_SPM @ observation_functionals).T
    start_time = time.time()
    ref_reduced_state_estimate = solve_pbdw_statement(ref_GloMatrix, secMbr)
    #ref_eta_n = observation_functionals @ ref_reduced_state_estimate[0:Mref:,0]
    #ref_z_n = ref_background_space @ ref_reduced_state_estimate[Mref:Mref+Nref:,0]
    #ref_state_estimate = ref_z_n + ref_eta_n
    ref_state_estimate = observation_functionals @ ref_reduced_state_estimate[0:Mref:,0] + ref_background_space @ ref_reduced_state_estimate[Mref:Mref+Nref:,0]
    times_pbdw.append(time.time()-start_time)
    ref_err = ref_state_estimate-true
    pbdw_H1_errors.append(norme(ref_err, ERR_NORM)/norme(true, ERR_NORM))
    pbdw_solutions[idid,:] = ref_state_estimate.reshape((nb_dof))


plt.figure(1)
plt.plot(list(online_test_set), pbdw_H1_errors, marker = markers[0], color = "b", label = "PBDW")
plt.plot(list(online_test_set), obsOnly_H1_errors, marker = markers[1], color = "g", label = "DAIROM")
plt.scatter(list(offline_train_set), ref_H1_errors, marker = markers[2], color = "r", label = "Offline PBDW")
plt.title("M="+str(Mref)+", eps_pod = "+str(eps_pod)+", N = "+str(Nref2))
plt.ylabel("Relative H1 error")#'||u*-u_true||/||u_true||')
plt.xlabel("Parameter index")
plt.yscale('log')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()

plt.figure(2)
plt.plot(1+np.arange(len(sg)), sg, marker = markers[4], color = "b")
plt.title("POD on ROM: M="+str(Mref)+", eps_pod = "+str(eps_pod)+", N = "+str(Nref))
plt.ylabel("Singular value")#'||u*-u_true||/||u_true||')
plt.xlabel("rank")
plt.yscale('log')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
#plt.legend()


plt.figure(3)
plt.plot(1+np.arange(len(sg2)), sg2, marker = markers[4], color = "b")
plt.title("POD on PBDW: M="+str(Mref)+", eps_pod = "+str(eps_pod)+", N = "+str(Nref2))
plt.ylabel("Singular value")#'||u*-u_true||/||u_true||')
plt.xlabel("rank")
plt.yscale('log')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
#plt.legend()

plt.show()

print("PBDW: Nref = ", Nref)
print("M = ", Mref)
print("DAIROM: N = ", Nref2)
print("eps_pod = ", eps_pod)
print("PBDW: cond(ref_GloMatrix) = ", np.linalg.cond(ref_GloMatrix))

print("cond(C^T C) : ", np.linalg.cond(CT_C))

print("~~~~~~~~~~~~~~ exec_times ~~~~~~~~~~~~~~~~")
mean_pbdw = np.sum(times_pbdw)/len(times_pbdw)
mean_dairom = np.sum(times_dairom)/len(times_dairom)
ratio = mean_pbdw/mean_dairom
print("PBDW online mean exec time = ", mean_pbdw)
print("DAIROM online mean exec time = ", mean_dairom)
print("ratio = ", ratio)
