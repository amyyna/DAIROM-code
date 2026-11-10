__Author__    = "Amina Benaceur(*^)"
__bibli__     = "To appear"
import os
import sys
import time
import shutil
import pickle
import numpy as np
from math import sqrt
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def gen_off_data(PRE_DIR, OFF_DIR, BK_DIR, MAT_DIR, TRUE_DIR):
    start = time.time()
    if (os.path.exists(OFF_DIR)==False):
        os.makedirs(OFF_DIR)
    if (os.path.exists(MAT_DIR)==True):
        shutil.rmtree(MAT_DIR)
    if (os.path.exists(BK_DIR)==True):
        shutil.rmtree(BK_DIR)
    if (os.path.exists(TRUE_DIR)==True):
        shutil.rmtree(TRUE_DIR)
    os.makedirs(MAT_DIR)
    os.makedirs(BK_DIR)
    os.makedirs(TRUE_DIR)
    aa = os.system("FreeFem++ -v 0 "+PRE_DIR+"/fem_obs_only_online.edp")
    if aa!=0:
        sys.exit("freefem failed to run offline computations")
    duration = time.time()-start
    file = open(PRE_DIR+"/log_file.txt","w") 
    file.write("Offline data generation lasted "+str(duration)+" seconds")
    file.close()
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def gen_observables(PRE_DIR, OFF_DIR, OBS_DIR):
    start = time.time()
    if (os.path.exists(OFF_DIR)==False):
        os.makedirs(OFF_DIR)
    if (os.path.exists(OBS_DIR)==True):
        shutil.rmtree(OBS_DIR)
    os.makedirs(OBS_DIR)
    for typ in ["LEAST_16", "EXTREMELY_COARSE_25", "VERY_COARSE_49", "COARSE_121", "LESS_225", "NORMAL_361",
                 "MORE_676", "DENSE_1521"]:
        file = open(PRE_DIR+"/observ.txt","w")
        file.write(typ+"\n")
        file.close()
        os.makedirs(OBS_DIR+"/"+typ)
        aa = os.system("FreeFem++ -v 0 "+PRE_DIR+"/riesz_observ.edp")
        if aa!=0:
            sys.exit("freefem failed to generate Riesz representations")
    duration = time.time()-start
    file = open(PRE_DIR+"/log_file.txt","w")
    file.write("Offline data generation lasted "+str(duration)+" seconds")
    file.close()
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def import_data(OFF_DIR, MAT_DIR, RIESZ_OBS_DIR):
    nb_dof  = int(np.loadtxt(OFF_DIR+"/nb_dof.txt"))
    """ matrices"""
    mass = np.matrix(np.loadtxt(MAT_DIR+"/mass.txt")).T
    stiff = np.matrix(np.loadtxt(MAT_DIR+"/stiff.txt")).T
    """ Storing all the matrices in sparse format """
    MASS = csc_matrix((np.array(mass[2,:]).reshape(max(np.shape(mass))),
                       (np.array(mass[0,:]).reshape(max(np.shape(mass))),
                        np.array(mass[1,:]).reshape(max(np.shape(mass))))),
                      shape = (nb_dof,nb_dof))
    STIFF  = csc_matrix((np.array(stiff[2,:]).reshape(max(np.shape(stiff))),
                         (np.array(stiff[0,:]).reshape(max(np.shape(stiff))),
                          np.array(stiff[1,:]).reshape(max(np.shape(stiff))))),
                      shape = (nb_dof,nb_dof))
    #number of observation functionals
    Mmax = len(os.listdir(RIESZ_OBS_DIR))#int(np.loadtxt(OFF_DIR+"/nb_observations.txt"))
    return MASS, STIFF, nb_dof, Mmax
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def build_observable_space(observation_set, RIESZ_OBS_DIR, nb_dof, SPM):
    Mmax = len(observation_set)
    observation_functionals = np.matrix(np.zeros((nb_dof, Mmax)))
    for idobs, obs_func in enumerate(observation_set):
        observation_functionals[:,idobs] = spsolve(SPM, np.matrix(np.loadtxt(RIESZ_OBS_DIR+
                                               "/riesz_"+str(obs_func)+".txt")).T)[:,None]
    return observation_functionals
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def read_solutions(DIR, train_set, nb_dof):
    TRUE_solutions = np.ndarray((len(train_set), nb_dof))
    for idj, jaux in enumerate(train_set):
        fo=open(DIR+'/sol'+str(int(jaux))+'.txt')
        TRUE_solutions[idj,:] = np.loadtxt(fo) #[0,0:nb_dof]
        fo.close()
    # print(type(TRUE_solutions[idj,:]), TRUE_solutions[idj,:])
    return TRUE_solutions
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def relative_POD(X_off, rel_ener, nbMod, SPM):
    """
    This function processes the POD for a given set of vectors X_off.
    It returns the first POD modes that represent an absolute energy of 
    value at least equal to ener
    INPUTS      Sc_prod_matrix  : Mass and stiffness matrices
                nbMod           : Maximum number of modes allowed
                ener            : Energy threshold
    OUTPUTS     Y               : Most significant modes
    """
    if (nbMod == None or nbMod>min(np.shape(X_off))):
        nbMod = min(np.shape(X_off))
    autoCorrM = X_off.T @ SPM @ X_off
    [eig_vals, sg_vecs]  = np.linalg.eigh(autoCorrM)
    # Sort singular values in descending order
    eig_vals = list(eig_vals)
    eig_vals = [(eig_value>0)*eig_value for eig_value in eig_vals]
    sg_vals = [sqrt(eig_value) for eig_value in eig_vals]
    sg_vals.reverse()
    # Calculate the first modes (POD basis vectors)
    N0 = len(sg_vals)
    sg_vecs = np.matrix(sg_vecs)
#    print "xoff :",np.shape(X_off)
    #Capture the most significant modes
    sum_sg = np.sum(sg_vals)
    print("sum_sg : ", sum_sg)
    print("sg_vals : ", sg_vals)
    jaux   = 0
    Ypod=None
    trunc_order=0
    energy = sg_vals[0]/sum_sg 
    jaux   = 1
    tmp    = X_off @ np.matrix(sg_vecs)[:,N0-1]
    Xpod   = tmp/sqrt(tmp.T @ SPM @ tmp)
    while (jaux<=(nbMod-1) and (1-energy)>=rel_ener):
        # find POD basis vector
        tmp  = X_off @ np.matrix(sg_vecs)[:,N0-1-jaux]
        # normalize  POD basis vector
        tmp = tmp/sqrt(tmp.T @ SPM @ tmp)
        Xpod = np.concatenate((Xpod,tmp), axis = 1)
        energy = energy+sg_vals[jaux]/sum_sg
        print("energy", jaux, " = ", energy)
        jaux  = jaux+1
    trunc_order = min(jaux,nbMod)
    Ypod = np.matrix(Xpod[:, 0:trunc_order])
    sg = sg_vals[0:min(jaux,nbMod)]
    return (Ypod, sg_vals, trunc_order)
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def norme(vec, SPM):
    return float(sqrt(vec.T @ SPM @ vec))
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def offline_pbdw_matrices(SPM, background_space, observation_functionals):
    """ Bibliography: PBDW Patera, Yano et al., matrix definitions page 9"""
    Bmn = background_space @ SPM @ observation_functionals #N*M
    Mmax = min(np.shape(observation_functionals))
    Nmax = min(np.shape(background_space))
    Amm = observation_functionals.T @ SPM @ observation_functionals
    GloMatrix = np.matrix(np.zeros((Mmax+Nmax, Mmax+Nmax)))
    GloMatrix[0:Mmax,0:Mmax] = Amm
    GloMatrix[Mmax:Mmax+Nmax,0:Mmax] = Bmn
    GloMatrix[0:Mmax,Mmax:Mmax+Nmax] = Bmn.T
    return Amm, Bmn, GloMatrix
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def solve_pbdw_statement(GloMatrix, secMbr):
    """ Bibliography: PBDW Patera, Yano et al., algebraic system page 9"""
    return np.linalg.solve(GloMatrix, secMbr)
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def POD_Greedy(U_0, BK_solutions, nb_par, nb_inst, SPM, dT, eps_pod):
    basis = U_0/sqrt(U_0.T @ SPM @ U_0)
    error = 2*eps_pod
    H1_norms = []
    sg = [1]
    for ind in range(nb_par):
        H1_norms.append(H1_norm(np.matrix(BK_solutions[ind,:,:]),dT, SPM))
    while (error>eps_pod):
        error=0
        dim_basis =  min(np.shape(basis))
        for iei in range(nb_par):
            Proj0 = S_tilde(BK_solutions[iei,:,:].T,basis, SPM, nb_inst,dim_basis)
            err1 = H1_norm(Proj0.T, dT, SPM)/H1_norms[iei]
            if err1>error:
                error = float(err1)
                catch=iei
                traj_to_add = Proj0
        pod, sg, _ = relative_POD(traj_to_add, eps_pod/sg[-1], 1, SPM)#np.matrix(np.eye(nb_dof)))
        print("greedy", dim_basis, "err : ", error, "ii", catch, "sg", sg[-1])
        basis = np.concatenate((basis, pod), axis = 1)
    print("The POD-greedy basis consists of ", min(np.shape(basis)), " vectors")
    return basis
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def H1_norm(TRAJ, dT, SPM):    
    nb_inst = min(np.shape(TRAJ))
    norm = 0
    for ins in range(nb_inst-1):
        norm = norm + dT[ins]*TRAJ[ins,:] @ SPM @ TRAJ[ins,:].T
    return sqrt(norm/np.sum(dT))
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def S_tilde(mathcalS, Theta, SPM, nb_inst, dim_theta):
    """ Reference PROGRESSIVE_RB, see my paper on PREIM """
    Proj_S = np.matrix(mathcalS)
    for ii in range(nb_inst):
        for jj in range(dim_theta):
            factor = mathcalS[:,ii].T @ SPM @ Theta[:,jj]/(Theta[:,jj].T @ SPM @ Theta[:,jj])
            Proj_S[:,ii] = Proj_S[:,ii] - float(factor) @ Theta[:,jj]
#        Proj_S[:,ii] = Proj_S[:,ii]/sqrt(Proj_S[:,ii].T @ Sc_pr_M @ Proj_S[:,ii])
    return Proj_S
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def build_background_space_idea2(PRE_DIR, BK_solutions, eps_rel_pod, SPM, nb_inst, dT, nb_par):
    start = time.time()
    U_0 = np.matrix(BK_solutions[0,0,:]).T
    basis = POD_Greedy(U_0, BK_solutions, nb_par, nb_inst, SPM, dT, eps_rel_pod)
    background_space = basis
    file = open(PRE_DIR+"/log_file.txt","w") 
    file.write("The POD-greedy lasted "+str(time.time()-start)+" seconds")
    file.close()
    return background_space
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def compute_inf_sup_constant(SPM, background_space, observation_functionals):
    Bmn = background_space.T @ SPM @ observation_functionals
#    Znn = background_space.T @ SPM @ background_space #=Identite en prenant SPM la matrice de rigidite
    Amm = observation_functionals.T @ SPM @ observation_functionals
    eigvals = np.linalg.eigh(Bmn @ np.linalg.inv(Amm) @ Bmn.T)[0]
    eigvals = [val*(abs(val)>1e-9) for val in eigvals]
    if min(eigvals)<0:
        print(eigvals)
    return sqrt(np.min(eigvals))
