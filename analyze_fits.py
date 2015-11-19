"""
Created on Mon Nov 16 21:49:05 2015

@author: douglas
"""

import ebsim as ebs
import run_ebsim as reb
import ebsim_results as ebres
import glob
import pdb
import pickle
import numpy as np
import matplotlib.pyplot as plt

def doug_test(network='doug'):
    bests = np.array(load_bestparams(network))
    trues = load_truevalues(network)
    initials = load_initialparams(network)
    nruns = len(bests)
    
    
    #standard error: (measured-real)/onesigma
    
    plt.figure()
    plt.ion()
    ival = 0 #m1
    ste_m1s = []
    i_times = []
    for nrun in range(nruns):
        ste_m1s.append((bests[nrun][ival][1] - trues[nrun][ival])/bests[nrun][ival][3])
        i_times.append(initials[nrun][1])
    plt.plot(i_times,ste_m1s, 'o')
    plt.xlabel('Integration Time')
    plt.ylabel('M1 Standard Error')
        
    
    

def load_bestparams(network='bellerophon'):
    """Loads the contents of all bestparams.txt's into an array""" 
    path = reb.get_path(network)
    files = glob.glob(path + '*/bestparams.txt')
    bests = []
    for bestparams in files:
        bests.append([float(val) for val in open(bestparams).readline().strip().replace("  "," ").split(" ")[1:]])
    #order: run#, vals[0],meds[0],modes[0],onesig[0],vals[1],meds[1],modes[1],onesig[1],vals[2],meds[2],
        #modes[2],onesig[2],vals[3],meds[3],modes[3],onesig[3],vals[4],meds[4],modes[4],onesig[4]
    #val order: (m1,m2,r1,r2,eccentricity)
    bests = np.reshape(bests, [len(bests),5,4])
    
    #output shape: runs x [m1,m2,r1,r2,e] x [val,med,mode,onesig]
    return bests

def load_truevalues(network='bellerophon'):
    """loads the contents of all ebpar.p's into an array"""
    path = reb.get_path(network)
    filenames = glob.glob(path + '*/')
    runs = []
    for name in filenames:
        runs.append(pickle.load( open( name+'ebpar.p', 'rb' ) ))
    
    trues = []
    for params in runs:
        trues.append([params['Mstar1'],params['Mstar2'],params['Rstar1'],params['Rstar2'],np.sqrt(params['ecosw']**2 + params['esinw']**2)])
    
    #shape: runs x [m1,m2,r1,r2,e]
    return trues

def load_initialparams(network='bellerophon'):
    """loads the contents of all initialparams.txt's into an array"""
    path = reb.get_path(network)
    filenames = glob.glob(path + '*/initialparams.txt')
    initials = []
    for name in filenames:
        initials.append([float(a.strip()) for a in open(name).readlines()])
    
    #shape: runs x [photometric noise, integration time (s), # of RV samples, radius ratio, impact parameter]
    return initials