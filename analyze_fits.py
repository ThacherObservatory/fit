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
import pandas as pd
import robust as rb

def doug_test(network='doug'):
    bests = np.array(load_bestparams(network))
    trues = load_truevalues(network)
    initials = load_initialparams(network)
    nruns = len(bests)
    
    
    #standard error: (measured-real)/(1/2 * onesigma)
    #confidence interval: (1/2 * onesigma)/(real value)
    
    plt.figure()
    plt.ion()
    ival = 0 #m1
    bins_dict = {}
    for nrun in range(nruns):
        ste = (bests[nrun][ival][1] - trues[nrun][ival])/(.5*bests[nrun][ival][3])
        CI = (.5*bests[nrun][ival][3])/(trues[nrun][ival]) * 100
        t = initials[nrun][1]
        if t in bins_dict.keys():
            bins_dict[t].append(CI)
        else:
            bins_dict[t] = [CI]
    for val in bins_dict.keys():
        mean = np.average(bins_dict[val])
        std = np.std(bins_dict[val])
        plt.plot([val]*len(bins_dict[val]), bins_dict[val], 'o')
        #plt.errorbar(v al, mean, yerr=std,fmt='o')
       # plt.plot()

    plt.xlabel('Integration Time')
    plt.ylabel('M1 Confidence Interval (%)')
    if network=='bellerophon':
        plt.savefig('/home/administrator/Desktop/dougtest.png')   
    
def plot_relative_error(input_param, stellar_param, network='bellerophon'):
    """Plots input param vs relative error % of the stellar param
    input params: ['period','photnoise','rvsamples','rratio','impact']
    stellar params: ['m1', 'm2', 'r1', 'r2', 'e']"""
    
    best_params = load_bestparams(network)
    true_values = load_truevalues(network)
    initial_params = load_initialparams(network)
    
    #short cadences
    input_vals = initial_params[input_param]
    rel_err = [50*run['onesig'][stellar_param] for run in best_params['short']]/(true_values['m1'])

    #bin the data
    bins_dict = {}
    for val, err in zip(input_vals, rel_err):
        if val in bins_dict.keys():
            bins_dict[val].append(err)
        else:
            bins_dict[val] = [err]  
    
    meds = []
    yerrs = []
    for val in bins_dict.keys():
        pts = bins_dict[val]
        meds.append(np.median(pts))
        yerrs.append(rb.std(np.array(pts)))
    plt.figure()
    plt.ion()
    plt.errorbar(bins_dict.keys(), meds, yerr=yerrs,fmt='o')
    plt.xlabel(input_param)
    plt.ylabel(stellar_param + ' % relative error')
    xmin = np.min(bins_dict.keys()) - np.ptp(bins_dict.keys())/5
    xmax = np.max(bins_dict.keys()) + np.ptp(bins_dict.keys())/5
    ymin = np.min(meds - yerrs) - np.ptp(meds)/5
    ymax = np.max(meds + yerrs) + np.ptp(meds)/5
    if input_param == 'photnoise':
        plt.xscale('log')
        xmin = .000005
        xmax = .015
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.show()
    plt.savefig(input_param + ' vs ' + stellar_param + '.png')
    
    
def load_bestparams(network='bellerophon'):
    """Loads the contents of all bestparams.txt's
    output shape: {short long} x runs x [Val Med Mode Onesig] x [M1 M2 R1 R2 E]""" 
    path = reb.get_path(network)
    shorts = glob.glob(path + 'short/*/bestparams.txt')
    longs = glob.glob(path + 'long/*/bestparams.txt')
    
    bests = {'short':np.ndarray(324, pd.DataFrame),'long':np.ndarray(324, pd.DataFrame)}
    for bestparams in shorts:
        run = int(bestparams.split('/')[-2])
        vals = [float(val) for val in open(bestparams).readline().strip().replace("  "," ").split(" ")[1:]]
        vals = np.reshape(vals, [5,4])
        bests['short'][run] = pd.DataFrame(vals,index=['m1', 'm2', 'r1', 'r2', 'e'], columns=['val', 'med', 'mode', 'onesig'])
        
    for bestparams in longs:
        run = int(bestparams.split('/')[-2])
        vals = [float(val) for val in open(bestparams).readline().strip().replace("  "," ").split(" ")[1:]]
        vals = np.reshape(vals, [5,4])
        bests['long'][run] = pd.DataFrame(vals,index=['m1', 'm2', 'r1', 'r2', 'e'], columns=['val', 'med', 'mode', 'onesig'])
    
    bests['short'] = [x for x in bests['short'] if x is not None]
    bests['long'] = [x for x in bests['long'] if x is not None]
    
    #output shape: {short long} x runs x [Val Med Mode Onesig] x [M1 M2 R1 R2 E] 
    return bests

def load_truevalues(network='bellerophon'):
    """loads the contents of all ebpar.p's into a sorted DataFrame
    output shape: runs x [m1,m2,r1,r2,e]"""
    path = reb.get_path(network)
    filenames = glob.glob(path + 'long/*/') #long and short trues identical
    trues = []
    runs = []
    for name in filenames:
        runs.append(int(name.split('/')[-2]))
        params = pickle.load( open( name+'ebpar.p', 'rb' ) )
        trues.append([params['Mstar1'],params['Mstar2'],params['Rstar1'],params['Rstar2'],np.sqrt(params['ecosw']**2 + params['esinw']**2)])
    
   
    #shape: runs x [m1,m2,r1,r2,e]
    return pd.DataFrame(trues, columns=['m1','m2','r1','r2','e'], index=runs).sort_index()

def load_initialparams(network='bellerophon'):
    """loads the contents of all initialparams.txt's into a sorted DataFrame
    output shape: runs x [period, photnoise, RVsamples, Rratio, impact]"""
    path = reb.get_path(network)
    filenames = glob.glob(path + 'long/*/initialparams.txt') #initialparams identical for longs and shorts
    initials = []
    runs = []
    for name in filenames:
        runs.append(int(name.split('/')[-2]))
        initials.append([float(a.strip()) for a in open(name).readlines()])
    
    #shape: runs x [period, photnoise, RVsamples, Rratio, impact]
    return pd.DataFrame(initials , columns=['period','photnoise','rvsamples','rratio','impact'], index=runs).sort_index()
