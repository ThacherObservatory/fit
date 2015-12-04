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
import os

def plot_suite(network='bellerophon-old'):
    input_params = ['period','photnoise','rvsamples','rratio','impact']
    stellar_params = ['m1', 'm2', 'r1', 'r2', 'e']
    for i in input_params:
        for s in stellar_params:
            plot_relative_error(i, s, network, view=False,cadence='short')
            plot_relative_error(i,s,network,view=False,cadence='long')

    
def plot_relative_error(input_param, stellar_param, network='bellerophon',view=True,cadence='short'):
    """Plots input param vs relative error % of the stellar param
    input params: ['period','photnoise','rvsamples','rratio','impact']
    stellar params: ['m1', 'm2', 'r1', 'r2', 'e']"""
    
    best_params = load_bestparams(network)
    true_values = load_truevalues(network)
    initial_params = load_initialparams(network)
    
    #short cadences
    input_vals = initial_params[input_param]
    rel_err = [50*run['onesig'][stellar_param] for run in best_params[cadence]]/(true_values['m1'])

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
    meds = np.array(meds)
    yerrs = np.array(yerrs)
    plt.clf()
    plt.ioff
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

    if view:
        plt.ion()
        plt.show()
    plt.savefig(reb.get_path(network) + 'plots/' + input_param + ' vs ' + stellar_param + '-'+cadence+'.png')
    
    
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
