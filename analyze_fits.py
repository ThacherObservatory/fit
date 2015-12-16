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
from scipy.interpolate import interp1d as interp
from plot_params import *

def plot_suite(network='bellerophon-old'):
    input_params = ['period','photnoise','rvsamples','rratio','impact']
    stellar_params = ['m1', 'm2', 'r1', 'r2', 'e']
    for i in input_params:
        for s in stellar_params:
            plot_relative_error(i, s, network, view=False,cadence='short')
            plot_relative_error(i,s,network,view=False,cadence='long')

    
def plot_relative_error(input_param, stellar_param, network='bellerophon',view=True,cadence='short'):
    """
    Plots input param vs relative error % of the stellar param
    input params: ['period','photnoise','rvsamples','rratio','impact']
    stellar params: ['m1', 'm2', 'r1', 'r2', 'e']
    """
    
    best_params = load_bestparams(network)
    true_values = load_truevalues(network)
    initial_params = load_initialparams(network)
    
    #short cadences
    input_vals = initial_params[input_param]

    # !!! is the 'm1' index a bug? !!!
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
    """
    Loads the contents of all bestparams.txt's
    output shape: {short long} x runs x [Val Med Mode Onesig] x [M1 M2 R1 R2 E]
    """ 

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


def noise_to_mag(noise_in,debug=False):
    from scipy.interpolate import interp1d

    if noise_in < 60.0 or noise_in > 22775.486:
        return np.nan
    
    # These data were taken from Sullivan et al. (2015) figure using GraphClick
    mag = np.array([4.000, 4.703, 5.382, 5.822, 6.493, 7.050, 7.539, 8.104, 
                    8.639, 9.170, 9.823, 10.653, 11.313, 11.942, 12.419, 13.065,
                    13.669,14.182, 14.646,15.328,15.784,16.025, 16.956])

    noise = np.array([60.000, 61.527, 62.717, 65.357, 68.866, 74.845, 82.670,	 
                      95.163, 114.670, 137.554, 180.130, 264.020, 375.210,	 
                      531.251, 708.058, 1055.581, 1535.950, 2333.945, 3338.925, 
                      5901.051, 8586.475, 10238.113, 22775.486])

    func = interp1d(noise,mag,kind='cubic')

    mag_out = func(noise_in)
    
    if debug:
        plt.ion()
        plt.figure(1)
        plt.clf()
        plt.plot(mag,noise,'o')
        plt.yscale('log')
        plt.plot([mag_out],[noise_in],'rx',markersize=20)

    return mag_out
    

def get_plot_data(input_param,stellar_param,cadence='short',network='external',nograze=True,
                  highrvs=True,accuracy=False):
    """
    Loads vectors of param, relative error or absolute error, and error on the distribution of
    errors.
    
    input params: ['period','photnoise','rvsamples','rratio','impact']
    stellar params: ['m1', 'm2', 'r1', 'r2', 'e']
    """

    # Load best, true and initial values
    best_params = load_bestparams(network)
    true_values = load_truevalues(network)
    initial_params = load_initialparams(network)
    
    # Short cadences
    input_vals = initial_params[input_param]

    # Get accuracy of fitted parameter, else get relative error.
    if accuracy:
        # !!! Check this !!!
        sig = [0.5*run['onesig'][stellar_param] for run in best_params[cadence]]
        err = ([run['med'][stellar_param] for run in best_params[cadence]] - \
               true_values[stellar_param])/ sig
    else:
    # Relative error in percent (half of 1 sigma interval)
        err = [0.5*100*run['onesig'][stellar_param] for run in best_params[cadence]]/ \
              (true_values[stellar_param])

    if nograze and highrvs:
        inds, = np.where((initial_params['impact'] < 1.0) & (initial_params['rvsamples'] > 11))
        in_val = input_vals[inds]
        err_val = err[inds]
    elif nograze and not highrvs:
        inds, = np.where(initial_params['impact'] < 1.0)
        in_val = input_vals[inds]
        err_val = err[inds]
    elif not nograze and highrvs:
        inds, = np.where(initial_params['rvsamples'] > 11)
        in_val = input_vals[inds]
        err_val = err[inds]
    else:
        in_val = input_vals[range(len(input_vals))]
        err_val = err[range(len(err))]

        
    # Bin the data
    bins_dict = {}
    for val, err in zip(in_val, err_val):
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

    return bins_dict.keys(), meds, yerrs


def accuracy_plot():
    # Get the plot data
    phot1, r1med, err1 = af.get_plot_data('photnoise','r1',cadence='short',network='external',
                                           nograze=False,highrvs=False,accuracy=True)
    phot2, r2med, err2 = af.get_plot_data('photnoise','r2',cadence='short',network='external',
                                           nograze=True,highrvs=False,accuracy=True)

    plt.figure(3)
    plt.clf()
    plt.gcf().subplots_adjust(bottom=0.15,left=0.15)
    plot_params(fontsize=20)
    
    plt.errorbar(phot1,r1med,yerr=err1,fmt='o',color='black',lw=3,markersize=15,label='Primary')
    plt.errorbar(phot2,r2med,yerr=err2,fmt='o',color='darkgoldenrod',lw=3,markersize=15,label='Secondary')
    plt.xscale('log')
    plt.xlim(5e-6,2e-2)
    plt.ylim(-2,3)
    plt.xlabel('Photometric Noise (ppm)',fontsize=20)
    plt.ylabel(r'$(R - R_{true})/\sigma$',fontsize=20)

    plt.legend(loc='upper left',fontsize=18,numpoints=1)
    plt.title('Stellar Radius Accuracy vs. Photometric Noise',fontsize=18)
    plt.axhline(y=0,ls='--',lw=3,color='purple')
 
    plt.savefig('Radius_Abs_Error.png',dpi=300)


    return


def mass_plots_old(network='external',nograze=True,view=True,region=True):

    """
    Plots input param vs relative error % of the stellar param
    input params: ['period','photnoise','rvsamples','rratio','impact']
    stellar params: ['m1', 'm2', 'r1', 'r2', 'e']
    """

    # Get the plot data
    nrvs1, rvmed1, err1 = get_plot_data('rvsamples','m1',cadence='short',network='external',
                                        nograze=True)
    nrvs2, rvmed2, err2 = get_plot_data('rvsamples','m2',cadence='short',network='external',
                                        nograze=True)
    
    plt.figure(1,figsize=(8,8))
    plt.clf()
    plt.ioff()
    plot_params(fontsize=20)
    
    if region:
        e1f = interp(nrvs1,rvmed1,kind='linear')
        e1ft = interp(nrvs1,rvmed1+err1,kind='linear')
        e1fb = interp(nrvs1,rvmed1-err1,kind='linear')
        x1 = np.linspace(np.min(nrvs1),np.max(nrvs1),1000)
        et1 = e1ft(x1)
        eb1 = e1fb(x1)
        plt.plot(x1,et1,'b-',linewidth=1)
        plt.plot(x1,eb1,'b-',linewidth=1)
        plt.fill_between(x1,et1,eb1,where=et1>=eb1, facecolor='blue', interpolate=True,alpha=0.5)        
        plt.plot(x1,e1f(x1),'b-',linewidth=5)
        plt.plot(nrvs1,rvmed1,'bo',markersize=20)
        plt.xlabel('Number of RV Data Points',fontsize=20)
        plt.ylabel('Relative Error (%)',fontsize=20)
        plt.title('Stellar Mass Precision vs. RV Samples',fontsize=20)
        plt.savefig('Mass_Rel_Error.png',dpi=300)

        """
        e2f = interp(nrvs2,rvmed2,kind='linear')
        e2ft = interp(nrvs2,rvmed2+err2,kind='linear')
        e2fb = interp(nrvs2,rvmed2-err2,kind='linear')
        x2 = np.linspace(np.min(nrvs2),np.max(nrvs2),1000)
        et2 = e2ft(x2)
        eb2 = e2fb(x2)
        plt.plot(x2,et,'-',color='darkgoldenrod',linewidth=1)
        plt.plot(x2,eb,'-',color='darkgoldenrod',linewidth=1)
        plt.fill_between(x2,et2,eb2,where=et2>=eb2, facecolor='darkgoldenrod', interpolate=True,alpha=0.5)        
        plt.plot(x2,e2f(x2),'-',color='darkgoldenrod',linewidth=5)
        plt.plot(nrvs2,rvmed2,'o',color='darkgoldenrod',markersize=20)
        """

        
    else:
        pass

    return


def mass_plots(network='external',nograze=True,view=True,region=True):

    """
    Plots input param vs relative error % of the stellar param
    input params: ['period','photnoise','rvsamples','rratio','impact']
    stellar params: ['m1', 'm2', 'r1', 'r2', 'e']
    """

    # Get the plot data
    nrvs1, m1med, err1 = get_plot_data('rvsamples','m1',cadence='short',network='external',
                                           nograze=True,highrvs=False)
    nrvs2, m2med, err2 = get_plot_data('rvsamples','m2',cadence='short',network='external',
                                           nograze=True,highrvs=False)
    n1val = 130
    m1val = 0.55
    s1 = np.argsort(nrvs1)
    n1 = np.append(np.array(nrvs1)[s1],np.array(n1val),)
    m1 = np.append(np.array(m1med)[s1],np.array(m1val))
    e1 = np.append(np.array(err1)[s1],np.array(0.1))

    n2val = 130
    m2val = 0.6
    s2 = np.argsort(nrvs2)
    n2 = np.append(np.array(nrvs2)[s2],np.array(n2val),)
    m2 = np.append(np.array(m2med)[s2],np.array(m2val))
    e2 = np.append(np.array(err2)[s2],np.array(0.1))


    
    plt.figure(1)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.clf()
    plot_params(fontsize=20)

    plt.plot(n1[:-1],m1[:-1],'ko',markersize=20)

    m1f = interp(n1,m1,kind='quadratic')
    x1 = np.linspace(np.min(n1[:-1]),np.max(n1[:-1]),1000)
    m1fit = m1f(x1)
    plt.plot(x1,m1fit,'k-',linewidth=5,label='Primary')


    plt.plot(n2[:-1],m2[:-1],'o',color='darkgoldenrod',markersize=20)

    m2f = interp(n2,m2,kind='quadratic')
    x2 = np.linspace(np.min(n2[:-1]),np.max(n2[:-1]),1000)
    m2fit = m2f(x2)
    plt.plot(x2,m2fit,'-',color='darkgoldenrod',linewidth=5,label='Secondary')

    plt.xlabel('Number of RV Data Points',fontsize=20)
    plt.ylabel('Relative Error (%)',fontsize=20)
    plt.title('Stellar Mass Precision vs. RV Samples',fontsize=20)
    plt.legend(loc='upper right',fontsize=18)
    plt.axhline(y=1,ls='--',lw=3,color='purple')
    
    plt.savefig('Mass_Rel_Error.png',dpi=300)


    return


def radius_plots(network='external',nograze=True,view=True,region=True,cadence='short'):

    """
    Plots input param vs relative error % of the stellar param
    input params: ['period','photnoise','rvsamples','rratio','impact']
    stellar params: ['m1', 'm2', 'r1', 'r2', 'e']
    """

    # Get the plot data
    phot1, r1med, err1 = get_plot_data('photnoise','r1',cadence=cadence,network='external',
                                       nograze=True,highrvs=True)
    phot2, r2med, err2 = get_plot_data('photnoise','r2',cadence=cadence,network='external',
                                       nograze=True,highrvs=True)
    p1val = 1e-6
    r1val = 0.5
    s1 = np.argsort(phot1)
    p1 = np.append(np.array(p1val),np.array(phot1)[s1])
    r1 = np.append(np.array(r1val),np.array(r1med)[s1])
    e1 = np.append(np.array(0.1),np.array(err1)[s1])

    p2val = 1e-6
    r2val = 0.8
    s2 = np.argsort(phot2)
    p2 = np.append(np.array(p2val),np.array(phot2)[s2])
    r2 = np.append(np.array(r2val),np.array(r2med)[s2])
    e2 = np.append(np.array(0.1),np.array(err2)[s2])

    
    plt.figure(2)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.clf()
    plot_params(fontsize=20)

    logphot2 = np.log10(p2)

    plt.plot(p2[1:]*1e6,r2[1:],'o',color='darkgoldenrod',markersize=20)
    plt.xscale('log')

    r2f = interp(logphot2,r2,kind='cubic')
    x2 = np.linspace(np.min(logphot2[1:]),np.max(logphot2[1:]),1000)
    r2fit = r2f(x2)
    plt.plot(10**x2*1e6,r2fit,'-',color='darkgoldenrod',linewidth=5)

    logphot1 = np.log10(p1)

    plt.plot(p1[1:]*1e6,r1[1:],'ko',markersize=20)
    plt.xscale('log')

    r1f = interp(logphot1,r1,kind='cubic')
    x1 = np.linspace(np.min(logphot1[1:]),np.max(logphot1[1:]),1000)
    r1fit = r1f(x1)
    plt.plot(10**x1*1e6,r1fit,'k-',linewidth=5,label='Primary')
    plt.plot([10],[0],'-',color='darkgoldenrod',linewidth=5,label='Secondary')
    
    
    plt.xlabel('Photometric Noise (ppm)',fontsize=20)
    plt.ylabel('Relative Error (%)',fontsize=20)

    plt.legend(loc='upper left',fontsize=18)
    plt.title('Stellar Radius Precision vs. Photometric Noise',fontsize=18)
    plt.axhline(y=1,ls='--',lw=3,color='purple')
 
    plt.savefig('Radius_Rel_Error_'+cadence+'.png',dpi=300)


    return
