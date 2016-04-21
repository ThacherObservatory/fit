import emcee, george, corner
from george.kernels import ExpSine2Kernel, ExpSquaredKernel
import numpy as np
import matplotlib.pyplot as plt
import sys,pdb
import time as tm

# Read out-of-eclipse data for KIC 10935310
time,flux,err = np.loadtxt("10935310_ooe_real.txt",unpack=True)

# Pick a fraction of the total amount of data
time = time[8000:8800]
flux = flux[8000:8800]
err  = err[8000:8800]

plt.ion()

def look(theta=[0.1,1,0.01,4],plot=False,verbose=False):

    # Plot data
    if plot:
        plt.ion()
        plt.figure(1)
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(time,flux,'ko',markersize=5)
        plt.xlabel('Time (BKJD)')
        plt.ylabel('Flux (ADU)')


    # Choose starting values for kernel
    k = theta[0]**2 * ExpSquaredKernel(theta[1]) * ExpSine2Kernel(theta[2],theta[3])
    #gp = george.GP(k,mean=np.mean(flux),solver=george.HODLRSolver)
    gp = george.GP(k,mean=np.mean(flux))

    # Predict data
    gp.compute(time,yerr=err,sort=True)
    flux_fit, cov_fit = gp.predict(flux, time)

    if plot:
 #       tsamp = np.linspace(np.min(time),np.max(time),10000)
 #       fsamp,cov_samp = gp.predict(flux,tsamp)
        plt.plot(time,flux_fit,'c.')
#        plt.plot(tsamp,fsamp,'.',color='purple')

    # Plot the predicted values
    if plot:
        plt.subplot(2,1,2)
        plt.plot(time,flux-flux_fit,'ko')
        plt.axhline(y=0,linestyle='-',color='red',lw=3)
        plt.xlabel('Time (BKJD)')
        plt.ylabel('Residuals (ADU)')
        plt.savefig(str(theta) + "_plot.png")

    # Compute log likelihood and compare to interal computation
    loglike1 = 1 #np.sum((flux_fit-flux)**2/(2*err**2))# + 0.5*np.log(2*np.pi*err**2))
    loglike2 = gp.lnlikelihood(flux, quiet=True) 

    if verbose:
        print 'Why does this lnlikelihood: %.3f, ...' % loglike1
        print '... not equal this lnlikelihood: %.3f' % loglike2

    return loglike2


def compare():
    avec = np.arange(1,10,1)
    pvec = np.arange(1,10,1)
    l1 = [] ; l2 = []
    for p in pvec:
        for a in avec:
            theta = np.array([a,1,0.01,p])
            ll1, ll2 = look(theta=theta)
            l1.append(ll1)
            l2.append(ll2)
        
    l1 = np.array(l1)
    l2 = np.array(l2)

    plt.figure(2)
    plt.clf()
    plt.plot(l1 - np.median(l1))
    plt.plot(l2 - np.median(l2))
    
    plt.figure(3)
    plt.clf()
    lratio = l2/l1
    plt.plot(lratio)

    return

def grid_search_lnlike():
    """performs a grid search of lnlike space and plot"""
    
    #values to gridsearch
    t1s = np.logspace(np.log10(.001),np.log10(1),8)
    t2s = np.logspace(np.log10(30),np.log10(100),8)
    t3s = np.logspace(np.log10(.01),np.log10(1),8)
    
    #estimates for 'good' values to use for testing
    t1 = .01
    t2 = 47
    t3 = .02
    t4 = 3.99
    
    t1likes = []
    t2likes = []
    t3likes = []

    for n in t1s:
        theta = [n,t2,t3,t4]
        t1likes.append(look(theta,plot=True))
        
    for n in t2s:
        theta = [t1,n,t3,t4]
        t2likes.append(look(theta,plot=True))
        
    for n in t3s:
        theta = [t1,t2,n,t4]
        t3likes.append(look(theta,plot=True))
        
    plt.clf()
    plt.figure(1)
    plt.plot(t1s, t1likes,'-o')
    plt.xscale('log')
    plt.figure(2)
    plt.plot(t2s, t2likes,'-o')
    plt.xscale('log')
    plt.figure(3)
    plt.plot(t3s, t3likes,'-o')
    plt.xscale('log')
