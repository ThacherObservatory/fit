# TO DO:
# Allow option to fit for limb darkening

import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import constants as c
import pdb
import scipy as sp
import time
import glob
import re
import get_limb_coeff as glc
import occultquad as oq
import transit as trans
from scipy.ndimage.filters import uniform_filter
import robust as rb
from scipy.optimize import fmin_l_bfgs_b as bfgs
import emcee
import os
from scipy.stats.kde import gaussian_kde
#from scipy.integrate import quad as integrate
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
import time


#----------------------------------------------------------------------
# prep_data:
#    get lightcurve
#    set global variables
#----------------------------------------------------------------------

def input_data(dir='/Users/jonswift/Astronomy/Caltech/MINERVA/Observing/Photometry/reductions/2014Sep17/',
               limbmodel='quad',target='Target',starprops=[1.0,1.0,5800.],filter="r'",
               tranprops=[0.1,1.0/24,0.0],inperiod=1.0,inephem=2454833.0,rprior=False,
               intime=None,influx=None,inerr=None,inexp=None,inttm=None):

    global c1
    global c2
    global t
    global ttm
    global flux
    global e_flux
    global exptime
    global path
    global name
    global params0
    global sdata
    global bjd
    global limb
    global rhostar0
    global ndim
    global period
    global jeffries
    global filt

    if inttm == None or intime == None or influx == None or inerr == None or inexp == None:
        print "Must supply input time, ttm, flux, error and exptime!"
        return

    t  = intime
    ttm = inttm
    flux = influx
    e_flux = inerr
    exptime = inexp
    filt = filter

    jeffries = rprior

    path = dir

    name = target

    period = inperiod

    ephem0 = inephem

    ndim = 4

# Star properties
    Mstar = starprops[0]*c.Msun
    Rstar = starprops[1]*c.Rsun
    Tstar = starprops[2]
    
    loggstar = np.log10( c.G * Mstar / Rstar**2. )

    rhostar0 = Mstar/((4./3.) * np.pi * Rstar**3)
    
    print "... getting LD coefficients for "+limbmodel+" model"
    a,b = glc.get_limb_coeff(Tstar,loggstar,limb=limbmodel,filter=filt)

 
    if limbmodel == 'quad':
        ltag = '_quad'
        c1 = a
        c2 = b
    elif limbmodel == 'sqrt':
        ltag = '_sqrt'
        c1 = b
        c2 = a
    else:
        print "Limb darkening law not recognized"
        return

    print  "Limb darkening coefficients:"
    aout = '     c1 = {0:.4f}'
    print aout.format(c1)
    bout = '     c2 = {0:.4f}'
    print bout.format(c2)
    print " "

    limb = limbmodel

    
# Intial guesses for MCMC 
    rprs0   = tranprops[0]
    dur0    = tranprops[1]
    impact0 = tranprops[2]
    
    params0 = [rprs0,dur0,impact0,ephem0]

    return 


#----------------------------------------------------------------------
# prep_data:
#    get lightcurve
#    set global variables
#----------------------------------------------------------------------

def prep_data(dir='/Users/jonswift/Astronomy/Caltech/MINERVA/Observing/Photometry/reductions/2014Feb16/',
              limbmodel='quad',target='Target',starprops=[1.0,1.0,5800.],
              tranprops=[0.1,1.0/24,0.0],inperiod=1.0,rprior=False):

    global c1
    global c2
    global t
    global ttm
    global flux
    global e_flux
    global exptime
    global path
    global name
    global params0
    global sdata
    global bjd
    global limb
    global rhostar0
    global ndim
    global period
    global jeffries

    jeffries = rprior

    path = dir

    name = target

    period = inperiod

    ndim = 4

# Star properties
    Mstar = starprops[0]*c.Msun
    Rstar = starprops[1]*c.Rsun
    Tstar = starprops[2]
    
    loggstar = np.log10( c.G * Mstar / Rstar**2. )

    rhostar0 = Mstar/((4./3.) * np.pi * Rstar**3)
    
    print "... getting LD coefficients for "+limbmodel+" model"
    a,b = glc.get_limb_coeff(Tstar,loggstar,limb=limbmodel,filter=filt)

 
    if limbmodel == 'quad':
        ltag = '_quad'
        c1 = a
        c2 = b
    elif limbmodel == 'sqrt':
        ltag = '_sqrt'
        c1 = b
        c2 = a
    else:
        print "Limb darkening law not recognized"
        return

    print  "Limb darkening coefficients:"
    aout = '     c1 = {0:.4f}'
    print aout.format(c1)
    bout = '     c2 = {0:.4f}'
    print bout.format(c2)
    print " "

    limb = limbmodel

# Load in data
    file = glob.glob(dir+'lightcurve.txt')
    if len(file) == 0:
        print "No data file present in "+dir
        return
    data = np.loadtxt(file[0])
    t = data[:,0]
    flux = data[:,1]
    e_flux = data[:,2]
    exptime = data[:,3]/(24.0*3600.0)

    
# Intial guesses for MCMC 
    rprs0   = tranprops[0]
    dur0    = tranprops[1]
    impact0 = tranprops[2]
    ephem0  = np.median(t)
    ttm     = t-ephem0
    
    params0 = [rprs0,dur0,impact0,ephem0]

    return t,ttm,flux,e_flux,exptime


#------------------------------------------------------------------------
# get_limb_curves:
#------------------------------------------------------------------------

def get_limb_curve(ldc,limbmodel='quad'):

    gamma = np.linspace(0,np.pi/2.0,1000,endpoint=True)
    theta = gamma*180.0/np.pi
    mu = np.cos(gamma)
 
    if limbmodel == 'nlin':
        c1 = ldc[0]
        c2 = ldc[1]
        c3 = ldc[2]
        c4 = ldc[3]
        Imu = 1.0 - c1*(1.0 - mu**0.5) - c2*(1.0 - mu) - \
            c3*(1.0 - mu**1.5) - c4*(1.0 - mu**2.0)
    elif limbmodel == 'quad':
        c1 = ldc[0]
        c2 = ldc[1]
        Imu = 1.0 - c1*(1.0 - mu) - c2*(1.0 - mu)**2.0
    elif limbmodel == 'sqrt':
        c1 = ldc[0]
        c2 = ldc[1]
        Imu = 1.0 - c1*(1.0 - mu) - c2*(1.0 - mu**0.5)
    else: pass

    return theta, Imu


#------------------------------------------------------------------------
# plot_limb_curves:
#------------------------------------------------------------------------

def plot_limb_curves(write=False):

    Mstar = sdata[0,0]
    Rstar = sdata[0,1]
    Tstar = sdata[0,2]

    loggstar = np.log10( c.G * Mstar / Rstar**2. )
    
    a1,a2,a3,a4 = glc.get_limb_coeff(Tstar,loggstar,limb='nlin',interp='linear',filter=filt)
    a,b = glc.get_limb_coeff(Tstar,loggstar,limb='quad',interp='linear',filter=filt)
    c,d = glc.get_limb_coeff(Tstar,loggstar,limb='sqrt',interp='linear',filter=filt)

    thetaq,Imuq = get_limb_curve([a,b],limbmodel='quad',filter=filt)
    thetas,Imus = get_limb_curve([c,d],limbmodel='sqrt',filter=filt)
    thetan,Imun = get_limb_curve([a1,a2,a3,a4],limbmodel='nlin',filter=filt)

    if write:
        plt.figure(10,figsize=(11,8.5),dpi=300)
    else:
        plt.ion()
        plt.figure()
    plt.plot(thetaq,Imuq,label='Quadratic LD Law')
    plt.plot(thetas,Imus,label='Root-Square LD Law')
    plt.plot(thetan,Imun,label='Non-Linear LD Law')

    plt.ylim([0,1.1])
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel(r"$\theta$ (degrees)",fontsize=18)
    plt.ylabel(r"$I(\theta)/I(0)$",fontsize=18)
    plt.title("KOI-"+str(koi)+" limb darkening",fontsize=20)
    plt.legend(loc=6)

    plt.annotate(r'$T_{\rm eff}$ = %.0f K' % sdata[0][2], [0.86,0.82],horizontalalignment='right',
                 xycoords='figure fraction',fontsize='large')
    plt.annotate(r'$\log(g)$ = %.2f' % loggstar, [0.86,0.77],horizontalalignment='right',
                 xycoords='figure fraction',fontsize='large')
 
    if write:
        plt.savefig(path+'MCMC/'+str(koi)+'_LD.png')
        plt.clf()
    else:
        plt.ioff()

    return


#----------------------------------------------------------------------
# foldtime:
#    function to fold time based on period and ephemeris
#----------------------------------------------------------------------

def foldtime(time,period=1.0,t0=0.0):

# Number of transits before t0 in data
    npstart = np.round((t0 - np.min(time))/period)

# Time of first transit in data
    TT0 = t0 - npstart*period

# Let cycle start 1/2 period before first transit
    tcycle0 = TT0 - period/2.0

# tcycle = 0 at 1/2 period before first transit
    tcycle = (time - tcycle0) % period

# Time to mid transit is 1/2 period from this starting point
    tfold  = (tcycle - period/2.0)

    return tfold


#----------------------------------------------------------------------
# bin_lc:
#    bin data 
#----------------------------------------------------------------------


def bin_lc(x,y,nbins=100):
    n, bin_edges = np.histogram(x, bins=nbins)
    sy, bin_edges = np.histogram(x, bins=nbins, weights=y)
    sy2, bin_edges = np.histogram(x, bins=nbins, weights=y*y)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)

    binvals = (bin_edges[1:] + bin_edges[:-1])/2
    yvals = mean
    yerr = std/np.sqrt(len(std))
    
    return binvals,yvals,yerr




def midtotot(dmid,rprs,impact):
    """
    ----------------------------------------------------------------------
    midtotot:
    --------------
    Function to convert midingress/egress transit duration to total and 
    full duration given a planet size and impact parameter

    TODO:
    -----
    Need to catch for high impact parameter solutions
    """
#    if impact**2 > (1.0-rl)**2:
    xtot  = np.sqrt((1+rprs)**2 - impact**2)
    arg = (1-rprs)**2 - impact**2
    inds, = np.where(arg < 0)
    if len(inds) > 0:
        print str(len(inds))+" grazing transits!"
        arg[inds] = 0.0
        xfull = np.sqrt(arg)
    xmid = (xtot+xfull)/2.

    dtot  = dmid*xtot/xmid
    dfull = dmid*xfull/xmid 
    
    return dtot,dfull


#----------------------------------------------------------------------
# compute_trans:
#    function to compute transit curve for given transit parameters
#----------------------------------------------------------------------

def compute_trans(rprs,aors,cosi,t0,unsmooth=False):


    # "Lens" radius = Planet radius/Star radius
    rl = rprs

    # Sampling of transit model will be at intervals equal to the 
    # integration time divided by modelfac
    modelfac = 21.0

    intmin = np.min(exptime)*24.*3600.
    intmax = np.max(exptime)*24.*3600.

    impact = np.abs(cosi)*aors

    sini = np.sqrt(1.0 - cosi**2)
    arg = ((1+rprs)**2 - impact**2) / (aors * sini)
    if arg > 1:
        print "planet is inside star!!"
        duration = period/2.0
        sztrans = 10
        modelt = sp.linspace(-1*period/2,period/2,sztrans*2)
        smoothmodel = np.ones(sztrans*2)
        return modelt+t0,smoothmodel
    elif arg < 0:
        duration = 0.0
        sztrans = 10
        modelt = sp.linspace(-1*period/2,period/2,sztrans*2)
        smoothmodel = np.ones(sztrans*2)
        return modelt+t0,smoothmodel
    else:
        duration = period/np.pi * np.arcsin(arg)

    # Number of integrations in 1/2 full duration (use minimum integration time)
    nint = duration*24.*3600./(2.0*intmin)

    # Number of transit samples = integer number of integrations * modelfac
    # extra 1 to account for one step = 2 values
    sztrans = np.ceil(nint)*modelfac + 1
    
    # Factor beyond full duration that an integer number of integrations extends
    ifac = np.ceil(nint)/nint

    # Linear time for sztrans samplings of transit curve    
    time0 = np.linspace(0,duration/2.*ifac,sztrans)
#    dt = time0[1]-time0[0]
    dt = intmin/(modelfac*24.0*3600.0)

    # Compute impact parameter for linear time intervals assuming circular orbit
    theta = 2.0*np.pi*time0/period
    h = aors*np.sin(theta)
    v = np.cos(theta)*impact
    b0 = np.sqrt(h**2 + v**2)

    if limb == 'quad':
        model = np.array(oq.occultquad(b0,c1,c2,rl))[0]
    elif limb == 'sqrt':
        model = np.array(trans.occultnonlin(b0,rl,np.array([c1,c2,0.0,0.0])))
    else:
        print "Limb darkening law not recognized"
        return
 

# Append additional ones 2 integration times outside of transit with same sample 
# rate as the rest of the transit (for smoothing)
    nadd = 2.0*modelfac
    addtime = np.linspace(dt,nadd*dt,nadd)+np.max(time0)  
    time0 = np.append(time0,addtime)
    model = np.append(model,np.ones(nadd))
    b0    = np.append(b0,np.zeros(nadd)+999)

# Append 1's far from transit (1/2 period away for linear interpolation
    time0 = np.append(time0,period/2)
    model = np.append(model,1.0)
    b0    = np.append(b0,999)

# Final model time and flux (flip and stitch)
    tmodel = np.append(-1*time0[::-1][:-1],time0)
    fmodel  = np.append(model[::-1][:-1],model)
    bmodel = np.append(b0[::-1][:-1],b0)

# Return unsmoothed model if requested
    if unsmooth:
        return tmodel+t0,fmodel

# Smooth to integration time
    sl = modelfac
    smoothmodel = uniform_filter(fmodel, np.int(sl))

#    if all:
#        return tmodel+t0,bmodel,fmodel,smoothmodel

    return tmodel+t0,smoothmodel



#----------------------------------------------------------------------
# lnprob:
#    function to compute logarithmic probability of data given model
#----------------------------------------------------------------------

def lnprob(x):

### Input parameters ###
    rprs     = x[0]
    aors     = 10.0**x[1]
    cosi     = x[2]
    t0       = x[3]

### Impose non-physical prior directly in  likelihood ###
#    if np.abs(per - pdata[0,4])/pdata[0,4] > 1e-4:
#        print "period out of range"
#        pdb.set_trace()
#        return -np.inf


#    if duration < 1e-4:
#        print "duration out of range"
#        pdb.set_trace()
#        return -np.inf

    if np.abs(t0) > (np.max(ttm)-np.min(ttm))/2.0:
#        print "t0 out of range"
#        pdb.set_trace()
        return -np.inf
        
    if rprs < 0 or rprs > 1:
#        print "rprs out of range"
#        pdb.set_trace()
        return -np.inf

#    if impact > (1 + rprs) or impact < 0:
#        print "impact parameter out of range"
#        pdb.set_trace()
#        return -np.inf

    if cosi > 1 or cosi < -1:
        return -np.inf

    if aors < 1:
        return -np.inf


    intmax = np.max(exptime)
    intmin = np.min(exptime)

### Compute transit model for given input parameters ###

    tmodel,model = compute_trans(rprs,aors,cosi,t0,unsmooth=True)

    # compare data to model in transit region
    outs  = np.ones(len(ttm),dtype=bool)

# t0 moves computed transit centroid by + t0
# compare model within one integration time outside of transit
    ininds = np.where((ttm >= tmodel[1]) & (ttm <= tmodel[-2]))
    outs[ininds] = False

# Bin data around integrations 
    ledge = ttm-exptime/2.0
    redge = ttm+exptime/2.0

    bins = np.append(ledge,redge)
    bins = bins[np.argsort(bins)]

    npts,bin = np.histogram(tmodel,bins=bins)
    sum,bin = np.histogram(tmodel,bins=bins,weights=model)

    tbin = (bin[1:] + bin[:-1])/2
    inds = np.arange(len(ttm))*2
 
    tbin = tbin[inds]
    sumbin = sum[inds]
    nptsbin = npts[inds]

    nz = nptsbin > 0
    mbin = np.ones(len(tbin))
    mbin[nz] = sumbin[nz]/nptsbin[nz]   
    mbin[outs] = 1.0

    lfi = -1*(mbin - flux)**2/(2.0*e_flux**2)
    lf = np.sum(lfi)

    if jeffries == True:
        lf = lf - 2.0*np.log(rprs)

    return lf


#----------------------------------------------------------------------
# transit_func
def transit_func(x, *args):
# outdated!!!

### Input parameters ###
    rprs     = x[0]
    duration = x[1]
    impact   = x[2]
    t0       = x[3]

    
    ttm = args[0]
    flux = args[1]
    e_flux = args[2]
    exptime = args[3]


    intmax = np.max(exptime)
    intmin = np.min(exptime)

### Compute transit model for given input parameters ###
    # t0 is time of mid trans (relative to nominal ephemeris)
    tmodel,model = compute_trans(rprs,duration,impact,t0,unsmooth=True)

    # compare data to model in transit region
    outs  = np.ones(len(ttm),dtype=bool)

# t0 moves computed transit centroid by + t0
# compare model within one integration time outside of transit
    ininds = np.where((ttm > (-0.5*duration - intmax/(24.*3600) + t0)) & 
                      (ttm < 0.5*duration + intmax/(24.*3600) + t0))
    outs[ininds] = False

# Bin data around integrations 
    ledge = ttm-exptime/2.0
    redge = ttm+exptime/2.0

    bins = np.append(ledge,redge)
    bins = bins[np.argsort(bins)]

    npts,bin = np.histogram(tmodel,bins=bins)
    sum,bin = np.histogram(tmodel,bins=bins,weights=model)

    tbin = (bin[1:] + bin[:-1])/2
    inds = np.arange(len(ttm))*2

    tbin = tbin[inds]
    sumbin = sum[inds]
    nptsbin = npts[inds]

    nz = nptsbin > 0
    mbin = np.ones(len(tbin))

    mbin[nz] = sumbin[nz]/nptsbin[nz]
    
    mbin[outs] = 1.0

    residuals = (mbin-flux)/e_flux
    return np.sum(residuals**2)


#----------------------------------------------------------------------
# residuals:
#    get residuals given moel lightcurve and mpfit transit parameters for koi
#    set global variables
#----------------------------------------------------------------------

def residuals(inp):
    import numpy as np
    import scipy as sp
    import pdb

    rprs = inp[0]
    aors = inp[1]
    cosi = inp[2]
    t0 = inp[3]

 
    intmax = np.max(exptime)
    intmin = np.min(exptime)

    tmodel,model = compute_trans(rprs,aors,cosi,0.0)

    # compare data to model in transit region
    outs  = np.ones(len(ttm),dtype=bool)

# t0 moves computed transit centroid by + t0
# compare model within one integration time outside of transit
    ininds, = np.where((ttm >= tmodel[1]) & (ttm <= tmodel[-2]))
    outs[ininds] = False

# Bin data around integrations 
    ledge = ttm-exptime/2.0
    redge = ttm+exptime/2.0

    bins = np.append(ledge,redge)
    bins = bins[np.argsort(bins)]

    npts,bin = np.histogram(tmodel,bins=bins)
    sum,bin = np.histogram(tmodel,bins=bins,weights=model)

    tbin = (bin[1:] + bin[:-1])/2
    inds = np.arange(len(ttm))*2

 
    tbin = tbin[inds]
    sumbin = sum[inds]
    nptsbin = npts[inds]

    nz = nptsbin > 0
    mbin = np.ones(len(tbin))
    mbin[nz] = sumbin[nz]/nptsbin[nz]   
    mbin[outs] = 1.0

    resid = flux - mbin
        
    return resid


#----------------------------------------------------------------------
# plot_model:
#    plot transit model given model params
#----------------------------------------------------------------------

def plot_model(modelparams,bestparams=None,tag='',markersize=5,nbins=100):
    from matplotlib import gridspec
    import matplotlib as mpl
    import robust as rb
    import os

    if bestparams == None:
        bestparams = modelparams

    mpl.rc('axes', linewidth=2)
    fs = 15
    lw = 2
    ms = 8

    directory = path

    tplt = ttm
    fplt = flux
    eplt = e_flux
    plt.figure(1,figsize=(11,8.5),dpi=300)
    gs = gridspec.GridSpec(3, 1,wspace=0.05,hspace=0.05)
    ax1 = plt.subplot(gs[0:2,0])
#    ax1.plot(tplt,fplt,'bo',markersize=ms,linewidth=lw)
    ax1.errorbar(tplt,fplt,yerr=eplt,fmt='o',color='k',markersize=ms,linewidth=lw)
    wid1 = np.max(tplt)
    wid2 = abs(np.min(tplt))
    if wid1 > wid2:
        wid = wid1
    else:
        wid = wid2

    ax1.set_xlim((-1*wid,wid))

    tmodel,model = compute_trans(modelparams[0],modelparams[1],modelparams[2],0.0,unsmooth=False)
    
    sig = rb.std(fplt)
    med = np.median(fplt)
    min = np.min(model)
    yrange = np.array([min-3*sig,med+7*sig])
    ymin = min-2.5*sig
    ymax = med+4*sig
    ax1.set_ylim(ymin,ymax)
#    decimal = "%i" % np.round(np.log10(10./(yrange[1]-yrange[0])))
#    locs,labels = plt.yticks()
#    plt.yticks(locs, map(lambda x: "%.3f" % x, locs))
#    plt.ylabel("Relative Flux")

#    ytickvals = np.linspace(yrange[0],yrange[1],10)
    ax1.yaxis.set_tick_params(length=5, width=lw, labelsize=fs)
#    ax1.xaxis.set_tick_params(length=5, width=lw, labelsize=fs)
    ax1.xaxis.set_tick_params(length=10, width=lw, labelsize=fs)
    ax1.set_xticklabels(())
    ax1.plot(tmodel,model,color='r',linewidth=lw)
    ax1.set_title(name+" Transit Fit",fontsize=fs)
    ax1.set_ylabel("Relative Flux",fontsize=fs)

    res = residuals(modelparams)
    
    dof = len(res) - ndim - 1.0
    chisq = np.sum((res/eplt)**2)/dof

#    aors    = 2.0 * np.sqrt(modelparams[0]) * period / \
#        (np.pi*np.sqrt(modelparams[1]**2 - tfull**2))
#    
#    rhostar =  3.0*np.pi/( c.G * (period*24.*3600.)**2 ) * aors**3
    

#    plt.annotate(r'$P$ = %.7f d' % modelparams[4], [0.5,0.87],horizontalalignment='center',
#                 xycoords='figure fraction',fontsize='large')

#    plt.annotate(r'$\rho_*$ (e=0) = %.3f' % rhostar, [0.87,0.81],horizontalalignment='right',
#                 xycoords='figure fraction',fontsize='large')
#    plt.annotate(r'$\rho_*$ (orig) = %.3f' % rhostar0, [0.87,0.77],horizontalalignment='right',
#                 xycoords='figure fraction',fontsize='large')
#    val = (modelparams[4]-pdata[0,4])*24.*3600.
#    plt.annotate(r'$\Delta P$ = %.3f s' % val, [0.15,0.85],
#                 xycoords='figure fraction',fontsize='large')
    ax1.annotate(r'$R_p/R_*$ = %.5f' % bestparams[0], [0.15,0.85],
                  xycoords='figure fraction',fontsize='large')   
    ax1.annotate('b = %.2f' % bestparams[2], [0.15,0.81],
                  xycoords='figure fraction',fontsize='large')


    plt.annotate(r'$\chi^2_r$ = %.5f' % chisq, [0.87,0.85],horizontalalignment='right',
                 xycoords='figure fraction',fontsize='large')

    val =  bestparams[1]*24.
    ax1.annotate(r'$\tau$ = %.4f h' % val, [0.87,0.81],horizontalalignment='right',
                  xycoords='figure fraction',fontsize='large')

#    t0out = (modelparams[3])*24.*60.*60.
    t0out = bestparams[3] + params0[3]
    ax1.annotate(r'$t_0$ = %.6f JD' % t0out, [0.87,0.77],horizontalalignment='right',
                  xycoords='figure fraction',fontsize='large')
    
    ax2 = plt.subplot(gs[2,0])
    s = np.argsort(tplt)
#    ax2.plot(tplt[s],res,'bo',markersize=ms,linewidth=lw)
    ax2.errorbar(tplt[s],res,yerr=eplt,color='k',fmt='o',markersize=ms,linewidth=lw)
    ax2.set_xlim((-1*wid,wid))
    sig = rb.std(res)
    med = np.median(res)
    ax2.set_ylim((-7*sig,7*sig))
    ax2.axhline(y=0,color='r',linewidth=lw)
    ax2.set_xlabel("Time from Mid Transit (days)",fontsize=fs)
    ax2.set_ylabel("Residuals",fontsize=fs)
    sval = sig*1e6
#    ax2.annotate(r'$\sigma$ = %.0f ppm' % sval , [0.87,0.40],horizontalalignment='right',
#                  xycoords='figure fraction',fontsize='large')
#    ax2.yaxis.set_tick_params(length=10, width=lw, labelsize=fs)
    ax2.yaxis.set_tick_params(length=5, width=lw, labelsize=fs)
    ax2.xaxis.set_tick_params(length=10, width=lw, labelsize=fs)
    plt.savefig(path+''.join(name.split())+tag+'.png')
    plt.clf()

    mpl.rc('axes', linewidth=1)

    return chisq



def quick_fit():
    initial_values = params0
    initial_values[3] = 0.0
    mybounds = [(0,1), (0,None), (0,1), (None,None)]
    
    x,f,d = bfgs(transit_func, x0=initial_values, args=(ttm,flux,e_flux,exptime), 
                 bounds=mybounds, approx_grad=True)

    return x,f,d



#----------------------------------------------------------------------
# MCMC_fit:
#    fit a single transit signal with specified mcmc parameters
#    return chains
#----------------------------------------------------------------------

def MCMC_fit(nwalkers=250,burnsteps=1000,mcmcsteps=1000,clobber=False,tag=''):

    global nw
    global bs
    global mcs

    nw = nwalkers
    bs = burnsteps
    mcs = mcmcsteps

    directory = path

# Do not redo MCMC unless clobber flag
    done = os.path.exists(path+'RpRs_chain.txt')
    if done and not clobber:
        return False,False

    print "Starting MCMC fitting routine"

# Set up MCMC sampler
    print "... initializing emcee sampler"
    tstart = time.time()
    sampler = emcee.EnsembleSampler(nw, ndim, lnprob)

    twomin = 2./(24.*60.)
    onesec = 1./(24.*60.*60.)

# Initial chain values

#Rp/Rs
    p0_1 = params0[0] + np.random.normal(0,0.1*params0[0],nw)
#    p0_2 = params0[1] + np.random.normal(0,0.1*params0[1], nw)
#    p0_3 = np.random.uniform(0., 0.95, nw)

# log(a/Rs)
    aors0 = np.sqrt( ((1+params0[0])**2 - params0[2]**2)/
                        ((np.sin(np.pi*params0[1]/period))**2) + params0[2]**2)
    p0_2 = np.random.uniform(0.99,1.01,nw) * np.log10(aors0)

# cosi
    p0_3 = np.random.uniform(0, 0.95, nw)/aors0

# t0
    p0_4 = np.random.uniform(-0.5*twomin, 0.5*twomin, nw)

    p0 = np.array([p0_1,p0_2,p0_3,p0_4]).T

# Run burn-in
    print "... running burn-in with "+str(bs)+" steps and "+str(nw)+" walkers"
    pos, prob, state = sampler.run_mcmc(p0, bs)
    done_in(tstart)

    variables =["Rp/R*","log(a/R*)","cos(i)","t0"]

# Calculate G-R scale factor for each variable
    Rs = GR_test(sampler.flatchain,nw,bs,variables=variables)

    print ""
    for var in np.arange(ndim):
        acout = "Autocorrelation time for "+variables[var]+" = {0:0.3f}"
        print acout.format(sampler.acor[var])

    print ""
    afout = "Mean acceptance fraction: {0:0.3f}"
    print afout.format(np.mean(sampler.acceptance_fraction))


# Reset sampler and run MCMC for reals
    print "... resetting sampler and running MCMC with "+str(mcs)+" steps"
    sampler.reset()
    posf, probf, statef = sampler.run_mcmc(pos, mcs)
    done_in(tstart)

    print " "
    afout = "Final mean acceptance fraction: {0:0.3f}"
    print afout.format(np.mean(sampler.acceptance_fraction))

 # Calculate G-R scale factor for each variable
    Rf = GR_test(sampler.flatchain,nw,mcs,variables=variables)

    for var in np.arange(ndim):
        acout = "Autocorrelation time for "+variables[var]+" = {0:0.3f}"
        print acout.format(sampler.acor[var])

    print " "
    print "Writing out autocorrelation times"
    np.savetxt(path+'acor_times'+tag+'.txt',sampler.acor)

    print " "
    print "Writing out final Gelman Rubin scale factors"
    np.savetxt(path+'GR_factors'+tag+'.txt',Rf)

    print "Writing MCMC chains to disk"
    rdist = sampler.flatchain[:,0]
    laorsdist = sampler.flatchain[:,1]
    cosidist = sampler.flatchain[:,2]
    tdist = sampler.flatchain[:,3]

    np.savetxt(path+'RpRs_chain'+tag+'.txt',rdist)
    np.savetxt(path+'logaors_chain'+tag+'.txt',laorsdist)
    np.savetxt(path+'cosi_chain'+tag+'.txt',cosidist)

    aorsdist = 10.0**laorsdist
    sinidist = np.sqrt(1.0 - cosidist**2)
    bdist    = aorsdist*np.abs(cosidist)
    ddist    = period/np.pi * np.arcsin( np.sqrt((1+rdist)**2 - bdist**2) / (aorsdist * sinidist))

    np.savetxt(path+'Tdur_chain'+tag+'.txt',ddist)
    np.savetxt(path+'b_chain'+tag+'.txt',bdist)
    np.savetxt(path+'T0_chain'+tag+'.txt',tdist)
    lp = sampler.lnprobability.flatten()
    np.savetxt(path+'lnProb'+tag+'.txt',lp)

    chains = sampler.flatchain

    return chains,lp


#----------------------------------------------------------------------
# GR_test:
#    Compute the Gelman-Rubin scale factor for each variable given input
#    flat chains
#----------------------------------------------------------------------


def GR_test(chains,nwalkers,nsamples,variables=False):

    nels = nsamples/2.
    Rs = np.zeros(ndim)
    for var in np.arange(ndim):
        distvec = chains[:,var].reshape(nwalkers,nsamples)
        
        mean_total = np.mean(distvec[:,nsamples/2:])
        means = np.zeros(nwalkers)
        vars = np.zeros(nwalkers)
        for i in np.arange(nwalkers):
            means[i] = np.mean(distvec[i,nsamples/2:])
            vars[i] = 1./(nels-1) * np.sum((distvec[i,nsamples/2.:] - means[i])**2)

        B = nels/(nwalkers-1) * np.sum((means - mean_total)**2)
        W = 1./nwalkers * np.sum(vars)

        V_hat = (nels-1.0)/nels * W + B/nels

        R =  np.sqrt(V_hat/W)

        if len(variables) == ndim:
            out = "Gelman Rubin scale factor for "+variables[var]+" = {0:0.3f}"
            print out.format(R)
            
        Rs[var] = R

    return Rs
    

#----------------------------------------------------------------------
# bestvals:
#    find the best values from the 1-d posterior pdfs
#    return best values and the posterior pdf for rp/rs
#----------------------------------------------------------------------

def bestvals(chains=False,lp=False,frac=0.001,nbins=100,sigfac=5,bindiv=30.0,tag=''):


#    Rstar = sdata[0,1]/c.Rsun
#    e_Rstar = sdata[1,1]/c.Rsun

## Define file tags if not already    
#    if short:
#        stag = '_short'
#    else:
#        stag = '_long'
#
#    if binned:
#        btag = '_bin'
#    else:
#        btag = ''


# Use supplied chains or read from disk
    if chains is not False:
        rprsdist = chains[:,0]
        ddist = chains[:,1]*24.
        bdist = chains[:,2]
        tdist = chains[:,3]*24.*60.*60.
    else:
        print '... importing MCMC chains'
        rprsdist = np.loadtxt(path+'RpRs_chain'+tag+'.txt')
        ddist = np.loadtxt(path+'Tdur_chain'+tag+'.txt')*24.
        bdist = np.loadtxt(path+'b_chain'+tag+'.txt')
        tdist = (np.loadtxt(path+'T0_chain'+tag+'.txt'))*24.*60.*60.
        laorsdist = np.loadtxt(path+'logaors_chain'+tag+'.txt')
        cosidist = np.abs(np.loadtxt(path+'cosi_chain'+tag+'.txt'))
        print '... done importing chains!'

    if lp is False:
        lp = np.loadtxt(path+'lnProb'+tag+'.txt')
        
    nsamp = len(ddist)

# Plot initial fit
#    plot_model([pdata[0,0],pdata[0,1],pdata[0,2],0.0,pdata[0,4]],tag='_mpfit',nbins=nbins)
#    plot_model_bin([pdata[0,0],pdata[0,1],pdata[0,2],0.0,pdata[0,4]],tag='_mpfit')


#  Get maximum likelihood values
    maxlike = np.max(lp)
    imax = np.array([i for i, j in enumerate(lp) if j == maxlike])
    if imax.size > 1:
        imax = imax[0]
    rprsval = np.float(rprsdist[imax])
    bval    = np.float(bdist[imax])
    dval    = np.float(ddist[imax])
    tval    = np.float(tdist[imax])
    aorsval = 10.0**np.float(laorsdist[imax])
    cosival = np.float(cosidist[imax])

    upper = np.linspace(.69,.999,100)
    lower = upper-0.6827

    print ''
    print 'Best fit parameters for '+name
    plt.figure(99,figsize=(8.5,11),dpi=300)    

    
    # Rp/Rstar
    sigsamp = 5.0
    rprsmin = np.min(rprsdist)
    rprsmax = np.max(rprsdist)
    rprssig = rb.std(rprsdist)
    bw = rprssig/sigsamp
    rprsmed = np.median(rprsdist)
    minval = max(rprsmed - sigfac*rprssig,rprsmin)
    maxval = min(rprsmed + sigfac*rprssig,rprsmax)
    if rprssig < 1.0e-5 or rprsval < 1.0e-6:
        print "It looks like there is no planet!!!"
        rprsmode = (maxval-minval)/2.0
        rprshi = maxval
        rprslo = minval
        nb = 100
        rprss = np.array([minval,maxval])
        rprspdf = np.array([1,1])
    else:
        rprss = np.linspace(rprsmin-2*np.std(rprsdist),rprsmax+2*np.std(rprsdist),1000)
        rprs_kde = gaussian_kde(rprsdist)
        rprspdf = rprs_kde(rprss)
        rprsdist_c = np.cumsum(rprspdf)/np.sum(rprspdf)
        rprsfunc = sp.interpolate.interp1d(rprsdist_c,rprss,kind='linear')
        rprshis = rprsfunc(upper)
        rprslos = rprsfunc(lower)
        ind     = np.argmin(rprshis-rprslos)
        rprshi  = rprshis[ind] 
        rprslo  = rprslos[ind] 
        rprsmode = rprss[np.argmax(rprspdf)]
        nb = np.ceil((rprsmax-rprsmin) / (np.abs(maxval-minval)/bindiv))

    rprsout = 'Rp/R*: max = {0:.5f}, med = {1:.5f}, mode = {2:.5f}, 1 sig int = {3:.5f}'
    print rprsout.format(rprsval, rprsmed, rprsmode, rprshi-rprslo)
    plt.subplot(4,1,1)
    plt.hist(rprsdist,bins=nb,normed=True)
#    plt.plot(rprss,rprspdf,color='c')
#    rprsmin = np.float(np.sort(rprsdist)[np.round(frac*nsamp)])
#    rprsmax = min(np.float(np.sort(rprsdist)[np.round((1-frac)*nsamp)]),rpmax)
    plt.xlim([minval,maxval])
#    plt.axvline(x=rprsval,color='r',linestyle='--')
#    plt.axvline(x=rprslo,color='c',linestyle='--')
#    plt.axvline(x=rprsmed,color='c',linestyle='--')
#    plt.axvline(x=rprshi,color='c',linestyle='--')
    plt.xlabel(r'$R_p/R_{*}$')
    plt.ylabel(r'$dP/d(R_p/R_{*})$')
    plt.title('Parameter Distributions for '+name)


  

    # Duration                           
    dmin = np.min(ddist)
    dmax = np.max(ddist)
    dsig = np.std(ddist)
    ds = np.linspace(dmin-dsig,dmax+dsig,1000)
    d_kde = gaussian_kde(ddist)
    dpdf = d_kde(ds)
    ddist_c = np.cumsum(dpdf)/np.sum(dpdf)
    dfunc = sp.interpolate.interp1d(ddist_c,ds,kind='linear')
    dhis = dfunc(upper)
    dlos = dfunc(lower)
    ind  = np.argmin(dhis-dlos)
    dhi  = dhis[ind] 
    dlo  = dlos[ind] 
    dmed = np.median(ddist)
    dmode = ds[np.argmax(dpdf)]
    rbsig = rb.std(ddist)
    minval = max(dmed - sigfac*rbsig,dmin)
    maxval = min(dmed + sigfac*rbsig,dmax)
    nb = np.ceil((dmax-dmin) / (np.abs(maxval-minval)/bindiv))
    
    dout = 'Transit duration: max = {0:.4f}, med = {1:.4f}, mode = {2:.4f}, 1 sig int = {3:.4f}'
    print dout.format(dval, dmed, dmode, dhi-dlo)

    plt.subplot(4,1,2)    
    plt.hist(ddist,bins=nb,normed=True)
#    plt.plot(ds,dpdf,color='c')
#    plt.axvline(x=dval,color='r',linestyle='--')
#    plt.axvline(x=dlo,color='c',linestyle='--')
#    plt.axvline(x=dmed,color='c',linestyle='--')
#    plt.axvline(x=dhi,color='c',linestyle='--')
#    dmin = np.float(np.sort(ddist)[np.round(frac*nsamp)])
#    dmax = min(np.float(np.sort(ddist)[np.round((1-frac)*nsamp)]),durmax)
    plt.xlim([minval,maxval])
    plt.xlabel(r'Transit Duration (hours)')
    plt.ylabel(r'$dP/d\tau$')


    # Impact parameter                        
    bmin = np.min(bdist)
    bmax = np.max(bdist)
    bsig = np.std(bdist)
    bss = np.linspace(bmin-bsig,bmax+bsig,1000)
    b_kde = gaussian_kde(bdist)
    bpdf = b_kde(bss)
    bdist_c = np.cumsum(bpdf)/np.sum(bpdf)
    bfunc = sp.interpolate.interp1d(bdist_c,bss,kind='linear')
    bhis = bfunc(upper)
    blos = bfunc(lower)
    ind  = np.argmin(bhis-blos)
    bhi  = bhis[ind] 
    blo  = blos[ind] 
    bmed = np.median(bdist)
    bmode = bss[np.argmax(bpdf)]
    rbsig = rb.std(bdist)
    minval = max(bmed - sigfac*rbsig,bmin)
    maxval = min(bmed + sigfac*rbsig,bmax)
    nb = np.ceil((bmax-bmin) / (np.abs(maxval-minval)/bindiv))

    bout = 'Impact parameter: max = {0:.3f}, med = {1:.3f}, mode = {2:.3f}, 1 sig int = {3:.3f}'
    print bout.format(bval, bmed, bmode, bhi-blo)


    plt.subplot(4,1,3)
    plt.hist(bdist,bins=nb,normed=True)
#    plt.plot(bss,bpdf,color='c')
#    bmin = np.float(np.sort(bdist)[np.round(frac*nsamp)])
#    bmax = np.float(np.sort(bdist)[np.round((1-frac)*nsamp)])
    plt.xlim([minval,maxval])
#    plt.axvline(x=bval,color='r',linestyle='--')
#    plt.axvline(x=blo,color='c',linestyle='--')
#    plt.axvline(x=bmed,color='c',linestyle='--')
#    plt.axvline(x=bhi,color='c',linestyle='--')
    plt.xlabel('Impact Parameter')
    plt.ylabel(r'$dP/db$')


    # t0 parameter                            
    tmin = np.min(tdist)
    tmax = np.max(tdist)
    tsig = np.std(tdist)
    ts = np.linspace(tmin-tsig,tmax+tsig,1000)
    t_kde = gaussian_kde(tdist)
    tpdf = t_kde(ts)
    tdist_c = np.cumsum(tpdf)/np.sum(tpdf)
    tfunc = sp.interpolate.interp1d(tdist_c,ts,kind='linear')
    this = tfunc(upper)
    tlos = tfunc(lower)
    ind  = np.argmin(this-tlos)
    thi  = this[ind] 
    tlo  = tlos[ind] 
    tmed = np.median(tdist)
    tmode = ts[np.argmax(tpdf)]
    rbsig = rb.std(tdist)
    minval = max(tmed - sigfac*rbsig,tmin)
    maxval = min(tmed + sigfac*rbsig,tmax)
    nb = np.ceil((tmax-tmin) / (np.abs(maxval-minval)/bindiv))
    
    tout = 'Delta t0: max = {0:.3f}, med = {1:.3f}, mode = {2:.3f}, 1 sig int = {3:.3f}'
    print tout.format(tval, tmed, tmode, thi-tlo)

    plt.subplot(4,1,4)
    
    plt.hist(tdist,bins=nb,normed=True)
#    plt.plot(ts,tpdf,color='c')
#    tmin = np.float(np.sort(tdist)[np.round(frac*nsamp)])
#    tmax = np.float(np.sort(tdist)[np.round((1-frac)*nsamp)])
    plt.xlim([minval,maxval])
#    plt.axvline(x=tval,color='r',linestyle='--')
#    plt.axvline(x=tlo,color='c',linestyle='--')
#    plt.axvline(x=tmed,color='c',linestyle='--')
#    plt.axvline(x=thi,color='c',linestyle='--')
    plt.xlabel(r'$\Delta t_0$ (seconds)')
    plt.ylabel(r'$dP/dt_0$')
    plt.annotate(r'$t_0$ = %.6f d' % (params0[3]), xy=(0.97,0.8),
                 ha="right",xycoords='axes fraction',fontsize='large')


    plt.subplots_adjust(hspace=0.55)

    plt.savefig(path+''.join(name.split())+'_params'+tag+'.png')
    plt.clf()

    vals    = [rprsval,dval/24.,bval,tval/(24.*3600.)+params0[3]]
    medvals = [rprsmed,dmed/24.,bmed,tmed/(24.*3600.)+params0[3]]
    modevals = [rprsmode,dmode/24.,bmode,tmode/(24.*3600.)]
    err = [rprssig,dsig/24.,bsig,tsig/(24.*3600.)]
#    errm = [rprsval-rprslo,(dval-dlo)/24.,bval-blo,(tval-tlo)/(24.*3600.)]

    bestvals = [[vals],[medvals],[modevals],[err]]

 # Plot up final fit
#    chisq = plot_model([vals[0],vals[1],vals[2],tval/(24.*3600.)],tag='_MCMCfit',nbins=nbins)
#    chisq = plot_model([vals[0],vals[1],vals[2],tval/(24.*3600.)], \
#                       bestparams=[rprsmed,dmed/24.0,bmed,tmed/(24.*3600.)],\
#                       tag='_MCMCfit',nbins=nbins)
    chisq = plot_model([vals[0],aorsval,cosival,tval/(24.*3600.)], \
                       bestparams=[rprsmed,dmed/24.0,bmed,tmed/(24.*3600.)],\
                       tag='_MCMCfit',nbins=nbins)


# Create KDE of Rp/Rs distribution    
#    print "Output Rp/Rs and Rp distributions"
#    rpdf = np.array([rprss,rprspdf])
#    np.savetxt(path+'MCMC/'+name+stag+ctag+ltag+'_RpRs.dat',rpdf.T)

# Create KDE of Rp distribution
#    rstarpdf = np.random.normal(Rstar,e_Rstar,nsamp)
#    rpdist = rstarpdf * rprsdist * c.Rsun/c.Rearth

#    rp = np.linspace(np.min(rpdist)*0.5,np.max(rpdist)*1.5,1000)
#    pdf_func = gaussian_kde(rpdist)
#    rkde = pdf_func(rp)
#    rpdf = np.array([rp,rkde])
#    np.savetxt(path+'MCMC/'+name+stag+ctag+ltag+'_Rp.dat',rpdf.T)
        
    outstr = name+', %.8f,  %.8f,  %.8f,  %.6f,  %.8f,  %.8f,  %.8f,  %.8f,  %.8f,  %.8f,  %.8f,  %.8f,  %.8f,  %.8f,  %.8f,  %.8f' %  (vals[0],medvals[0],modevals[0],err[0],vals[1],medvals[1],modevals[1],err[1],vals[2],medvals[2],modevals[2],err[2],vals[3],medvals[3],modevals[3],err[3])

    f = open(path+''.join(name.split())+'_fitparams.txt','w')

    f.write(outstr+'\n')
    f.closed


    return bestvals




def triangle_plot(chains=False,lp=False,thin=False,fontsize=15,linewidth=2,
                  tickfont=10,bindiv=30,sigsamp=5.0,sigfac=3.0,tag=''):
    import matplotlib as mpl
    global tfs, tlw, ttf
    
    mpl.rc('axes', linewidth=2)
    tfs = fontsize
    tlw = linewidth
    ttf = tickfont

    tmaster = time.time()
    print "Reading in MCMC chains"
    if chains is not False:
        rprsdist = chains[:,0]
        ddist = chains[:,1]*24.
        bdist = chains[:,2]
        tdist = chains[:,3]*24.*60.*60.
    else:
        print '... importing MCMC chains'
        rprsdist = np.loadtxt(path+'RpRs_chain'+tag+'.txt')
        ddist = np.loadtxt(path+'Tdur_chain'+tag+'.txt')*24.
        bdist = np.loadtxt(path+'b_chain'+tag+'.txt')
        tdist = (np.loadtxt(path+'T0_chain'+tag+'.txt'))*24.*60.*60.
        print '... done importing chains!'

    if lp is False:
        lp    = np.loadtxt(path+'lnProb'+tag+'.txt')

    done_in(tmaster)

    print "Determining maximum likelihood values"
    maxlike = np.max(lp)
    imax = np.array([i for i, j in enumerate(lp) if j == maxlike])
    if imax.size > 1:
        imax = imax[0]
    rprsval = np.float(rprsdist[imax])
    dval = np.float(ddist[imax])
    bval = np.float(bdist[imax])
    tval = np.float(tdist[imax])

    if thin:
        rprsdist = rprsdist[0::thin]
        ddist = ddist[0::thin]
        tdist = tdist[0::thin]
        bdist = bdist[0::thin]

    print " "
    print "Starting grid of posteriors..."
    plt.figure(5,figsize=(8.5,8.5))
    nx = 4
    ny = 4

    gs = gridspec.GridSpec(nx,ny,wspace=0.1,hspace=0.1)
    print " "
    print "... top plot of first column"
    tcol = time.time()
    top_plot(rprsdist,gs[0,0],val=rprsval,bindiv=bindiv,sigfac=sigfac)
    done_in(tcol)
    t = time.time()
    print "... first 2-d posterior plot"
    column_plot(rprsdist,ddist,gs[1,0],val1=rprsval,val2=dval,ylabel=r'$\tau$ (h)',sigfac=sigfac,sigsamp=sigsamp)
    done_in(t)
    column_plot(rprsdist,tdist,gs[2,0],val1=rprsval,val2=tval,ylabel=r'$\Delta t_0$ (s)',sigfac=sigfac,sigsamp=sigsamp)
    corner_plot(rprsdist,bdist,gs[3,0],val1=rprsval,val2=bval,\
                xlabel=r'$R_p/R_{*}$',ylabel=r'$b$',sigfac=sigfac,sigsamp=sigsamp)
    print "First column: "
    done_in(tcol)

    print "... second column"
    t2 = time.time()
    top_plot(ddist,gs[1,1],val=dval,sigfac=sigfac,bindiv=bindiv)    
    middle_plot(ddist,tdist,gs[2,1],val1=dval,val2=tval,sigfac=sigfac,sigsamp=sigsamp)
    row_plot(ddist,bdist,gs[3,1],val1=dval,val2=bval,xlabel=r'$\tau$ (h)',sigfac=sigfac,sigsamp=sigsamp)
    done_in(t2)

    print "... third column"
    t3 = time.time()
    top_plot(tdist,gs[2,2],val=tval,sigfac=sigfac,bindiv=bindiv)    
    row_plot(tdist,bdist,gs[3,2],val1=tval,val2=bval,xlabel=r'$\Delta t_0$ (s)',sigfac=sigfac,sigsamp=sigsamp)
    done_in(t3)

    print "... last plot"
    t5 = time.time()
    top_plot(bdist,gs[3,3],val=bval,xlabel=r'$b$',sigfac=sigfac,bindiv=bindiv)    
    done_in(t5)


    print "Saving output figures"
    plt.savefig(path+name+'_triangle'+tag+'.png', dpi=300)
    plt.savefig(path+name+'_triangle'+tag+'.eps', format='eps', dpi=600)

    print "Procedure finished!"
    done_in(tmaster)

    mpl.rc('axes', linewidth=1)

    return




def top_plot(dist,position,val=False,minval=False,maxval=False,sigfac=5.0,bindiv=20,aspect=1,xlabel=False):

    med = np.median(dist)
    sig = rb.std(dist)
    datamin = np.min(dist)
    datamax = np.max(dist)
    if not minval:
        minval = max(med - sigfac*sig,datamin)
    if not maxval:
        maxval = min(med + sigfac*sig,datamax)
    
    nb = np.ceil( (np.max(dist)-np.min(dist))*bindiv / np.abs(maxval-minval))
    ax = plt.subplot(position)
    plt.hist(dist,bins=nb,normed=True,color='black')
    if not xlabel: 
        ax.set_xticklabels(())
    ax.yaxis.set_tick_params(length=5, width=tlw)
    ax.xaxis.set_tick_params(length=5, width=tlw)
    ax.set_yticklabels(())
    ax.set_xlim(minval,maxval)
    xlimits = ax.get_xlim()
    ylimits = ax.get_ylim()
    ax.set_aspect(abs((xlimits[1]-xlimits[0])/(ylimits[1]-ylimits[0]))/aspect)
    if val:
#        plt.axvline(x=val,color='w',linestyle='--',linewidth=2)
        pass
    if xlabel:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(ttf) 
            tick.label.set_rotation('vertical')
        ax.set_xlabel(xlabel,fontsize=tfs)
        
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    return



def column_plot(dist1,dist2,position,val1=False,val2=False,sigfac=3.0,sigsamp=5.0,
                min1=False,max1=False,min2=False,max2=False,ylabel=None):
    from statsmodels.nonparametric.kernel_density import KDEMultivariate as KDE

    med1 = np.median(dist1)
    sig1 = rb.std(dist1)
    datamin1 = np.min(dist1)
    datamax1 = np.max(dist1)
    if not min1:
        min1 = max(med1 - sigfac*sig1,datamin1)
    if not max1:
        max1 = min(med1 + sigfac*sig1,datamax1)

#    len2 = np.size(dist2)
#    min2 = np.float(np.sort(dist2)[np.round(frac*len2)])
#    max2 = np.float(np.sort(dist2)[np.round((1.-frac)*len2)])
    med2 = np.median(dist2)
    sig2 = rb.std(dist2)
    datamin2 = np.min(dist2)
    datamax2 = np.max(dist2)
    if not min2:
        min2 = max(med2 - sigfac*sig2,datamin2)
    if not max2:
        max2 = min(med2 + sigfac*sig2,datamax2)

    aspect = (max1-min1)/(max2-min2)
    X, Y = np.mgrid[min1:max1:100j, min2:max2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([dist1, dist2])

    kernel = KDE(values,var_type='cc',bw=[sig1/sigsamp,sig2/sigsamp])
    Z = np.reshape(kernel.pdf(positions).T, X.shape)

#    kernel = gaussian_kde(values)
#    Z = np.reshape(kernel(positions).T, X.shape)

    ax = plt.subplot(position)
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,aspect=aspect,\
              extent=[min1, max1, min2, max2],origin='upper')
    clev = np.exp(np.log(np.max(Z))-0.5)
    cset = ax.contour(X,Y,Z,[clev],colors='w',linewidth=5,linestyles='dotted')
#    ax.plot(val1,val2, 'wx', markersize=3)
#    ax.set_xlim(min1, max1)
#    ax.set_ylim(min2, max2)
    ax.set_xticklabels(())
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ttf) 
    ax.yaxis.set_tick_params(length=5, width=tlw)
    ax.xaxis.set_tick_params(length=5, width=tlw)
    ax.set_ylabel(ylabel,fontsize=tfs)

    return



def row_plot(dist1,dist2,position,val1=False,val2=False,xlabel=None,sigfac=3.5,sigsamp=5.0,
             min1=False,max1=False,min2=False,max2=False):
    from statsmodels.nonparametric.kernel_density import KDEMultivariate as KDE

#    len1 = np.size(dist1)
#    min1 = np.float(np.sort(dist1)[np.round(frac*len1)])
#    max1 = np.float(np.sort(dist1)[np.round((1.-frac)*len1)])
    med1 = np.median(dist1)
    sig1 = rb.std(dist1)
    datamin1 = np.min(dist1)
    datamax1 = np.max(dist1)
    if not min1:
        min1 = max(med1 - sigfac*sig1,datamin1)
    if not max1:
        max1 = min(med1 + sigfac*sig1,datamax1)

    
#    len2 = np.size(dist2)
#    min2 = np.float(np.sort(dist2)[np.round(frac*len2)])
#    max2 = np.float(np.sort(dist2)[np.round((1.-frac)*len2)])
    med2 = np.median(dist2)
    sig2 = rb.std(dist2)
    datamin2 = np.min(dist2)
    datamax2 = np.max(dist2)
    if not min2:
        min2 = max(med2 - sigfac*sig2,datamin2)
    if not max2:
        max2 = min(med2 + sigfac*sig2,datamax2)
    
    aspect = (max1-min1)/(max2-min2)
    X, Y = np.mgrid[min1:max1:100j, min2:max2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([dist1, dist2])

    kernel = KDE(values,var_type='cc',bw=[sig1/sigsamp,sig2/sigsamp])
    Z = np.reshape(kernel.pdf(positions).T, X.shape)

#    kernel = gaussian_kde(values)
#    Z = np.reshape(kernel(positions).T, X.shape)

    ax = plt.subplot(position)
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,aspect=aspect,\
              extent=[min1, max1, min2, max2],origin='upper')
    clev = np.exp(np.log(np.max(Z))-0.5)
    cset = ax.contour(X,Y,Z,[clev],colors='w',linewidth=5,linestyles='dotted')
#    ax.plot(val1,val2, 'wx', markersize=3)
    ax.set_xlim(min1, max1)
    ax.set_ylim(min2, max2)
    ax.set_yticklabels(())
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(ttf) 
        tick.label.set_rotation('vertical')
    ax.set_xlabel(xlabel,fontsize=tfs)
    ax.yaxis.set_tick_params(length=5, width=tlw)
    ax.xaxis.set_tick_params(length=5, width=tlw)
    return

def middle_plot(dist1,dist2,position,val1=False,val2=False,sigfac=3.5,sigsamp=5.0,
                min1=False,max1=False,min2=False,max2=False):
    from statsmodels.nonparametric.kernel_density import KDEMultivariate as KDE


#    len1 = np.size(dist1)
#    min1 = np.float(np.sort(dist1)[np.round(frac*len1)])
#    max1 = np.float(np.sort(dist1)[np.round((1.-frac)*len1)])
    med1 = np.median(dist1)
    sig1 = rb.std(dist1)
    datamin1 = np.min(dist1)
    datamax1 = np.max(dist1)
    if not min1:
        min1 = max(med1 - sigfac*sig1,datamin1)
    if not max1:
        max1 = min(med1 + sigfac*sig1,datamax1)

#    len2 = np.size(dist2)
#    min2 = np.float(np.sort(dist2)[np.round(frac*len2)])
#    max2 = np.float(np.sort(dist2)[np.round((1.-frac)*len2)])
    med2 = np.median(dist2)
    sig2 = rb.std(dist2)
    datamin2 = np.min(dist2)
    datamax2 = np.max(dist2)
    if not min2:
        min2 = max(med2 - sigfac*sig2,datamin2)
    if not max2:
        max2 = min(med2 + sigfac*sig2,datamax2)

    aspect = (max1-min1)/(max2-min2)
    X, Y = np.mgrid[min1:max1:100j, min2:max2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([dist1, dist2])

    kernel = KDE(values,var_type='cc',bw=[sig1/sigsamp,sig2/sigsamp])
    Z = np.reshape(kernel.pdf(positions).T, X.shape)

#    kernel = gaussian_kde(values)
#    Z = np.reshape(kernel(positions).T, X.shape)
 
    ax = plt.subplot(position)
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,aspect=aspect,\
              extent=[min1, max1, min2, max2],origin='upper')
    clev = np.exp(np.log(np.max(Z))-0.5)
    cset = ax.contour(X,Y,Z,[clev],colors='w',linewidth=5,linestyles='dotted')
#    ax.plot(val1,val2, 'wx', markersize=3)
    ax.set_xlim(min1, max1)
    ax.set_ylim(min2, max2)
    ax.set_xticklabels(())
    ax.set_yticklabels(())
    ax.yaxis.set_tick_params(length=5, width=tlw)
    ax.xaxis.set_tick_params(length=5, width=tlw)

    return



def corner_plot(dist1,dist2,position,val1=False,val2=False,sigfac=3.5,sigsamp=5.0,
                min1=False,max1=False,min2=False,max2=False,
                xlabel=None,ylabel=None):
    from statsmodels.nonparametric.kernel_density import KDEMultivariate as KDE

    med1 = np.median(dist1)
    sig1 = rb.std(dist1)
    datamin1 = np.min(dist1)
    datamax1 = np.max(dist1)
    if not min1:
        min1 = max(med1 - sigfac*sig1,datamin1)
    if not max1:
        max1 = min(med1 + sigfac*sig1,datamax1)

    
#    len2 = np.size(dist2)
#    min2 = np.float(np.sort(dist2)[np.round(frac*len2)])
#    max2 = np.float(np.sort(dist2)[np.round((1.-frac)*len2)])
    med2 = np.median(dist2)
    sig2 = rb.std(dist2)
    datamin2 = np.min(dist2)
    datamax2 = np.max(dist2)
    if not min2:
        min2 = max(med2 - sigfac*sig2,datamin2)
    if not max2:
        max2 = min(med2 + sigfac*sig2,datamax2)

    aspect = (max1-min1)/(max2-min2)
    X, Y = np.mgrid[min1:max1:100j, min2:max2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([dist1, dist2])

    kernel = KDE(values,var_type='cc',bw=[sig1/sigsamp,sig2/sigsamp])
    Z = np.reshape(kernel.pdf(positions).T, X.shape)

#    kernel = gaussian_kde(values)
#    Z = np.reshape(kernel(positions).T, X.shape) 

    ax = plt.subplot(position)
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,aspect=aspect,\
              extent=[min1, max1, min2, max2],origin='upper')
    clev = np.exp(np.log(np.max(Z))-0.5)
    cset = ax.contour(X,Y,Z,[clev],colors='w',linewidth=5,linestyles='dotted')
#    ax.plot(val1,val2, 'wx', markersize=3)
    ax.set_xlim(min1, max1)
    ax.set_ylim(min2, max2)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ttf) 
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(ttf) 
        tick.label.set_rotation('vertical')
    ax.yaxis.set_tick_params(length=5, width=tlw)
    ax.xaxis.set_tick_params(length=5, width=tlw)
    ax.set_xlabel(xlabel,fontsize=tfs)
    ax.set_ylabel(ylabel,fontsize=tfs)

    return

   

def done_in(tmaster):

    t = time.time()
    hour = (t - tmaster)/3600.
    if np.floor(hour) == 1:
        hunit = "hour"
    else:
        hunit = "hours"

    minute = (hour - np.floor(hour))*60.
    if np.floor(minute) == 1:
        munit = "minute"
    else:
        munit = "minutes"

    sec = (minute - np.floor(minute))*60.


    if np.floor(hour) == 0 and np.floor(minute) == 0:
        tout = "done in {0:.2f} seconds"
        print tout.format(sec)
    elif np.floor(hour) == 0:
        tout = "done in {0:.0f} "+munit+" {1:.2f} seconds"
        print tout.format(np.floor(minute),sec)
    else:
        tout = "done in {0:.0f} "+hunit+" {1:.0f} "+munit+" {2:.2f} seconds"
        print tout.format(np.floor(hour),np.floor(minute),sec)

    print " "

    return
 
#----------------------------------------------------------------------
# do_full_fit:
#    pick a KOI and do all fits!
#----------------------------------------------------------------------
            

def do_full_fit(koi,nwalkers=500,burnsteps=1000,mcmcsteps=2000,clobber=False,lcnb=False,network=None,thin=False):
    
    print "starting KOI-"+str(koi)

    for planet in np.arange(1,9):
        check1 = isthere(koi,planet,short=1,network=network)
        check2 = isthere(koi,planet,network=network)
        check3 = isthere(koi,planet,network=network)
        if check3 and lcnb:
            lc,pdata,sdata = get_koi_info(koi,planet,network=network)
            chains,lp = fit_single(nwalkers=nwalkers,burnsteps=burnsteps,\
                                    mcmcsteps=mcmcsteps,clobber=clobber)
            fit,rpdf = bestvals(chains=chains)
            triangle_plot(thin=thin)
        if check2:
            lc,pdata,sdata = get_koi_info(koi,planet,network=network)
            chains,lp = fit_single(nwalkers=nwalkers,burnsteps=burnsteps,\
                                    mcmcsteps=mcmcsteps,clobber=clobber)
            fit,rpdf = bestvals(chains=chains)
            triangle_plot(thin=thin)
        if check1:
            lc,pdata,sdata = get_koi_info(koi,planet,short=1,network=network)
            chains,lp = fit_single(nwalkers=nwalkers,burnsteps=burnsteps,\
                                    mcmcsteps=mcmcsteps,clobber=clobber)
            fit,rpdf = bestvals(chains=chains)
            triangle_plot(thin=thin)
        else: pass
 
    return

def do_short_fits(koi,nwalkers=500,burnsteps=1000,mcmcsteps=2000,clobber=False,network=None,thin=False):
    
    import numpy as np
    import time
    import os
    import constants as c

    print "starting KOI-"+str(koi)

    for planet in np.arange(1,9):
        check1 = isthere(koi,planet,short=1,network=network)
        if check1:
            lc,pdata,sdata = get_koi_info(koi,planet,short=1,network=network)
            chains,lp = fit_single(nwalkers=nwalkers,burnsteps=burnsteps,\
                                    mcmcsteps=mcmcsteps,clobber=clobber)
            fit,rpdf = bestvals(chains=chains)
            triangle_plot(thin=thin)
        else: pass
 
    return


#----------------------------------------------------------------------
# batchfit:
#    routine allowing the batch fitting of several KOIs
#----------------------------------------------------------------------

def batchfit(start=0,stop=103,nwalkers=1000,burnsteps=500,mcmcsteps=1000,clip=1,\
                 clobber=False,network=None,short=False,limbmodel='quad',list=False):

    import numpy as np
    import time
    import os
    import constants as c
    
    kois = np.sort(koilist(network=network))

    nkois = len(kois)

    if list:
        for number in np.arange(nkois):
            print number,kois[number]
        return

    else:
        tstart = time.time()
        for i in np.arange(start,stop+1):
            koi = kois[i]
            npl,koiname = numplanet(koi,network=network)
            for planet in np.arange(npl)+1:
                print "Starting MCMC fitting for KOI-"+str(koi)+".0"+str(planet)
                do_fit(koi,planet,nwalkers=nwalkers,burnsteps=burnsteps,mcmcsteps=mcmcsteps,\
                           clobber=clobber,network=network,limbmodel=limbmodel,short=short,clip=clip)
    return



def do_fit(koi,planet,nwalkers=250,burnsteps=1000,mcmcsteps=1000,clobber=False,\
               network=None,limbmodel='quad',short=False,clip=1):
    
    import numpy as np
    import time
    import os
    import constants as c

    print "starting KOI-"+str(koi)

    check1 = isthere(koi,planet,short=short,network=network,clip=clip)
    if check1:
        lc,pdata,sdata = get_koi_info(koi,planet,short=short,network=network,\
                                          clip=clip,limbmodel=limbmodel)
        chains,lp = fit_single(nwalkers=nwalkers,burnsteps=burnsteps,\
                                   mcmcsteps=mcmcsteps,clobber=clobber)
    else:
        print "planet "+str(planet)+" not found"
 
    return




def batchplot(start=0,stop=103,network=None,clip=1,limbmodel='quad',thin=10,frac=0.001,short=False,dotriangle=True,list=False):

    import numpy as np
    import time
    import os
    import constants as c
    
    kois = np.sort(koilist(network=network))

    nkois = len(kois)

    if list:
        for number in np.arange(nkois):
            print number,kois[number]
        return

    else:
        tstart = time.time()
        for i in np.arange(start,stop+1):
            koi = kois[i]
            npl,koiname = numplanet(koi,network=network)
            for planet in np.arange(npl)+1:
                print "Starting plot routine for KOI-"+str(koi)+".0"+str(planet)
                do_plots(koi,planet,network=network,limbmodel=limbmodel,short=short,clip=clip,dotriangle=dotriangle)
    return



def do_plots(koi,planet,network=None,clip=1,limbmodel='quad',thin=10,frac=0.001,short=False,dotriangle=False):
    
    import numpy as np
    import time
    import os
    import constants as c

    print "starting KOI-"+str(koi)

    check1 = isthere(koi,planet,short=short,network=network,clip=clip)
    if check1:
        lc,pdata,sdata = get_koi_info(koi,planet,short=short,network=network,clip=clip,limbmodel=limbmodel)
        fit,rpdf = bestvals(frac=frac)
        if dotriangle:
            triangle_plot(thin=thin,frac=frac)
        plot_limb_curves(write=True)
    else:
        print "planet "+str(planet)+" not found"
 
    return





















































#def do_long_fit(koi,planet,nwalkers=250,burnsteps=1000,mcmcsteps=1000,clobber=False,network=None,thin=False,clip=1,limbmodel='quad',frac=0.001):
#    
#    import numpy as np
#    import time
#    import os
#    import constants as c
#
#    print "starting KOI-"+str(koi)
#
#    check1 = isthere(koi,planet,short=False,network=network,clip=clip)
#    if check1:
#        lc,pdata,sdata = get_koi_info(koi,planet,short=False,network=network,\
#                                          clip=clip,limbmodel=limbmodel)
#        chains,lp = fit_single(nwalkers=nwalkers,burnsteps=burnsteps,\
#                                   mcmcsteps=mcmcsteps,clobber=clobber)
#        fit,rpdf = bestvals(chains=chains)
#        triangle_plot(thin=thin,frac=frac)
#        plot_limb_curves(write=True)
#    else:
#        print "planet "+str(planet)+" not found"
# 
#    return

#def do_short_fit(koi,planet,nwalkers=250,burnsteps=1000,mcmcsteps=1000,clobber=False,network=None,thin=False,clip=1,limbmodel='quad',frac=0.001):
#    
#    import numpy as np
#    import time
#    import os
#    import constants as c
#
#    print "starting KOI-"+str(koi)
#
#    check1 = isthere(koi,planet,short=True,network=network,clip=clip)
#    if check1:
#        lc,pdata,sdata = get_koi_info(koi,planet,short=1,network=network,clip=clip,limbmodel=limbmodel)
#        chains,lp = fit_single(nwalkers=nwalkers,burnsteps=burnsteps,\
#                                   mcmcsteps=mcmcsteps,clobber=clobber)
#        fit,rpdf = bestvals(chains=chains,frac=frac)
#        triangle_plot(thin=thin,frac=frac)
#    else:
#        print "planet "+str(planet)+" not found"
# 
#    return
