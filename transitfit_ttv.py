# To Do:
# plot_ttv_model: fold transits on TTV times and plot

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
import os
import robust as rb
from scipy.stats.kde import gaussian_kde

def koilist(network=None):

    """ 
    ----------------------------------------------------------------------
    koilist:
    --------
    Utility for checking the list of KOIs with preliminary data. Will check
    a directory corresponding to what network you are on.
    
    example:
    --------
    In[1]: kois = koilist(network=None)
    ----------------------------------------------------------------------
    """

    if network == 'astro':
        dir = '/scr2/jswift/Mdwarfs/outdata/'
    if network == 'gps':
        dir = '/home/jswift/Mdwarfs/outdata/'
    if network == None:
        dir = '/Users/jonswift/Astronomy/Exoplanets/Mdwarfs/All/outdata/'

    files = glob.glob(dir+'[1-9]*')

    koilist = np.zeros(len(files))
    for i in xrange(0,len(files)):
        ind = files[i].find('outdata/')+8
        koilist[i] = files[i][ind:]

    koilist = np.array(koilist).astype(int)
    return koilist




def isthere(koiin,plin,short=False,network=None,clip=False):

    """ 
    ----------------------------------------------------------------------
    isthere:
    --------
    Utility for checking if the data file for a KOI planet is on disk. Will 
    distinguish between SC and LC data and data that has been sigma clipped.
    
    example:
    --------
    In[1]: isthere(247,1,short=True,network='gps',clip=True)
    ----------------------------------------------------------------------
    """

    prefix = str(koiin)+'.0'+str(plin)

   # Define file tags 
    if short:
        type = '_short'
    else:
        type = '_long'
        
    if clip: 
        fname = prefix+type+'_clip.dat'
    else:
        fname = prefix+type+'.dat'

    if network == 'astro':
        path = '/scr2/jswift/Mdwarfs/outdata/'+str(koiin)+'/Refine/'
    if network == 'gps':
        path = '/home/jswift/Mdwarfs/outdata/'+str(koiin)+'/Refine/'
    if network == None:
        path = '/Users/jonswift/Astronomy/Exoplanets/Mdwarfs/All/outdata/'+ \
        str(koiin)+'/Refine/'
    if network == 'koi':
        path = '/Users/jonswift/Astronomy/Exoplanets/KOI'+str(koiin)+ \
        '/lc_fit/outdata/'+str(koiin)+'/Refine/'

    test = os.path.exists(path+fname)

    return test


def numplanet(koiin,network=None):
    
    """
    ----------------------------------------------------------------------
    numplanets:
    -----------
    return the number of planets in KOI system does not depend on any global variables
    
    example:
    In[1]: npl,names = numplanet(952,network=None)

    """
    if network == 'astro':
        path = '/scr2/jswift/Mdwarfs/outdata/'+str(koiin)+'/Refine/'
    if network == 'gps':
        path = '/home/jswift/Mdwarfs/outdata/'+str(koiin)+'/Refine/'
    if network == None:
        path = '/Users/jonswift/Astronomy/Exoplanets/Mdwarfs/All/outdata/'+ \
            str(koiin)+'/Refine/'
    if network == 'koi':
        path = '/Users/jonswift/Astronomy/Exoplanets/KOI'+str(koiin)+ \
            '/lc_fit/outdata/'+str(koiin)+'/Refine/'
  
    info1 = np.loadtxt(path+str(koiin)+'_long.out',delimiter=',',ndmin=2)
    npl = info1.shape[0]
    names = []
    for i in xrange(0,npl):
        names.append("%.2f" % (info1[i,0]))

    return npl,names

def tottomid(dtot,rprs,impact):
    """
    ----------------------------------------------------------------------
    tottomid:
    --------------
    Function to convert total transit duration to midingress/egress transit 
    given a planet size and impact parameter

    TODO:
    -----
    Need to catch for high impact parameter solutions
    """
    xtot  = np.sqrt((1+rprs)**2 - impact**2)
    xfull = np.sqrt((1-rprs)**2 - impact**2)
    xmid = (xtot+xfull)/2.

    dmid  = dtot*xmid/xtot
    return dmid

def qtou(q1,q2,limb='quad'):
    if limb == 'quad':
        u1 =  2*np.sqrt(q1)*q2
        u2 =  np.sqrt(q1)*(1-2*q2)
    if limb == 'sqrt':
        u1 = np.sqrt(q1)*(1-2*q2)
        u2 = 2*np.sqrt(q1)*q2

    return u1, u2


def utoq(u1,u2,limb='quad'):
    
    if limb == 'quad':
        q1 = (u1+u2)**2
        q2 = u1/(2.0*(u1+u2))
    if limb == 'sqrt':
        q1 = (u1+u2)**2
        q2 = u2/(2.0*(u1+u2))

    return q1, q2

def get_koi_info(koiin,planet,short=False,network=None,clip=False,limbmodel='quad',
                 getsamp=False,rprior=False,lprior=False,errfac=3,fitall=False):

    """
    ----------------------------------------------------------------------
    get_koi_info:
    -------------
    Get lightcurve and preliminary transit parameters for KOI. Also set a
    bunch of global variables to be used by transit fitting routines.
    
    Must run this routine first, before transit fitting


    example:
    --------
    In[1]: lc,pdata,sdata = get_koi_info(248,4,short=True,network=None,
                                  clip=True,limbmodel='quad')


    to do:
    ------
    Create a dictionary with all relevant information instead of clunky 
    information arrays.
    ----------------------------------------------------------------------
    """

# Import limb coefficient module
    import get_limb_coeff as glc
    import constants as c

# Define global variables
    global lc
    global integration
    global path
    global name
    global koi
    global pdata
    global sdata
    global stag
    global onastro
    global bjd
    global ctag
    global ltag, lptag, atag, doall
    global limb
    global ndim
    global rtag
    global jeffries
    global tindex
    global ntrans
    global sampfac, adapt

# Short cadence data vs. long cadence data    
    int = 6.019802903270
    read = 0.518948526144
    if short == True:
        stag = '_short'
        integration = int*9.0+read*8.0
    else:
        stag = '_long'
        integration = int*270.0 + read*269.0

    koi = koiin

# "Kepler time"
    bjd = 2454833.0

# Setup path and info specific to KOI
    if network == 'astro':
        path = '/scr2/jswift/Mdwarfs/outdata/'+str(koi)+'/'
    if network == 'gps':
        path = '/home/jswift/Mdwarfs/outdata/'+str(koi)+'/'
    if network == None:        
        path = '/Users/jonswift/Astronomy/Exoplanets/Mdwarfs/All/outdata/'+str(koi)+'/'
    if network == 'koi':
        path = '/Users/jonswift/Astronomy/Exoplanets/KOI'+str(koi)+'/lc_fit/outdata/'+str(koi)+'/'


# Get number of planets and their names
    npl,names = numplanet(koi,network=network)

    if planet == 0 or planet > npl:
        sys.exit("Planet number = "+str(planet)+'. But only '+str(npl)+" planets in system")

# Planet index
    pli = planet-1

        
# Use clipped data if asked for
    if clip: 
        ctag = '_clip'
    else:
        ctag = ''

# Use Jeffries prior on Rp/Rstar
    if fitall:
        atag = '_all'
        doall = True

    if rprior:
        rtag = '_rp'
        jeffries = True
    else:
        rtag = ''
        jeffries = False

    if lprior:
        lptag = '_lp'
    else:
        lptag = ''

# Star properties
    print "... getting star properties"
    info2 = np.loadtxt(path+'Fits/'+names[pli]+stag+'_fixlimb_fitparams.dat',dtype='string')
    Mstar = np.float(info2[1])*c.Msun
    eMstar = np.float(info2[2])*c.Msun
    Rstar = np.float(info2[3])*c.Rsun
    eRstar = np.float(info2[4])*c.Rsun
    Tstar = np.float(info2[5])
    eTstar = np.float(info2[6])

    loggstar = np.log10( c.G * Mstar / Rstar**2. )

    
    print "... getting LD coefficients for "+limbmodel+" model"
    if network == 'koi':
        net = None
    else:
        net = network

    if limbmodel == 'nlin':
        a,b,c,d = glc.get_limb_coeff(Tstar,loggstar,network=net,limb=limbmodel,interp='nearest')
    else:
        a,b = glc.get_limb_coeff(Tstar,loggstar,network=net,limb=limbmodel,interp='nearest')
  

    if limbmodel == 'quad':
        ltag = '_quad'
        c1 = a
        c2 = b
        ldc = [c1,c2]
        eldc = [0,0]
        print  "Limb darkening coefficients:"
        aout = '     c1 = {0:.4f}'
        print aout.format(a)
        bout = '     c2 = {0:.4f}'
        print bout.format(b)
        print " "
    elif limbmodel == 'sqrt':
        ltag = '_sqrt'
        c1 = b
        c2 = a
        ldc = [c1,c2]
        eldc = [0,0]
        print  "Limb darkening coefficients:"
        aout = '     c1 = {0:.4f}'
        print aout.format(c1)
        bout = '     c2 = {0:.4f}'
        print bout.format(c2)
        print " "
    elif limbmodel == 'nlin':
        ltag = '_nlin'
        c1 = a
        c2 = b
        c3 = c
        c4 = d
        ldc = [c1,c2,c3,c4]
        eldc = [0,0,0,0]
        print  "Limb darkening coefficients:"
        aout = '     c1 = {0:.4f}'
        print aout.format(a)
        bout = '     c2 = {0:.4f}'
        print bout.format(b)
        cout = '     c3 = {0:.4f}'
        print cout.format(c)
        dout = '     c4 = {0:.4f}'
        print dout.format(d)
        print " "    
    else:
        print "Limb darkening law not recognized"
        return

    limb = limbmodel

    
# MCMC fit parameters 
    info3 = np.loadtxt(path+'MCMC/'+names[pli]+stag+ctag+rtag+ltag+'fit_fitparams.txt',dtype='string')
    # Take modes
    rprs0    = np.float(info3[3])
    erprs0   = np.float(info3[4])
    dur0     = np.float(info3[7]) 
    edur0    = np.float(info3[8]) 
    impact0  = np.float(info3[11])
    eimpact0 = np.float(info3[12])
    period0  = np.float(info3[19])
    eperiod0 = np.float(info3[20])
    ephem0   = np.float(info3[15])
    eephem0  = np.float(info3[16])


# Determine number of samples per integration time by interpolation using
# LM fit parameters as a guide.
    if getsamp == True:
        adapt = True
        # From int_sample.py
        rsz = 50
        dsz = 50
        rmin = 0.006
        rmax = 0.3
        dmin = 0.015
        dmax = 0.5
        rgrid = np.linspace(rmin,rmax,rsz)
        dgrid = np.linspace(dmin,dmax,dsz)
        vals = np.loadtxt('sample_grid'+stag+'.txt')
        f = interp2d(dgrid,rgrid,vals,kind='cubic')
        sampfac, = np.round(f(min(max(dur0,dmin),dmax),
                              min(max(rprs0,rmin),rmax)))
#        if sampfac % 2 == 0:
#            sampfac += 1
        sampfac = sampfac*2 + 1
        print "Using a sampling of "+str(sampfac)+ \
            " to account for integration time of "+str(integration)+" s"
    else:
        adapt = False
    name = names[pli]
    file = name+stag+ctag+'.dat'
    print "Reading data from "+file
    data = np.loadtxt(path+'Refine/'+file)

    t = data[:,0]
    flux = data[:,2]
    e_flux = data[:,3]
    tnum = data[:,4]

    tindex = np.unique(tnum)

    if limb == 'nlin':
        ndim = 7 + len(tindex)
    else:
        ndim = 5 + len(tindex)

# Make planet data array
    pinfo =[rprs0,dur0,impact0,ephem0,period0]
    perr = [erprs0,edur0,eimpact0,eephem0,eperiod0]

    pdata = np.array([pinfo,perr])
        
# Make star data array
    if limbmodel == 'nlin':
        sinfo = [Mstar,Rstar,Tstar,c1,c2,c3,c4]
        serr  = [eMstar,eRstar,eTstar,-999,-999,-999,-999]    
    else:
        sinfo = [Mstar,Rstar,Tstar,c1,c2]
        serr  = [eMstar,eRstar,eTstar,-999,-999]
    
    sdata = np.array([sinfo,serr])

# Number of measured transit times
    info4 = np.loadtxt(path+'TTVs/'+names[pli]+'_long'+ctag+rtag+'_TTVs.dat')
    ninds, = np.where(np.isfinite(info4[:,1]) == True)
    ntrans = len(ninds)
    print "There are "+str(ntrans)+" transit mid times measured for KOI-"+str(koi)+".0"+str(planet)
    print "Recommend using at least %.0f walkers in MCMC fit" % np.int(2.0*(ntrans+6))

# Make light curve array
    ttm = foldtime(t,period=period0,t0=ephem0)

    lc = np.array([t,ttm,flux,e_flux,tnum])

    return lc,pdata,sdata



def foldtime(time,period=1.0,t0=0.0):

    """ 
    ----------------------------------------------------------------------
    foldtime:
    ---------
    Basic utility to fold time based on period and ephemeris

    example:
    --------
    In[1]: ttm = foldtime(time,period=2.35884,t0=833.5123523)
    ----------------------------------------------------------------------
    """

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



def foldttv(maxval=False):
    
    # Get raw data
    time = lc[0,:]
    flux = lc[2,:]
    e_flux = lc[3,:]
    tnum = lc[4,:]

    # Get nominal period and ephemeris
    ephem0 = pdata[0,3]
    period0 = pdata[0,4]

    # Derive reference time from mid transit
    ttm = foldtime(time,period=period0,t0=ephem0)
    
    # get TTV data
    ttvdata = np.loadtxt(path+'TTVs/'+name+stag+ctag+rtag+ltag+'_MCMC_TTVs.dat')
    index  = ttvdata[:,0]
    ttime  = ttvdata[:,1]
    ttvmed = ttvdata[:,2]
    ttvval = ttvdata[:,3]
    ttvhi  = ttvdata[:,4]
    ttvlo  = ttvdata[:,5]

    ttmout = []
    fout = []
    eout = []

    for i in np.arange(len(index)):
        inds, = np.where(tnum == index[i])
        if len(inds) >= 1:
            if maxval:
                dt = ttvval[i]
            else:
                dt = ttvmed[i]

            ttmout = np.append(ttmout,ttm[inds]-dt)
            fout = np.append(fout,flux[inds])
            eout = np.append(eout,e_flux[inds])

    return ttmout,fout,eout



def bin_lc(x,y,nbins=100):

    """
    ----------------------------------------------------------------------    
    bin_lc:
    -------
    Utility to bin data and return standard deviation in each bin
    

    example:
    --------
    tbin,fbin,errbin = bin_lc(ttm,flux,nbins=200)
    ----------------------------------------------------------------------
    """

    n, _ = np.histogram(x, bins=nbins)
    sy, _ = np.histogram(x, bins=nbins, weights=y)
    sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)

    binvals = (_[1:] + _[:-1])/2
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
    xfull = np.sqrt((1-rprs)**2 - impact**2)
    xmid = (xtot+xfull)/2.

    dtot  = dmid*xtot/xmid
    dfull = dmid*xfull/xmid 
    return dtot,dfull


def tottomid(dtot,rprs,impact):
    """
    ----------------------------------------------------------------------
    tottomid:
    --------------
    Function to convert total transit duration to midingress/egress transit 
    given a planet size and impact parameter

    TODO:
    -----
    Need to catch for high impact parameter solutions
    """
    xtot  = np.sqrt((1+rprs)**2 - impact**2)
    xfull = np.sqrt((1-rprs)**2 - impact**2)
    xmid = (xtot+xfull)/2.

    dmid  = dtot*xmid/xtot
    return dmid





def compute_trans(rprs,duration,impact,t0,ldc,unsmooth=False,
                  all=False,modelfac=21.0):

    """
    ----------------------------------------------------------------------
    compute_trans:
    --------------
    Function to compute transit curve for given transit parameters. Returns
    model time and model flux. 

    options:
    --------
    getingress: In addition to the model data, also return the tII to tIII 
                duration.
    unsmooth:   In addition to the model data, also return the unsmoothed 
                light curve for comparison

    examples:
    ---------
    In[1]: tmodel,model = compute_trans(rprs,duration,impact,t0,per)

    In[2]: tmodel,model,tfull = compute_trans(rprs,duration,impact,t0,per,
                                                    getingress=True)
    
    In[3]: tmodel,model,rawmodel = compute_trans(rprs,duration,impact,t0,
                                                 per,unsmooth=True)
    ----------------------------------------------------------------------

    """

# t0 is offset from nominal ephemeris
    import occultquad as oq
    import transit as trans
    from scipy.ndimage.filters import uniform_filter1d, uniform_filter

    period = pdata[0,4]

    if limb == 'nlin':
        a1 = ldc[0]
        a2 = ldc[1]
        a3 = ldc[2]
        a4 = ldc[3]
    else:        
        q1in = ldc[0]
        q2in = ldc[1]
        u1,u2 = qtou(q1in,q2in,limb=limb)
        
   # "Lens" radius = Planet radius/Star radius
    rl = rprs

    # Sampling of transit model will be at intervals equal to the 
    # integration time divided by modelfac
    # Needs to be an odd number!
    if adapt == True:
        modelfac = sampfac
    
    if modelfac % 2 == 0:
        print "Modelfac = %.1f oversampling of integration time must "+ \
            "be an odd number! " % modelfac
        pdb.set_trace()

    # Sample transit at "modelfac" times the integration time else if impact
    # parameter is < 1-rl
    if impact >= (1+rl):
        duration = 0.0
        sztrans = 10
        modelt = sp.linspace(-1*period/2,period/2,sztrans*2)
        smoothmodel = np.ones(sztrans*2)
        if unsmooth:
            return modelt+t0,smoothmodel,smoothmodel
        else:
            return modelt+t0,smoothmodel
    
    # a/R* assuming inclined, circular orbit
    aors = np.sqrt( ((1+rl)**2 - impact**2)/
                    ((np.sin(np.pi*duration/period))**2) + impact**2)    
    
    # Number of integrations in 1/2 full duration
    nint = duration*24.*3600./(2.0*integration)
 
    # Number of transit samples = integer number of integrations * modelfac
    # extra 1 to account for one step = 2 points
    sztrans = np.ceil(nint)*modelfac + 1

    # Factor beyond full duration that an integer number of integrations extends
    ifac = np.ceil(nint)/nint

    # Linear time for sztrans samplings of transit curve    
    time0 = np.linspace(0,duration/2.*ifac,sztrans)
#    dt = time0[1]-time0[0]
    dt = integration/(modelfac*24.0*3600.0)

    # Compute impact parameter for linear time intervals assuming circular orbit
    theta = 2.0*np.pi*time0/period
    h = aors*np.sin(theta)
    v = np.cos(theta)*impact
    b0 = np.sqrt(h**2 + v**2)

    if limb == 'quad':
        model = np.array(oq.occultquad(b0,u1,u2,rl))[0]
    elif limb == 'sqrt':
        model = np.array(trans.occultnonlin(b0,rl,np.array([u1,u2,0.0,0.0])))
    elif limb == 'nlin':
        model = np.array(trans.occultnonlin(b0,rl,np.array([a1,a2,a3,a4])))
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

# Smooth to integration time
    sl = modelfac
    smoothmodel = uniform_filter1d(fmodel, np.int(sl))

# Return unsmoothed model if requested
    if unsmooth:
        return tmodel+t0,smoothmodel,fmodel

    if all:
        return tmodel+t0,bmodel,fmodel,smoothmodel

    return tmodel+t0,smoothmodel



def lnprob(x):

    """
    ----------------------------------------------------------------------
    lnprob:
    -------
    Function to compute logarithmic probability of data given model. This
    function sets prior constaints explicitly and calls compute_trans to
    compare the data with the model. Only data within the smoothed transit
    curve is compared to model. 
    ----------------------------------------------------------------------

    """

    import robust as rb

# Input parameters
    rprs     = x[0]
    duration = x[1]
    impact   = x[2]
    q1       = x[3]
    q2       = x[4]
    if limb == 'nlin':
        q3 = x[5]
        q4 = x[6]
        ldc = [q1,q2,q3,q4]
        deltats = x[7:]
    else:
        ldc = [q1,q2]
        deltats = x[5:]

# Priors

    if duration < 1e-4 or duration > pdata[0,4]/2.0:
#        print "duration out of range"
#        pdb.set_trace()
        return -np.inf

    if rprs <= 0 or rprs > 0.5:
#        print "rprs out of range"
#        pdb.set_trace()
        return -np.inf

    if impact > (1 + rprs) or impact < 0:
#        print "impact parameter out of range"
#        pdb.set_trace()
        return -np.inf

    if not limb == 'nlin':
        if q1 > 1 or q1 < 0 or q2 > 1 or q2 < 0:
#            print "LD law out of range"
#            pdb.set_trace()
            return -np.inf
        
        
    time = lc[0,:]
    ttm = lc[1,:]
    flux = lc[2,:]
    e_flux = lc[3,:]
    tnum = lc[4,:]

    lfi = []
# loop through all transit events
    for i in np.arange(len(tindex)):
        t0 = deltats[i]
        trans = tindex[i]        
        use = np.where(tnum == trans)
        t = ttm[use]
        f = flux[use]
        e = e_flux[use]

        val = np.abs(t0)
        ref = min(np.abs(np.max(t)/2.),np.abs(np.min(t)/2))

        if val > ref:
            arg = np.ones(len(t))*10.0
            lfi = np.append(lfi,np.sum(arg**2/2))
            return -np.inf
        else:
            
            
### Compute transit model for given input parameters ###
    # t0 is time of mid trans (relative to nominal ephemeris)
            tmodel,smoothmodel = compute_trans(rprs,duration,impact,t0,ldc)

    # compare data to model in only transit region
#    tlo = np.max(tmodel[np.where((tmodel < t0) & (smoothmodel > 0.99999999))])
#    thi = np.min(tmodel[np.where((tmodel > t0) & (smoothmodel > 0.99999999))])
            tlo = np.min(tmodel[1:-1])
            thi = np.max(tmodel[1:-1])
            ins  = np.zeros(len(t),dtype=bool)
            ininds = np.where((t >= tlo) & (t <= thi))
            ins[ininds] = True
    
    # interpolate model onto data values
            cfunc = sp.interpolate.interp1d(tmodel,smoothmodel,kind='linear')
            mfit = np.ones(len(t),dtype=float)
        
            mfit[ins] = cfunc(t[ins])
            
    # Log likelihood function
            lfi = np.append(lfi,-1*(mfit - f)**2/(2.0*e**2))

#        t = []
#        f = []
#        e = []
#        use = []
#        trans = []
#        t0 = []
        
# Log likelihood
    lf = np.sum(lfi)

    if jeffries == True:
        lf = lf - 2*np.log(rprs)

    return lf


def residuals(inp):
    """ 
    ----------------------------------------------------------------------
    resuduals:
    ----------
    Calculate residuals given an input model.
    ----------------------------------------------------------------------
    """

# Input parameters
    rprs = inp[0]
    duration = inp[1]
    impact = inp[2]
    t0 = inp[3]

    # Get raw data
    time = lc[0,:]
    flux = lc[2,:]
    e_flux = lc[3,:]
    tnum = lc[4,:]
    
# Limb darkening params
    c1       = inp[4][0]
    c2       = inp[4][1]
    if limb == 'nlin':
        c3 = inp[4][2]
        c4 = inp[4][3]
        ldc = [c1,c2,c3,c4]
    else:
        ldc = [c1,c2]

# Compute model with zero t0
    tmodel,smoothmodel = compute_trans(rprs,duration,impact,t0,ldc)

# Impose ephemeris offset in data folding    
    tfit = foldtime(time,period=pdata[0,4],t0=pdata[0,3])
    ffit = flux
    efit = e_flux

# Interpolate model at data values
    s = np.argsort(tfit)
    tfits = tfit[s]
    ffits = ffit[s]    
    cfunc = sp.interpolate.interp1d(tmodel,smoothmodel,kind='linear')
    mfit = cfunc(tfits)
    
# Residuals
    resid = ffits - mfit

    return resid




def plot_model(modelparams,short=False,tag='',markersize=5,smallmark=2,
               nbins=100,errorbars=False):

    """
    ----------------------------------------------------------------------
    plot_model:
    -----------
    Plot transit model given model params.

    ----------------------------------------------------------------------
    """

    import gc
# Check for output directory   
    directory = path+'MCMC/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get raw data
    time = lc[0,:]
    flux = lc[2,:]
    e_flux = lc[3,:]
    tnum = lc[4,:]

    # Get nominal period and ephemeris
    ephem0 = pdata[0,3]
    period0 = pdata[0,4]

    # Derive reference time from mid transit
    ttm = foldtime(time,period=period0,t0=ephem0)

    tplt = ttm
    fplt = flux
    eplt = e_flux

# Bin data (for visual aid)
    tfit, yvals, yerror = bin_lc(tplt,fplt,nbins=nbins)

    plt.figure(2,figsize=(11,8.5),dpi=300)
    plt.subplot(2,1,1)
    plt.plot(tplt,(fplt-1)*1e6,'.',color='gray',markersize=smallmark)
    if errorbars:
        plt.errorbar(tfit,(yvals-1)*1e6,yerr=yerror*1e6,fmt='o',color='blue',
                     markersize=markersize)
    else:
        plt.plot(tfit,(yvals-1)*1e6,'bo',markersize=markersize)

# Center transit in plot
    wid1 = np.max(tplt)
    wid2 = abs(np.min(tplt))
    if wid1 < wid2:
        wid = wid1
    else:
        wid = wid2
    plt.xlim(np.array([-1,1])*wid)

    ldc = modelparams[4]

# Get model, raw (unsmoothed) model

    tmodel,model,rawmodel = compute_trans(modelparams[0],modelparams[1],modelparams[2],\
                                              0.0,ldc,unsmooth=True)

#    tfull = tottofull(modelparams[1],modelparams[0],modelparams[2],modelparams[4])
 
    sig = rb.std((fplt-1)*1e6)
    med = np.median((fplt-1)*1e6)
#    min = max(np.min((model-1)*1e6),-4*sig)
# For deep transits
    min = np.min((model-1)*1e6)
#    yrange = np.array([min-3*sig,med+15*sig])
    yrange = np.array([min-4*sig,med+12*sig])
    plt.ylim(yrange) 
    locs,labels = plt.yticks()
    plt.yticks(locs, map(lambda x: "%.0f" % x, locs))
    plt.ylabel("ppm")

    plt.plot(tmodel,(rawmodel-1)*1e6,'g')
    plt.plot(tmodel,(model-1)*1e6,'r')
    plt.title(name+" Transit Fit")

    res = residuals(modelparams)

    dof = len(res) - ndim - 1.
    chisq = np.sum((res/eplt)**2)/dof

    rhoi = sdata[0,0]/(4./3.*np.pi*sdata[0,1]**3)

    aors = np.sqrt( ((1+modelparams[0])**2 - modelparams[2]**2)/
                    ((np.sin(np.pi*modelparams[1]/period0))**2) + modelparams[2]**2)    

#    aors    = 2.0 * np.sqrt(modelparams[0]) * modelparams[4] / \
#        (np.pi*np.sqrt(ttot**2 - tfull**2))
    

    rhostar =  3.0*np.pi/( c.G * (period0*24.*3600.)**2 ) * aors**3
    

    plt.annotate(r'$P$ = %.7f d' % period0, [0.5,0.87],horizontalalignment='center',
                 xycoords='figure fraction',fontsize='large')

    plt.annotate(r'$\chi^2_r$ = %.5f' % chisq, [0.87,0.85],horizontalalignment='right',
                 xycoords='figure fraction',fontsize='large')

    plt.annotate(r'$\rho_*$ (e=0) = %.3f' % rhostar, [0.87,0.81],horizontalalignment='right',
                 xycoords='figure fraction',fontsize='large')

    plt.annotate(r'$\rho_*$ (orig) = %.3f' % rhoi, [0.87,0.77],horizontalalignment='right',
                 xycoords='figure fraction',fontsize='large')
    val =  modelparams[1]*24.
    plt.annotate(r'$\tau_{\rm tot}$ = %.4f h' % val, [0.87,0.73],horizontalalignment='right',
                  xycoords='figure fraction',fontsize='large')

#    val = (modelparams[4]-pdata[0,4])*24.*3600.
#    plt.annotate(r'$\Delta P$ = %.3f s' % val, [0.15,0.85],
#                 xycoords='figure fraction',fontsize='large')

    plt.annotate(r'$R_p/R_*$ = %.5f' % modelparams[0], [0.15,0.85],
                  xycoords='figure fraction',fontsize='large')   

    plt.annotate('b = %.2f' % modelparams[2], [0.15,0.81],
                  xycoords='figure fraction',fontsize='large')

#    t0out = (modelparams[3])*24.*60.*60.
#    plt.annotate(r'$\Delta t_0$ = %.3f s' % t0out, [0.15,0.73],
#                  xycoords='figure fraction',fontsize='large')

   # Limb darkening parameters
    if limb == 'nlin':
        a1 = ldc[0]
        a2 = ldc[1]
        a3 = ldc[2]
        a4 = ldc[3]
        # need annotation for non-linear LD fits
    else:        
        q1in = ldc[0]
        q2in = ldc[1]
        u1, u2 = qtou(q1in,q2in,limb=limb)

        u1out = u1
        plt.annotate(r'$u_1$ = %.2f' % u1out, [0.15,0.59],
                     xycoords='figure fraction',fontsize='large')
        u2out = u2
        plt.annotate(r'$u_2$ = %.2f' % u2out, [0.15,0.55],
                     xycoords='figure fraction',fontsize='large')

    plt.subplot(2,1,2)
    s = np.argsort(tplt)
    plt.plot(tplt[s],res*1e6,'.',markersize=smallmark,color='gray')
    tres, yres, yreserr = bin_lc(tplt[s],res,nbins=nbins)
    if errorbars:
        plt.errorbar(tres,yres*1e6,yerr=yreserr*1e6,fmt='o',color='blue',markersize=markersize)
    else:
        plt.plot(tres,yres*1e6,'bo',markersize=markersize)

    plt.xlim(np.array([-1,1])*wid)
    sig = rb.std(res*1e6)
    med = np.median(res*1e6)
    plt.ylim(np.array([-5*sig,5*sig]))
    plt.axhline(y=0,color='r')
    locs,labels = plt.yticks()
    plt.yticks(locs, map(lambda x: "%.0f" % x, locs))
    plt.title("Residuals")
    plt.xlabel("Time from Mid Transit (days)")
    plt.ylabel("ppm")

    plt.savefig(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit'+tag+'.png', dpi=300)
    plt.clf()

    gc.collect()
    return



def plot_ttv_model(modelparams,short=False,tag='',markersize=5,smallmark=2,
               nbins=100,errorbars=False):

    """
    ----------------------------------------------------------------------
    plot_ttv_model:
    -----------
    Plot transit model given model params and folded on TTVs 

    ----------------------------------------------------------------------
    """
    import gc

    tplt,fplt,eplt = foldttv(maxval=False)


    # Get nominal period and ephemeris
    ephem0 = pdata[0,3]
    period0 = pdata[0,4]



# Bin data (for visual aid)
    tfit, yvals, yerror = bin_lc(tplt,fplt,nbins=nbins)

    plt.figure(222,figsize=(11,8.5),dpi=300)
    plt.subplot(2,1,1)
    plt.plot(tplt,(fplt-1)*1e6,'.',color='gray',markersize=smallmark)
    if errorbars:
        plt.errorbar(tfit,(yvals-1)*1e6,yerr=yerror*1e6,fmt='o',color='blue',
                     markersize=markersize)
    else:
        plt.plot(tfit,(yvals-1)*1e6,'bo',markersize=markersize)

# Center transit in plot
    wid1 = np.max(tplt)
    wid2 = abs(np.min(tplt))
    if wid1 < wid2:
        wid = wid1
    else:
        wid = wid2
    plt.xlim(np.array([-1,1])*wid)

    ldc = modelparams[4]

# Get model, raw (unsmoothed) model
    tmodel,model,rawmodel = compute_trans(modelparams[0],modelparams[1],modelparams[2],\
                                              0.0,modelparams[4],unsmooth=True)
    
    sig = rb.std((fplt-1)*1e6)
    med = np.median((fplt-1)*1e6)
#    min = max(np.min((model-1)*1e6),-4*sig)
# For deep transits
    min = np.min((model-1)*1e6)
#    yrange = np.array([min-3*sig,med+15*sig])
    yrange = np.array([min-4*sig,med+12*sig])
    plt.ylim(yrange) 
    locs,labels = plt.yticks()
    plt.yticks(locs, map(lambda x: "%.0f" % x, locs))
    plt.ylabel("ppm")

    plt.plot(tmodel,(rawmodel-1)*1e6,'g')
    plt.plot(tmodel,(model-1)*1e6,'r')
    plt.title(name+" TTV Transit Fit")

    res = residuals(modelparams)

    dof = len(res) - ndim - 1.
    chisq = np.sum((res/eplt)**2)/dof

    rhoi = sdata[0,0]/(4./3.*np.pi*sdata[0,1]**3)

    aors = np.sqrt( ((1+modelparams[0])**2 - modelparams[2]**2)/
                    ((np.sin(np.pi*modelparams[1]/period0))**2) + modelparams[2]**2)    

#    aors    = 2.0 * np.sqrt(modelparams[0]) * modelparams[4] / \
#        (np.pi*np.sqrt(ttot**2 - tfull**2))
    
    rhostar =  3.0*np.pi/( c.G * (period0*24.*3600.)**2 ) * aors**3
    

    plt.annotate(r'$P$ = %.7f d' % period0, [0.5,0.87],horizontalalignment='center',
                 xycoords='figure fraction',fontsize='large')

    plt.annotate(r'$\chi^2_r$ = %.5f' % chisq, [0.87,0.85],horizontalalignment='right',
                 xycoords='figure fraction',fontsize='large')
    plt.annotate(r'$\rho_*$ (e=0) = %.3f' % rhostar, [0.87,0.81],horizontalalignment='right',
                 xycoords='figure fraction',fontsize='large')
    plt.annotate(r'$\rho_*$ (orig) = %.3f' % rhoi, [0.87,0.77],horizontalalignment='right',
                 xycoords='figure fraction',fontsize='large')
#    val = (modelparams[4]-pdata[0,4])*24.*3600.
#    plt.annotate(r'$\Delta P$ = %.3f s' % val, [0.15,0.85],
#                 xycoords='figure fraction',fontsize='large')
    plt.annotate(r'$R_p/R_*$ = %.5f' % modelparams[0], [0.15,0.85],
                  xycoords='figure fraction',fontsize='large')   
    plt.annotate('b = %.2f' % modelparams[2], [0.15,0.81],
                  xycoords='figure fraction',fontsize='large')
    val =  modelparams[1]*24.
    plt.annotate(r'$\tau_{\rm tot}$ = %.4f h' % val, [0.15,0.77],
                  xycoords='figure fraction',fontsize='large')
#    t0out = (modelparams[3])*24.*60.*60.
#    plt.annotate(r'$\Delta t_0$ = %.3f s' % t0out, [0.15,0.73],
#                  xycoords='figure fraction',fontsize='large')

   # Limb darkening parameters
    if limb == 'nlin':
        a1 = ldc[0]
        a2 = ldc[1]
        a3 = ldc[2]
        a4 = ldc[3]
        # need annotation for non-linear LD fits
    else:        
        q1in = ldc[0]
        q2in = ldc[1]
        u1, u2 = qtou(q1in,q2in,limb=limb)

        u1out = u1
        plt.annotate(r'$u_1$ = %.2f' % u1out, [0.15,0.59],
                     xycoords='figure fraction',fontsize='large')
        u2out = u2
        plt.annotate(r'$u_2$ = %.2f' % u2out, [0.15,0.55],
                     xycoords='figure fraction',fontsize='large')

    plt.subplot(2,1,2)
    s = np.argsort(tplt)
    plt.plot(tplt[s],res*1e6,'.',markersize=smallmark,color='gray')
    tres, yres, yreserr = bin_lc(tplt[s],res,nbins=nbins)
    if errorbars:
        plt.errorbar(tres,yres*1e6,yerr=yreserr*1e6,fmt='o',color='blue',markersize=markersize)
    else:
        plt.plot(tres,yres*1e6,'bo',markersize=markersize)

    plt.xlim(np.array([-1,1])*wid)
    sig = rb.std(res*1e6)
    med = np.median(res*1e6)
    plt.ylim(np.array([-5*sig,5*sig]))
    plt.axhline(y=0,color='r')
    locs,labels = plt.yticks()
    plt.yticks(locs, map(lambda x: "%.0f" % x, locs))
    plt.title("Residuals")
    plt.xlabel("Time from Mid Transit (days)")
    plt.ylabel("ppm")

    plt.savefig(path+'TTVs/'+name+stag+ctag+rtag+lptag+ltag+'fit'+tag+'_ttvfold.png', dpi=300)
    plt.clf()

    gc.collect()

    return
                           


def fit_all(nwalkers=100,burnsteps=100,mcmcsteps=100,clobber=False):

    """
    fit_all:
    -----------
    Fit a single transit signal with specified mcmc parameters return 
    chains and log likelihood.

    """

    import emcee
    import robust as rb
    import time
    import constants as c

    nw = nwalkers
    bs = burnsteps
    mcs = mcmcsteps

    directory = path+'TTVs/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = path+'TTVs/chains/'
    if not os.path.exists(directory):
        os.makedirs(directory)

# Do not redo MCMC unless clobber flag
    done = os.path.exists(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_rchain.txt')
    if done == True and clobber == False:
        print "MCMC run already completed"
        return False,False


    print "Starting MCMC fitting routine for "+name

# Set up MCMC sampler
    print "... initializing emcee sampler"
    tstart = time.time()
    sampler = emcee.EnsembleSampler(nw, ndim, lnprob)

    twomin = 2./(24.*60.)
    onesec = 1./(24.*60.*60.)

    q1,q2 = utoq(sdata[0,3],sdata[0,4],limb=limb)

# Initial chain values
    # Rp/rs
    p0_init = np.array([np.random.uniform(0,2*pdata[0,0],nw)])
    # duration
    p0_init = np.append(p0_init,[np.random.uniform(0.5*pdata[0,1],2*pdata[0,1], nw)],axis=0)
    # impact
    p0_init = np.append(p0_init,[np.random.uniform(0.,0.99, nw)],axis=0)
    if limb == 'nlin':
        p0_init = np.append(p0_init,[sdata[0,3] + 
                                     np.random.normal(0.5,0.05,nw)],axis=0)
        p0_init = np.append(p0_init,[sdata[0,3] + 
                                     np.random.normal(0.5,0.05,nw)],axis=0)
        p0_init = np.append(p0_init,[sdata[0,3] +
                                     np.random.normal(0.5,0.05,nw)],axis=0)
        p0_init = np.append(p0_init,[sdata[0,3] +
                                     np.random.normal(0.5,0.05,nw)],axis=0)
        variables = ["Rp/R*","duration","impact","a1","a2","a3","a4"]
    else:
        # q1
        p0_init = np.append(p0_init,[np.random.uniform(0,1,nw)],axis=0)
        # q2
        p0_init = np.append(p0_init,[np.random.uniform(0,1,nw)],axis=0)
        variables = ["Rp/R*","duration","impact","q1","q2"]

    for i in np.arange(len(tindex)):
        p0_init = np.append(p0_init,[np.random.normal(0,twomin,nw)],axis=0)
        variables = np.append(variables,'t'+str(np.int(tindex[i])))
        
    p0 = np.array(p0_init).T

# Run burn-in
    print "... running burn-in with "+str(bs)+" steps and "+str(nw)+" walkers"
    pos, prob, state = sampler.run_mcmc(p0, bs)
    done_in(tstart)

# Calculate G-R scale factor for each variable
    Rs = GR_test(sampler.flatchain,nw,bs,variables=variables)

    for var in np.arange(ndim):
        acout = "Autocorrelation time for "+variables[var]+" = {0:0.3f}"
        print acout.format(sampler.acor[var])

    afout = "Mean acceptance fraction: {0:0.3f}"
    print afout.format(np.mean(sampler.acceptance_fraction))

# Save burn in stats
    burn = np.append(Rs,sampler.acor)
    burn = np.append(burn,sampler.acceptance_fraction)
    np.savetxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_burnstats.txt',burn)


# Reset sampler and run MCMC for reals
    print "... resetting sampler and running MCMC with "+str(mcs)+" steps"
    sampler.reset()
    posf, probf, statef = sampler.run_mcmc(pos, mcs)
    done_in(tstart)

 # Calculate G-R scale factor for each variable
    Rs = GR_test(sampler.flatchain,nw,mcs,variables=variables)

# Autocorrelation times
    for var in np.arange(ndim):
        acout = "Autocorrelation time for "+variables[var]+" = {0:0.3f}"
        print acout.format(sampler.acor[var])

    afout = "Final mean acceptance fraction: {0:0.3f}"
    print afout.format(np.mean(sampler.acceptance_fraction))

# Save final in stats
    stats = np.append(Rs,sampler.acor)
    stats = np.append(stats,sampler.acceptance_fraction)
    np.savetxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_finalstats.txt',stats)

#    print "Writing out final Gelman Rubin scale factors"
#    np.savetxt(path+'TTVs/'+name+stag+ctag+rtag+ltag+'fit_GR.txt',Rs)


    print "Writing MCMC chains to disk"
    rdist  = sampler.flatchain[:,0]
    ddist  = sampler.flatchain[:,1]
    bdist  = sampler.flatchain[:,2]
    c1dist = sampler.flatchain[:,3]
    c2dist = sampler.flatchain[:,4]
    np.savetxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_rchain.txt',rdist)
    np.savetxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_dchain.txt',ddist)
    np.savetxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_bchain.txt',bdist)
    np.savetxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_q1chain.txt',c1dist)
    np.savetxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_q2chain.txt',c2dist)
    istart = 5
    if limb == 'nlin':
        c3dist = sampler.flatchain[:,5]
        c4dist = sampler.flatchain[:,6]
        np.savetxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_q3chain.txt',c3dist)
        np.savetxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_q4chain.txt',c4dist)
        istart = 7
    for i in np.arange(len(tindex)):
        dist = sampler.flatchain[:,i+istart]
        np.savetxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_t'+str(np.int(tindex[i]))+'chain.txt',dist)

    lp = sampler.lnprobability.flatten()
    np.savetxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_lnprob.txt',lp)

    chains = sampler.flatchain

    return chains,lp



def GR_test(chains,nwalkers,nsamples,variables=False):

    """
    ----------------------------------------------------------------------
    GR_test:
    --------
    Compute the Gelman-Rubin scale factor for each variable given input
    flat chains
    ----------------------------------------------------------------------
    """


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

    



def bestvals(chains=False,lp=False,network=None,bindiv=20.,frac=0.01,nbins=50,rpmax=1,durmax=10,sigfac=5.0):

    """
    ----------------------------------------------------------------------
    bestvals:
    ---------
    Find the best values from the 1-d posterior pdfs return best values 
    and the posterior pdf for rp/rs
    ----------------------------------------------------------------------
    """
    
    import robust as rb
    from scipy.stats.kde import gaussian_kde

    Rstar = sdata[0,1]/c.Rsun
    e_Rstar = sdata[1,1]/c.Rsun

# Use supplied chains or read from disk
    if chains is not False:
        rprsdist = chains[:,0]
        ddist = chains[:,1]*24.
        bdist = chains[:,2]
        q1dist = chains[:,3]
        q2dist = chains[:,4]
        if limb == 'nlin':
            q3dist = chains[:,5]
            q4dist = chains[:,6]
    else:
        print '... importing MCMC chains'
        rprsdist = np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_rchain.txt')
        ddist = np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_dchain.txt')*24.
        bdist = np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_bchain.txt')

        q1dist = np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_q1chain.txt')
        q2dist = np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_q2chain.txt')
        if limb == 'nlin':
            q3dist = np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_q3chain.txt')
            q4dist = np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_q4chain.txt')

        print '... done importing chains!'


    if lp is False:
        lp = np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_lnprob.txt')
        
    nsamp = len(ddist)

# Plot initial fit
    if limb == 'nlin':
        plot_model([pdata[0,0],pdata[0,1],pdata[0,2],0.0,\
                        [sdata[0,3],sdata[0,4],sdata[0,5],sdata[0,6]]],tag='_ttv_mpfit',nbins=nbins)
    else:
        q1,q2 = utoq(sdata[0,3],sdata[0,4],limb=limb)
        plot_model([pdata[0,0],pdata[0,1],pdata[0,2],0.0,[q1,q2]],tag='_ttv_mpfit',nbins=nbins)


#  Get maximum likelihood values
    maxlike = np.max(lp)
    imax = np.array([i for i, j in enumerate(lp) if j == maxlike])
    if imax.size > 1:
        imax = imax[0]
    rprsval = np.float(rprsdist[imax])
    bval    = np.float(bdist[imax])
    dval    = np.float(ddist[imax])
    q1val    = np.float(q1dist[imax])
    q2val    = np.float(q2dist[imax])
    if limb == 'nlin':
        q3val = np.float(q3dist[imax])
        q4val = np.float(q4dist[imax])
    print ''
    print 'Best fit parameters for '+name
    plt.figure(4,figsize=(8.5,11),dpi=300)    
   
    upper = np.linspace(.69,.999,100)
    lower = upper-0.6827

    # Rp/Rstar
    rprsmin = np.min(rprsdist)
    rprsmax = np.max(rprsdist)
    rprssig = np.std(rprsdist)
    rprsmed = np.median(rprsdist)
    minval = max(rprsmed - sigfac*rprssig,rprsmin)
    maxval = min(rprsmed + sigfac*rprssig,rprsmax)
    rprss = np.linspace(rprsmin-rprssig,rprsmax+rprssig,1000)
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
    nb = np.ceil((np.max(rprsdist)-np.min(rprsdist)) / (np.abs(maxval-minval)/bindiv))
    
    rprsout = 'Rp/R*: max = {0:.5f}, med = {1:.5f}, mode = {2:.5f}, 1 sig int = {3:.5f}'
    print rprsout.format(rprsval, rprsmed, rprsmode, rprshi-rprslo)


    plt.subplot(3,1,1)
    plt.hist(rprsdist,bins=nb,normed=True)
#    plt.plot(rprss,rprspdf,color='c')
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
    dmed = np.median(ddist)
    minval = max(dmed - sigfac*dsig,dmin)
    maxval = min(dmed + sigfac*dsig,dmax)
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
    dmode = ds[np.argmax(dpdf)]
    nb = np.ceil((np.max(ddist)-np.min(ddist)) / (np.abs(maxval-minval)/bindiv))
    
    dout = 'Transit duration: max = {0:.4f}, med = {1:.4f}, mode = {2:.4f}, 1 sig int = {3:.4f}'
    print dout.format(dval, dmed, dmode, dhi-dlo)

    plt.subplot(3,1,2)    
    plt.hist(ddist,bins=nb,normed=True)
#    plt.plot(ds,dpdf,color='c')
#    plt.axvline(x=dval,color='r',linestyle='--')
#    plt.axvline(x=dlo,color='c',linestyle='--')
#    plt.axvline(x=dmed,color='c',linestyle='--')
#    plt.axvline(x=dhi,color='c',linestyle='--')
    plt.xlim([minval,maxval])
    plt.xlabel(r'Transit Duration (hours)')
    plt.ylabel(r'$dP/d\tau$')


   # Impact parameter                        
    bmin = np.min(bdist)
    bmax = np.max(bdist)
    bsig = np.std(bdist)
    bmed = np.median(bdist)
    minval = max(bmed - sigfac*bsig,bmin)
    maxval = min(bmed + sigfac*bsig,bmax)
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
    bmode = bss[np.argmax(bpdf)]
    nb = np.ceil((np.max(bdist)-np.min(bdist)) / (np.abs(maxval-minval)/bindiv))
    
    bout = 'Impact parameter: max = {0:.3f}, med = {1:.3f}, mode = {2:.3f}, 1 sig int = {3:.3f}'
    print bout.format(bval, bmed, bmode, bhi-blo)


    plt.subplot(3,1,3)
    plt.hist(bdist,bins=nb,normed=True)
#    plt.plot(bss,bpdf,color='c')
    plt.xlim([minval,maxval])
#    plt.axvline(x=bval,color='r',linestyle='--')
#    plt.axvline(x=blo,color='c',linestyle='--')
#    plt.axvline(x=bmed,color='c',linestyle='--')
#    plt.axvline(x=bhi,color='c',linestyle='--')
    plt.xlabel('Impact Parameter')
    plt.ylabel(r'$dP/db$')

    plt.subplots_adjust(hspace=0.4)

    plt.savefig(path+'TTVs/'+name+stag+ctag+rtag+ltag+'fit_params1.png', dpi=300)
    plt.clf()

# Second set of parameters

    plt.figure(5,figsize=(8.5,11),dpi=300)    

    plt.subplot(2,1,1)

 # q1
    q1min = np.min(q1dist)
    q1max = np.max(q1dist)
    q1sig = np.std(q1dist)
    q1med = np.median(q1dist)
    minval = max(q1med - sigfac*q1sig,q1min)
    maxval = min(q1med + sigfac*q1sig,q1max)
    q1s = np.linspace(-0.1,1.1,1000)
    q1_kde = gaussian_kde(q1dist)
    q1pdf = q1_kde(q1s)
    q1dist_c = np.cumsum(q1pdf)/np.sum(q1pdf)
    q1func = sp.interpolate.interp1d(q1dist_c,q1s,kind='linear')
    q1his = q1func(upper)
    q1los = q1func(lower)
    ind  = np.argmin(q1his-q1los)
    q1hi  = q1his[ind] 
    q1lo  = q1los[ind] 
    q1mode = q1s[np.argmax(q1pdf)]
    nb = np.ceil((np.max(q1dist)-np.min(q1dist)) / (np.abs(maxval-minval)/bindiv))
    
    q1out = 'Kipping q1 = max = {0:.3f}, med = {1:.3f}, mode = {2:.3f}, 1 sig int = {3:.3f}'
    print q1out.format(q1val,q1med, q1mode, q1hi-q1lo)

    
    plt.hist(q1dist,bins=nb,normed=True)
 #    plt.plot(q1s,q1pdf,color='c')
    q1min = np.float(np.sort(q1dist)[np.round(frac*nsamp)])
    q1max = np.float(np.sort(q1dist)[np.round((1-frac)*nsamp)])
    plt.xlim([q1min,q1max])
#    plt.axvline(x=q1val,color='r',linestyle='--')
#    plt.axvline(x=q1lo,color='c',linestyle='--')
#    plt.axvline(x=q1med,color='c',linestyle='--')
#    plt.axvline(x=q1hi,color='c',linestyle='--')
    plt.xlabel(r'$q_1$')
    plt.ylabel(r'$dP/q_1$')


# q2
    plt.subplot(2,1,2)

    q2min = np.min(q2dist)
    q2max = np.max(q2dist)
    q2sig = np.std(q2dist)
    q2med = np.median(q2dist)
    minval = max(q2med - sigfac*q2sig,q2min)
    maxval = min(q2med + sigfac*q2sig,q2max)
    q2s = np.linspace(-0.1,1.1,1000)
    q2_kde = gaussian_kde(q2dist)
    q2pdf = q2_kde(q2s)
    q2dist_c = np.cumsum(q2pdf)/np.sum(q2pdf)
    q2func = sp.interpolate.interp1d(q2dist_c,q2s,kind='linear')
    q2his = q2func(upper)
    q2los = q2func(lower)
    ind  = np.argmin(q2his-q2los)
    q2hi  = q2his[ind] 
    q2lo  = q2los[ind] 
    q2med = np.median(q2dist)
    q2mode = q2s[np.argmax(q2pdf)]
    nb = np.ceil((np.max(q2dist)-np.min(q2dist)) / (np.abs(maxval-minval)/bindiv))
    
    q2out = 'Kipping q2 = max = {0:.3f}, med = {1:.3f}, mode = {2:.3f}, 1 sig int = {3:.3f}'
    print q2out.format(q2val,q2med, q2mode, q2hi-q2lo)

    
    plt.hist(q2dist,bins=nb,normed=True)
#    plt.plot(q2s,q2pdf,color='c')
    q2min = np.float(np.sort(q2dist)[np.round(frac*nsamp)])
    q2max = np.float(np.sort(q2dist)[np.round((1-frac)*nsamp)])
    plt.xlim([q2min,q2max])
#    plt.axvline(x=q2val,color='r',linestyle='--')
#    plt.axvline(x=q2lo,color='c',linestyle='--')
#    plt.axvline(x=q2med,color='c',linestyle='--')
#    plt.axvline(x=q2hi,color='c',linestyle='--')
    plt.xlabel(r'$q_2$')
    plt.ylabel(r'$dP/q_2$')

    plt.subplots_adjust(hspace=0.55)

    plt.savefig(path+'TTVs/'+name+stag+ctag+rtag+ltag+'fit_ttv_params2.png',dpi=300)
    plt.clf()

   # calculate limb darkening parameters
    if limb == 'quad' or limb == 'sqrt':
        u1,u2 = qtou(q1val,q2val,limb=limb)
        ldc = [u1,u2]
    if limb == 'nlin':
        ldc = [q1val,q2val,q3val,q4val]
    
    plot_limb_curves(ldc=ldc,limbmodel=limb,write=True,network=network)

 
# vals
# 0 = rprsval
# 1 = duration in days
# 2 = impact parameter
# 3 = offset time in days (kepler time)
# 4 = period in days
# 5 = Kipping q1 val
# 6 = Kipping q2 val

    vals = [rprsval,dval/24.,bval,q1val,q2val]
    meds = [rprsmed,dmed/24.,bmed,q1med,q2med]
    modes = [rprsmode,dmode/24.,bmode,q1mode,q2mode]
    onesig = [rprshi-rprslo,(dhi-dlo)/24.,bhi-blo,q1hi-q1lo,q2hi-q2lo]
    if limb == 'nlin':
        np.append(vals,q3val,q4val)
        np.append(meds,q3med,q4med)
        np.append(modes,q3mode,q4mode)
        np.append(onesig,q3hi-q3lo,q4hi-q4lo)

    bestvals = [[vals],[meds],[modes],[onesig]]


# Get TTV solutions and plot
    index, ttime, ttvmed, ttvval, ttvhi, ttvlo = get_ttvs()
    plot_ttvs(ttime,ttvmed,ttvhi,ttvlo)


# Plot up final fit
    if limb == 'nlin':
        plot_ttv_model([vals[0],vals[1],vals[2],0.0,\
                        [vals[3],vals[4],vals[5],vals[6]]],tag='_ttv_MCMCfit')
    else:
        plot_ttv_model([vals[0],vals[1],vals[2],0.0,[vals[3],vals[4]]],tag='_ttv_MCMCfit')


# Create KDE of Rp/Rs distribution    
    print "Output Rp/Rs and Rp distributions"
    rpdf = np.array([rprss,rprspdf])
    np.savetxt(path+'TTVs/'+name+stag+ctag+rtag+ltag+'fit_ttv_RpRs.dat',rpdf.T)

# Create KDE of Rp distribution
    rstarpdf = np.random.normal(Rstar,e_Rstar,nsamp)
    rpdist = rstarpdf * rprsdist * c.Rsun/c.Rearth

    rp = np.linspace(np.min(rpdist)*0.5,np.max(rpdist)*1.5,1000)
    pdf_func = gaussian_kde(rpdist)
    rkde = pdf_func(rp)
    rpdf = np.array([rp,rkde])
    np.savetxt(path+'TTVs/'+name+stag+ctag+rtag+ltag+'fit_ttv_Rp.dat',rpdf.T)

    if limb == 'nlin':
        outstr = name+ ' %.5f  %.5f  %.5f  %.5f  %.3f  %.3f  %.3f  %.3f  %.2f  %.2f  %.2f  %.2f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f' %  (vals[0],meds[0],modes[0],onesig[0],vals[1],meds[1],modes[1],onesig[1],vals[2],meds[2],modes[2],onesig[2],vals[3],meds[3],modes[3],onesig[3],vals[4],meds[4],modes[4],onesig[4],vals[5],meds[5],modes[5],onesig[5],vals[6],meds[6],modes[6],onesig[6])
    else:
        outstr = name+ ' %.5f  %.5f  %.5f  %.5f  %.3f  %.3f  %.3f  %.3f  %.2f  %.2f  %.2f  %.2f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f' % (vals[0],meds[0],modes[0],onesig[0],vals[1],meds[1],modes[1],onesig[1],vals[2],meds[2],modes[2],onesig[2],vals[3],meds[3],modes[3],onesig[3],vals[4],meds[4],modes[4],onesig[4])

    f = open(path+'TTVs/'+name+stag+ctag+ltag+'fit_ttv_fitparams.txt','w')

    f.write(outstr+'\n')
    f.closed

    return bestvals,rpdf


def get_refindex(t0,duration):

    time = lc[0,:]
    indices = lc[4,:]
    inds, = np.where((time > t0 - duration) & (time < t0 + duration))
    index = np.median(indices[inds])

    return index


def get_ttvs(thin=10,bindiv=30):

    ephem0 = pdata[0,3]
    period0 = pdata[0,4]
    duration0 = pdata[0,2]
    # TTV time in BJD is (index - refindex)*period0 + ephem0 + tval
    refindex = get_refindex(ephem0,duration0)
    
    upper = np.linspace(.69,.999,100)
    lower = upper-0.6827
    print "... loading ln prob chain"
    lp = np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_lnprob.txt')
    maxlike = np.max(lp)
    imax = np.array([i for i, j in enumerate(lp) if j == maxlike])
    if imax.size > 1:
        imax = imax[0]

    ttime = []
    ttvmed = []
    ttvval = []
    ttvhi = []
    ttvlo = []
    index = []

    files = glob.glob(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_t[1-9]*chain.txt')
    for f in files:
        tindex = np.int(np.float(f.split('chain.txt')[0].split('ttv_t')[-1]))
        print "... loading chain for transit index "+str(tindex)
        tdist = np.loadtxt(f)
        tval = tdist[imax]
        tdist = tdist[0::thin]
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
        nb = np.ceil((np.max(tdist)-np.min(tdist)) / (np.abs(thi-tlo)/bindiv))
        
        tout = 'Delta t0: max = {0:.3f}, med = {1:.3f}, mode = {2:.3f}, 1 sig int = {3:.3f}'
        print tout.format(tval, tmed, tmode, thi-tlo)
        
        # ttime is expected time according to linear ephemeris
        ttime = np.append(ttime,(tindex - refindex)*period0 + ephem0)
        ttvmed = np.append(ttvmed,tmed)
        ttvval = np.append(ttvval,tval)
        ttvhi = np.append(ttvhi,thi)
        ttvlo = np.append(ttvlo,tlo)
        index = np.append(index,tindex)

        # Write out TTVs in appropriate directory
        sz = len(ttime)
        s = np.argsort(index)
        data = np.zeros((sz,6))
        data[:,0] = index[s]
        data[:,1] = ttime[s]
        data[:,2] = ttvmed[s]
        data[:,3] = ttvval[s]
        data[:,4] = ttvhi[s]
        data[:,5] = ttvlo[s]
        np.savetxt(path+'TTVs/'+name+stag+ctag+rtag+ltag+'_MCMC_TTVs.dat',data)

    return index, ttime, ttvmed, ttvval, ttvhi, ttvlo


def plot_ttvs(ttime,ttv,ttvhi,ttvlo):
    import matplotlib as mpl
    mpl.rc('axes', linewidth=2)
    fs = 15
    lw = 2
    ms = 12

    ephem0 = pdata[0,3]
    period0 = pdata[0,4]

    yerrp = np.abs(ttvhi - ttv)
    yerrn = np.abs(ttv - ttvlo)
    
    plt.figure(223,figsize=(11,8.5),dpi=300)
    ax = plt.subplot(111)
    ax.errorbar(ttime,ttv*24.*60.0,yerr=[yerrn*24.*60.0,yerrp*24.*60.0],fmt='o',linewidth=lw)
    ax.set_xlabel(r'BJD - 2454833.0',fontsize=fs)
    ax.set_ylabel(r'$(O-C)$ (min)',fontsize=fs)
    ax.set_title('MCMC TTVs for '+name,fontsize=fs)
    ax.yaxis.set_tick_params(length=10, width=lw, labelsize=fs)
    ax.xaxis.set_tick_params(length=10, width=lw, labelsize=fs)
 
    sig = np.std(ttv)*24.0*60.0
    minval = np.min(ttv-yerrn)*24.0*60.0 - sig
    maxval = np.max(ttv+yerrp)*24.0*60.0 + 3*sig
    ax.set_ylim(minval,maxval)
    plt.annotate(r'$P$ = %.6f d' % period0, [0.15,0.85],
                  xycoords='figure fraction',fontsize='large')
    plt.annotate(r'$t_0$ = %.6f' % ephem0, [0.15,0.81],
                  xycoords='figure fraction',fontsize='large')

    ax.axhline(y=0,color='r',linestyle='--',linewidth=lw)
    plt.savefig(path+'TTVs/'+name+stag+ctag+rtag+ltag+'fit_TTVs.png',dpi=300)
    plt.clf()
    mpl.rc('axes', linewidth=1)
    return


def distparams(dist):

    vals = np.linspace(np.min(dist)*0.5,np.max(dist)*1.5,1000)
    kde = gaussian_kde(dist)
    pdf = b_kde(vals)
    dist_c = np.cumsum(pdf)/np.sum(pdf)
    func = sp.interpolate.interp1d(dist_c,vals,kind='linear')
    lo = np.float(func(math.erfc(1./np.sqrt(2))))
    hi = np.float(func(math.erf(1./np.sqrt(2))))
    med = np.float(func(0.5))
    mode = vals[np.argmax(pdf)]

    disthi = np.linspace(.684,.999,100)
    distlo = distthi-0.6827
    disthis = func(distthi)
    distlos = func(disttlo)
    
    interval = np.min(rhohis-rholos)

    return med,mode,interval,lo,hi



def triangle_plot(chains=False,lp=False,thin=False,frac=0.001):
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import robust as rb
    import sys
    import math
    from scipy.stats.kde import gaussian_kde
    import pdb
    import time
    
    bindiv = 10
    
    tmaster = time.time()
    print "Reading in MCMC chains"
    if chains is not False:
        rprsdist = chains[:,0]
        ddist = chains[:,1]*24.
        bdist = chains[:,2]
        tdist = chains[:,3]*24.*60.*60.
        pdist = (chains[:,4]-pdata[0,4])*24.*3600.
        q1dist = chains[:,5]
        q2dist = chains[:,6]
    else:
        print '... importing MCMC chains'
        rprsdist = np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_rchain.txt')
        ddist = np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_dchain.txt')*24.
        bdist = np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_bchain.txt')
        tdist = (np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_t0chain.txt'))*24.*60.*60.
        pdist = (np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_pchain.txt')-pdata[0,4])*24.*3600.
        q1dist = np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_q1chain.txt')
        q2dist = np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_q2chain.txt')
        print '... done importing chains!'

    if lp is False:
        lp    = np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_ttv_lnprob.txt')

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
    pval = np.float(pdist[imax])
    q1val = np.float(q1dist[imax])
    q2val = np.float(q2dist[imax])

    if thin:
        rprsdist = rprsdist[0::thin]
        ddist = ddist[0::thin]
        tdist = tdist[0::thin]
        bdist = bdist[0::thin]
        pdist = pdist[0::thin]
        q1dist = q1dist[0::thin]
        q2dist = q2dist[0::thin]

    print " "
    print "Starting grid of posteriors..."
    plt.figure(6,figsize=(8.5,8.5))
    nx = 7
    ny = 7

    gs = gridspec.GridSpec(nx,ny,wspace=0.1,hspace=0.1)
    print " "
    print "... top plot of first column"
    tcol = time.time()
    top_plot(rprsdist,gs[0,0],val=rprsval,frac=frac)
    done_in(tcol)
    t = time.time()
    print "... first column"
    column_plot(rprsdist,ddist,gs[1,0],val1=rprsval,val2=dval,ylabel=r'$\tau$ (h)',frac=frac)
    done_in(t)
    column_plot(rprsdist,tdist,gs[2,0],val1=rprsval,val2=tval,ylabel=r'$\Delta t_0$ (s)',frac=frac)
    column_plot(rprsdist,pdist,gs[3,0],val1=rprsval,val2=pval,ylabel=r'$\Delta P$ (s)',frac=frac)
    column_plot(rprsdist,q1dist,gs[4,0],val1=rprsval,val2=q1val,ylabel=r'$q_1$',frac=frac)
    column_plot(rprsdist,q2dist,gs[5,0],val1=rprsval,val2=q2val,ylabel=r'$q_2$',frac=frac)
    corner_plot(rprsdist,bdist,gs[6,0],val1=rprsval,val2=bval,\
                xlabel=r'$R_p/R_{*}$',ylabel=r'$b$',frac=frac)
    print "First column: "
    done_in(tcol)

    print "... second column"
    t2 = time.time()
    top_plot(ddist,gs[1,1],val=dval,frac=frac)    
    middle_plot(ddist,tdist,gs[2,1],val1=dval,val2=tval,frac=frac)
    middle_plot(ddist,pdist,gs[3,1],val1=pval,val2=pval,frac=frac)
    middle_plot(ddist,q1dist,gs[4,1],val1=q1val,val2=q1val,frac=frac)
    middle_plot(ddist,q2dist,gs[5,1],val1=q2val,val2=q2val,frac=frac)
    row_plot(ddist,bdist,gs[6,1],val1=dval,val2=bval,xlabel=r'$\tau$ (h)',frac=frac)
    done_in(t2)

    print "... third column"
    t3 = time.time()
    top_plot(tdist,gs[2,2],val=tval,frac=frac)    
    middle_plot(tdist,pdist,gs[3,2],val1=tval,val2=pval,frac=frac)
    middle_plot(tdist,q1dist,gs[4,2],val1=tval,val2=q1val,frac=frac)
    middle_plot(tdist,q2dist,gs[5,2],val1=tval,val2=q2val,frac=frac)
    row_plot(tdist,bdist,gs[6,2],val1=tval,val2=bval,xlabel=r'$\Delta t_0$ (s)',frac=frac)
    done_in(t3)

    print "... fourth column"
    t4 = time.time()
    top_plot(pdist,gs[3,3],val=pval,frac=frac)    
    middle_plot(pdist,q1dist,gs[4,3],val1=pval,val2=q1val,frac=frac)
    middle_plot(pdist,q2dist,gs[5,3],val1=pval,val2=q2val,frac=frac)
    row_plot(pdist,bdist,gs[6,3],val1=tval,val2=bval,xlabel=r'$\Delta P$ (s)',frac=frac)
    done_in(t4)


    print "... fifth column"
    t5 = time.time()
    top_plot(q1dist,gs[4,4],val=q1val,frac=frac)    
    middle_plot(q1dist,q2dist,gs[5,4],val1=q1val,val2=q2val,frac=frac)
    row_plot(q1dist,bdist,gs[6,4],val1=q1val,val2=bval,xlabel=r'$q_1$',frac=frac)
    done_in(t5)

    print "... sixth column"
    t6 = time.time()
    top_plot(q2dist,gs[5,5],val=q2val,frac=frac)    
    row_plot(q2dist,bdist,gs[6,5],val1=q2val,val2=bval,xlabel=r'$q_2$',frac=frac)
    done_in(t6)
  

    print "... last plot"
    t7 = time.time()
    top_plot(bdist,gs[6,6],val=bval,xlabel=r'$b$',frac=frac)    
    done_in(t7)


    print "Saving output figures"
    plt.savefig(path+'TTVs/'+name+stag+ctag+rtag+ltag+'fit_triangle.png', dpi=300)
    plt.savefig(path+'TTVs/'+name+stag+ctag+rtag+ltag+'fit_triangle.eps', format='eps', dpi=600)

    print "Procedure finished!"
    done_in(tmaster)


    return



def top_plot(dist,position,val=False,frac=0.001,bindiv=10,aspect=1,xlabel=False):
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.stats.kde import gaussian_kde
    from scipy.interpolate import interp1d
    import time

    len = np.size(dist)
    min = np.float(np.sort(dist)[np.round(frac*len)])
    max = np.float(np.sort(dist)[np.round((1.-frac)*len)])
    dists = np.linspace(np.min(dist)*0.5,np.max(dist)*1.5,1000)
    kde = gaussian_kde(dist)
    pdf = kde(dists)
    cumdist = np.cumsum(pdf)/np.sum(pdf)
    func = interp1d(cumdist,dists,kind='linear')
    lo = np.float(func(math.erfc(1./np.sqrt(2))))
    hi = np.float(func(math.erf(1./np.sqrt(2))))
    nb = np.ceil((np.max(dist)-np.min(dist)) / (np.abs(hi-lo)/bindiv))
    ax = plt.subplot(position)
    plt.hist(dist,bins=nb,normed=True,color='black')
    if not xlabel: 
        ax.set_xticklabels(())
    ax.set_yticklabels(())
    ax.set_xlim(min,max)
    xlimits = ax.get_xlim()
    ylimits = ax.get_ylim()
    ax.set_aspect(abs((xlimits[1]-xlimits[0])/(ylimits[1]-ylimits[0]))/aspect)
    if val:
        plt.axvline(x=val,color='w',linestyle='--',linewidth=2)
    if xlabel:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(8) 
            tick.label.set_rotation('vertical')
        ax.set_xlabel(xlabel,fontsize=12)
 
    return



def column_plot(dist1,dist2,position,val1=False,val2=False,frac=0.001,ylabel=None):
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.stats.kde import gaussian_kde
    from scipy.interpolate import interp1d
    import pdb
    import time


    len1 = np.size(dist1)
    min1 = np.float(np.sort(dist1)[np.round(frac*len1)])
    max1 = np.float(np.sort(dist1)[np.round((1.-frac)*len1)])

    len2 = np.size(dist2)
    min2 = np.float(np.sort(dist2)[np.round(frac*len2)])
    max2 = np.float(np.sort(dist2)[np.round((1.-frac)*len2)])

    aspect = (max1-min1)/(max2-min2)
    X, Y = np.mgrid[min1:max1:100j, min2:max2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([dist1, dist2])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
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
        tick.label.set_fontsize(8) 
    ax.set_ylabel(ylabel,fontsize=12)

    return



def row_plot(dist1,dist2,position,val1=False,val2=False,frac=0.001,xlabel=None):
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.stats.kde import gaussian_kde
    from scipy.interpolate import interp1d
    import pdb
    import time


    len1 = np.size(dist1)
    min1 = np.float(np.sort(dist1)[np.round(frac*len1)])
    max1 = np.float(np.sort(dist1)[np.round((1.-frac)*len1)])
    
    len2 = np.size(dist2)
    min2 = np.float(np.sort(dist2)[np.round(frac*len2)])
    max2 = np.float(np.sort(dist2)[np.round((1.-frac)*len2)])
    
    aspect = (max1-min1)/(max2-min2)
    X, Y = np.mgrid[min1:max1:100j, min2:max2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([dist1, dist2])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
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
        tick.label.set_fontsize(8) 
        tick.label.set_rotation('vertical')
    ax.set_xlabel(xlabel,fontsize=12)
    return


def middle_plot(dist1,dist2,position,val1=False,val2=False,frac=0.001):
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.stats.kde import gaussian_kde
    from scipy.interpolate import interp1d
    import pdb
    import time



    len1 = np.size(dist1)
    min1 = np.float(np.sort(dist1)[np.round(frac*len1)])
    max1 = np.float(np.sort(dist1)[np.round((1.-frac)*len1)])

    len2 = np.size(dist2)
    min2 = np.float(np.sort(dist2)[np.round(frac*len2)])
    max2 = np.float(np.sort(dist2)[np.round((1.-frac)*len2)])

    aspect = (max1-min1)/(max2-min2)
    X, Y = np.mgrid[min1:max1:100j, min2:max2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([dist1, dist2])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
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
    return



def corner_plot(dist1,dist2,position,val1=False,val2=False,\
                frac=0.001,xlabel=None,ylabel=None):
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.stats.kde import gaussian_kde
    from scipy.interpolate import interp1d
    import pdb
    import time

    len1 = np.size(dist1)
    min1 = np.float(np.sort(dist1)[np.round(frac*len1)])
    max1 = np.float(np.sort(dist1)[np.round((1.-frac)*len1)])

    len2 = np.size(dist2)
    min2 = np.float(np.sort(dist2)[np.round(frac*len2)])
    max2 = np.float(np.sort(dist2)[np.round((1.-frac)*len2)])

    aspect = (max1-min1)/(max2-min2)
    X, Y = np.mgrid[min1:max1:100j, min2:max2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([dist1, dist2])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    ax = plt.subplot(position)
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,aspect=aspect,\
              extent=[min1, max1, min2, max2],origin='upper')
    clev = np.exp(np.log(np.max(Z))-0.5)
    cset = ax.contour(X,Y,Z,[clev],colors='w',linewidth=5,linestyles='dotted')
#    ax.plot(val1,val2, 'wx', markersize=3)
    ax.set_xlim(min1, max1)
    ax.set_ylim(min2, max2)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
        tick.label.set_rotation('vertical')
    ax.set_xlabel(xlabel,fontsize=12)
    ax.set_ylabel(ylabel,fontsize=12)

    return

   


def get_limb_curve(ldc,limbmodel='quad'):

    """
    get_limb_curve:
    ---------------
    Function to compute limb darkening curve given models and parameters

    """

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



def plot_limb_curves(ldc=False,limbmodel='quad',write=False,network=None):

    """
    
    plot_limb_curves:
    -----------------
    

    """
    import get_limb_coeff as glc
    import constants as c

    if network == 'koi':
        net = None
    else:
        net = network


    Mstar = sdata[0,0]
    Rstar = sdata[0,1]
    Tstar = sdata[0,2]

    loggstar = np.log10( c.G * Mstar / Rstar**2. )
    
    a1,a2,a3,a4 = glc.get_limb_coeff(Tstar,loggstar,limb='nlin',interp='nearest',network=net)
    a,b = glc.get_limb_coeff(Tstar,loggstar,limb='quad',interp='nearest',network=net)
    c,d = glc.get_limb_coeff(Tstar,loggstar,limb='sqrt',interp='nearest',network=net)

    thetaq,Imuq = get_limb_curve([a,b],limbmodel='quad')
    thetas,Imus = get_limb_curve([c,d],limbmodel='sqrt')
    thetan,Imun = get_limb_curve([a1,a2,a3,a4],limbmodel='nlin')
    if ldc:
        thetain,Iin = get_limb_curve(ldc,limbmodel=limbmodel)

    if write:
        plt.figure(1,figsize=(11,8.5),dpi=300)
    else:
        plt.ion()
        plt.figure()
    plt.plot(thetaq,Imuq,label='Quadratic LD Law')
    plt.plot(thetas,Imus,label='Root-Square LD Law')
    plt.plot(thetan,Imun,label='Non-Linear LD Law')

    if ldc:
        if limbmodel == 'nlin':
            label = '{0:0.2f}, {1:0.2f}, {2:0.2f}, {3:0.2f} ('+limbmodel+')'
            plt.plot(thetain,Iin,label=label.format((ldc[0],ldc[1],ldc[2],ldc[3])))
        else:
            label = '%.2f, ' % ldc[0] + '%.2f' % ldc[1]+' ('+limbmodel+')'
            plt.plot(thetain,Iin,label=label.format((ldc[0],ldc[1])))

    plt.ylim([0,1.0])
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel(r"$\theta$ (degrees)",fontsize=18)
    plt.ylabel(r"$I(\theta)/I(0)$",fontsize=18)
    plt.title("KOI-"+str(koi)+" limb darkening",fontsize=20)
    plt.legend(loc=3)
    plt.annotate(r'$T_{\rm eff}$ = %.0f K' % sdata[0][2], [0.86,0.82],horizontalalignment='right',
                 xycoords='figure fraction',fontsize='large')
    plt.annotate(r'$\log(g)$ = %.2f' % loggstar, [0.86,0.77],horizontalalignment='right',
                 xycoords='figure fraction',fontsize='large')
   
    if write:
        plt.savefig(path+'TTVs/'+str(koi)+stag+ctag+rtag+'_'+limbmodel+'fit_ttv.png')
        plt.clf()
    else:
        plt.ioff()

    return


def thin_chains(koi,planet,thin=10,short=False,network=None,clip=False,limbmodel='quad',rprior=False):
    
    lc,pdata,sdata = get_koi_info(koi,planet,short=short,network=network,\
                                      clip=clip,limbmodel=limbmodel,rprior=rprior)
    t = time.time()
    print 'Importing MCMC chains'
    rprsdist = np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_rchain.txt')
    ddist    = np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_dchain.txt')*24.
    bdist    = np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_bchain.txt')
    tdist    = (np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_t0chain.txt')) * 24.0 * 3600.0 + pdata[0,3]
    pdist    = (np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_pchain.txt')-pdata[0,4])*24.*3600.
    q1dist   = np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_q1chain.txt')
    q2dist   = np.loadtxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_q2chain.txt')
    done_in(t)

    rprsdist = rprsdist[0::thin]
    ddist    = ddist[0::thin]
    tdist    = tdist[0::thin]
    bdist    = bdist[0::thin]
    pdist    = pdist[0::thin]
    q1dist   = q1dist[0::thin]
    q2dist   = q2dist[0::thin]

    t = time.time()
    print 'Exporting thinned chains'
    np.savetxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_thin_rchain.txt',rdist)
    np.savetxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_thin_dchain.txt',ddist)
    np.savetxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_thin_bchain.txt',bdist)
    np.savetxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_thin_t0chain.txt',tdist)
    np.savetxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_thin_pchain.txt',pdist)
    np.savetxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_thin_q1chain.txt',q1dist)
    np.savetxt(path+'TTVs/chains/'+name+stag+ctag+rtag+ltag+'fit_thin_q2chain.txt',q2dist)
    done_in(t)
    
    return



def done_in(tmaster):
    import time
    import numpy as np

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
# do_full_koi:
#    pick a KOI and do all fits!
#----------------------------------------------------------------------
            

def do_full_koi(koi,nwalkers=1000,burnsteps=1000,mcmcsteps=2000,clobber=False,network=None,thin=False):
    
    import numpy as np
    import time
    import os
    import constants as c

    print "starting KOI-"+str(koi)

    for planet in np.arange(1,9):
        check1 = isthere(koi,planet,short=True,network=network)
        check2 = isthere(koi,planet,network=network)
        check3 = isthere(koi,planet,network=network)
        if check3 and lcnb:
            lc,pdata,sdata = get_koi_info(koi,planet,network=network)
            chains,lp = fit_all(nwalkers=nwalkers,burnsteps=burnsteps,\
                                    mcmcsteps=mcmcsteps,clobber=clobber)
            fit,rpdf = bestvals(chains=chains)
            triangle_plot(thin=thin)
        if check2:
            lc,pdata,sdata = get_koi_info(koi,planet,network=network)
            chains,lp = fit_all(nwalkers=nwalkers,burnsteps=burnsteps,\
                                    mcmcsteps=mcmcsteps,clobber=clobber)
            fit,rpdf = bestvals(chains=chains)
            triangle_plot(thin=thin)
        if check1:
            lc,pdata,sdata = get_koi_info(koi,planet,short=True,network=network)
            chains,lp = fit_all(nwalkers=nwalkers,burnsteps=burnsteps,\
                                    mcmcsteps=mcmcsteps,clobber=clobber)
            fit,rpdf = bestvals(chains=chains)
            triangle_plot(thin=thin)
        else: pass
 
    return





#----------------------------------------------------------------------
# batchfit:
#    routine allowing the batch fitting of several KOIs
#----------------------------------------------------------------------

def batchfit(start=0,stop=16,nwalkers=1000,burnsteps=1000,mcmcsteps=1000,
             clobber=False,thin=10,frac=0.001,doplots=False,short=False,
             list=False,network=None,clip=True,rprior=True):

    import numpy as np
    import time
    import os
    import constants as c
    
    data = np.loadtxt('ttv_fit_list_validated.dat',dtype='string')
    tmp = np.array(data[:,0])
    kois = np.array([int(n) for n in tmp])
    
    tmp2 = np.array(data[:,1])
    planet = np.array([int(n) for n in tmp2])
    
    valid = np.array(data[:,2])
    inds, = np.where((valid == 'v') | (valid == 'e'))
    
    kois = kois[inds]
    planet = planet[inds]

    nkois = len(kois)

    if list:
        for number in np.arange(nkois):
            print number,kois[number],planet[number]
        return


    tstart = time.time()
    for i in np.arange(start,stop+1):
        koi = kois[i]
        pl = planet[i]
        do_fit(koi,pl,nwalkers=nwalkers,burnsteps=burnsteps,
               mcmcsteps=mcmcsteps,network=network,clobber=clobber,
               thin=thin,frac=frac,doplots=doplots,short=short,clip=clip)
        
    return



def do_fit(koi,planet,nwalkers=100,burnsteps=100,mcmcsteps=100,clobber=False,
                network=None,thin=20,frac=0.001,clip=True,limbmodel='quad',doplots=False,
                short=False,rprior=True,notriangle=True,rpmax=0.5,durmax=10,bindiv=30):
    
    import numpy as np
    import time
    import os
    import constants as c

    print ""
    print ""    
    print "Starting MCMC fitting for KOI-"+str(koi)+".0"+str(planet)

    check1 = isthere(koi,planet,short=short,network=network,clip=clip)
    if check1:
        lc,pdata,sdata = get_koi_info(koi,planet,short=short,network=network,\
                                          clip=clip,limbmodel=limbmodel,rprior=rprior)
        if nwalkers < 2.0*(ntrans+6):
            print "Using less than the recommended number of walkers"
            return

        chains,lp = fit_all(nwalkers=nwalkers,burnsteps=burnsteps,mcmcsteps=mcmcsteps,
                               clobber=clobber)
        if doplots != False:
            fit,rpdf = bestvals(chains=chains,frac=frac,rpmax=rpmax,durmax=durmax,bindiv=bindiv)
            if notriangle != True:
                triangle_plot(thin=thin,frac=frac)

    else: pass
 
    return

