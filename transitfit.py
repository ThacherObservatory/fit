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
import gc
from scipy.stats.kde import gaussian_kde
from length import length

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
    return the number of planets in KOI system does not depend on any global 
    variables
    
    example:
    In[1]: npl,names = numplanet(952,network=None)

    """
    from length import length

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
  
    fname = path+str(koiin)+'_long.out'
    raw = np.loadtxt(fname,delimiter=',',usecols=[0])
    npl = length(raw)
    names = []
    if npl == 1:
        names.append("%.2f" % (raw))
    else:
        for i in xrange(0,npl):
            names.append("%.2f" % (raw[i]))

    return npl,names



def get_limb_coeff(Tstar,loggstar,filter='Kp',plot=False,network=None,limb='quad',interp='linear'):
    """ 
    ----------------------------------------------------------------------
    get_limb_coeff:
    -----------
    return the LD coefficients from Claret et al. (2012) for a give Teff and
    logg. 

    Warning:
    --------
    limb = 'nlin' does not currently work!!!
    need to download and incorporate another data file.

    """

    from scipy.interpolate import griddata
    import pylab as pl
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    from scipy.interpolate import RectBivariateSpline as bspline

    global ldc1func,ldc2func,ldc3func,ldc4func

    plt.rcdefaults()

# Account for gap in look up tables between 4800 and 5000K
    if (Tstar > 4800 and Tstar <= 4900):
        Tstar = 4800
    if (Tstar > 4900 and Tstar < 5000):
        Tstar = 5000
    
# Choose proper file to read
    if limb == 'nlin':
        skiprows = 49
        filtcol = 8
        metcol = 9
        mercol = 10
        col1 = 4
        col2 = 5
        col3 = 6
        col4 = 7
        if (Tstar <= 4800):
            file = 'Claret_cool_nlin.dat'
        if (Tstar >= 5000):
            file = 'Claret_hot_nlin.dat'
    else:
        skiprows = 58
        filtcol = 4
        metcol = 5
        mercol = 6
        col1 = 9
        col2 = 10
        col3 = 11
        col4 = 12
        if (Tstar <= 4800):
            file = 'Claret_cool.dat'
        if (Tstar >= 5000):
            file = 'Claret_hot.dat'
                
    if network == 'gps':
        path = '/home/jswift/Mdwarfs/'
    if network == 'astro':
        path = '/home/jswift/Mdwarfs/'
    if network == None:
        path = '/Users/jonswift/Astronomy/Exoplanets/TransitFits/'

    limbdata = np.loadtxt(path+file,dtype='string', delimiter='|',skiprows=skiprows)

    logg = limbdata[:,0].astype(np.float).flatten()
    Teff = limbdata[:,1].astype(np.float).flatten()
    Z = limbdata[:,2].astype(np.float).flatten()
    xi = limbdata[:,3].astype(np.float).flatten()
    filt = np.char.strip(limbdata[:,filtcol].flatten())
    method = limbdata[:,metcol].flatten()
    avec = limbdata[:,col1].astype(np.float).flatten()
    bvec = limbdata[:,col2].astype(np.float).flatten()
    cvec = limbdata[:,col3].astype(np.float).flatten()
    dvec = limbdata[:,col4].astype(np.float).flatten()

# Select out the limb darkening coefficients
#    inds = np.where((filt == 'Kp') & (Teff == 3000) & (logg == 5.0) & (method == 'L'))

    idata, = np.where((filt == filter) & (method == 'L'))
    
    npts = idata.size

    uTeff = np.unique(Teff[idata])
    ulogg = np.unique(logg[idata])
    
#    agrid0 = np.zeros((len(uTeff),len(ulogg)))
#    for i in np.arange(len(uTeff)):
#        for ii in np.arange(len(ulogg)):
#            ind, = np.where((Teff[idata] == uTeff[i]) & (logg[idata] == ulogg[ii]))
#            val = avec[idata[ind]]
#            if len(val) > 0:
#                agrid0[i,ii] = val[0]
#            else:
#                pass #pdb.set_trace()


    locs = np.zeros(2*npts).reshape(npts,2)
    locs[:,0] = Teff[idata].flatten()
    locs[:,1] = logg[idata].flatten()
    
    vals = np.zeros(npts)
    vals[:] = avec[idata]

    agrid = np.zeros((len(uTeff),len(ulogg)))
    for i in np.arange(len(uTeff)):
        for ii in np.arange(len(ulogg)):
            eval  = np.array([uTeff[i],ulogg[ii]]).reshape(1,2)
            val = griddata(locs,vals,eval,method='cubic')
            if len(val) > 0:
                agrid[i,ii] = val[0]
            else:
                pass #pdb.set_trace()

    ldc1func = bspline(uTeff, ulogg, agrid, kx=1, ky=1, s=0)    
    aval = ldc1func(Tstar,loggstar)[0][0]

    if plot:      
        plt.figure(1)
        plt.clf()
        plt.imshow(agrid,interpolation='none',
                   extent=[np.min(ulogg),np.max(ulogg),np.min(uTeff),np.max(uTeff)],
                   aspect=1./1000,vmin=np.min(agrid),vmax=np.max(agrid))
        plt.colorbar()

#------------------------------
# Second coefficient
#------------------------------
    vals = np.zeros(npts)
    vals[:] = bvec[idata]

    bgrid = np.zeros((len(uTeff),len(ulogg)))
    for i in np.arange(len(uTeff)):
        for ii in np.arange(len(ulogg)):
            eval  = np.array([uTeff[i],ulogg[ii]]).reshape(1,2)
            val = griddata(locs,vals,eval,method='cubic')
            if len(val) > 0:
                bgrid[i,ii] = val[0]
            else:
                pass 

    ldc2func = bspline(uTeff, ulogg, bgrid, kx=1, ky=1, s=0)
    bval = ldc2func(Tstar,loggstar)[0][0]

    if plot:      
        plt.figure(2)
        plt.clf()
        plt.imshow(bgrid,interpolation='none',
                   extent=[np.min(ulogg),np.max(ulogg),np.min(uTeff),np.max(uTeff)],
                   aspect=1./1000,vmin=np.min(bgrid),vmax=np.max(bgrid))
        plt.colorbar()

#------------------------------
# Third coefficient
#------------------------------
    vals = np.zeros(npts)
    vals[:] = cvec[idata]

    cgrid = np.zeros((len(uTeff),len(ulogg)))
    for i in np.arange(len(uTeff)):
        for ii in np.arange(len(ulogg)):
            eval  = np.array([uTeff[i],ulogg[ii]]).reshape(1,2)
            val = griddata(locs,vals,eval,method='cubic')
            if len(val) > 0:
                cgrid[i,ii] = val[0]
            else:
                pass 

    ldc3func = bspline(uTeff, ulogg, cgrid, kx=1, ky=1, s=0)
    cval = ldc3func(Tstar,loggstar)[0][0]

    if plot:      
        plt.figure(3)
        plt.clf()
        plt.imshow(cgrid,interpolation='none',
                   extent=[np.min(ulogg),np.max(ulogg),np.min(uTeff),np.max(uTeff)],
                   aspect=1./1000,vmin=np.min(cgrid),vmax=np.max(cgrid))
        plt.colorbar()


#------------------------------
# Fourth coefficient
#------------------------------

    vals = np.zeros(npts)
    vals[:] = dvec[idata]

    dgrid = np.zeros((len(uTeff),len(ulogg)))
    for i in np.arange(len(uTeff)):
        for ii in np.arange(len(ulogg)):
            eval  = np.array([uTeff[i],ulogg[ii]]).reshape(1,2)
            val = griddata(locs,vals,eval,method='cubic')
            if len(val) > 0:
                dgrid[i,ii] = val[0]
            else:
                pass 

    ldc4func = bspline(uTeff, ulogg, dgrid, kx=1, ky=1, s=0)
    dval = ldc4func(Tstar,loggstar)[0][0]

    if plot:      
        plt.figure(4)
        plt.clf()
        plt.imshow(dgrid,interpolation='none',
                   extent=[np.min(ulogg),np.max(ulogg),np.min(uTeff),np.max(uTeff)],
                   aspect=1./1000,vmin=np.min(dgrid),vmax=np.max(dgrid))
        plt.colorbar()

    if limb == 'quad':
        return aval, bval

    if limb == 'sqrt':
        return cval, dval

    if limb == 'nlin':
        return aval, bval, cval, dval



def get_koi_info(koiin,planet,short=False,network=None,clip=False,getsamp=False,
                 lprior=False,rprior=False,limbmodel='quad',errfac=3):

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

    ----------------------------------------------------------------------
    """

# Import limb coefficient module
    import constants as c
    from scipy.interpolate import interp2d

# Define global variables
    global t, flux, e_flux
    global integration
    global path, name, koi
    global pdata
    global sdata
    global bjd
    global limb
    global ndim
    global jeffries, limbprior, lerr
    global stag,ctag,ltag,rtag,lptag
    global sampfac, adapt
    global net

# Short cadence data vs. long cadence data    
    # Integration and read time from FITS header
    int = 6.019802903270
    read = 0.518948526144
    if short == True:
        stag = '_short'
        integration = int*9.0+read*8.0
    else:
        stag = '_long'
        integration = int*270.0 + read*269.0

    koi = koiin

# "Keplerp time"
    bjd = 2454833.0

# Factor by which errors in Teff and logg are inflated for limb darkening prior
    lerr = errfac

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
    plnumber = []
    for nm in names:
        plnumber = np.append(plnumber,np.int(nm.split('.')[-1]))

    plind, = np.where(planet == plnumber)
    if length(plind) > 1:
        sys.exit("Multiple planets found for identical designation!")

    if length(plind) == 0:
        sys.exit("No planet found for given designation!")

# Planet index
#    pli = planet-1
    pli = plind[0]

# Use clipped data if asked for
    if clip: 
        ctag = '_clip'
    else:
        ctag = ''

# Use Jeffries prior on Rp/Rstar
    if rprior:
        rtag = '_rp'
        jeffries = True
    else:
        rtag = ''
        jeffries = False

# Use Claret prior on limb darkening
    if lprior:
        lptag = '_lp'
        limbprior = True
    else:
        lptag = ''
        limbprior = False

# Star properties
    print "... getting star properties"
    info2 = np.loadtxt(path+'Fits/'+names[pli]+stag+
                       '_fixlimb_fitparams.dat',dtype='string')
    Mstar = np.float(info2[1])*c.Msun
    eMstar = np.float(info2[2])*c.Msun
    Rstar = np.float(info2[3])*c.Rsun
    eRstar = np.float(info2[4])*c.Rsun
    Tstar = np.float(info2[5])
    eTstar = np.float(info2[6])

    loggstar = np.log10( c.G * Mstar / Rstar**2. )
    dm = eMstar/(Mstar*np.log(10))
    dr = 2.0*eRstar/(Rstar*np.log(10))
    e_loggstar = np.sqrt(dm**2 + dr**2)
        
# Limb darkening parameters
    print "... getting LD coefficients for "+limbmodel+" model"
    if network == 'koi':
        net = None
    else:
        net = network

    if limbmodel == 'nlin':
        a,b,c,d = get_limb_coeff(Tstar,loggstar,network=net,
                                     limb=limbmodel)
    else:
        a,b = get_limb_coeff(Tstar,loggstar,network=net,
                                 limb=limbmodel)
  
    print ""
    if limbmodel == 'quad' or limbmodel == 'sqrt':
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
        ndim = 7
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
        ndim = 9
    else:
        print "Limb darkening law not recognized"
        return

    limb = limbmodel

    
# Preliminary Levenburg-Marquardt fit parameters 
    rprs0   = np.float(info2[16])
    erprs0  = np.float(info2[17])
    dur0    = np.float(info2[14])/24.
    edur0   = np.float(info2[15])/24.
    impact0 = np.float(info2[13])

    period0  = np.float(info2[8])
    eperiod0 = np.float(info2[9])
    ephem0_tmp   = np.float(info2[10])
    eephem0  = np.float(info2[11])


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
        vals = np.loadtxt('/Users/jonswift/Astronomy/Exoplanets/Mdwarfs/All/sample_grid'+stag+'.txt')
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
        sampfac = 21.0

    name = names[pli]
    file = name+stag+ctag+'.dat'
    print "Reading data from "+file
    data = np.loadtxt(path+'Refine/'+file)

    t = data[:,0]
    flux = data[:,2]
    e_flux = data[:,3]
    
# Take ephemeris nearest to the midpoint of time series (max - min)
    tmid = (np.max(t) - np.min(t))/2.0 + np.min(t)
    nps = np.arange(0.0,10000.0,1.0,dtype='float64')
    ttimes = ephem0_tmp + nps*period0
    diff = ttimes - tmid
    arg = (np.where(np.abs(diff) == np.min(np.abs(diff))))[0][0]
    ephem0 = ttimes[arg]

# Make planet data array
    pinfo =[rprs0,dur0,impact0,ephem0,period0]
    perr = [erprs0,edur0,-999,eephem0,eperiod0]

    pdata = np.array([pinfo,perr])
        
# Make star data array
    if limbmodel == 'nlin':
        sinfo = [Mstar,Rstar,Tstar,c1,c2,c3,c4,loggstar]
        serr  = [eMstar,eRstar,eTstar,-999,-999,-999,-999,e_loggstar]    
    else:
        sinfo = [Mstar,Rstar,Tstar,c1,c2,loggstar]
        serr  = [eMstar,eRstar,eTstar,-999,-999,e_loggstar]
    
    sdata = np.array([sinfo,serr])


# Get fiducial limb darkening reference points
#    print ""
#    if lprior:
#        print "Getting distribution of LD coefficients from "+ \
#            "stellar parameters for LD prior"
#        q1v,q2v = get_qdists(sdata,sz=1000,errfac=errfac)
#        inds = np.where((q1v >= 0) & (q1v <= 1) & (q2v >= 0) & (q2v <= 1))
#        q1v = q1v[inds]
#        q2v = q2v[inds]
#        write = True
#        get_limb_spread(q1v,q2v,sdata=sdata,limbmodel=limbmodel,network=net, 
#                        plot=write,write=write,factor=errfac)
#    else:
#        write = False

# Make light curve array
    lc = np.array([t,flux,e_flux])

    return lc,pdata,sdata





def foldtime(time,period=1.0,t0=0.0):

    """ 
    ----------------------------------------------------------------------
    foldtime:
    ---------
    Basic utility to fold time based on ephemeris

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




def bin_lc(x,y,nbins=100):

    """
    ----------------------------------------------------------------------    
    bin_lc:
    -------
    Utility to bin data and return standard deviation in each bin
    
    For visual aid in plots, mostly

    example:
    --------
    tbin,fbin,errbin = bin_lc(ttm,flux,nbins=200)

    """

    n, I = np.histogram(x, bins=nbins)
    sy, I = np.histogram(x, bins=nbins, weights=y)
    sy2, I = np.histogram(x, bins=nbins, weights=y*y)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)

    binvals = (I[1:] + I[:-1])/2
    yvals = mean
    yerr = std/np.sqrt(len(std))
    
    return binvals,yvals,yerr



def tottofull(dtot,rprs,impact,period):
    """
    ----------------------------------------------------------------------
    tottofull:
    --------------
    Function to convert total transit duration to full duration given a 
    planet size, impact parameter, and period

    To do:
    ------
    Check if this is indeed correct! Currently this routine is not used in
    this package.

    """

    
    aors  = np.sqrt( ((1+rprs)**2 - impact**2)/
                    ((np.sin(np.pi*dtot/period))**2) + impact**2)    

    arg = (1-rprs)**2 - impact**2
    inds, = np.where(arg < 0)
    if len(inds) > 0:
        if len(inds) > 1:
            s = 's'
        else:
            s = ''
        print str(len(inds))+" grazing transit"+s+"!"
        arg[inds] = 0.0
        ginds, = np.where(arg >= 0)
        
    sini = np.sqrt(1.0 - (impact/aors)**2)

    dfull = (period/np.pi)*np.arcsin(arg/(aors*sini))

    if len(inds) > 0:
        dfull[inds] = -999
    
    return dfull




def compute_trans(rprs,duration,impact,t0,period,ldc,unsmooth=False,
                  all=False,modelfac=21.0):

    """
    ----------------------------------------------------------------------
    compute_trans:
    --------------
    Function to compute transit curve for given transit parameters. Returns
    model time and model flux. 

    options:
    --------
    unsmooth:   In addition to the model data, also return the unsmoothed 
                light curve for comparison

    examples:
    ---------
    In[1]: tmodel,model = compute_trans(rprs,duration,impact,t0,per)

    In[2]: tmodel,model,rawmodel = compute_trans(rprs,duration,impact,t0,
                                                 per,unsmooth=True)

    """

# t0 is offset from nominal ephemeris
    import occultquad as oq
    import transit as trans
    from scipy.ndimage.filters import uniform_filter1d, uniform_filter

    
    if limb == 'nlin':
        a1 = ldc[0]
        a2 = ldc[1]
        a3 = ldc[2]
        a4 = ldc[3]
    else:        
        q1in = ldc[0]
        q2in = ldc[1]
        u1, u2 = qtou(q1in,q2in,limb=limb)
        
    # "Lens" radius = Planet radius/Star radius
    posrl = rprs > 0
    rl = np.abs(rprs)

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

# If "negative" radius of planet then add the flux
    if not posrl:
        model = 2.0 - model
        
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

    """

# Input parameters
    rprs     = x[0]
    duration = x[1]
    impact   = x[2]
    t0       = x[3]
    per      = x[4]

    # Sample Teff and logg if limb prior is set
    if limbprior == True:
        Teff = x[5]
        logg = x[6]
        if limb == 'quad':
            u1 = ldc1func(Teff,logg)[0][0]
            u2 = ldc2func(Teff,logg)[0][0]
        if limb == 'sqrt':
            u1 = ldc3func(Teff,logg)[0][0]
            u2 = ldc4func(Teff,logg)[0][0]
        if limb == 'nlin':
            u1 = ldc1func(Teff,logg)[0][0]
            u2 = ldc2func(Teff,logg)[0][0]
            u3 = ldc3func(Teff,logg)[0][0]
            u4 = ldc4func(Teff,logg)[0][0]
            ldc = [q1,q2,q3,q4]
        else:
            q1,q2 = utoq(u1,u2,limb=limb)            
            ldc = [q1,q2]
    else:
        q1       = x[5]
        q2       = x[6]
        if limb == 'nlin':
            q3 = x[7]
            q4 = x[8]
            ldc = [q1,q2,q3,q4]
        else:
            ldc = [q1,q2]

# Priors

    if np.abs(per - pdata[0,4])/pdata[0,4] > 1e-2:
#        print "period out of range"
#        pdb.set_trace()
        return -np.inf

    if duration < 1e-4 or duration > pdata[0,4]/2.0:
#        print "duration out of range"
#        pdb.set_trace()
        return -np.inf

    if np.abs(t0) > pdata[0,1]/2.0:
#        print "t0 out of range"
#        pdb.set_trace()
        return -np.inf
        
    if rprs > 0.5:
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
        

#    if limbprior == True:
#        print Teff, logg, q1, q2
#    else:
#        print q1,q2

### Compute transit model for given input parameters ###
    # t0 is time of mid trans (relative to nominal ephemeris)
    tmodel,smoothmodel = compute_trans(rprs,duration,impact,t0,per,ldc)

    # compute model only in transit region
    ttm = foldtime(t,period=per,t0=pdata[0,3])
    ins  = np.zeros(len(ttm),dtype=bool)
    ininds = np.where((ttm >= tmodel[1]) & (ttm <= tmodel[-2]))
    ins[ininds] = True
    
    # interpolate model onto data values
    cfunc = sp.interpolate.interp1d(tmodel,smoothmodel,kind='linear')    
    mfit = np.ones(len(ttm),dtype=float)

    mfit[ins] = cfunc(ttm[ins])

# Log likelihood function
    lfi = -1*(mfit - flux)**2/(2.0*e_flux**2)

# Log likelihood
    lf = np.sum(lfi)

    if jeffries == True:
        lf = lf - 2.0*np.log(np.abs(rprs))

#    if limbprior == True:
#        lf = lf + np.log(q1pdf_func(q1)) + np.log(q2pdf_func(q2))

    if limbprior == True:
        Tstar0 = sdata[0,2]
        eTstar = sdata[1,2]

        if limb == 'nlin':
            logg0 = sdata[0,7]
            elogg = sdata[1,7]
        else:
            logg0 = sdata[0,5]
            elogg = sdata[1,5]

        lf = lf - (Teff - Tstar0)**2/(2.0*(eTstar*lerr)**2) \
            - (logg - logg0)**2/(2.0*(elogg*lerr)**2)

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
    per = inp[4]

# Limb darkening params

    c1       = inp[5][0]
    c2       = inp[5][1]
    if limb == 'nlin':
        c3 = inp[5][2]
        c4 = inp[5][3]
        ldc = [c1,c2,c3,c4]
    else:
        ldc = [c1,c2]


# Compute model with zero t0
    tmodel,smoothmodel = compute_trans(rprs,duration,impact,0.0,per,ldc)

# Impose ephemeris offset in data folding    
    tfit = foldtime(t,period=per,t0=pdata[0,3]+t0)
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
               nbins=100,errorbars=False,pdf=False):

    """
    ----------------------------------------------------------------------
    plot_model:
    -----------
    Plot transit model given model params.

    ----------------------------------------------------------------------
    """
    import gc

    plt.rcdefaults()

# Check for output directory   
    directory = path+'MCMC/'
    if not os.path.exists(directory):
        os.makedirs(directory)

# Fold data on input parameters
    tplt = foldtime(t,period=modelparams[4],t0=pdata[0,3]+modelparams[3])
    fplt = flux
    eplt = e_flux

# Bin data (for visual aid)
    tfit, yvals, yerror = bin_lc(tplt,flux,nbins=nbins)

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

    ldc = modelparams[5]

# Get model, raw (unsmoothed) model, and tfull

    tmodel,model,rawmodel = compute_trans(modelparams[0],modelparams[1],modelparams[2],\
                                              0.0,modelparams[4],\
                                              ldc,unsmooth=True)

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
                    ((np.sin(np.pi*modelparams[1]/modelparams[4]))**2) + modelparams[2]**2)    

#    aors    = 2.0 * np.sqrt(modelparams[0]) * modelparams[4] / \
#        (np.pi*np.sqrt(ttot**2 - tfull**2))
    
    rhostar =  3.0*np.pi/( c.G * (modelparams[4]*24.*3600.)**2 ) * aors**3
    

    plt.annotate(r'$P$ = %.7f d' % modelparams[4], [0.5,0.87],horizontalalignment='center',
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
    val = (modelparams[4]-pdata[0,4])*24.*3600.
    plt.annotate(r'$\Delta P$ = %.3f s' % val, [0.15,0.85],
                 xycoords='figure fraction',fontsize='large')
    plt.annotate(r'$R_p/R_*$ = %.5f' % modelparams[0], [0.15,0.81],
                  xycoords='figure fraction',fontsize='large')   
    plt.annotate('b = %.2f' % modelparams[2], [0.15,0.77],
                  xycoords='figure fraction',fontsize='large')
    t0out = (modelparams[3])*24.*60.*60.
    plt.annotate(r'$\Delta t_0$ = %.3f s' % t0out, [0.15,0.73],
                  xycoords='figure fraction',fontsize='large')

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

    if pdf:
        ftype = '.pdf'
    else:
        ftype = '.png'

    plt.savefig(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit'+tag+ftype, dpi=300)
    plt.clf()

    gc.collect()
    return


def plot_final(modelparams,dispparams=False,short=False,tag='',ms=10,sm=8,
               errorbars=False,pdf=False):

    """
    ----------------------------------------------------------------------
    plot_final:
    -----------
    Plot transit model given model params.

    ----------------------------------------------------------------------
    """

    import gc
    from matplotlib import gridspec
    import matplotlib as mpl
    plt.rcdefaults()

    if stag == '_long':
        titletxt = 'Long Cadence'
        alpha = 0.5
        ysigmin = 2.5
        ysigmax = 4
        ms = 10
        sm = 8
        intfac = 1.0
    elif stag == '_short':
        titletxt = 'Short Cadence'
        alpha = 0.25
        ysigmin = 3.5
        ysigmax = 5.5
        ms = 8
        sm = 5
        intfac = 5.0


    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.major.size'] = 5
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['xtick.labelsize'] = 18
    mpl.rcParams['ytick.labelsize'] = 18
    fs = 20
    lw = 1.5

# Check for output directory   
    directory = path+'MCMC/'
    if not os.path.exists(directory):
        os.makedirs(directory)

# Fold data on input parameters
    tplt = foldtime(t,period=modelparams[4],t0=pdata[0,3]+modelparams[3])
    fplt = flux
    eplt = e_flux

# Center transit in plot
    wid1 = np.max(tplt)
    wid2 = abs(np.min(tplt))
    if wid1 < wid2:
        wid = wid1
    else:
        wid = wid2

    nbins = 2.0*wid*24.0*3600.0/(integration*intfac)

# Bin data (for visual aid)
    tfit, yvals, yerror = bin_lc(tplt,flux,nbins=nbins)

    plt.figure(2,figsize=(11,8.5),dpi=300)
    gs = gridspec.GridSpec(3, 1,wspace=0)
    ax1 = plt.subplot(gs[0:2, 0])    

    ax1.plot(tplt,(fplt-1)*1e6,'.',color='gray',markersize=sm,alpha=0.5)
    if errorbars:
        ax1.errorbar(tfit,(yvals-1)*1e6,yerr=yerror*1e6,fmt='o',color='black',
                     markersize=ms)
    else:
        ax1.plot(tfit,(yvals-1)*1e6,'o',color='black',markersize=ms)

# Center transit in plot
    ax1.set_xlim(np.array([-1,1])*wid)

    ldc = modelparams[5]

# Get model, raw (unsmoothed) model, and tfull

    tmodel,model,rawmodel = compute_trans(modelparams[0],modelparams[1],modelparams[2],\
                                              0.0,modelparams[4],\
                                              ldc,unsmooth=True)

#    tfull = tottofull(modelparams[1],modelparams[0],modelparams[2],modelparams[4])
 
    sig = rb.std((fplt-1)*1e6)
    med = np.median((fplt-1)*1e6)
#    min = max(np.min((model-1)*1e6),-4*sig)
# For deep transits
    min = np.min((model-1)*1e6)
#    yrange = np.array([min-3*sig,med+15*sig])
    yrange = np.array([min-ysigmin*sig,med+ysigmax*sig])
    ax1.set_ylim(yrange) 
    locs = ax1.yaxis.get_majorticklocs()
    ax1.yaxis.set_ticks(locs, map(lambda x: "%.0f" % x, locs))
    ax1.set_ylabel("ppm",fontsize=fs)
    ax1.set_xticklabels(())
    ax1.xaxis.set_tick_params(length=10, width=lw, labelsize=fs)
    ax1.yaxis.set_tick_params(length=10, width=lw, labelsize=fs)

    ax1.plot(tmodel,(rawmodel-1)*1e6,'b--',linewidth=lw)
    ax1.plot(tmodel,(model-1)*1e6,'r',linewidth=lw)
    ax1.set_title(name+" "+titletxt+" Fit",fontsize=fs)

    res = residuals(modelparams)

    dof = len(res) - ndim - 1.
    chisq = np.sum((res/eplt)**2)/dof

    rhoi = sdata[0,0]/(4./3.*np.pi*sdata[0,1]**3)

#    rprs     = dispparams[0] 
#    duration = dispparams[1]
#    impact   = dispparams[2]
#    period   = dispparams[4]
#    aors     = np.sqrt( ((1+rprs)**2 - impact**2)/
#                        ((np.sin(np.pi*duration/period))**2) + impact**2)    
    
#    rhostar =  3.0*np.pi/( c.G * (period*24.*3600.)**2 ) * aors**3
    
#    ax1.annotate(r'$P$ = %.5f d' % period, [0.55,0.86],horizontalalignment='center',
#                 xycoords='figure fraction',fontsize=fs-1)
#    ax1.annotate(r'$\chi^2_r$ = %.3f' % chisq, [0.9,0.83],horizontalalignment='right',
#                 xycoords='figure fraction',fontsize=fs-1)
#    ax1.annotate(r'$\rho_*$ (e=0) = %.2f g/cc' % rhostar, [0.9,0.78],horizontalalignment='right',
#                 xycoords='figure fraction',fontsize=fs-1)
#    ax1.annotate(r'$\rho_{*,0}$ = %.2f g/cc' % rhoi, [0.9,0.73],horizontalalignment='right',
#                 xycoords='figure fraction',fontsize=fs-1)
#    ax1.annotate(r'$R_p/R_*$ = %.3f' % dispparams[0], [0.2,0.83],
#                  xycoords='figure fraction',fontsize=fs-1)   
#    val =  duration*24.
#    ax1.annotate(r'$\tau_{\rm tot}$ = %.2f h' % val, [0.2,0.78],
#                  xycoords='figure fraction',fontsize=fs-1)
#    t0out = (dispparams[3]) + pdata[0,3] # + 2454833.0
#    ax1.annotate(r'$t_0$ = %.4f BKJD' % t0out, [0.2,0.73],
#                  xycoords='figure fraction',fontsize=fs-1)

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

#        u1out = u1
#        ax1.annotate(r'$u_1$ = %.2f' % u1out, [0.15,0.59],
#                     xycoords='figure fraction',fontsize='large')
        u2out = u2
#        ax1.annotate(r'$u_2$ = %.2f' % u2out, [0.15,0.55],
#                     xycoords='figure fraction',fontsize='large')

    ax2 = plt.subplot(gs[2, 0])
    s = np.argsort(tplt)
    ax2.plot(tplt[s],res*1e6,'.',markersize=sm,color='gray',alpha=0.5)
    tres, yres, yreserr = bin_lc(tplt[s],res,nbins=nbins)
    if errorbars:
        ax2.errorbar(tres,yres*1e6,yerr=yreserr*1e6,fmt='o',color='black',markersize=ms)
    else:
        ax2.plot(tres,yres*1e6,'o',color='black',markersize=ms)

    ax2.set_xlim(np.array([-1,1])*wid)
    sig = rb.std(res*1e6)
    med = np.median(res*1e6)
    ax2.set_ylim(np.array([-5*sig,5*sig]))
    ax2.axhline(y=0,color='r',linewidth=lw)
    locs = ax2.yaxis.get_majorticklocs()
    ax2.yaxis.set_ticks(locs, map(lambda x: "%.0f" % x, locs))
#    ax2.set_title("Residuals")
    ax2.set_xlabel("Time from Mid-Transit (days)",fontsize=fs)
    ax2.set_ylabel("ppm",fontsize=fs)
    ax2.xaxis.set_tick_params(length=10, width=lw, labelsize=fs)
    ax2.yaxis.set_tick_params(length=10, width=lw, labelsize=fs)

    plt.subplots_adjust(hspace=0.15,left=0.15,right=0.95,top=0.92)

    if pdf:
        ftype='.pdf'
    else:
        ftype='.png'
    plt.savefig(directory+name+stag+ctag+rtag+lptag+ltag+'fit'+tag+'_final'+ftype, dpi=300)

#    plt.rcdefaults()
    plt.subplots_adjust(hspace=0.2,left=0.125,right=0.9,top=0.9)
    plt.clf()


    out = np.array([tplt,(fplt-1)*1e6])
    np.savetxt(directory+name+stag+ctag+rtag+lptag+ltag+'_data.txt',out.T)
    out = np.array([tfit,(yvals-1)*1e6])
    np.savetxt(directory+name+stag+ctag+rtag+lptag+ltag+'_databin.txt',out.T)
    out = np.array([tmodel,(model-1)*1e6])
    np.savetxt(directory+name+stag+ctag+rtag+lptag+ltag+'_model.txt',out.T)
    out = np.array([tplt[s],res*1e6])
    np.savetxt(directory+name+stag+ctag+rtag+lptag+ltag+'_res.txt',out.T)
    out = np.array([tres,yres*1e6])
    np.savetxt(directory+name+stag+ctag+rtag+lptag+ltag+'_resbin.txt',out.T)
    
    
    gc.collect()
    return



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

                           
def get_limb_qs(Mstar=0.5,Rstar=0.5,Tstar=3800.0,limb='quad',network=None):
    import constants as c

    Ms = Mstar*c.Msun
    Rs = Rstar*c.Rsun
    loggstar = np.log10( c.G * Ms / Rs**2. )
 
    
    if limb == 'nlin':
        a,b,c,d = get_limb_coeff(Tstar,loggstar,network=network,limb=limb,interp='linear')
        return a,b,c,d
    else:
        a,b = get_limb_coeff(Tstar,loggstar,network=network,limb=limb,interp='linear')
        q1,q2 = utoq(a,b,limb=limb)
        return q1, q2
 


def get_qdists(sdata,sz=100,errfac=1):
    from scipy.stats.kde import gaussian_kde
    global q1pdf_func
    global q2pdf_func

    Mstar = sdata[0,0]/c.Msun
    eMstar = sdata[1,0]/c.Msun
    Rstar = sdata[0,1]/c.Rsun
    eRstar = sdata[1,1]/c.Rsun
    Tstar = sdata[0,2]
    eTstar = sdata[1,2]
    
    print "... using error factor of "+str(errfac)
    Ms = np.random.normal(Mstar,eMstar*errfac,sz)
    Rs = np.random.normal(Rstar,eRstar*errfac,sz)
    Ts = np.random.normal(Tstar,eTstar*errfac,sz)

    q1v = []
    q2v = []

    print "... generating LD distribution of "+str(sz)+" values"
    for i in range(sz):
        q1,q2 = get_limb_qs(Mstar=max(Ms[i],0.001),Rstar=max(Rs[i],0.001),Tstar=max(Ts[i],100),limb=limb)
        q1v = np.append(q1v,q1)
        q2v = np.append(q2v,q2)
        
    q1v = q1v[~np.isnan(q1v)]
    q2v = q2v[~np.isnan(q2v)]

    vals  = np.linspace(0,1,10000)
    q1kde = gaussian_kde(q1v)
    q1pdf = q1kde(vals)
    q1pdf_func = sp.interpolate.interp1d(vals,q1pdf,kind='nearest')

    q2kde = gaussian_kde(q2v)
    q2pdf = q2kde(vals)
    q2pdf_func = sp.interpolate.interp1d(vals,q2pdf,kind='nearest')


    return q1v, q2v



def get_qvals(q1v,q2v,nsamp=100):
    from scipy.stats.kde import gaussian_kde
    global q1pdf_func
    global q2pdf_func

    vals  = np.linspace(0,1,1000)
    q1kde = gaussian_kde(q1v)
    q1pdf = q1kde(vals)
    q1pdf_func = sp.interpolate.interp1d(vals,q1pdf,kind='linear')
    q1c   = np.cumsum(q1pdf)/np.sum(q1pdf)
    q1func = sp.interpolate.interp1d(q1c,vals,kind='linear')
    q1samp = q1func(np.random.uniform(0,1,nsamp))

    q2kde = gaussian_kde(q2v)
    q2pdf = q2kde(vals)
    q2pdf_func = sp.interpolate.interp1d(vals,q2pdf,kind='linear')
    q2c   = np.cumsum(q2pdf)/np.sum(q2pdf)
    q2func = sp.interpolate.interp1d(q2c,vals,kind='linear')
    q2samp = q2func(np.random.uniform(0,1,nsamp))

    return q1samp, q2samp




def fit_single(nwalkers=250,burnsteps=1000,mcmcsteps=1000,clobber=False):

    """
    fit_single:
    -----------
    Fit a single transit signal with specified mcmc parameters return 
    chains and log likelihood.

    """

    import emcee
    import robust as rb
    import time
    import constants as c

    global nw, bs, mcs
    global q1s, q2s

    nw = nwalkers
    bs = burnsteps
    mcs = mcmcsteps

    directory = path+'MCMC/'
    if not os.path.exists(directory):
        os.makedirs(directory)

# Do not redo MCMC unless clobber flag is set
    done = os.path.exists(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_rchain.txt')
    if done == True and clobber == False:
        print "MCMC run already completed"
        return False,False

    print ""
    os.system('date')
    print "Starting MCMC fitting routine for "+name


# Set up MCMC sampler
    print "... initializing emcee sampler"
    tstart = time.time()
    sampler = emcee.EnsembleSampler(nw, ndim, lnprob)

    twomin = 2./(24.*60.)
    onesec = 1./(24.*60.*60.)

#    q1 = (sdata[0,3]+sdata[0,4])**2
#    q2 = sdata[0,3]/(2.0*(sdata[0,3]+sdata[0,4]))

# Initial chain values
    print ""
    print "Deriving starting values for chains"
    p0_1 = np.random.uniform(0,2*pdata[0,0],nw)
    p0_2 = np.random.uniform(0.5*pdata[0,1],1.0*pdata[0,1], nw)
    p0_3 = np.random.uniform(0.,0.99, nw)
    p0_4 = np.random.uniform(-0.5*twomin, 0.5*twomin, nw)
    p0_5 = pdata[0,4] + np.random.uniform(-1*onesec,onesec,nw)
    if limbprior == True:
        p0_6 = np.random.normal(sdata[0,2],lerr*sdata[1,2],nw)
        if limb == 'nlin':
            ind = 7
        else:
            ind = 5
        p0_7 = np.random.normal(sdata[0,ind],lerr*sdata[1,ind],nw)

        p0 = np.array([p0_1,p0_2,p0_3,p0_4,p0_5,p0_6,p0_7]).T
        variables =["Rp/R*","duration","impact","t0","Period","Teff","logg"]

    else:
        p0_6 = np.random.uniform(0,1,nw)
        p0_7 = np.random.uniform(0,1,nw)
        if limb == 'nlin':
            p0_6 = sdata[0,3] + np.zeros(nw)
            p0_7 = sdata[0,4] + np.zeros(nw)
            p0_8 = sdata[0,5] + np.zeros(nw)
            p0_9 = sdata[0,6] + np.zeros(nw)
            p0 = np.array([p0_1,p0_2,p0_3,p0_4,p0_5,p0_6,p0_7,p0_8,p0_9]).T
            variables =["Rp/R*","duration","impact","t0","Period","a1","a2","a3","a4"]
        else:
            p0 = np.array([p0_1,p0_2,p0_3,p0_4,p0_5,p0_6,p0_7]).T
            variables =["Rp/R*","duration","impact","t0","Period","q1","q2"]
        
# Run burn-in
    print ""
    print "Running burn-in with "+str(bs)+" steps and "+str(nw)+" walkers"
    pos, prob, state = sampler.run_mcmc(p0, bs)
    print done_in(tstart)

# Calculate G-R scale factor for each variable
    Rs = GR_test(sampler.chain,variables=variables)
    
    for var in np.arange(ndim):
        acout = "Autocorrelation time for "+variables[var]+" = {0:0.3f}"
        print acout.format(sampler.acor[var])

    afout = "Mean acceptance fraction: {0:0.3f}"
    print afout.format(np.mean(sampler.acceptance_fraction))

# Save burn in stats
    burn = np.append(Rs,sampler.acor)
    burn = np.append(burn,np.mean(sampler.acceptance_fraction))
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_burnstats.txt',burn)

# Reset sampler and run MCMC for reals
    print "getting pdfs for LD coefficients"
    print "... resetting sampler and running MCMC with "+str(mcs)+" steps"
    sampler.reset()
    posf, probf, statef = sampler.run_mcmc(pos, mcs)
    print done_in(tstart)

 # Calculate G-R scale factor for each variable
    Rs = GR_test(sampler.chain,variables=variables)

# Autocorrelation times
    for var in np.arange(ndim):
        acout = "Autocorrelation time for "+variables[var]+" = {0:0.3f}"
        print acout.format(sampler.acor[var])

    afout = "Final mean acceptance fraction: {0:0.3f}"
    print afout.format(np.mean(sampler.acceptance_fraction))

    stats = np.append(Rs,sampler.acor)
    stats = np.append(stats,np.mean(sampler.acceptance_fraction))
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_finalstats.txt',stats)


    print "Writing MCMC chains to disk"
    rdist  = sampler.flatchain[:,0]
    ddist  = sampler.flatchain[:,1]
    bdist  = sampler.flatchain[:,2]
    tdist  = sampler.flatchain[:,3]
    pdist  = sampler.flatchain[:,4]
    if limbprior == True:
        teffdist = sampler.flatchain[:,5]
        loggdist = sampler.flatchain[:,6]
        np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_teffchain.txt',teffdist)
        np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_loggchain.txt',loggdist)
        print "Making LD coefficient chains from Teff and logg chains"
        if limb == 'nlin':
            c1dist = []
            c2dist = []
            c3dist = []
            c4dist = []
            for i in np.arange(len(teffdist)):
                c1dist = np.append(c1dist,ldc1func(teffdist[i],loggdist[i])[0][0])
                c2dist = np.append(c2dist,ldc2func(teffdist[i],loggdist[i])[0][0])
                c3dist = np.append(c3dist,ldc3func(teffdist[i],loggdist[i])[0][0])
                c4dist = np.append(c4dist,ldc4func(teffdist[i],loggdist[i])[0][0])

            np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_q3chain.txt',c3dist)
            np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_q4chain.txt',c4dist)
    
        else:
            c1dist = []
            c2dist = []
            for i in np.arange(len(teffdist)):
                if limb =='quad':
                    u1 = ldc1func(teffdist[i],loggdist[i])[0][0]
                    u2 = ldc2func(teffdist[i],loggdist[i])[0][0]
                elif limb == 'sqrt':
                    u1 = ldc3func(teffdist[i],loggdist[i])[0][0]
                    u2 = ldc4func(teffdist[i],loggdist[i])[0][0]

                c1,c2 = utoq(u1,u2,limb=limb)
                c1dist = np.append(c1dist,c1)
                c2dist = np.append(c2dist,c2)
    else:
        c1dist = sampler.flatchain[:,5]
        c2dist = sampler.flatchain[:,6]
        if limb == 'nlin':
            c3dist = sampler.flatchain[:,7]
            c4dist = sampler.flatchain[:,8]
            np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_q3chain.txt',c3dist)
            np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_q4chain.txt',c4dist)

    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_rchain.txt',rdist)
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_dchain.txt',ddist)
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_bchain.txt',bdist)
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_t0chain.txt',tdist)
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_pchain.txt',pdist)
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_q1chain.txt',c1dist)
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_q2chain.txt',c2dist)
        
    lp = sampler.lnprobability.flatten()
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_lnprob.txt',lp)

    chains = sampler.flatchain

    return chains,lp



def GR_test(chains,variables=False):

    """
    ----------------------------------------------------------------------
    GR_test:
    --------
    Compute the Gelman-Rubin scale factor for each variable given input
    flat chains
    ----------------------------------------------------------------------
    """

    nwalkers = np.float(np.shape(chains)[0])
    nsamples = np.float(np.shape(chains)[1])
    ndims    = np.shape(chains)[2]
    Rs = np.zeros(ndims)
    for var in np.arange(ndims):
        psi = chains[:,:,var]
        psichainmean = np.mean(psi,axis=1)
        psimean = np.mean(psi)

        B = nsamples/(nwalkers-1.0)*np.sum((psichainmean - psimean)**2)

        s2j = np.zeros(nwalkers)
        for j in range(np.int(nwalkers)):
            s2j[j] = 1.0/(nsamples-1.0)*np.sum((psi[j,:] - psichainmean[j])**2)

        W = np.mean(s2j)

        varbarplus = (nsamples-1.0)/nsamples * W + 1/nsamples * B

        R = np.sqrt(varbarplus/W)

        if len(variables) == ndims:
            out = "Gelman Rubin scale factor for "+variables[var]+" = {0:0.3f}"
            print out.format(R)

        Rs[var] = R

    return Rs

    

def bestvals(chains=False,lp=False,network=None,bindiv=30.0,thin=False,
             frac=0.003,nbins=100,rpmax=1,durmax=10,sigfac=5.0,pdf=False):

    """
    ----------------------------------------------------------------------
    bestvals:
    ---------
    Find the best values from the 1-d posterior pdfs return best values 
    and the posterior pdf for rp/rs
    ----------------------------------------------------------------------
    """
    from statsmodels.nonparametric.kde import KDEUnivariate as KDE_U
    import robust as rb
    from scipy.stats.kde import gaussian_kde
    plt.rcdefaults()

    Rstar = sdata[0,1]/c.Rsun
    e_Rstar = sdata[1,1]/c.Rsun

# Use supplied chains or read from disk
    try:
        rprsdist = chains[:,0]
        ddist = chains[:,1]*24.
        bdist = chains[:,2]
        tdist = chains[:,3]*24.*60.*60.
        pdist = (chains[:,4]-pdata[0,4])*24.*3600.
        if limbprior == True:
            teffdist = sampler.flatchain[:,5]
            loggdist = sampler.flatchain[:,6]
            print "Making LD coefficient chains from Teff and logg chains"
            if limb == 'nlin':
                q1dist = []
                q2dist = []
                q3dist = []
                q4dist = []
                for i in np.arange(len(teffdist)):
                    q1dist = np.append(q1dist,ldc1func(teffdist[i],loggdist[i])[0][0])
                    q2dist = np.append(q2dist,ldc2func(teffdist[i],loggdist[i])[0][0])
                    q3dist = np.append(q3dist,ldc3func(teffdist[i],loggdist[i])[0][0])
                    q4dist = np.append(q4dist,ldc4func(teffdist[i],loggdist[i])[0][0])
    
            else:
                q1dist = []
                q2dist = []
                for i in np.arange(len(teffdist)):
                    u1 = ldc1func(teffdist[i],loggdist[i])[0][0]
                    u2 = ldc2func(teffdist[i],loggdist[i])[0][0]
                    c1,c2 = utoq(u1,u2,limb=limb)
                    q1dist = np.append(q1dist,c1)
                    q2dist = np.append(q2dist,c2)
        else:
            q1dist = sampler.flatchain[:,5]
            q2dist = sampler.flatchain[:,6]
            if limb == 'nlin':
                q3dist = sampler.flatchain[:,7]
                q4dist = sampler.flatchain[:,8]
    except:
        print '... importing MCMC chains'
        rprsdist = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_rchain.txt')
        ddist = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_dchain.txt')*24.
        bdist = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_bchain.txt')
        tdist = (np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_t0chain.txt'))*24.*60.*60.
        pdist = (np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_pchain.txt')-pdata[0,4])*24.*3600.
        q1dist = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_q1chain.txt')
        q2dist = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_q2chain.txt')
        if limb == 'nlin':
            q3dist = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_q3chain.txt')
            q4dist = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_q4chain.txt')
        print '... done importing chains!'


    try:
        test = len(lp)
    except:
        lp = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_lnprob.txt')


# Plot initial fit
#    tmid = tottomid(pdata[0,1],pdata[0,0],pdata[0,2])

    if limb == 'nlin':
        plot_model([pdata[0,0],pdata[0,1],pdata[0,2],0.0,pdata[0,4],\
                        [sdata[0,3],sdata[0,4],sdata[0,5],sdata[0,6]]],tag='_mpfit',nbins=nbins,pdf=pdf)
    else:
        q1,q2 = utoq(sdata[0,3],sdata[0,4],limb=limb)
        plot_model([pdata[0,0],pdata[0,1],pdata[0,2],0.0,pdata[0,4],[q1,q2]],tag='_mpfit',nbins=nbins,pdf=pdf)
        

#  Get maximum likelihood values
    maxlike = np.max(lp)
    imax = np.array([i for i, j in enumerate(lp) if j == maxlike])
    if imax.size > 1:
        imax = imax[0]
    rprsval = np.float(rprsdist[imax])
    bval    = np.float(bdist[imax])
    dval    = np.float(ddist[imax])
    tval    = np.float(tdist[imax])
    pval    = np.float(pdist[imax])
    q1val    = np.float(q1dist[imax])
    q2val    = np.float(q2dist[imax])
    if limb == 'nlin':
        q3val = np.float(q3dist[imax])
        q4val = np.float(q4dist[imax])
        
    if thin:
        rprsdist = rprsdist[0::thin]
        ddist = ddist[0::thin]
        tdist = tdist[0::thin]
        bdist = bdist[0::thin]
        pdist = pdist[0::thin]
        q1dist = q1dist[0::thin]
        q2dist = q2dist[0::thin]
        lp     = lp[0::thin]

    nsamp = len(ddist)


    print ''
    print 'Best fit parameters for '+name
    plt.figure(4,figsize=(8.5,11),dpi=300)    
   
    upper = np.linspace(.685,.999,100)
    lower = upper-0.6827

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
    plt.subplot(3,1,1)
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

    plt.subplot(3,1,2)    
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


    plt.subplot(3,1,3)
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

    plt.subplots_adjust(hspace=0.4)

    plt.savefig(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_params1.png', dpi=300)
    plt.clf()



# Second set of parameters

    plt.figure(5,figsize=(8.5,11),dpi=300)    

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

    plt.subplot(4,1,1)
    
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
    plt.annotate(r'$t_0$ = %.6f d' % (pdata[0,3] + bjd), xy=(0.97,0.8),
                 ha="right",xycoords='axes fraction',fontsize='large')

 # Period
    pmin = np.min(pdist)
    pmax = np.max(pdist)
    psig = np.std(pdist)
    ps = np.linspace(pmin-psig,pmax+psig,1000)
    p_kde = gaussian_kde(pdist)
    ppdf = p_kde(ps)
    pdist_c = np.cumsum(ppdf)/np.sum(ppdf)
    pfunc = sp.interpolate.interp1d(pdist_c,ps,kind='linear')
    phis = pfunc(upper)
    plos = pfunc(lower)
    ind  = np.argmin(phis-plos)
    phi  = phis[ind] 
    plo  = plos[ind] 
    pmed = np.median(pdist)
    pmode = ps[np.argmax(ppdf)]
    rbsig = rb.std(pdist)
    minval = max(pmed - sigfac*rbsig,pmin)
    maxval = min(pmed + sigfac*rbsig,pmax)
    nb = np.ceil((pmax-pmin) / (np.abs(maxval-minval)/bindiv))
    
    pout = 'Delta Period: max = {0:.3f}, med = {1:.3f}, mode = {2:.3f}, 1 sig int = {3:.3f}'
    print pout.format(pval, pmed, pmode, phi-plo)

    plt.subplot(4,1,2)
    
    plt.hist(pdist,bins=nb,normed=True)
#    plt.plot(ps,ppdf,color='c')
#    pmin = np.float(np.sort(pdist)[np.round(frac*nsamp)])
#    pmax = np.float(np.sort(pdist)[np.round((1-frac)*nsamp)])
    plt.xlim([minval,maxval])
#    plt.axvline(x=pval,color='r',linestyle='--')
#    plt.axvline(x=plo,color='c',linestyle='--')
#    plt.axvline(x=pmed,color='c',linestyle='--')
#    plt.axvline(x=phi,color='c',linestyle='--')
    plt.xlabel(r'$\Delta Period$ (seconds)')
    plt.ylabel(r'$dP/d(Period)$')
    plt.annotate(r'$P$ = %.6f d' % pdata[0,4], xy=(0.97,0.8),
                 ha="right",xycoords='axes fraction',fontsize='large')

 # q1
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
    q1med = np.median(q1dist)
    q1mode = q1s[np.argmax(q1pdf)]
    rbsig = rb.std(q1dist)
    q1min = np.min(q1dist)
    q1max = np.max(q1dist)
    minval = max(q1med - sigfac*rbsig,q1min)
    maxval = min(q1med + sigfac*rbsig,q1max)
    nb = np.ceil((q1max-q1min) / (np.abs(maxval-minval)/bindiv))
    
    q1out = 'Kipping q1 = max = {0:.3f}, med = {1:.3f}, mode = {2:.3f}, 1 sig int = {3:.3f}'
    print q1out.format(q1val,q1med, q1mode, q1hi-q1lo)

    plt.subplot(4,1,3)
    
    plt.hist(q1dist,bins=nb,normed=True)
 #    plt.plot(q1s,q1pdf,color='c')
    q1min = np.float(np.sort(q1dist)[np.round(frac*nsamp)])
    q1max = np.float(np.sort(q1dist)[np.round((1-frac)*nsamp)])
    plt.xlim([minval,maxval])
#    plt.axvline(x=q1val,color='r',linestyle='--')
#    plt.axvline(x=q1lo,color='c',linestyle='--')
#    plt.axvline(x=q1med,color='c',linestyle='--')
#    plt.axvline(x=q1hi,color='c',linestyle='--')
    plt.xlabel(r'$q_1$')
    plt.ylabel(r'$dP/q_1$')

# q2
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
    rbsig = rb.std(q2dist)
    q2min = np.min(q2dist)
    q2max = np.max(q2dist)
    minval = max(q2med - sigfac*rbsig,q2min)
    maxval = min(q2med + sigfac*rbsig,q2max)
    nb = np.ceil((q2max-q2min) / (np.abs(maxval-minval)/bindiv))
    
    q2out = 'Kipping q2 = max = {0:.3f}, med = {1:.3f}, mode = {2:.3f}, 1 sig int = {3:.3f}'
    print q2out.format(q2val,q2med, q2mode, q2hi-q2lo)

    plt.subplot(4,1,4)
    
    plt.hist(q2dist,bins=nb,normed=True)
#    plt.plot(q2s,q2pdf,color='c')
    q2min = np.float(np.sort(q2dist)[np.round(frac*nsamp)])
    q2max = np.float(np.sort(q2dist)[np.round((1-frac)*nsamp)])
    plt.xlim([minval,maxval])
#    plt.axvline(x=q1val,color='r',linestyle='--')
#    plt.axvline(x=q1lo,color='c',linestyle='--')
#    plt.axvline(x=q1med,color='c',linestyle='--')
#    plt.axvline(x=q1hi,color='c',linestyle='--')
    plt.xlabel(r'$q_2$')
    plt.ylabel(r'$dP/q_2$')

    plt.subplots_adjust(hspace=0.55)

    plt.savefig(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_params2.png',dpi=300)
    plt.clf()

   # calculate limb darkening parameters
    if limb == 'quad' or limb == 'sqrt':
        u1, u2 = qtou(q1val,q2val,limb=limb)
        ldc = [u1,u2]
    elif limb == 'nlin':
        ldc = [q1val,q2val,q3val,q4val]
    else:
        pdb.set_trace()

    plot_limb_curves(ldc=ldc,limbmodel=limb,write=True,network=network)

 
# vals
# 0 = rprsval
# 1 = duration in days
# 2 = impact parameter
# 3 = offset time in days (kepler time)
# 4 = period in days
# 5 = Kipping q1 val
# 6 = Kipping q2 val

    vals = [rprsval,dval/24.,bval,tval/(24.*3600.)+pdata[0,3],pval/(24.*3600.)+pdata[0,4],q1val,q2val]
    meds = [rprsmed,dmed/24.,bmed,tmed/(24.*3600.)+pdata[0,3],pmed/(24.*3600.)+pdata[0,4],q1med,q2med]
    modes = [rprsmode,dmode/24.,bmode,tmode/(24.*3600.)+pdata[0,3],pmode/(24.*3600.)+pdata[0,4],q1mode,q2mode]
    onesig = [rprshi-rprslo,(dhi-dlo)/24.,bhi-blo,(thi-tlo)/(24.*3600.),(phi-plo)/(24.*3600.),q1hi-q1lo,q2hi-q2lo]
    if limb == 'nlin':
        np.append(vals,q3val,q4val)
        np.append(meds,q3med,q4med)
        np.append(modes,q3mode,q4mode)
        np.append(onesig,q3hi-q3lo,q4hi-q4lo)

    bestvals = [[vals],[meds],[modes],[onesig]]

 # Plot up final fit
    if name == '4971.01':
        fmodelvals = [meds[0],meds[1],meds[2],meds[3]-pdata[0,3],meds[4],[meds[5],meds[6]]]
    else:
        fmodelvals = [vals[0],vals[1],vals[2],vals[3]-pdata[0,3],vals[4],[vals[5],vals[6]]]

    if limb == 'nlin':
        plot_model([vals[0],vals[1],vals[2],vals[3]-pdata[0,3],vals[4],\
                        [vals[5],vals[6],vals[7],vals[8]]],tag='_MCMCfit',pdf=pdf)
    else:
        plot_model(fmodelvals,tag='_MCMCfit',pdf=pdf)
        plot_final(fmodelvals,tag='_MCMCfit',pdf=pdf)


# Create KDE of Rp/Rs distribution    
    print "Output Rp/Rs and Rp distributions"
    rpdf = np.array([rprss,rprspdf])
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_RpRs.dat',rpdf.T)

# Create KDE of Rp distribution
    rstarpdf = np.random.normal(Rstar,e_Rstar,nsamp)
    rpdist = rstarpdf * rprsdist * c.Rsun/c.Rearth

    rp = np.linspace(np.min(rpdist)*0.5,np.max(rpdist)*1.5,1000)
    pdf_func = gaussian_kde(rpdist)
    rkde = pdf_func(rp)
    rpdf = np.array([rp,rkde])
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_Rp.dat',rpdf.T)

    if limb == 'nlin':
        outstr = name+ ' %.5f  %.5f  %.5f  %.5f  %.4f  %.4f  %.4f  %.4f  %.2f  %.2f  %.2f  %.2f  %.8f  %.8f  %.8f  %.8f  %.8f  %.8f  %.8f  %.8f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.0f' %  (vals[0],meds[0],modes[0],onesig[0],vals[1],meds[1],modes[1],onesig[1],vals[2],meds[2],modes[2],onesig[2],vals[3],meds[3],modes[3],onesig[3],vals[4],meds[4],modes[4],onesig[4],vals[5],meds[5],modes[5],onesig[5],vals[6],meds[6],modes[6],onesig[6],vals[7],meds[7],modes[7],onesig[7],vals[8],meds[8],modes[8],onesig[8],sampfac)
    else:
        outstr = name+ ' %.5f  %.5f  %.5f  %.5f  %.4f  %.4f  %.4f  %.4f  %.2f  %.2f  %.2f  %.2f  %.8f  %.8f  %.8f  %.8f  %.8f  %.8f  %.8f  %.8f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.0f' % (vals[0],meds[0],modes[0],onesig[0],vals[1],meds[1],modes[1],onesig[1],vals[2],meds[2],modes[2],onesig[2],vals[3],meds[3],modes[3],onesig[3],vals[4],meds[4],modes[4],onesig[4],vals[5],meds[5],modes[5],onesig[5],vals[6],meds[6],modes[6],onesig[6],sampfac)

    f = open(path+'MCMC/'+name+stag+ctag+rtag+ltag+'fit_fitparams.txt','w')
    f.write(outstr+'\n')
    f.closed

    # collect garbage
    gc.collect()
    return bestvals,rpdf




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



def triangle_plot(chains=False,lp=False,thin=False,sigfac=3.0,bindiv=75,sigsamp=5.0,pdf=False):
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
    import gc
    import matplotlib as mpl
    plt.rcdefaults()

    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.major.size'] = 5
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    fs = 14
    lw = 1.5



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
        rprsdist = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_rchain.txt')
        ddist = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_dchain.txt')*24.
        bdist = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_bchain.txt')
        tdist = (np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_t0chain.txt'))*24.*60.*60.
        pdist = (np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_pchain.txt')-pdata[0,4])*24.*3600.
        q1dist = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_q1chain.txt')
        q2dist = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_q2chain.txt')
        print '... done importing chains!'

    if lp is False:
        lp    = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_lnprob.txt')

    print done_in(tmaster)

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
    top_plot(rprsdist,gs[0,0],val=rprsval,sigfac=sigfac,bindiv=bindiv)
    print done_in(tcol)
    t = time.time()
    print "... first column plot"
    column_plot(rprsdist,ddist,gs[1,0],val1=rprsval,val2=dval,ylabel=r'$\tau_{\rm tot}$ (h)',sigfac=sigfac)
    print done_in(t)
    column_plot(rprsdist,tdist,gs[2,0],val1=rprsval,val2=tval,ylabel=r'$\Delta t_0$ (s)',sigfac=sigfac,sigsamp=sigsamp)
    column_plot(rprsdist,pdist,gs[3,0],val1=rprsval,val2=pval,ylabel=r'$\Delta P$ (s)',sigfac=sigfac,sigsamp=sigsamp)
    column_plot(rprsdist,q1dist,gs[4,0],val1=rprsval,val2=q1val,ylabel=r'$q_1$',sigfac=sigfac,sigsamp=sigsamp)
    column_plot(rprsdist,q2dist,gs[5,0],val1=rprsval,val2=q2val,ylabel=r'$q_2$',sigfac=sigfac,sigsamp=sigsamp)
    corner_plot(rprsdist,bdist,gs[6,0],val1=rprsval,val2=bval,\
                xlabel=r'$R_p/R_{*}$',ylabel=r'$b$',sigfac=sigfac)
    print "First column: "
    print done_in(tcol)

    print "... starting second column"
    t2 = time.time()
    top_plot(ddist,gs[1,1],val=dval,sigfac=sigfac,bindiv=bindiv)    
    middle_plot(ddist,tdist,gs[2,1],val1=dval,val2=tval,sigfac=sigfac,sigsamp=sigsamp)
    middle_plot(ddist,pdist,gs[3,1],val1=pval,val2=pval,sigfac=sigfac,sigsamp=sigsamp)
    middle_plot(ddist,q1dist,gs[4,1],val1=q1val,val2=q1val,sigfac=sigfac,sigsamp=sigsamp)
    middle_plot(ddist,q2dist,gs[5,1],val1=q2val,val2=q2val,sigfac=sigfac,sigsamp=sigsamp)
    row_plot(ddist,bdist,gs[6,1],val1=dval,val2=bval,xlabel=r'$\tau_{\rm tot}$ (h)',sigfac=sigfac,sigsamp=sigsamp)
    print done_in(t2)

    print "... starting third column"
    t3 = time.time()
    top_plot(tdist,gs[2,2],val=tval,sigfac=sigfac,bindiv=bindiv)    
    middle_plot(tdist,pdist,gs[3,2],val1=tval,val2=pval,sigfac=sigfac,sigsamp=sigsamp)
    middle_plot(tdist,q1dist,gs[4,2],val1=tval,val2=q1val,sigfac=sigfac,sigsamp=sigsamp)
    middle_plot(tdist,q2dist,gs[5,2],val1=tval,val2=q2val,sigfac=sigfac,sigsamp=sigsamp)
    row_plot(tdist,bdist,gs[6,2],val1=tval,val2=bval,xlabel=r'$\Delta t_0$ (s)',sigfac=sigfac,sigsamp=sigsamp)
    print done_in(t3)

    print "... starting fourth column"
    t4 = time.time()
    top_plot(pdist,gs[3,3],val=pval,sigfac=sigfac,bindiv=bindiv)    
    middle_plot(pdist,q1dist,gs[4,3],val1=pval,val2=q1val,sigfac=sigfac,sigsamp=sigsamp)
    middle_plot(pdist,q2dist,gs[5,3],val1=pval,val2=q2val,sigfac=sigfac,sigsamp=sigsamp)
    row_plot(pdist,bdist,gs[6,3],val1=tval,val2=bval,xlabel=r'$\Delta P$ (s)',sigfac=sigfac,sigsamp=sigsamp)
    print done_in(t4)


    print "... starting fifth column"
    t5 = time.time()
    top_plot(q1dist,gs[4,4],val=q1val,sigfac=sigfac,bindiv=bindiv)    
    middle_plot(q1dist,q2dist,gs[5,4],val1=q1val,val2=q2val,sigfac=sigfac,sigsamp=sigsamp)
    row_plot(q1dist,bdist,gs[6,4],val1=q1val,val2=bval,xlabel=r'$q_1$',sigfac=sigfac,sigsamp=sigsamp)
    print done_in(t5)

    print "... starting sixth column"
    t6 = time.time()
    top_plot(q2dist,gs[5,5],val=q2val,sigfac=sigfac,bindiv=bindiv)    
    row_plot(q2dist,bdist,gs[6,5],val1=q2val,val2=bval,xlabel=r'$q_2$',sigfac=sigfac,sigsamp=sigsamp)
    print done_in(t6)
  

    print "... starting the last plot"
    t7 = time.time()
    top_plot(bdist,gs[6,6],val=bval,xlabel=r'$b$',sigfac=sigfac,bindiv=bindiv)    
    print done_in(t7)

    if stag == '_long':
        titletxt = 'Long Cadence'
    elif stag == '_short':
        titletxt = 'Short Cadence'

    plt.suptitle(name+" "+titletxt+" Fit Posterior Distributions",fontsize=fs)

    print "Saving output figures"
    if pdf:
        ftype='.pdf'
    else:
        ftype='.png'
    plt.savefig(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_triangle'+ftype,dpi=300)
    plt.savefig(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_triangle.eps', format='eps',dpi=300)

    print "Procedure finished!"
    print done_in(tmaster)

    gc.collect()

    mpl.rcdefaults()

    return



def top_plot(dist,position,val=False,sigfac=4.0,minval=False,maxval=False,bindiv=20,aspect=1,xlabel=False):
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.stats.kde import gaussian_kde
    from scipy.interpolate import interp1d
    import time

#    len = np.size(dist)
#    min = np.float(np.sort(dist)[np.round(frac*len)])
#    max = np.float(np.sort(dist)[np.round((1.-frac)*len)])
#    dists = np.linspace(np.min(dist)*0.5,np.max(dist)*1.5,1000)
#    kde = gaussian_kde(dist)
#    pdf = kde(dists)
#    cumdist = np.cumsum(pdf)/np.sum(pdf)
#    func = interp1d(cumdist,dists,kind='linear')
#    lo = np.float(func(math.erfc(1./np.sqrt(2))))
#    hi = np.float(func(math.erf(1./np.sqrt(2))))
    med = np.median(dist)
#    pdb.set_trace()
    sig = rb.std(dist)
    if sig < 1.0e-5 or med < 1.0e-10:
        nb = 10
    else:
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
    ax.set_yticklabels(())
    ax.set_xlim(minval,maxval)
    xlimits = ax.get_xlim()
    ylimits = ax.get_ylim()
#    print "x range for top plot:"
#    print xlimits
    ax.set_aspect(abs((xlimits[1]-xlimits[0])/(ylimits[1]-ylimits[0]))/aspect)
    if val:
        pass
#        plt.axvline(x=val,color='w',linestyle='--',linewidth=2)
    if xlabel:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(8) 
            tick.label.set_rotation('vertical')
        ax.set_xlabel(xlabel,fontsize=12)
 
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    return


def column_plot(dist1,dist2,position,val1=False,val2=False,sigfac=2.0,sigsamp=5.0,
                min1=False,max1=False,min2=False,max2=False,ylabel=None):
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
#    print "x range for column plot:"
#    print min1,max1
#    ax.plot(val1,val2, 'wx', markersize=3)
#    ax.set_xlim(min1, max1)
#    ax.set_ylim(min2, max2)
    ax.set_xticklabels(())
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
    ax.set_ylabel(ylabel,fontsize=12)

    return



def row_plot(dist1,dist2,position,val1=False,val2=False,sigfac=3.5,sigsamp=5.0,
             min1=False,max1=False,min2=False,max2=False,xlabel=None):
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
        tick.label.set_fontsize(8) 
        tick.label.set_rotation('vertical')
    ax.set_xlabel(xlabel,fontsize=12)
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
    import constants as c
    plt.rcdefaults()

    if network == 'koi':
        net = None
    else:
        net = network


    Mstar = sdata[0,0]
    Rstar = sdata[0,1]
    Tstar = sdata[0,2]

    loggstar = np.log10( c.G * Mstar / Rstar**2. )
    
    a1,a2,a3,a4 = get_limb_coeff(Tstar,loggstar,limb='nlin',interp='nearest',network=net)
    a,b = get_limb_coeff(Tstar,loggstar,limb='quad',interp='nearest',network=net)
    c,d = get_limb_coeff(Tstar,loggstar,limb='sqrt',interp='nearest',network=net)

    thetaq,Imuq = get_limb_curve([a,b],limbmodel='quad')
    thetas,Imus = get_limb_curve([c,d],limbmodel='sqrt')
    thetan,Imun = get_limb_curve([a1,a2,a3,a4],limbmodel='nlin')
    if ldc:
        thetain,Iin = get_limb_curve(ldc,limbmodel=limbmodel)

    if write:
        plt.figure(1,figsize=(11,8.5))
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
        plt.savefig(path+'MCMC/'+str(koi)+stag+ctag+rtag+lptag+'_'+limbmodel+'.png')
        plt.clf()
    else:
        plt.ioff()

    return




def get_limb_spread(q1s,q2s,sdata=None,factor=1,limbmodel='quad',
                    fontsz=18,write=False,plot=True,network=None):

    """
    
    get_limb_spread:
    -----------------

    To Do:
    ------
    Write out distributions of q1 values so that it does not have
    to be done every refit.

    """
    plt.rcdefaults()

    if sdata == None:
        print "Must supply stellar data!"
        return
    
    if limbmodel == 'quad':
        lname = 'Quadratic'
    if limbmodel == 'sqrt':
        lname = 'Root Square'
    if limbmodel == 'nlin':
        lname = '4 Parameter'

    Mstar  = sdata[0][0]
    eMstar = sdata[1][0]/c.Msun * factor

    Rstar  = sdata[0][1]
    eRstar = sdata[1][1]/c.Rsun * factor

    Tstar  = sdata[0][2]
    eTstar = sdata[1][2] * factor

    loggstar = np.log10( c.G * Mstar / Rstar**2. )


    if plot:
        if write:
            plt.figure(101)
        else:
            plt.ion()
            plt.figure(123)
            plt.clf()

    sz = len(q1s)
    for i in range(sz):
        u1,u2 = qtou(q1s[i],q2s[i],limb=limb)
        theta,Imu = get_limb_curve([u1,u2],limbmodel=limbmodel)
        plt.plot(theta,Imu,lw=0.1,color='blue')
        
    plt.ylim([0,1.4])
    plt.tick_params(axis='both', which='major', labelsize=fontsz-2)
    plt.xlabel(r"$\theta$ (degrees)",fontsize=fontsz)
    plt.ylabel(r"$I(\theta)/I(0)$",fontsize=fontsz)
    plt.title("KOI-"+str(koi)+" limb darkening prior distribution",fontsize=fontsz)
#    plt.legend(loc=3)

    plt.annotate(r'$\Delta T_{\rm eff}$ = %.0f K' % eTstar, [0.86,0.82],horizontalalignment='right',
                 xycoords='figure fraction',fontsize='large')
    plt.annotate(r'$\Delta M_\star$ = %.2f M$_\odot$' % eMstar, [0.86,0.77],horizontalalignment='right',
                 xycoords='figure fraction',fontsize='large')
    plt.annotate(r'$\Delta R_\star$ = %.2f R$_\odot$' % eRstar, [0.86,0.72],horizontalalignment='right',
                 xycoords='figure fraction',fontsize='large')
 
    plt.annotate(r'$T_{\rm eff}$ = %.0f K' % sdata[0][2], [0.16,0.82],horizontalalignment='left',
                 xycoords='figure fraction',fontsize='large')
    plt.annotate(r'$\log(g)$ = %.2f' % loggstar, [0.16,0.77],horizontalalignment='left',
                 xycoords='figure fraction',fontsize='large')
    plt.annotate(lname, [0.16,0.72],horizontalalignment='left',
                 xycoords='figure fraction',fontsize='large')
    

    if write:
        directory = path+'MCMC/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory+str(koi)+stag+ctag+rtag+lptag+'_'+limbmodel+'fit_LDspread.png')
        plt.clf()


    plt.hist(q1s[~np.isnan(q1s)],bins=sz/70,normed=True,label=r'$q_1$')
    plt.hist(q2s[~np.isnan(q2s)],bins=sz/70,normed=True,label=r'$q_2$')
    plt.tick_params(axis='both', which='major', labelsize=fontsz-2)
    plt.title('Distribution of Kipping $q$ values',fontsize=fontsz)
    plt.xlabel(r'$q$ value',fontsize=fontsz)
    plt.ylabel('Normalized Frequency',fontsize=fontsz)
    plt.legend(loc='upper right',prop={'size':fontsz-2},shadow=True)
    plt.xlim(0,1)

    if write:
        plt.savefig(directory+str(koi)+stag+ctag+rtag+lptag+'_'+limbmodel+'qdist.png',dpi=300)
        plt.clf()
    else:
        plt.ioff()

 
    return 



def thin_chains(koi,planet,thin=10,short=False,network=None,clip=False,limbmodel='quad',rprior=False):
    
    lc,pdata,sdata = get_koi_info(koi,planet,short=short,network=network,\
                                      clip=clip,limbmodel=limbmodel,rprior=rprior)
    t = time.time()
    print 'Importing MCMC chains'
    rprsdist = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_rchain.txt')
    ddist    = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_dchain.txt')*24.
    bdist    = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_bchain.txt')
    tdist    = (np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_t0chain.txt')) * 24.0 * 3600.0 + pdata[0,3]
    pdist    = (np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_pchain.txt')-pdata[0,4])*24.*3600.
    q1dist   = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_q1chain.txt')
    q2dist   = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_q2chain.txt')
    print done_in(t)

    rprsdist = rprsdist[0::thin]
    ddist    = ddist[0::thin]
    tdist    = tdist[0::thin]
    bdist    = bdist[0::thin]
    pdist    = pdist[0::thin]
    q1dist   = q1dist[0::thin]
    q2dist   = q2dist[0::thin]

    t = time.time()
    print 'Exporting thinned chains'
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_thin_rchain.txt',rdist)
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_thin_dchain.txt',ddist)
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_thin_bchain.txt',bdist)
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_thin_t0chain.txt',tdist)
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_thin_pchain.txt',pdist)
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_thin_q1chain.txt',q1dist)
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_thin_q2chain.txt',q2dist)
    print done_in(t)
    
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
        out = tout.format(sec)
#        print tout.format(sec)
    elif np.floor(hour) == 0:
        tout = "done in {0:.0f} "+munit+" {1:.2f} seconds"
        out = tout.format(np.floor(minute),sec)
#        print tout.format(np.floor(minute),sec)
    else:
        tout = "done in {0:.0f} "+hunit+" {1:.0f} "+munit+" {2:.2f} seconds"
        out = tout.format(np.floor(hour),np.floor(minute),sec)
#        print tout.format(np.floor(hour),np.floor(minute),sec)

    print " "

    return out
 



def get_rhostar(koi,planet,short=False,network=None,clip=False,limbmodel='quad',
                rprior=False,lprior=False,thin=100,bins=100):
    plt.rcdefaults()

    lc,pdata,sdata = get_koi_info(koi,planet,short=short,network=network,\
                                  clip=clip,limbmodel=limbmodel,rprior=rprior,lprior=lprior)
    t = time.time()

    print 'Importing MCMC chains'
    rprsdist = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_rchain.txt')
    print done_in(t)
    ddist    = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_dchain.txt')
    bdist    = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_bchain.txt')
    pdist    = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_pchain.txt')
    q1dist   = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_q1chain.txt')
    q2dist   = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_q2chain.txt')
    print done_in(t)

    if thin:
        rprsdist = rprsdist[0::thin]
        ddist = ddist[0::thin]
        pdist = pdist[0::thin]
        bdist = bdist[0::thin]
        q1dist = q1dist[0::thin]
        q2dist = q2dist[0::thin]


#    ttotdist,tfulldist = midtotot(ddist,rprsdist,bdist)

    ttotdist = ddist

#    t = time.time()
#    tfulldist =[]
#    for i in np.arange(len(rprsdist)):
#        tmodel,smoothmodel,tcontact = compute_trans(rprsdist[i],ddist[i],bdist[i],0.0,
#                                                 pdist[i],[q1dist[i],q2dist[i]],
#                                                 getcontact=True)
#        ttot=tcontact[0]
#        tfull=tcontact[1]
#
#        tfulldist = np.append(tfulldist,tfull)
#        ttotdist  = np.append(ttotdist,ttot)
#        if (i+1) % 500 == 0:
#            out = done_in(t)
#            pdone = (i+1.0)/len(rprsdist) * 100.0
#            print str(pdone)+"% "+out

    aorsdist = np.sqrt( ((1+rprsdist)**2 - bdist**2)/((np.sin(np.pi*ttotdist/pdist))**2) + bdist**2)
#    aorsdist = np.sqrt( ((1-rprsdist)**2 - bdist**2)/((np.sin(np.pi*tfulldist/pdist))**2) + bdist**2)

#    aorsdist = 2.0 * np.sqrt(rprsdist) * pdist / (np.pi*np.sqrt(ttotdist**2 - tfulldist**2))
    
    rhodist =  3.0*np.pi/( c.G * (pdist*24.*3600.)**2 ) * aorsdist**3

    plt.figure(999,figsize=(8.5,11))
    plt.clf()
    plt.subplot(2,1,1)
    
    plt.hist(aorsdist,bins=bins)
    nsamp = len(aorsdist)
    frac = 0.003
    aorsmin = np.float(np.sort(aorsdist)[np.round(frac*nsamp)])
    aorsmax = np.float(np.sort(aorsdist)[np.round((1-frac)*nsamp)])
    plt.xlim([aorsmin,aorsmax])
    plt.xlim([10.8,11.1])
#    plt.axvline(x=q1val,color='r',linestyle='--')
#    plt.axvline(x=q1lo,color='c',linestyle='--')
#    plt.axvline(x=q1med,color='c',linestyle='--')
#    plt.axvline(x=q1hi,color='c',linestyle='--')
    plt.xlabel(r'$a/R_{\star}$')
    plt.ylabel(r'$N$')

    plt.subplot(2,1,2)
    
    plt.hist(rhodist,bins=bins)
    nsamp = len(rhodist)
    frac = 0.003
    rhomin = np.float(np.sort(rhodist)[np.round(frac*nsamp)])
    rhomax = np.float(np.sort(rhodist)[np.round((1-frac)*nsamp)])
    plt.xlim([rhomin,rhomax])
    plt.xlim([3.95,4.3])
#    plt.axvline(x=q1val,color='r',linestyle='--')
#    plt.axvline(x=q1lo,color='c',linestyle='--')
#    plt.axvline(x=q1med,color='c',linestyle='--')
#    plt.axvline(x=q1hi,color='c',linestyle='--')
    plt.xlabel(r'$\rho_{\star}$')
    plt.ylabel(r'$N$')
    
    plt.savefig(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_rhostar.png')

    pdb.set_trace()
    return rhodist



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
            lc,pdata,sdata = get_koi_info(koi,planet,short=True,network=network)
            chains,lp = fit_single(nwalkers=nwalkers,burnsteps=burnsteps,\
                                    mcmcsteps=mcmcsteps,clobber=clobber)
            fit,rpdf = bestvals(chains=chains)
            triangle_plot(thin=thin)
        else: pass
 
    return

#----------------------------------------------------------------------
# batchfit:
#    routine for the batch fitting of several KOIs
#----------------------------------------------------------------------
# 1-8 (-1, 255)
# 9-17 (0, done)
# 19-   on 936
# on 1141
# on 1867
# on 2329
# on 2793
# on 3444

def batchfit(start=0,stop=107,nwalkers=1000,burnsteps=1000,mcmcsteps=1000,
             clobber=False,thin=50,tthin=100,sigfac=3.5,doplots=False,short=False,
             list=False,network=None,clip=True,rprior=True,lprior=False,
             notriangle=True,bindiv=75,durmax=10,rpmax=0.5,errfac=3,
             getsamp=True,limbmodel='quad',sigsamp=5.0,pdf=False):

    import numpy as np
    import time
    import os
    import constants as c
    
    kois = np.sort(koilist(network=network))

    nkois = len(kois)
    
  
    if list:
        ns = 0
        nks = 0
        for number in np.arange(nkois):
            npl,fullname = numplanet(kois[number],network=network)
            check = isthere(kois[number],1,short=True,network=network,clip=clip)   
            if check:
                s = 's'
                ns += 1
                nks += 1*npl
            else:
                s = ''
            print number,kois[number],npl,s
        print str(ns)+' KOIs, and '+str(nks)+' planets with short cadence data '
        return

    else:
        tstart = time.time()
        for i in np.arange(start,stop+1):
            koi = kois[i]
            npl,koiname = numplanet(koi,network=network)
            for planet in np.arange(npl)+1:
                do_fit(koi,planet,nwalkers=nwalkers,burnsteps=burnsteps,
                       mcmcsteps=mcmcsteps,network=network,clobber=clobber,
                       thin=thin,sigfac=sigfac,doplots=doplots,short=short,clip=clip,
                       rprior=rprior,notriangle=notriangle,bindiv=bindiv,lprior=lprior,
                       errfac=errfac,getsamp=getsamp,rpmax=rpmax,durmax=durmax,
                       limbmodel=limbmodel,sigsamp=sigsamp,tthin=tthin,pdf=pdf)
    return



def do_fit(koi,planet,nwalkers=1000,burnsteps=1000,mcmcsteps=1000,clobber=False,
           network=None,thin=50,tthin=100,sigfac=3.5,clip=True,limbmodel='quad',doplots=False,
           short=False,rprior=True,lprior=False,notriangle=True,getsamp=True,
           errfac=3,rpmax=0.5,durmax=10,bindiv=75,sigsamp=5.0,pdf=False):
    
    import numpy as np
    import time
    import os
    import constants as c

    print ""
    print ""    
    print "Starting MCMC fitting for KOI-"+str(koi)+".0"+str(planet)

    check1 = isthere(koi,planet,short=short,network=network,clip=clip)
    if check1:
        lc,pdata,sdata = get_koi_info(koi,planet,short=short,network=network, \
                                      clip=clip,limbmodel=limbmodel,getsamp=getsamp, \
                                      rprior=rprior,lprior=lprior,errfac=errfac)
        chains,lp = fit_single(nwalkers=nwalkers,burnsteps=burnsteps, \
                                   mcmcsteps=mcmcsteps,clobber=clobber)
        if doplots != False:
            fit,rpdf = bestvals(chains=chains,lp=lp,thin=thin,rpmax=rpmax,
                                durmax=durmax,bindiv=bindiv,sigfac=sigfac,pdf=pdf)
            if notriangle != True:
                triangle_plot(thin=tthin,sigfac=sigfac,bindiv=bindiv,sigsamp=sigsamp,pdf=pdf)

    else: pass
 
    return
