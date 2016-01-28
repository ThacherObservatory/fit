# TO DO:
#-------

import sys,math,pdb,time,glob,re,os,eb,emcee
import numpy as np
import matplotlib.pyplot as plt
import constants as c
import scipy as sp
import robust as rb
from scipy.io.idl import readsav
from length import length
from kepler_tools import *
import pyfits as pf
from statsmodels.nonparametric.kernel_density import KDEMultivariate as KDE
from stellar import rt_from_m, flux2mag, mag2flux

#----------------------------------------------------------------------
# GET_PATH
#----------------------------------------------------------------------
def get_path(network=None):
    """
    get_path:
    ---------
    set path given network choice

    inputs:
    -------
    "network": name of network choices are:
               None: local path
               astro: Caltech astronomy network
               gps: Caltech geology and planetary science network
               eb: local path but for an individual KIC target
    example:
    --------
    path = get_path(network='astro')

    """

    # Set correct paths here
    if network == 'astro':
        path = '/scr2/jswift/EBs/outdata/'
    if network == 'gps':
        path = '/home/jswift/EBs/outdata/'
    if network == None:
        path = '/Users/jonswift/Astronomy/EBs/outdata/'
    if network == 'eb':
        path = '/Users/jonswift/Astronomy/EBs/'    

    return path
    


#----------------------------------------------------------------------
# EBLIST
#----------------------------------------------------------------------
def eblist(network=None):

    """ 
    ----------------------------------------------------------------------
    eblist:
    --------
    Check the list of EBs with preliminary data.

    inputs:
    -------
    "network": choose correct path (see get_path routine)
    
    example:
    --------
    In[1]: ebs = eblist(network=None)
    ----------------------------------------------------------------------
    """

    # Directories correspond to KIC numbers
    files = glob.glob(get_path(network=network)+'[1-9]*')

    # Extract KIC numbers from string
    eblist = np.zeros(len(files))
    for i in xrange(0,len(files)):
        ind = files[i].find('outdata/')+8
        eblist[i] = files[i][ind:]

    # Format output
    eblist = np.array(eblist).astype(int)

    return eblist


#----------------------------------------------------------------------
# ISTHERE
#----------------------------------------------------------------------

def isthere(ebin,short=False,network=None,clip=False,running=False):

    """ 
    ----------------------------------------------------------------------
    isthere:
    --------
    Utility for checking if the data file for an EB is on disk. Will 
    distinguish between SC and LC data and data that has been sigma 
    clipped.

    inputs:
    -------
    "ebin": KIC number of target source
    "short": search for short cadence data
    "network": keyword to set correct path
    "clip": Look for sigma clipped data
    "running": Look for data such that individual eclipse pairs can be fit
    
    example:
    --------
    In[1]: isthere(10935310,short=True,network='gps',clip=True)
    ----------------------------------------------------------------------
    """

    prefix = str(ebin)

   # Define file tags 
    type = '_short' if short else '_long'
    ctag = '_clip' if clip else ''

    path = get_path(network=network)+str(ebin)+'/Joint/' if network != 'eb' else \
        get_path(network=network)+str(ebin)+'KIC'+str(ebin)+ \
        '/lc_fit/outdata/'+str(ebin)+'/Joint/'

    if running:
        path = '/Users/jonswift/Astronomy/EBs/outdata/'+str(ebin)+'/Refine/'
        fname = str(ebin)+type+'_full.dat'
    else:
        path +='all/'
        fname = str(ebin)+'_all.dat'

    test = os.path.exists(path+fname)

    return test



def get_eb_info(kicin,short=False,limbmodel='quad',errfac=3,network=None,L3=0.0):


    """
    ----------------------------------------------------------------------
    get_eb_info:
    -------------
    Get lightcurve and a priori parameters for EB. Also set a
    bunch of global variables to be used by eb fitting routines.
    
    This routine must be run before fitting

    example:
    --------
    In[1]: info = get_eb_info(9821078,short=True,network=None,
                              clip=True,limbmodel='quad')
    
    ----------------------------------------------------------------------
    """

# Import limb coefficient module
    from scipy.interpolate import interp2d

# Define global variables
    global integration
    global path, name, kic
    global period0,ephem1,ephem2,dur1,dur2
    global ndim,limb,bjd
    global jeffries, limbprior, lerr
    global stag,ctag,ltag,rtag,lptag
    global sampfac, adapt
    global net, colors
    global lcf, lcm, ebpar0, rvdata1, rvdata2

    name = str(kicin)

    getsamp = False
    clip = False
    lprior = False
    rprior = False
    
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

    kic = kicin

# "Kepler time"
    bjd = 2454833.0

# Factor by which errors in Teff and logg are inflated for limb darkening prior
    lerr = errfac


# Setup path and info specific to EB
    path = get_path(network=network)+str(kic)+'/' if network != 'eb' else \
        get_path(network=network)+'KIC'+str(kic)+'/lc_fit/outdata/'+str(kic)+'/'

# Check if data exists
#    test = isthere(kic,short=False,network=None,clip=False,running=running)
#    if test == False:
#        sys.exit("No data for KIC "+str(kic)+"!")
        
# Use clipped data if asked for
    if clip: 
        ctag = '_clip'
    else:
        ctag = ''

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
    print "... Looking for star properties in RVfit output"
    test = os.path.exists(path+'RVs/'+str(kic)+'_rvfitparams.txt')
    if test == True:
        info2 = np.loadtxt(path+'RVs/'+str(kic)+'_rvfitparams.txt')
        Mstar1 = np.float(info2[1])*c.Msun
        eMstar1 = np.float(info2[4])*c.Msun
        Mstar2 = np.float(info2[5])*c.Msun
        eMstar2 = np.float(info2[8])*c.Msun
    else:
        Mstar1 = 0.5*c.Msun
        eMstar1 = 0.1*c.Msun
        Mstar2 = 0.4*c.Msun
        eMstar2 = 0.1*c.Msun

    # Estimate expected radius and temperature from isochrones
    Rstar1,Tstar1 = rt_from_m(Mstar1/c.Msun)
    Rstar1 *= c.Rsun
    eRstar1 = Rstar1*eMstar1/Mstar1
    eTstar1  = Tstar1*eMstar1/Mstar1
    
    Rstar2,Tstar2 = rt_from_m(Mstar2/c.Msun)
    Rstar2 *= c.Rsun
    eRstar2 = Rstar2*eMstar2/Mstar2
    eTstar2  = Tstar2*eMstar2/Mstar2

    loggstar1 = np.log10( c.G * Mstar1 / Rstar1**2. )
    dm1 = eMstar1/(Mstar1*np.log(10))
    dr1 = 2.0*eRstar1/(Rstar1*np.log(10))
    e_loggstar1 = np.sqrt(dm1**2 + dr1**2)

    loggstar2 = np.log10( c.G * Mstar2 / Rstar2**2. )
    dm2 = eMstar2/(Mstar2*np.log(10))
    dr2 = 2.0*eRstar2/(Rstar2*np.log(10))
    e_loggstar2 = np.sqrt(dm2**2 + dr2**2)


# Limb darkening parameters
    print "... getting LD coefficients for "+limbmodel+" model"
    if network == 'eb':
        net = None
    else:
        net = network

    if limbmodel == 'nlin':
        a1,b1,c1,d1 = get_limb_coeff(Tstar1,loggstar1,network=net,
                                     limb=limbmodel)
        a2,b2,c2,d2 = get_limb_coeff(Tstar2,loggstar2,network=net,
                                     limb=limbmodel)
    else:
        a1,b1 = get_limb_coeff(Tstar1,loggstar1,network=net,
                                 limb=limbmodel)
        a2,b2 = get_limb_coeff(Tstar2,loggstar2,network=net,
                                 limb=limbmodel)
  
    print ""
    if limbmodel == 'quad':
        ltag = '_quad'
        ldc1  = [a1,b1]
        eldc1 = [0,0]
        print  "Limb darkening coefficients for primary star:"
        out = '     u1 = {0:.4f}, u2 = {1:.4f}'
        print out.format(a1,b1)

        ldc2  = [a2,b2]
        eldc2 = [0,0]
        print  "Limb darkening coefficients for secondary star:"
        out = '     u1 = {0:.4f}, u2 = {1:.4f}'
        print out.format(a2,b2)
        print " "        
    elif limbmodel == 'sqrt':
        ltag = '_sqrt'
        ldc1  = [b1,a1]
        eldc1 = [0,0]
        print  "Limb darkening coefficients for primary star:"
        out = '     u1 = {0:.4f}, u2 = {0:.4f}'
        print out.format(b1,a1)

        ldc2  = [b2,a2]
        eldc2 = [0,0]
        print  "Limb darkening coefficients for secondary star:"
        out = '     u1 = {0:.4f}, u2 = {0:.4f}'
        print out.format(b2,a2)
        print " "
    elif limbmodel == 'nlin':
        ltag = '_nlin'
        ldc1 = [a1,b1,c1,d1]
        eldc1 = [0,0,0,0]
        print  "Limb darkening coefficients for primary star:"
        out = '     c1 = {0:.4f},  c2 = {0:.4f, c3 = {0:.4f}, c4 = {0:.4f}'
        print out.format(a1,b1,c1,d1)
        ldc2 = [a2,b2,c2,d2]
        eldc2 = [0,0,0,0]
        print  "Limb darkening coefficients for primary star:"
        out = '     c1 = {0:.4f},  c2 = {0:.4f, c3 = {0:.4f}, c4 = {0:.4f}'
        print out.format(a2,b2,c2,d2)
    else:
        print "Limb darkening law not recognized"
        return

    limb = limbmodel


# Determine number of samples per integration time by interpolation using
# LM fit parameters as a guide.
# !!!
# Need to run test to create grid for the EBs
# !!!
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
        if sampfac % 2 == 0:
            sampfac += 1
        print "Using a sampling of "+str(sampfac)+ \
            " to account for integration time of "+str(integration)+" s"
    else:
        print 'Will use default number of samples per integration time'

# Kepler magnitude
    colors = kic_colors(kic)
    kpmag = colors["kepmag"][0]

# Eclipse information
    einfo = np.loadtxt(path+'Refine/'+str(kic)+'.out',delimiter=',')
    period0 = einfo[1]
    eperiod0 = einfo[2]
    ephem1 = einfo[3]
    eephem1 = einfo[4]
    ephem2 = einfo[9]
    eephem2 = einfo[10]
    dur1 = einfo[7]/24.0
    dur2 = einfo[13]/24.0
    ecosw0 = einfo[15]
    esinw0 = einfo[15]
    ecc0 = np.sqrt(ecosw0**2+esinw0**2)
    sma0 = ((period0*24.0*3600)**2*c.G*(Mstar1+Mstar2)/(4.0*np.pi**2))**(1.0/3.0)
    dt12 = (ephem2-ephem1) % period0
    rinfo  = np.loadtxt(path+'Rotation/'+str(kic)+'_rot.out',delimiter=',',dtype='string')
    prot = np.float(rinfo[4])

# Read in RV data
    rvdata = get_rv_data(kic,network=network)
    vsys = rvdata['vsys']

# Approximate orbital parameters (for initial guesses)
    esq = ecosw0**2+esinw0**2
    roe = np.sqrt(1.0-esq)
    sini = 1 #np.sqrt(1.0-cosi**2)
    qpo = 1.0+Mstar2/Mstar1
    gamma = vsys
    omega = 2.0*np.pi*(1.0 + gamma*1000/eb.LIGHT) / (period0*86400.0)
    ktot = (c.G*(Mstar1+Mstar2) * omega * sini)**(1.0/3.0)*roe / 1e5


# Make EB data array
    ebpar0 = {'J':(Tstar2/Tstar1)**4, 'Rsum_a':(Rstar1+Rstar2)/sma0, 'Rratio':Rstar2/Rstar1,
              'Mratio':Mstar2/Mstar1, 'LDlin1':a1, 'LDnon1':b1, 'LDlin2':a2, 'LDnon2':b2,
              'GD1':0.32, 'Ref1':0.4, 'GD2':0.32, 'Ref2':0.4, 'Rot1':prot/period0,
              'ecosw':ecosw0, 'esinw':esinw0, 'Period':period0, 't01':ephem1, 't02':ephem2, 
              'et01':ephem1, 'et02':eephem2, 'dt12':dt12, 'tdur1':dur1, 'tdur2':dur2, 
              'mag0':kpmag,'vsys':vsys, 'Mstar1':Mstar1/c.Msun, 'Mstar2':Mstar2/c.Msun,
              'ktot':ktot, 'L3':L3,'Period':period0, 'ePeriod':eperiod0,
              'integration':integration}

    return ebpar0



def get_lc_data(kic,short=False,quarter=None,exclude=None,
                sap=False,sourcefile=False):

    # lcf[0,:]  = time
    # lcf[1,:] = median normalized flux
    # lcf[2,:] = eflux

    global lcf, lcm

    data = readdata(kic,short=False,quarter=None,exclude=None,
             sap=False,sourcefile=False)

    colors = kic_colors(kic)
    
    refflux = np.median(data['norm'])
    flux = data['flux']
    e_flux = data['eflux']
    refmag = colors['kepmag'][0]
    mag,e_mag = flux2mag(flux,e_flux,refmag,refflux)
    
    lcf = np.array([data['time'],flux,e_flux,data['norm']])
    lcm = np.array([data['time'],mag,e_mag])
    
    return lcf,lcm



def get_rv_data(kic,network=None):

    global rvdata1, rvdata2, vsys

# Read in RV data
    path = get_path(network=network)+str(kic)+'/' if network != 'eb' else \
        get_path(network=network)+'KIC'+str(kic)+'/lc_fit/outdata/'+str(kic)+'/'

    test1 = os.path.exists(path+'RVs/'+str(kic)+'_comp1.dat')
    test2 = os.path.exists(path+'RVs/'+str(kic)+'_comp2.dat')
    if test1 and test2:
        rvdata1 = np.loadtxt(path+'RVs/'+str(kic)+'_comp1.dat',usecols=[0,1,2])
        rvdata1[:,0] -= bjd
        rvdata1[:,2][np.isnan(rvdata1[:,2])] = np.nanmax(rvdata1[:,2])
        rvdata2 = np.loadtxt(path+'RVs/'+str(kic)+'_comp2.dat',usecols=[0,1,2])
        rvdata2[:,0] -= bjd
        rvdata2[:,2][np.isnan(rvdata2[:,2])] = np.nanmax(rvdata2[:,2])        
        maxval = max(np.max(rvdata1[:,1]),np.max(rvdata2[:,1]))
        minval = min(np.min(rvdata1[:,1]),np.min(rvdata2[:,1]))
        vsys = (maxval+minval)/2.0
    else:
        rvdata1, rvdata2 = False, False
        vsys = 0.0

    data = {'RV1':rvdata1, 'RV2':rvdata2, 'vsys':vsys}

    return data


def eclipse_times(lcf,info):
    """
    eclipse_times:
    --------------
    Returns the mid primary eclipse times for the entire Kepler data set in BKJD

    """

    t  = lcf[0,:]
    tdur1 = info['tdur1']
    tdur2 = info['tdur2']
    integration = info['integration']
    
    period = info['Period']
    t01 = info['t01']
    t02 = info['t02']
    dt12 = info['dt12']

    # Get the mid-times of the primary eclipses
    tpe    = (np.arange(4000)-2000.0) * period0 + ephem1 - bjd
    tse    = (np.arange(4000)-2000.0) * period0 + ephem2 - bjd
    ifirst = np.where(tpe > np.min(t))[0][0]
    ilast  = np.where(tpe+dt12 < max(t))[0][-1]
    if ilast < ifirst:
        tpiter = [-999]
    else:
        tpiter = tpe[ifirst:ilast]  

    return tpiter



#----------------------------------------------------------------------
# SELECT_ECLIPSE
#----------------------------------------------------------------------

def select_eclipse(enum,info,durfac=2.25,gfrac=0.9,fbuf=1.2,order=3,despot=False,
                   maxchi=False,thin=1,plot=True):

    """
    select_eclipse:
    ---------------
    Select the data from a given eclipse number. Eclipses are numbered starting at the
    first eclipse observed by Kepler.

    inputs:
    -------
    "durfac": fraction of duration that is fit
    "gfrac" : fraction of good data points per eclipse to fit
    "fbuf"  : buffer for full sampling of the light curve outside of both primary and 
              secondary eclipses
    "order" : polynomial order to fit out spots.
    "despot": this was an attempt to correct for out of eclipse brightenss due to spot 
              variations on one star. DO NOT USE: this cannot be done apriori.
    """

    index = enum
    
    t   = lcf[0,:]
    x   = lcf[1,:]
    ex  = lcf[2,:]
    norm = lcf[3,:]


    tdur1 = info['tdur1']
    tdur2 = info['tdur2']
    integration = info['integration']

    period = info['Period']
    t01 = info['t01']
    t02 = info['t02']
    dt12 = info['dt12']

    tpiter = eclipse_times(lcf,info)
    
    tref = tpiter[index]
    trefsec = tref + dt12

    keep, = np.where( (t >= (tref - durfac*tdur1)) & (t <= trefsec + durfac*tdur2))
      
    if length(keep) > 0:
        tsnip = t[keep]
        xsnip = x[keep]
        exsnip = ex[keep]
        if length(np.unique(norm[keep])) > 1:
            print "Normalization over eclipse region varies! Do not use this eclipse"
            return {'status':-1}
        nsnip = np.median(norm[keep])
    else:
        print 'No data for eclipse #'+str(index)
        return {'status':-1}      
           
    if plot:
        plt.figure(99)
        plt.clf()
        plt.plot(tsnip,xsnip/nsnip,'ko')
        plt.xlabel('BKJD (days)')
        plt.ylabel('Relative Flux')
        
# Identify regions of interest in the snippet
    binds, = np.where(tsnip < (tref - fbuf*tdur1))
    bct = length(binds)
    
    pinds, = np.where( (tsnip >= (tref - fbuf*tdur1)) &
                       (tsnip <= (tref + fbuf*tdur1)) )
    pct = length(pinds)
    
    minds, = np.where( (tsnip >= (tref + fbuf*tdur1)) &
                       (tsnip <= (trefsec - fbuf*tdur2)) )
    mct = length(minds)
    
    sinds, = np.where( (tsnip >= (trefsec - fbuf*tdur2)) &
                       (tsnip <= (trefsec + fbuf*tdur2)) )
    sct = length(sinds)
    
    einds, = np.where( (tsnip >= (trefsec + fbuf*tdur2/2.0)) )
    ect = length(einds)
    

    if pct < gfrac*tdur1*24*3600./integration or sct < gfrac*tdur2*24*3600./integration:
        print 'Not enough data in primary or secondary eclipses for eclipse #'+str(index)
        return {'status':-1}

    if plot:
        plt.plot(tsnip[binds],xsnip[binds]/nsnip,'ro')
        plt.plot(tsnip[minds],xsnip[minds]/nsnip,color='orange',marker='o')
        plt.plot(tsnip[einds],xsnip[einds]/nsnip,'ro')
        
# Extract only the primary and secondary eclipses 
    pi, =  np.where( (tsnip >= (tref - durfac*tdur1)) &
                     (tsnip <= (tref + durfac*tdur1)) )
    pct2 = length(pi)

    si, =  np.where( (tsnip >= (trefsec - durfac*tdur2)) &
                     (tsnip <= (trefsec + durfac*tdur2)) )
    sct2 = length(si)
    
    pfiti = np.array(list((set(pi)-set(pinds))))
    sfiti = np.array(list((set(si)-set(sinds))))

    # Enforce specified number of points outside of eclipse      
    outmin = 8
    if length(pfiti) < outmin or length(sfiti) < outmin:
        return {'status':-1}
      
    # Detrend Primary
    tpfit = tsnip[pfiti] - tref
    xpfit = xsnip[pfiti]
    p_fit = np.polyfit(tpfit,xpfit,order)
    fitcurve = np.polyval(p_fit,tsnip[pi]-tref)
    rescurve = np.polyval(p_fit,tsnip[pfiti]-tref)
    fm1 = np.polyval(p_fit,0)/nsnip

    tprim = tsnip[pi]
    xprim = xsnip[pi]
    eprim = exsnip[pi]
    ol1prim = fitcurve
    if plot:
        plt.plot(tprim,fitcurve/nsnip,'g',linewidth=2)

# Do not use "despot" option... it does not work!
    if despot:
        sys.exit("Please don't use the 'despot' feature!")
#        cont = fitcurve - nsnip
#        xprim = (xsnip[pi] - cont)/(fitcurve - cont)
#        eprim = np.zeros(length(pi))+np.std(xsnip[pfiti]/rescurve)
    else:
        xprim_cor = xsnip[pi]/fitcurve
        eprim_cor = np.zeros(length(pi))+np.std(xsnip[pfiti]/rescurve)

    chisqp = np.sum(((xpfit-rescurve)**2/exsnip[pfiti]**2)/(np.float(length(pfiti))-np.float(order)-1))

    if maxchi:
        if chisqp > maxchi:
            return {'status':-1}

    # Detrend Secondary
    tsfit = tsnip[sfiti] - tref
    xsfit = xsnip[sfiti]
    s_fit = np.polyfit(tsfit,xsfit,order)
    fitcurve = np.polyval(s_fit,tsnip[si]-tref)
    rescurve = np.polyval(s_fit,tsnip[sfiti]-tref)
    fm2 = np.polyval(s_fit,0)/nsnip

    tsec = tsnip[si]
    xsec = xsnip[si]
    esec = exsnip[si]
    ol1sec = fitcurve

    
    if plot:
        directory = path+'MCMC/singlefits/E'+str(enum)+'/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.plot(tsnip[si],fitcurve/nsnip,'g',linewidth=2)
        plt.title('Eclipse #'+str(enum))
        plt.savefig(get_path()+name+'/MCMC/singlefits/E'+str(enum)+'/'+name+stag+'_E'+str(enum)+'.pdf')

    if despot:
        sys.exit("Please don't use the 'despot' feature!")
    else:
        xsec_cor = xsnip[si]/fitcurve
        esec_cor = np.zeros(length(si))+np.std(xsnip[sfiti]/rescurve)

    chisqs = np.sum(((xsfit-rescurve)**2/exsnip[sfiti]**2)/(np.float(length(sfiti))-np.float(order)-1))

    if maxchi:
        if chisqs > maxchi:
            return {'status':-1}


# indices for the out of eclipse regions
    eoeinds = np.array(list(binds[0::thin]) + list(minds[0::thin]) + list(einds[0::thin]))
    tout  = tsnip[eoeinds]
    xout  = xsnip[eoeinds]
    exout = exsnip[eoeinds]
    

    keep = np.array(list(binds[0::thin]) + list(pinds) + list(minds[0::thin]) + 
                    list(sinds) + list(einds[0::thin]))
    tall = tsnip[keep]
    xall = xsnip[keep]
    eall = exsnip[keep]

    xmag,exmag  = flux2mag(xall/nsnip,eall/nsnip,colors['kepmag'][0],1)

    status = 0


    eclipse = {'time':tall, 'flux':xall, 'err':eall, 'mag':xmag, 'magerr':exmag,
               'flev1':fm1, 'flev2':fm2, 'status':status, 
               'thin':thin, 'despot':despot, 'norm': nsnip, 'chisqp':chisqp, 'chisqs':chisqs, 
               'tout':tout, 'xout':xout, 'exout':exout, 'enum':enum,
               'tprim':tprim, 'xprim':xprim, 'eprim':eprim,
               'xprim_cor':xprim_cor, 'eprim_cor': eprim_cor,
               'xsec_cor':xsec_cor, 'esec_cor': esec_cor,
               'tsec':tsec, 'xsec':xsec, 'esec':esec,
               'pfit':p_fit, 'sfit':s_fit,'tref1':tref, 'tref2':trefsec}
  
    return eclipse



def vec_to_params(x):
    """
    ----------------------------------------------------------------------
    vec_to_params:
    --------------
    Function to convert a vector of input parameters into a parameter vector
    that is compatible with eb.model

    x[0]  = surface brightness ratio
    x[1]  = fractional sum of stellar radii (R1 + R2)/a
    x[2]  = ratio of the radii (R2/R1)
    x[3]  = cosine of the inclination 
    x[4]  = ecosw
    x[5]  = esinw
    x[6]  = magoff
    x[7]  = ephemeris t0
    x[8]  = period
    x[9]  = LD lin star 1 
    x[10]  = LD non-lin star 1 
    x[11] = LD lin star 2 
    x[12] = LD non-lin star 2
    x[13] = mass ratio
    x[14] = third light
    *** Spot models ***
    * Star 1
    x[15] = rotation parameter (Prot/Porbit)    
    x[16] = fraction of spots eclipsed
    x[17] = base spottedness out of eclipse
    x[18] = amplitude of sin component 
    x[19] = amplitude of cos component   
    x[20] = amplitude of sincos component
    x[21] = amplitude of (cos^2 - sin^2) component
    * Star 2
    x[22] = rotation parameter (Prot/Porbit)    
    x[23] = fraction of spots eclipsed
    x[24] = base spottedness out of eclipse
    x[25] = amplitude of sin component 
    x[26] = amplitude of cos component   
    x[27] = amplitude of sincos component
    x[28] = amplitude of (cos^2 - sin^2) component
    *** system velocity and mag offset is last since it is not in any of the eb attributes
    x[29] = total radial velocity amplitude
    x[30] = system velocity

    To do:
    ------
    Check if this is indeed correct!!!!

    """
    parm = np.zeros(eb.NPAR, dtype=np.double)
    # These are the basic parameters of the model.
    parm[eb.PAR_J]      =  x[0]  # J surface brightness ratio
    parm[eb.PAR_RASUM]  =  x[1]  # (R_1+R_2)/a
    parm[eb.PAR_RR]     =  x[2]  # R_2/R_1
    parm[eb.PAR_COSI]   =  x[3]  # cos i

    # Orbital parameters.
    parm[eb.PAR_ECOSW]  =  x[4]   # ecosw
    parm[eb.PAR_ESINW]  =  x[5]   # esinw

    # Period
    try:
        parm[eb.PAR_P] = x[variables == 'period'][0]
    except:
        parm[eb.PAR_P] = ebpar0["Period"]
        
    # T0
    try:
        parm[eb.PAR_T0] =  x[variables == 't0'][0]   # T0 (epoch of primary eclipse)
    except:
        parm[eb.PAR_T0] = ebpar0['t01']-bjd

    # offset magnitude
    try:
        parm[eb.PAR_M0] =  x[variables == 'magoff'][0]  
    except:
        parm[eb.PAR_M0] = ebpar0['mag0']

        
    # Limb darkening paramters for star 1
    try:
        q1a = x[variables == 'q1a'][0]  
        q2a = x[variables == 'q2a'][0]  
        a1, b1 = qtou(q1a,q2a,limb=limb)
        parm[eb.PAR_LDLIN1] = a1  # u1 star 1
        parm[eb.PAR_LDNON1] = b1  # u2 star 1
    except:
        parm[eb.PAR_LDLIN1] = ebpar0["LDlin1"]   # u1 star 1
        parm[eb.PAR_LDNON1] = ebpar0["LDnon1"]   # u2 star 1


    # Limb darkening paramters for star 2
    try:
        q1b = x[variables == 'q1b'][0]  
        q2b = x[variables == 'q2b'][0]  
        a2, b2 = qtou(q1b,q2b)
        parm[eb.PAR_LDLIN2] = a2  # u1 star 2
        parm[eb.PAR_LDNON2] = b2  # u2 star 2
    except:
        parm[eb.PAR_LDLIN2] = ebpar0["LDlin2"]   # u1 star 2
        parm[eb.PAR_LDNON2] = ebpar0["LDnon2"]   # u2 star 2


    # Mass ratio is used only for computing ellipsoidal variation and
    # light travel time.  Set to zero to disable ellipsoidal.
    try:
        parm[eb.PAR_Q]  = x[variables == 'massratio'][0]
        ktot  = x[variables == 'ktot'][0]
        vsys  = x[variables == 'vsys'][0]
    except:
        parm[eb.PAR_Q]  = ebpar0['Mratio']
        ktot = ebpar0['ktot']
        vsys = ebpar0['vsys']

    try:
        parm[eb.PAR_L3] = x[variables == 'L3'][0]
    except:
        parm[eb.PAR_L3] = ebpar0["L3"]
    
    # Light travel time coefficient.
    if fitlighttravel:        
        try:
            cltt = ktot / eb.LIGHT
            parm[eb.PAR_CLTT]   =  cltt      # ktot / c
        except:
            print "Cannot perform light travel time correction (no masses)"
            ktot = 0.0
#    else:
#        ktot = 0.0


    if usegravdark:
        parm[eb.PAR_GD1]    = ebpar0['GD1']   # gravity darkening, std. value
        parm[eb.PAR_GD2]    = ebpar0['GD2']   # gravity darkening, std. value

    if usereflection:
        parm[eb.PAR_REFL1]  = ebpar0['Ref1']  # albedo, std. value
        parm[eb.PAR_REFL2]  = ebpar0['Ref2']  # albedo, std. value


    # Spot model
    try: 
        parm[eb.PAR_ROT1]   = x[variables == 'Rot1'][0] # rotation parameter (1 = sync.)
        parm[eb.PAR_OOE1O]  = x[variables == 'spBase1'][0]  # base spottedness out of eclipse
        parm[eb.PAR_OOE11A] = x[variables == 'spSin1'][0] # amplitude of sine component
        parm[eb.PAR_OOE11B] = x[variables == 'spCos1'][0] # amplitude of cosine component
        parm[eb.PAR_OOE12A] = x[variables == 'spSinCos1'][0] # amplitude of sincos cross term
        parm[eb.PAR_OOE12B] = x[variables == 'spSqSinCos1'][0] # amplitude of sin^2 + cos^2 term
    except:
        pass

    try:
        parm[eb.PAR_FSPOT1] = x[variables == 'spFrac1'][0]  # fraction of spots eclipsed
    except:
        pass
    
    try: 
        parm[eb.PAR_ROT2]   = x[variables == 'Rot2'][0] # rotation parameter (1 = sync.)
        parm[eb.PAR_OOE2O]  = x[variables == 'spBase2'][0]  # base spottedness out of eclipse
        parm[eb.PAR_OOE21A] = x[variables == 'spSin2'][0] # amplitude of sine component
        parm[eb.PAR_OOE21B] = x[variables == 'spCos2'][0] # amplitude of cosine component
        parm[eb.PAR_OOE22A] = x[variables == 'spSinCos2'][0] # amplitude of sincos cross term
        parm[eb.PAR_OOE22B] = x[variables == 'spSqSinCos2'][0] # amplitude of sin^2+cos^2 term
    except:
        pass

    try:
        parm[eb.PAR_FSPOT2] = x[variables == 'spFrac2'][0]  # fraction of spots eclipsed
    except:
        pass
    

    # OTHER NOTES:
    #
    # To do standard transit models (a'la Mandel & Agol),
    # set J=0, q=0, cltt=0, albedo=0.
    # This makes the secondary dark, and disables ellipsoidal and reflection.
    #
    # The strange parameterization of radial velocity is to retain the
    # flexibility to be able to model just light curves, SB1s, or SB2s.
    #
    # For improved precision, it's best to subtract most of the "DC offset"
    # from T0 and the time array (e.g. take off the nominal value of T0 or
    # the midtime of the data array) and add it back on at the end when
    # printing parm[eb.PAR_T0] and vder[eb.PAR_TSEC].  Likewise the period
    # can cause scaling problems in minimization routines (because it has
    # to be so much more precise than the other parameters), and may need
    # similar treatment.
    
    # Simple (but not astronomer friendly) dump of model parameters.

#    print "Model parameters:"
#    for name, value, unit in zip(eb.parnames, parm, eb.parunits):
#        print "{0:<10} {1:14.6f} {2}".format(name, value, unit)
        
    # Derived parameters.
    vder = eb.getvder(parm, vsys, ktot)

#    print "stop in vec_to_parm"
#    pdb.set_trace()
#    print mass1, mass2, ktot, vsys

#    print "Derived parameters:"
#    for name, value, unit in zip(eb.dernames, vder, eb.derunits):
#        print "{0:<10} {1:14.6f} {2}".format(name, value, unit)
    return parm, vder



def compute_eclipse(t,parm,modelfac=11.0,fitrvs=False,tref=None,
                    period=None,ooe1fit=None,ooe2fit=None,unsmooth=False):

    """
    ----------------------------------------------------------------------
    compute_eclipse:
    --------------
    Function to compute eclipse light curve and RV data for given input model.

    options:
    --------

    examples:
    ---------
            
    """
    
    if fitrvs:
        typ = np.empty_like(t, dtype=np.uint8)
        typ.fill(eb.OBS_VRAD1)        
        rv = eb.model(parm, t, typ)
        return rv

    else:

        if not tref or not period:
            print "Must supply a reference time AND period for "\
                +"calculation of a light curve"

            
        # Create array of time offset values that we will average over to account
        # for integration time
        dvec = np.linspace(-1*(modelfac-1)/2,(modelfac-1)/2,num=modelfac)
        dvec *= integration/(np.float(modelfac)*24.0*3600.0)
    
        # Expand t vector into an array of the same dimensions
        ones = np.ones(len(t))
        darr = np.outer(dvec,ones)    
        tarr = np.reshape(np.tile(t,modelfac),(modelfac,len(t)))
        
        # Combine differential array and time array to get "modelfac" number of
        # time sequences
        tdarr = darr + tarr

        phiarr = foldtime(tdarr,t0=tref,period=period)/period

        if (np.max(phiarr) - np.min(phiarr)) > 0.5:
            phiarr[phiarr < 0] += 1.0
        
        # Compute ol1 and ol2 vectors if needed
        if np.shape(ooe1fit):
            ol1 = np.polyval(ooe1fit,phiarr)
        else:
            ol1 = None

        if np.shape(ooe2fit):
            ol2 = np.polyval(ooe2fit,phiarr)
        else:
            ol2 = None

        typ = np.empty_like(tdarr, dtype=np.uint8)

        # can use eb.OBS_LIGHT to get light output or
        # eb.OBS_MAG to get mag output        
        typ.fill(eb.OBS_LIGHT)
        if length(ol1) == length(phiarr) and length(ol2) == length(phiarr):
            yarr = eb.model(parm, phiarr, typ, eb.FLAG_PHI, ol1=ol1, ol2=ol2)
        if length(ol1) == length(phiarr) and not length(ol2) == length(phiarr):
            yarr = eb.model(parm, phiarr, typ, eb.FLAG_PHI, ol1=ol1)
        if length(ol2) == length(phiarr) and not length(ol1) == length(phiarr):
            yarr = eb.model(parm, phiarr, typ, eb.FLAG_PHI, ol2=ol2)
            
        # Average over each integration time
        smoothmodel = np.sum(yarr,axis=0)/np.float(modelfac)
        model = yarr[(modelfac-1)/2,:]

        # Return unsmoothed model if requested
        if unsmooth:
            return model
        else:
            return smoothmodel



def get_teffs_loggs(parm,vsys,ktot):

    # Get physical parameters
    ecosw = parm[eb.PAR_ECOSW]
    esinw = parm[eb.PAR_ESINW]
    cosi = parm[eb.PAR_COSI]
    mrat = parm[eb.PAR_Q]
    period = parm[eb.PAR_P]
    rrat = parm[eb.PAR_RR]
    rsum = parm[eb.PAR_RASUM]
    sbr = parm[eb.PAR_J]
    esq = ecosw * ecosw + esinw * esinw
    roe = np.sqrt(1.0 - esq)
    sini = np.sqrt(1.0 - cosi*cosi)
    qpo = 1.0 + mrat
    # Corrects for doppler shift of period
    omega = 2.0*np.pi*(1.0 + vsys*1000.0/eb.LIGHT) / (period*86400.0)
    tmp = ktot*1000.0 * roe
    sma = tmp / (eb.RSUN*omega*sini)
    mtot = tmp*tmp*tmp / (eb.GMSUN*omega*sini)
    m1 = mtot / qpo
    m2 = mrat * m1

    r1 = sma*rsum/(1+rrat)
    r2 = rrat*r1

    # extra 100 because logg is in cgs!
    logg1 = np.log10(100.0*eb.GMSUN*m1/(r1*eb.RSUN)**2)
    
    T1 = np.random.normal(4320,200,1)[0]
        
    # extra 100 because logg is in cgs!
    logg2 = np.log10(100.0*eb.GMSUN*m2/(r2*eb.RSUN)**2)

    # Assumes surface brightness ratio is effective temperature ratio to the 4th power
    T2 = T1 * sbr**(1.0/4.0)

    return T1,logg1,T2,logg2

def lnprob_single(x):

    """
    ----------------------------------------------------------------------
    lnprob:
    -------
    Function to compute logarithmic probability of data given model. This
    function sets prior constaints explicitly and calls compute_trans to
    compare the data with the model. Only data within the smoothed transit
    curve is compared to model. 

    """

    parm,vder = vec_to_params(x)
    
    vsys = x[-1]
    ktot = x[-2]

    if claret:
        T1,logg1,T2,logg2 = get_teffs_loggs(parm,vsys,ktot)

        u1a = ldc1func(T1,logg1)[0][0]
        u2a = ldc2func(T1,logg1)[0][0]
        
        u1b = ldc1func(T2,logg2)[0][0]
        u2b = ldc2func(T2,logg2)[0][0]
        
        q1a,q2a = utoq(u1a,u2a,limb=limb)        
        q1b,q2b = utoq(u1b,u2b,limb=limb)
        
        parm[eb.PAR_LDLIN1] = u1a  # u1 star 1
        parm[eb.PAR_LDNON1] = u2a  # u2 star 1
        parm[eb.PAR_LDLIN2] = u1b  # u1 star 2
        parm[eb.PAR_LDNON2] = u2b  # u2 star 2

    elif fitlimb:
        q1a = x[variables == 'q1a'][0]  
        q2a = x[variables == 'q2a'][0]  
        q1b = x[variables == 'q1b'][0]  
        q2b = x[variables == 'q2b'][0]  
        u1a = parm[eb.PAR_LDLIN1]
        u2a = parm[eb.PAR_LDNON1]
        u1b = parm[eb.PAR_LDLIN2]
        u2b = parm[eb.PAR_LDNON2]
    else:
        q1a = 0.5 ; q2a = 0.5 ; q1b = 0.5 ; q2b = 0.5
        u1a = 0 ; u2a = 0 ; u1b = 0 ; u2b = 0
        
    # Exclude conditions that give unphysical limb darkening parameters
    if q1b > 1 or q1b < 0 or q2b > 1 or q2b < 0 or np.isnan(q1b) or np.isnan(q2b):
        return -np.inf        
    if q1a > 1 or q1a < 0 or q2a > 1 or q2a < 0 or np.isnan(q1a) or np.isnan(q2a):
        return -np.inf        
    
    # Sometimes the gridding function fails
    if np.isnan(u1a) or np.isnan(u2a) or np.isnan(u1b) or np.isnan(u2b):
        return -np.inf

    # Priors
    if fitL3:
        if parm[eb.PAR_L3] > 1 or parm[eb.PAR_L3] < 0:
            return -np.inf

    # Need to understand exactly what this parameter is!!
    if fitsp1:
        if parm[eb.PAR_FSPOT1] < 0 or parm[eb.PAR_FSPOT1] > 1:
            return -np.inf
        coeff1 = []
        for i in range(fitorder+1):
            coeff1 = np.append(coeff1,x[variables == 'c'+str(i)+'_1'])
        
### Compute eclipse model for given input parameters ###

    massratio = parm[eb.PAR_Q]
    if not fitellipsoidal:
        parm[eb.PAR_Q] = 0.0

    # Primary eclipse
    t0 = parm[eb.PAR_T0]
    period = parm[eb.PAR_P]
    tprim = fitdict['tprim']
    norm = fitdict['norm']
    xprim = fitdict['xprim']/norm
    eprim = fitdict['eprim']/norm

    sm1  = compute_eclipse(tprim,parm,fitrvs=False,tref=t0,period=period,ooe1fit=coeff1)

    # Log Likelihood Vector
    lfi1 = -1.0*(sm1 - xprim)**2/(2.0*eprim**2)

    # Log likelihood
    lf1 = np.sum(lfi1)

    # Secondary eclipse
    tsec = fitdict['tsec']
    xsec = fitdict['xsec']/norm
    esec = fitdict['esec']/norm

    if fitsp1:
        coeff2 = []
        for i in range(fitorder+1):
            coeff2 = np.append(coeff2,x[variables == 'c'+str(i)+'_2'])

    sm2  = compute_eclipse(tsec,parm,fitrvs=False,tref=t0,period=period,ooe1fit=coeff2)
    
    # Log Likelihood Vector
    lfi2 = -1.0*(sm2 - xsec)**2/(2.0*esec**2)
    
    # Log likelihood
    lf2 = np.sum(lfi2)

    lf = lf1+lf2

    
    # need this for the RVs!
    parm[eb.PAR_Q] = massratio

    if fitrvs:
        if (vsys > max(np.max(rvdata1[:,1]),np.max(rvdata2[:,1]))) or \
           (vsys < min(np.min(rvdata1[:,1]),np.min(rvdata2[:,1]))): 
            return -np.inf
        rvmodel1 = compute_eclipse(rvdata1[:,0],parm,fitrvs=True)
        k2 = ktot/(1+massratio)
        k1 = k2*massratio
        rv1 = rvmodel1*k1 + vsys
        rvmodel2 = compute_eclipse(rvdata2[:,0],parm,fitrvs=True)
        rv2 = -1.0*rvmodel2*k2 + vsys
        lfrv1 = -np.sum((rv1 - rvdata1[:,1])**2/(2.0*rvdata1[:,2]))
        lfrv2 = -np.sum((rv2 - rvdata2[:,1])**2/(2.0*rvdata2[:,2]))
        lfrv = lfrv1 + lfrv2
        lf  += lfrv

    debug = False
    if debug:
        print "Model parameters:"
        for nm, vl, unt in zip(eb.parnames, parm, eb.parunits):
            print "{0:<10} {1:14.6f} {2}".format(nm, vl, unt)

        vder = eb.getvder(parm, vsys, ktot)
        print "Derived parameters:"
        for nm, vl, unt in zip(eb.dernames, vder, eb.derunits):
            print "{0:<10} {1:14.6f} {2}".format(nm, vl, unt)

        plt.ion()
        plt.figure(91)
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.plot(tprim,xprim,'ko')
        plt.plot(tprim,sm1,'ro')
        tcomp = np.linspace(np.min(tprim),np.max(tprim),10000)
        compmodel = compute_eclipse(tcomp,parm,fitrvs=False,tref=t0,period=period,ooe1fit=coeff1)
        plt.plot(tcomp,compmodel,'r-')
        plt.ylim(0.5,1.1)
        chi1 = -1*lf1
        plt.annotate(r'$\chi^2$ = %.0f' % chi1, [0.1,0.1],horizontalalignment='left',
                     xycoords='axes fraction',fontsize='large')

        plt.subplot(2, 2, 2)
        plt.plot(tsec,xsec,'ko')
        plt.plot(tsec,sm2,'ro')
        tcomp = np.linspace(np.min(tsec),np.max(tsec),10000)
        compmodel = compute_eclipse(tcomp,parm,fitrvs=False,tref=t0,period=period,ooe1fit=coeff2)
        plt.plot(tcomp,compmodel,'r-')
        plt.ylim(0.7,1.1)
        chi2 = -1*lf2
        plt.annotate(r'$\chi^2$ = %.0f' % chi2, [0.1,0.1],horizontalalignment='left',
                     xycoords='axes fraction',fontsize='large')
        
        plt.subplot(2, 1, 2)
        phi1 = foldtime(rvdata1[:,0],t0=t0,period=period)/period
        plt.plot(phi1,rvdata1[:,1],'ko')
        plt.plot(phi1,rv1,'kx')
        tcomp = np.linspace(-0.5,0.5,10000)*period+t0
        rvmodel1 = compute_eclipse(tcomp,parm,fitrvs=True)
        k2 = ktot/(1+massratio)
        k1 = k2*massratio
        rvcomp1 = rvmodel1*k1 + vsys
        plt.plot(np.linspace(-0.5,0.5,10000),rvcomp1,'k--')
        plt.annotate(r'$\chi^2$ = %.0f' % -lfrv, [0.05,0.85],horizontalalignment='left',
                     xycoords='axes fraction',fontsize='large')
  
        phi2 = foldtime(rvdata2[:,0],t0=t0,period=period)/period
        plt.plot(phi2,rvdata2[:,1],'ro')
        plt.plot(phi2,rv2,'rx')
        tcomp = np.linspace(-0.5,0.5,10000)*period+t0
        rvmodel2 = compute_eclipse(tcomp,parm,fitrvs=True)
        rvcomp2 = -1.0*rvmodel2*k2 + vsys
        plt.plot(np.linspace(-0.5,0.5,10000),rvcomp2,'r--')
        plt.xlim(-0.5,0.5)
        plt.suptitle('Eclipse Fit')

        gamma = np.linspace(0,np.pi/2.0,1000,endpoint=True)
        theta = gamma*180.0/np.pi
        mu = np.cos(gamma)
        Imu1 = 1.0 - u1a*(1.0 - mu) - u2a*(1.0 - mu)**2.0
        Imu2 = 1.0 - u1b*(1.0 - mu) - u2b*(1.0 - mu)**2.0


        plt.figure(92)
        plt.clf()
        label1 = '%.2f, ' % u1a + '%.2f' % u2a +' (Primary)'
        label2 = '%.2f, ' % u1b + '%.2f' % u2b +' (Secondary)'
        plt.plot(theta,Imu1,label=label1)
        plt.plot(theta,Imu2,label=label2)
        plt.ylim([0,1.0])
        plt.xlabel(r"$\theta$ (degrees)",fontsize=18)
        plt.ylabel(r"$I(\theta)/I(0)$",fontsize=18)
        plt.annotate(r'$T_{\rm eff,1}$ = %.0f K' % T1, [0.16,0.6],horizontalalignment='left',
                     xycoords='figure fraction',fontsize='large')
        plt.annotate(r'$M_1$ = %.2f M$_\odot$' % m1, [0.16,0.55],horizontalalignment='left',
                     xycoords='figure fraction',fontsize='large')
        plt.annotate(r'$R_1$ = %.2f R$_\odot$' % r1, [0.16,0.5],horizontalalignment='left',
                     xycoords='figure fraction',fontsize='large')
        plt.annotate(r'$\log(g)_1$ = %.2f' % logg1, [0.16,0.45],horizontalalignment='left',
                     xycoords='figure fraction',fontsize='large')

        plt.annotate(r'$T_{\rm eff,2}$ = %.0f K' % T2, [0.16,0.35],horizontalalignment='left',
                     xycoords='figure fraction',fontsize='large')
        plt.annotate(r'$M_2$ = %.2f M$_\odot$' % m2, [0.16,0.3],horizontalalignment='left',
                     xycoords='figure fraction',fontsize='large')
        plt.annotate(r'$R_2$ = %.2f R$_\odot$' % r2, [0.16,0.25],horizontalalignment='left',
                     xycoords='figure fraction',fontsize='large')
        plt.annotate(r'$\log(g)_2$ = %.2f' % logg2, [0.16,0.2],horizontalalignment='left',
                     xycoords='figure fraction',fontsize='large')
        plt.title('Limb Darkening')
        
        plt.legend()

        print q1a,q2a,q1b,q2b

        pdb.set_trace()
        
    return lf


def single_fit(eclipse,info,nwalkers=1000,burnsteps=1000,mcmcsteps=1000,clobber=False,
               fit_period=False,fit_limb=False,fit_rvs=True,fit_sp1=True,fit_L3=False,
               fit_sp2=False,full_spot=False,fit_ellipsoidal=False,fit_lighttravel=False,
               use_gravdark=False,use_reflection=False,write=True,order=3,reduce=10,
               claret_limb=False):

    import emcee
    global ndim, variables,tfit,mfit,e_mfit
    global nw, bs, mcs
    global fitrvs, fitlimb, fitsp1, fitsp2, claret
    global fullspot, fitellipsoidal, fitlighttravel, fitL3
    global usegravdark, usereflection
    global fitdict, fitorder

    fitorder = order

    fitdict = eclipse

    claret = claret_limb
    
    thintag = '_thin'+str(fitdict['thin']) if fitdict['thin'] > 1 else ''

    fitrvs = fit_rvs
    fitlimb = fit_limb
    fitsp1 = fit_sp1
    fitsp2 = fit_sp2
    fullspot = full_spot
    fitellipsoidal = fit_ellipsoidal
    fitlighttravel = fit_lighttravel
    fitL3 = fit_L3
    usegravdark = use_gravdark
    usereflection = use_reflection

    nw = nwalkers
    bs = burnsteps
    mcs = mcmcsteps

    enum = fitdict['enum']

    directory = path+'MCMC/singlefits/E'+str(enum)+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    print ""
    print "Starting MCMC fitting routine for "+name

    twomin = 2./(24.*60.)
    onesec = 1./(24.*60.*60.)

#    bestvals, meds, modes, sigints, chisq = best_spotvals(fitdict,info,doprint=False,doplot=False)
    
# Initial chain values
    print ""
    print "Deriving starting values for chains"
    p0_0  = np.random.uniform(ebpar0['J']*0.5,ebpar0['J']*2.0,nw)             # surface brightness
    p0_1  = np.random.uniform(ebpar0['Rsum_a']*0.5,ebpar0['Rsum_a']*2, nw)    # fractional radius
    p0_2  = np.random.uniform(ebpar0['Rratio']*0.5,ebpar0['Rratio']*2, nw)    # radius ratio
    p0_3  = np.random.uniform(0,ebpar0['Rsum_a'], nw)                         # cos i
    p0_4  = np.random.uniform(ebpar0['ecosw']*0.5,min(ebpar0['ecosw']*2,1), nw) # ecosw
    p0_5  = np.random.uniform(ebpar0['esinw']*0.5,min(ebpar0['esinw']*2,1), nw) # esinw
    p0_6  = np.random.normal(ebpar0['mag0'],0.1, nw)                          # mag zpt
    p0_7  = np.random.normal(ebpar0['t01']-bjd,twomin,nw)                     # ephemeris
    p0_8  = np.random.normal(ebpar0['Period'],onesec,nw    )                  # Period
    p0_9  = np.random.uniform(0,1,nw)                                         # Limb darkening
    p0_10 = np.random.uniform(0,1,nw)                                         # Limb darkening
    p0_11 = np.random.uniform(0,1,nw)                                         # Limb darkening
    p0_12 = np.random.uniform(0,1,nw)                                         # Limb darkening
    p0_13 = np.abs(np.random.normal(ebpar0['Mratio'],0.05,nw))                # Mass ratio
    p0_14 = np.random.uniform(0,0.5,nw)                                       # Third Light
    p0_15 = np.random.normal(ebpar0['Rot1'],0.001,nw)                         # Star 1 rotation
    p0_16 = np.random.uniform(0,1,nw)                                         # Fraction of spots eclipsed
    p0_17 = np.random.normal(0,0.001,nw)                                      # base spottedness
    p0_18 = np.random.normal(0,0.0001,nw)                                     # Sin amplitude
    p0_19 = np.random.normal(0,0.0001,nw)                                     # Cos amplitude
    p0_20 = np.random.normal(0,0.0001,nw)                                     # SinCos amplitude
    p0_21 = np.random.normal(0,0.0001,nw)                                     # Cos^2-Sin^2 amplitude
    p0_22 = np.random.uniform(ebpar0['Rot1'],0.001,nw)                        # Star 2 rotation
    p0_23 = np.random.uniform(0,1,nw)                                         # Fraction of spots eclipsed
    p0_24 = np.random.normal(0,0.001,nw)                                      # base spottedness
    p0_25 = np.random.normal(0,0.001,nw)                                      # Sin amplitude
    p0_26 = np.random.normal(0,0.001,nw)                                      # Cos amplitude
    p0_27 = np.random.normal(0,0.001,nw)                                      # SinCos amplitude
    p0_28 = np.random.normal(0,0.001,nw)                                      # Cos^2-Sin^2 amplitude
    p0_29 = np.abs(np.random.normal(ebpar0['ktot'],ebpar0['ktot']*0.2,nw))    # Total radial velocity amp
    p0_30 = np.random.normal(ebpar0['vsys'],5.0,nw)                           # System velocity


# L3 at 14 ... 14 and beyond + 1

#    p0_init = np.array([p0_0,p0_1,p0_2,p0_3,p0_4,p0_5,p0_6,p0_7])
#    variables =["J","Rsum","Rratio","cosi","ecosw","esinw","magoff","t0"]

    p0_init = np.array([p0_0,p0_1,p0_2,p0_3,p0_4,p0_5,p0_7])
    variables =["J","Rsum","Rratio","cosi","ecosw","esinw","t0"]

#    if fitperiod:
#        p0_init = np.append(p0_init,[p0_8],axis=0)
#        variables.append("period")

    if fitlimb and claret:
        sys.exit('Cannot fit for LD parameters and constrain them according to the other fit parameters!')

    if fitlimb:
        limb0 = np.array([p0_9,p0_10,p0_11,p0_12])
        lvars = ["q1a", "q2a", "q1b", "q2b"]
        p0_init = np.append(p0_init,limb0,axis=0)
        for var in lvars:
            variables.append(var)

    if fitL3:
        p0_init = np.append(p0_init,[p0_14],axis=0)
        variables.append('L3')

    if fitsp1:
        p0_init = np.append(p0_init,[p0_16],axis=0)
        variables.append('spFrac1')
        for i in range(fitorder+1):
            p0_init = np.append(p0_init,[np.random.normal(0,0.05,nw)],axis=0)
            variables.append('c'+str(i)+'_1')
        for i in range(fitorder+1):
            p0_init = np.append(p0_init,[np.random.normal(0,0.05,nw)],axis=0)
            variables.append('c'+str(i)+'_2')

    if fitsp2:
        sys.exit('Not ready for this yet!')

    
#    if fitsp1:
#        sys.exit('conflicting inputs!')
#        spot0 = np.array([p0_15,p0_16,p0_17,p0_18,p0_19])
#        spvars = ["Rot1","spFrac1","spBase1","spSin1","spCos1"]
#        p0_init = np.append(p0_init,spot0,axis=0)
#        for var in spvars:
#            variables.append(var)
#        if fullspot:
#            spot0 = np.array([p0_20,p0_21])
#            spvars = ["spSinCos1","spSqSinCos1"]
#            p0_init = np.append(p0_init,spot0,axis=0)
#            for var in spvars:
#                variables.append(var)

#    if fitsp2 and not fitsp1:
#        sys.exit('conflicting inputs!')
#        spot0 = np.array([p0_15,p0_16,p0_17,p0_18,p0_19])
#        spvars = ["Rot2","spFrac2","spBase2","spSin2","spCos2"]
#        p0_init = np.append(p0_init,spot0,axis=0)
#        for var in spvars:
#            variables.append(var)
#        if fullspot:
#            spot0 = np.array([p0_20,p0_21])
#            spvars = ["spSinCos2","spSqSinCos2"]
#            p0_init = np.append(p0_init,spot0,axis=0)
#            for var in spvars:
#                variables.append(var)
#
#    if fitsp2 and fitsp1:
#        sys.exit('conflicting inputs!')
#        spot0 = np.array([p0_22,p0_23,p0_24,p0_25,p0_26])
#        spvars = ["Rot2","spFrac2","spBase2","spSin2","spCos2"]
#        p0_init = np.append(p0_init,spot0,axis=0)
#        for var in spvars:
#            variables.append(var)
#        if fullspot:
#            spot0 = np.array([p0_27,p0_28])
#            spvars = ["spSinCos2","spSqSinCos2"]
#            p0_init = np.append(p0_init,spot0,axis=0)
#            for var in spvars:
#                variables.append(var)

    if fitrvs:
        p0_init = np.append(p0_init,[p0_13],axis=0)
        variables.append('massratio')
        p0_init = np.append(p0_init,[p0_29],axis=0)
        variables.append('ktot')
        p0_init = np.append(p0_init,[p0_30],axis=0)
        variables.append("vsys")

    variables = np.array(variables)

# Transpose array of initial guesses
    p0 = np.array(p0_init).T
    
# Number of dimensions in the fit.
    ndim = np.shape(p0)[1]

# Do not redo MCMC unless clobber flag is set
    done = os.path.exists(directory+name+stag+thintag+'_Jchain_E'+str(enum)+'.txt')
    if done == True and clobber == False:
        print "MCMC run already completed"
        return False,False,variables


    def lnprob(x):
        return lnprob_single(x)


# Set up MCMC sampler
    print "... initializing emcee sampler"
    tstart = time.time()
    sampler = emcee.EnsembleSampler(nw, ndim, lnprob)

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
    burn = np.append(burn,sampler.acceptance_fraction)
    np.savetxt(directory+name+stag+thintag+'_burnstats_E'+str(enum)+'.txt',burn)

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
    stats = np.append(stats,sampler.acceptance_fraction)
    np.savetxt(directory+name+stag+thintag+'_finalstats_E'+str(enum)+'.txt',stats)

# Write out chains to disk
    if write:
        print "Writing MCMC chains to disk"
        lp = sampler.lnprobability.flatten()
        np.savetxt(directory+name+stag+thintag+'_lnprob_E'+str(enum)+'.txt',lp[0::reduce])
        for i in np.arange(len(variables)):
            np.savetxt(directory+name+stag+thintag+'_'+variables[i]+'chain_E'+str(enum)+'.txt',sampler.flatchain[0::reduce,i])


    return sampler.lnprobability.flatten(),sampler.flatchain,variables



def best_single_vals(eclipse,info,chains=False,lp=False,network=None,bindiv=20.0,
                thin=False,frac=0.001,nbins=100,rpmax=1,
                durmax=10,sigrange=5.0):

    """
    ----------------------------------------------------------------------
    best_single_vals:
    ---------
    Find the best values from the 1-d posterior pdfs of a fit to a single
    primary and secondary eclipse pair
    ----------------------------------------------------------------------
    """
    
    import robust as rb
    from scipy.stats.kde import gaussian_kde
    import matplotlib as mpl
    from plot_params import plot_params, plot_defaults
    
    plot_params(linewidth=1.5,fontsize=12)

    nsamp = nw*mcs

    ethin = eclipse['thin']
    thintag = '_thin'+str(ethin) if ethin > 1 else ''

    enum = eclipse['enum']
    
    # Use supplied chains or read from disk
    if not np.shape(chains):
        for i in np.arange(len(variables)):
            try:
                print "Reading MCMC chains for "+variables[i]
                tmp = np.loadtxt(path+'MCMC/singlefits/E'+str(enum)+'/' \
                                 +name+stag+thintag+'_'+variables[i]+'chain_E'+str(enum)+\
                                 '.txt')
                if i == 0:
                    chains = np.zeros((len(tmp),len(variables)))

                chains[:,i] = tmp
            except:
                print name+stag+thintag+'_'+variables[i]+'chain.txt does not exist on disk !'

    if not np.shape(lp):
        try:
            print "Reading ln(prob) chain"
            lp = np.loadtxt(path+'MCMC/singlefits/E'+str(enum)+'/'+name+stag+thintag+'_lnprob_E'+str(enum)+'.txt')
        except:
            print name+stag+thintag+'_lnprob_E'+str(enum)+'.txt does not exist. Exiting'
            return

#  Get maximum likelihood values
    bestvals = np.zeros(len(variables))
    meds = np.zeros(len(variables))
    modes = np.zeros(len(variables))
    onesigs = np.zeros(len(variables))

    maxlike = np.max(lp)
    imax = np.array([i for i, j in enumerate(lp) if j == maxlike])
    if imax.size > 1:
        imax = imax[0]
    for i in np.arange(len(variables)):
        bestvals[i] = chains[imax,i]
        
    if thin:
        print "Thinning chains by a factor of "+str(thin)
        nsamp /= thin
        thinchains = np.zeros((nsamp,len(variables)))
        for i in np.arange(len(variables)):
            thinchains[:,i] = chains[0::thin,i]
        lp = lp[0::thin]
        chains = thinchains 

    varnames = varnameconv(variables)

# Primary Variables
    priminds, = np.where((np.array(variables) == 'J') ^ (np.array(variables) =='Rsum') ^ 
                         (np.array(variables) == 'Rratio') ^ (np.array(variables) == 'ecosw') ^ 
                         (np.array(variables) == 'esinw') ^ (np.array(variables) == 'cosi'))

    plt.ioff()
    plt.figure(4,figsize=(8.5,11),dpi=300)    
    plt.clf()
    plotnum = 0
    for i in priminds:
        print ''
        dist = chains[:,i]
        med,mode,interval,lo,hi = distparams(dist)
        meds[i] = med
        modes[i] = mode
        onesigs[i] = interval
        minval = np.min(dist)
        maxval = np.max(dist)
        sigval = rb.std(dist)
        maxval = med + sigrange*np.abs(hi-med)
        minval = med - sigrange*np.abs(med-lo)
        nb = np.ceil((maxval-minval) / (interval/bindiv))
        print 'Best fit parameters for '+variables[i]        
        out = variables[i]+': max = {0:.5f}, med = {1:.5f}, mode = {2:.5f}, 1 sig int = {3:.5f}'
        print out.format(bestvals[i], med, mode, interval)
        
        # do plot
        plotnum += 1
        plt.subplot(len(priminds),1,plotnum)
        print "Computing histogram of data"
        pinds, = np.where((dist >= minval) & (dist <= maxval))
        plt.hist(dist[pinds],bins=nb,normed=True)
        #    plt.xlim([minval,maxval])
#        plt.axvline(x=bestvals[i],color='r',linestyle='--')
#        plt.axvline(x=medval,color='c',linestyle='--')
        plt.xlabel(varnames[i])
        plt.ylabel(r'$dP$')
        if plotnum == 1:
            plt.title('Parameter Distributions for KIC '+name)

    plt.subplots_adjust(hspace=0.55)
    plt.savefig(path+'MCMC/singlefits/E'+str(enum)+'/'+name+stag+thintag+
                '_params1_E'+str(enum)+'.png', dpi=300)
    plt.clf()




# Second set of parameters
#    secinds, = np.where((np.array(variables) == 't0') ^ (np.array(variables) =='q1a') ^ 
#                        (np.array(variables) == 'q2a') ^ (np.array(variables) == 'q1b') ^ 
#                        (np.array(variables) == 'q2b') ^ (np.array(variables) == 'massratio') ^ 
#                        (np.array(variables) == 'ktot') ^ (np.array(variables) == 'vsys'))

    secinds, = np.where((np.array(variables) == 't0') ^ (np.array(variables) == 'massratio') ^ 
                        (np.array(variables) == 'ktot') ^ (np.array(variables) == 'vsys'))

    plt.figure(5,figsize=(8.5,11),dpi=300)    
    plt.clf()
    plotnum = 0
    for i in secinds:
        print ''
        if variables[i] == 't0':
            dist   = (chains[:,i] - (ebpar0["t01"] - bjd))*3600.0
            t0val = bestvals[i]
            bestvals[i] = (t0val -(ebpar0["t01"] - bjd))*3600.0
        else:
            dist = chains[:,i]

        med,mode,interval,lo,hi = distparams(dist)
        meds[i] = med
        modes[i] = mode
        onesigs[i] = interval
        minval = np.min(dist)
        maxval = np.max(dist)
        sigval = rb.std(dist)
        maxval = med + sigrange*np.abs(hi-med)
        minval = med - sigrange*np.abs(med-lo)
        nb = np.ceil((maxval-minval) / (interval/bindiv))
        print 'Best fit parameters for '+variables[i]        
        out = variables[i]+': max = {0:.5f}, med = {1:.5f}, mode = {2:.5f}, 1 sig int = {3:.5f}'
        print out.format(bestvals[i], med, mode, interval)
        
        # do plot
        plotnum += 1
        plt.subplot(len(secinds),1,plotnum)
        print "Computing histogram of data"
        pinds, = np.where((dist >= minval) & (dist <= maxval))
        plt.hist(dist[pinds],bins=nb,normed=True)
        #    plt.xlim([minval,maxval])
#        plt.axvline(x=bestvals[i],color='r',linestyle='--')
#        plt.axvline(x=medval,color='c',linestyle='--')
        plt.xlabel(varnames[i])
        plt.ylabel(r'$dP$')
        if plotnum == 1:
            plt.title('Parameter Distributions for KIC '+name)
        if variables[i] == 't0':
            plt.annotate(r'$t_0$ = %.6f BJD' % ebpar0["t01"], xy=(0.96,0.8),
                         ha="right",xycoords='axes fraction',fontsize='large')
            bestvals[i] = t0val

    plt.subplots_adjust(hspace=0.55)
    plt.savefig(path+'MCMC/singlefits/E'+str(enum)+'/'+name+stag+thintag+'_params2_E'+str(enum)+'.png', dpi=300)
    plt.clf()

# Third set of indices
    thinds, = np.where((np.array(variables) == 'spFrac1') ^ (np.array(variables) == 'c0_1') ^
                        (np.array(variables) =='c1_1') ^ (np.array(variables) == 'c2_1') ^
                        (np.array(variables) == 'c3_1'))

    plt.figure(5,figsize=(8.5,11),dpi=300)    
    plt.clf()
    plotnum = 0
    for i in thinds:
        print ''
        dist = chains[:,i]

        med,mode,interval,lo,hi = distparams(dist)
        meds[i] = med
        modes[i] = mode
        onesigs[i] = interval
        minval = np.min(dist)
        maxval = np.max(dist)
        sigval = rb.std(dist)
        maxval = med + sigrange*np.abs(hi-med)
        minval = med - sigrange*np.abs(med-lo)
        nb = np.ceil((maxval-minval) / (interval/bindiv))
        print 'Best fit parameters for '+variables[i]        
        out = variables[i]+': max = {0:.5f}, med = {1:.5f}, mode = {2:.5f}, 1 sig int = {3:.5f}'
        print out.format(bestvals[i], med, mode, interval)
        
        # do plot
        plotnum += 1
        plt.subplot(len(thinds),1,plotnum)
        print "Computing histogram of data"
        pinds, = np.where((dist >= minval) & (dist <= maxval))
        plt.hist(dist[pinds],bins=nb,normed=True)
        #    plt.xlim([minval,maxval])
#        plt.axvline(x=bestvals[i],color='r',linestyle='--')
#        plt.axvline(x=medval,color='c',linestyle='--')
        plt.xlabel(varnames[i])
        plt.ylabel(r'$dP$')
        if plotnum == 1:
            plt.title('Parameter Distributions for KIC '+name)

    plt.subplots_adjust(hspace=0.55)
    plt.savefig(path+'MCMC/singlefits/E'+str(enum)+'/'+name+stag+thintag+'_params3_E'+str(enum)+'.png', dpi=300)
    plt.clf()

# Fourth set of parameters
    finds, = np.where((np.array(variables) == 'c0_2') ^ (np.array(variables) == 'c1_2') ^ 
                        (np.array(variables) == 'c2_2') ^ (np.array(variables) == 'c3_2'))    

    plt.figure(5,figsize=(8.5,11),dpi=300)    
    plt.clf()
    plotnum = 0
    for i in finds:
        print ''
        dist = chains[:,i]

        med,mode,interval,lo,hi = distparams(dist)
        meds[i] = med
        modes[i] = mode
        onesigs[i] = interval
        minval = np.min(dist)
        maxval = np.max(dist)
        sigval = rb.std(dist)
        maxval = med + sigrange*np.abs(hi-med)
        minval = med - sigrange*np.abs(med-lo)
        nb = np.ceil((maxval-minval) / (interval/bindiv))
        print 'Best fit parameters for '+variables[i]        
        out = variables[i]+': max = {0:.5f}, med = {1:.5f}, mode = {2:.5f}, 1 sig int = {3:.5f}'
        print out.format(bestvals[i], med, mode, interval)
        
        # do plot
        plotnum += 1
        plt.subplot(len(finds),1,plotnum)
        print "Computing histogram of data"
        pinds, = np.where((dist >= minval) & (dist <= maxval))
        plt.hist(dist[pinds],bins=nb,normed=True)
        #    plt.xlim([minval,maxval])
#        plt.axvline(x=bestvals[i],color='r',linestyle='--')
#        plt.axvline(x=medval,color='c',linestyle='--')
        plt.xlabel(varnames[i])
        plt.ylabel(r'$dP$')
        if plotnum == 1:
            plt.title('Parameter Distributions for KIC '+name)

    plt.subplots_adjust(hspace=0.55)
    plt.savefig(path+'MCMC/singlefits/E'+str(enum)+'/'+name+stag+thintag+'_params4_E'+str(enum)+'.png', dpi=300)
    plt.clf()


    plotnum = 0
# For the remaining variables
    allinds = np.array(list(priminds) + list(secinds) + list(thinds) + list(finds))
    allinds = np.sort(allinds)
    if len(allinds) < len(variables):
        print "Starting plots for remaining variables"
        plt.figure(6,figsize=(8.5,11),dpi=300)
        vinds = np.arange(len(variables))
        missedi = []
        for vi in vinds:
            try:
                leni = len(np.where(vi == allinds)[0])
                if leni == 0:
                    missedi.append(vi)
            except:
                pass
        for i in missedi:
            print ''
            dist   = chains[:,i]
            if variables[i][0] == 'q':
                minval = 0.0
                maxval = 1.0
            else:
                sigval = rb.std(dist)
                minval = np.min(dist) - sigval
                maxval = np.max(dist) + sigval


            med,mode,interval,lo,hi = distparams(dist)
            meds[i] = med
            modes[i] = mode
            onesigs[i] = interval
            minval = np.min(dist)
            maxval = np.max(dist)
            sigval = rb.std(dist)
            maxval = med + sigrange*np.abs(hi-med)
            minval = med - sigrange*np.abs(med-lo)
            nb = np.ceil((maxval-minval) / (interval/bindiv))
            print 'Best fit parameters for '+variables[i]        
            out = variables[i]+': max = {0:.5f}, med = {1:.5f}, mode = {2:.5f}, 1 sig int = {3:.5f}'
            print out.format(bestvals[i], med, mode, interval)
            
            # do plot
            plotnum += 1
            plt.subplot(len(missedi),1,plotnum)
            print "Computing histogram of data"
            pinds, = np.where((dist >= minval) & (dist <= maxval))
            plt.hist(dist[pinds],bins=nb,normed=True)
            #    plt.xlim([minval,maxval])
#            plt.axvline(x=bestvals[i],color='r',linestyle='--')
#            plt.axvline(x=medval,color='c',linestyle='--')
            plt.xlabel(varnames[i])
            plt.ylabel(r'$dP$')
            if plotnum == 1:
                plt.title('Parameter Distributions for KIC '+name)
            
        plt.subplots_adjust(hspace=0.55)
        plt.savefig(path+'MCMC/singlefits/E'+str(enum)+'/'+name+stag+thintag+'_params5_E'+str(enum)+'.png', dpi=300)
        plt.clf()
 
   # calculate limb darkening parameters
#    if limb == 'quad':
#        ind1, = np.where(variables == 'q1a')
#        ind2, = np.where(variables == 'q2a')        
#        u1a,u2a =  qtou(bestvals[ind1],bestvals[ind2],limb=limb)
#        ldc = [u1a,u2a]
#        plot_limb_curves(ldc=ldc,limbmodel=limb,write=True,network=network)
#        ind1, = np.where(variables == 'q1b')
#        ind2, = np.where(variables == 'q2b')        
#        u1a,u2a =  qtou(bestvals[ind1],bestvals[ind2],limb=limb)
#        ldc = [u1a,u2a]
#        plot_limb_curves(ldc=ldc,limbmodel=limb,write=True,network=network)
#
#
#
#    vals = [[bestvals],[meds],[modes],[onesigs]]

    plot_single_model(bestvals,eclipse,tag='_MCMC')

    f = open(path+'MCMC/singlefits/E'+str(enum)+'/'+name+stag+thintag+'_fitparams_E'+str(enum)+'.txt','w')
    for i in np.arange(len(variables)):
        outstr = []
        fmt = []
        outstr.append(variables[i])
        outstr.append("{:.8f}".format(bestvals[i]))
        outstr.append("{:.8f}".format(meds[i]))
        outstr.append("{:.8f}".format(modes[i]))
        outstr.append("{:.8f}".format(onesigs[i]))
        f.write(', '.join(outstr)+'\n')
        
    f.closed

    plot_defaults()

    return bestvals


def desired_params_from_fits(file=None):
    file = '/Users/jonswift/Astronomy/EBs/outdata/10935310/MCMC/singlefits/Claret/E0/10935310_long_fitparams_E0.txt'

    names,vars = np.loadtxt(file,delimiter=',',usecols=(0,1),
                            dtype={'names':('name', 'value'),
                                   'formats':('|S15',np.float)},
                            unpack=True)
    return

def plot_single_model(vals,eclipse,markersize=5,smallmark=2,nbins=100,errorbars=False,durfac=5,enum=1,tag=''):

    """
    ----------------------------------------------------------------------
    plot_model_single:
    ------------------
    Plot transit model given model params.

    ----------------------------------------------------------------------
    """
    from matplotlib import gridspec
    import matplotlib as mpl
    from plot_params import plot_params, plot_defaults

    plot_params(fontsize=10,linewidth=1.2)
    
    # Check for output directory   
    directory = path+'MCMC/singlefits/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    parm,vder = vec_to_params(vals)
    
    
    ethin = eclipse['thin']
    thintag = '_thin'+str(ethin) if ethin > 1 else ''
    enum = eclipse['enum']
    
    tag = '_E'+str(enum)
    
    vsys = vals[variables == 'vsys'][0]
    ktot = vals[variables == 'ktot'][0]

    massratio = parm[eb.PAR_Q]

    if claret:
        T1,logg1,T2,logg2 = get_teffs_loggs(parm,vsys,ktot)
        
        u1a = ldc1func(T1,logg1)[0][0]
        u2a = ldc2func(T1,logg1)[0][0]
        
        u1b = ldc1func(T2,logg2)[0][0]
        u2b = ldc2func(T2,logg2)[0][0]
        
        q1a,q2a = utoq(u1a,u2a,limb=limb)        
        q1b,q2b = utoq(u1b,u2b,limb=limb)
        
        parm[eb.PAR_LDLIN1] = u1a  # u1 star 1
        parm[eb.PAR_LDNON1] = u2a  # u2 star 1
        parm[eb.PAR_LDLIN2] = u1b  # u1 star 2
        parm[eb.PAR_LDNON2] = u2b  # u2 star 2

    elif fitlimb:
        q1a = vals[variables == 'q1a'][0]  
        q2a = vals[variables == 'q2a'][0]  
        q1b = vals[variables == 'q1b'][0]  
        q2b = vals[variables == 'q2b'][0]  
        u1a = parm[eb.PAR_LDLIN1]
        u2a = parm[eb.PAR_LDNON1]
        u1b = parm[eb.PAR_LDLIN2]
        u2b = parm[eb.PAR_LDNON2]

    print "Model parameters:"
    for vname, value, unit in zip(eb.parnames, parm, eb.parunits):
        print "{0:<10} {1:14.6f} {2}".format(vname, value, unit)

    print "Derived parameters:"
    for vname, value, unit in zip(eb.dernames, vder, eb.derunits):
        print "{0:<10} {1:14.6f} {2}".format(vname, value, unit)


######################################################################
# Light curve model
######################################################################
    if not fitellipsoidal:
        parm[eb.PAR_Q] = 0.0        

    # Phases of contact points
    (ps, pe, ss, se) = eb.phicont(parm)

    # Primary eclipse
    t0 = parm[eb.PAR_T0]
    period = parm[eb.PAR_P]
    tprim = fitdict['tprim']
    norm = fitdict['norm']
    xprim = fitdict['xprim']/norm
    eprim = fitdict['eprim']/norm

    if fitsp1:
        coeff1 = []
        for i in range(fitorder+1):
            coeff1 = np.append(coeff1,vals[variables == 'c'+str(i)+'_1'])

    model1  = compute_eclipse(tprim,parm,fitrvs=False,tref=t0,period=period,ooe1fit=coeff1)

    tcomp1 = np.linspace(np.min(tprim),np.max(tprim),10000)
    compmodel1 = compute_eclipse(tcomp1,parm,fitrvs=False,tref=t0,period=period,ooe1fit=coeff1)
        
    phiprim  = foldtime(tprim,t0=t0,period=period)/period
    phicomp1 = foldtime(tcomp1,t0=t0,period=period)/period


    # Secondary eclipse
    tsec = fitdict['tsec']
    xsec = fitdict['xsec']/norm
    esec = fitdict['esec']/norm

    if fitsp1:
        coeff2 = []
        for i in range(fitorder+1):
            coeff2 = np.append(coeff2,vals[variables == 'c'+str(i)+'_2'])

    model2  = compute_eclipse(tsec,parm,fitrvs=False,tref=t0,period=period,ooe1fit=coeff2)

    tcomp2 = np.linspace(np.min(tsec),np.max(tsec),10000)
    compmodel2 = compute_eclipse(tcomp2,parm,fitrvs=False,tref=t0,period=period,ooe1fit=coeff2)
        
    phisec  = foldtime(tsec,t0=t0,period=period)/period
    phisec[phisec < 0] += 1.0
    
    phicomp2 = foldtime(tcomp2,t0=t0,period=period)/period
    phicomp2[phicomp2 < 0] += 1.0
    
    parm[eb.PAR_Q] = massratio

    if fitrvs:
        rvmodel1 = compute_eclipse(rvdata1[:,0],parm,fitrvs=True)
        k2 = ktot/(1+massratio)
        k1 = k2*massratio
        rv1 = rvmodel1*k1 + vsys
        rvmodel2 = compute_eclipse(rvdata2[:,0],parm,fitrvs=True)

# Does dof need another - 1 ???
#    dof = np.float(len(tfit)) - np.float(len(variables))
#    chisquare = np.sum((res/e_ffit)**2)/dof
    

#------------------------------
# PLOT

# Primary eclipse
    fig = plt.figure(109,dpi=300)
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.plot(phiprim,xprim,'ko')
#    plt.plot(phiprim,model1,'rx')
    plt.plot(phicomp1,compmodel1,'r-')
    plt.axvline(x=ps-1.0,color='b',linestyle='--')
    plt.axvline(x=pe,color='b',linestyle='--')
    ymax = np.max(np.array(list(xprim)+list(compmodel1)))
    ymin = np.min(np.array(list(xprim)+list(compmodel1)))
    ytop = ymax + (ymax-ymin)*0.1
    ybot = ymin - (ymax-ymin)*0.1
    plt.ylim(ybot,ytop)
    plt.ylabel('Flux (normalized)')
    plt.xlabel('Phase')
    plt.title('Primary Eclipse',fontsize=12)

    plt.subplot(2, 2, 2)
    plt.plot(phisec,xsec,'ko')
#    plt.plot(phisec,model2,'xo')
    plt.plot(phicomp2,compmodel2,'r-')
    plt.axvline(x=ss,color='b',linestyle='--')
    plt.axvline(x=se,color='b',linestyle='--')
    ymax = np.max(np.array(list(xsec)+list(compmodel2)))
    ymin = np.min(np.array(list(xsec)+list(compmodel2)))
    ytop = ymax + (ymax-ymin)*0.1
    ybot = ymin - (ymax-ymin)*0.1
    plt.ylim(ybot,ytop)
    plt.xlabel('Phase')
    plt.title('Secondary Eclipse',fontsize=12)

    
    plt.subplot(2, 1, 2)
    phi1 = foldtime(rvdata1[:,0],t0=t0,period=period)/period
    plt.plot(phi1,rvdata1[:,1],'ko')
#    plt.plot(phi1,rv1,'kx')
    tcomp = np.linspace(-0.5,0.5,10000)*period+t0
    rvmodel1 = compute_eclipse(tcomp,parm,fitrvs=True)
    k2 = ktot/(1+massratio)
    k1 = k2*massratio
    rvcomp1 = rvmodel1*k1 + vsys
    plt.plot(np.linspace(-0.5,0.5,10000),rvcomp1,'k--')
    
    phi2 = foldtime(rvdata2[:,0],t0=t0,period=period)/period
    plt.plot(phi2,rvdata2[:,1],'ro')
#    plt.plot(phi2,rv2,'rx')
    tcomp = np.linspace(-0.5,0.5,10000)*period+t0
    rvmodel2 = compute_eclipse(tcomp,parm,fitrvs=True)
    rvcomp2 = -1.0*rvmodel2*k2 + vsys
    plt.plot(np.linspace(-0.5,0.5,10000),rvcomp2,'r--')
    plt.xlim(-0.5,0.5)
    plt.ylabel('Radial Velocity (km/s)')
    plt.xlabel('Phase')

    plt.suptitle('Eclipse '+str(enum)+' Fitting Results',fontsize=14)
    
    plt.savefig(path+'MCMC/singlefits/E'+str(enum)+'/'+name+stag+thintag+'_MCMCfit_E'+str(enum)+'.png')

    plot_defaults()

    return # chisquare




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



def varnameconv(variables):

    varmatch = np.array(["J","Rsum","Rratio","cosi",
                         "ecosw","esinw",
                         "magoff","t0","period",
                         "q1a", "q2a", "q1b", "q2b","massratio","L3",
                         "Rot1","spFrac1","spBase1","spSin1","spCos1","spSinCos1","spSqSinCos1",
                         "Rot2","spFrac2","spBase2","spSin2","spCos2","spSinCos2","spSqSinCos2",
                         "c0_1","c1_1","c2_1","c3_1","c4_1","c5_1",
                         "c0_2","c1_2","c2_2","c3_2","c4_2","c5_2",
                         "ktot","vsys"])

    varnames = np.array(["Surf. Br. Ratio", r"$(R_1+R_2)/a$", r"$R_2/R_1$", r"$\cos i$", 
                         r"$e\cos\omega$",r"$e\sin\omega$",
                         r"$\Delta m_0$", r"$\Delta t_0$ (s)","$\Delta P$ (s)",
                         "$q_1^p$","$q_2^p$","$q_1^s$","$q_2^s$","$M_2/M_1$", "$L_3$",
                         "$P_{rot}^p$","Sp. Frac. 1", "Sp. Base 1", "Sin Amp 1", "Cos Amp 1", "SinCos Amp 1", "Cos$^2$-Sin$^2$ Amp 1",
                         "$P_{rot}^s$","Sp. Frac. 2", "Sp. Base 2", "Sin Amp 2", "Cos Amp 2", "SinCos Amp 2", "Cos$^2$-Sin$^2$ Amp 2",
                         "$C_0$ (eclipse 1)","$C_1$ (eclipse 1)","$C_2$ (eclipse 1)",
                         "$C_3$ (eclipse 1)","$C_4$ (eclipse 1)","$C_5$ (eclipse 1)",
                         "$C_0$ (eclipse 2)","$C_1$ (eclipse 2)","$C_2$ (eclipse 2)",
                         "$C_3$ (eclipse 2)","$C_4$ (eclipse 2)","$C_5$ (eclipse 2)",
                         r"$K_{\rm tot}$ (km/s)", r"$V_{\rm sys}$ (km/s)"])

    varvec = []
    for var in variables:
        ind, = np.where(var == varmatch)
        if length(ind) == 1:
            varvec.append(varnames[ind[0]])

    return varvec




def residuals_orig(inp):

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

    



def distparams(dist):
    from scipy.stats.kde import gaussian_kde

    vals = np.linspace(np.min(dist)*0.5,np.max(dist)*1.5,1000)
    try:
        kde = gaussian_kde(dist)
        pdf = kde(vals)
        dist_c = np.cumsum(pdf)/np.nansum(pdf)
        func = sp.interpolate.interp1d(dist_c,vals,kind='linear')
        lo = np.float(func(math.erfc(1./np.sqrt(2))))
        hi = np.float(func(math.erf(1./np.sqrt(2))))
        med = np.float(func(0.5))
        mode = vals[np.argmax(pdf)]
        disthi = np.linspace(.684,.999,100)
        distlo = disthi-0.6827
        disthis = func(disthi)
        distlos = func(distlo)
        interval = np.min(disthis-distlos)
    except:
        print 'KDE analysis failed! Using "normal" stats.'
        interval = 2.0*np.std(dist)
        med = np.median(dist)
        mode = med
        lo = med-interval/2.0
        hi = med+interval/2.0
    
    return med,mode,np.abs(interval),lo,hi


def params_of_interest(eclipse,chains=False,lp=False):

    ethin = eclipse['thin']
    thintag = '_thin'+str(ethin) if ethin > 1 else ''

    enum = eclipse['enum']

    print "Deriving values for parameters of interest"

#    tmaster = time.time()
    if not np.shape(chains):
        print "Reading in MCMC chains"        
        for i in np.arange(len(variables)):
            try:
                print "Reading MCMC chains for "+variables[i]
                tmp = np.loadtxt(path+'MCMC/singlefits/E'+str(enum)+'/' \
                                 +name+stag+thintag+'_'+variables[i]+'chain_E'+str(enum)+\
                                 '.txt')
                if i == 0:
                    chains = np.zeros((len(tmp),len(variables)))

                chains[:,i] = tmp
            except:
                print name+stag+'_'+variables[i]+'chain.txt does not exist on disk !'

#            print done_in(tmaster)


    if not np.shape(lp):
        try:
            print "Reading ln(prob) chain"
            lp = np.loadtxt(path+'MCMC/singlefits/E'+str(enum)+'/'+name+stag+thintag+'_lnprob_E'+str(enum)+'.txt')
        except:
            print "lnprob chain does not exist. Exiting"
            return

    print "Converting posteriors into physical quantities"
    ind, = np.where(np.array(variables) == 'Rsum')[0]
    rsumdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'Rratio')[0]
    rratdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'cosi')[0]
    cosidist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'ecosw')[0]
    ecoswdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'esinw')[0]
    esinwdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'massratio')[0]
    mratdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'vsys')[0]
    vsysdist = chains[:,ind]

    try:
        ind, = np.where(np.array(variables) == 'period')[0]
        pdist = chains[:,ind]
    except:
        pdist = ebpar0['Period']

    ind, = np.where(np.array(variables) == 'ktot')[0]
    ktotdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'J')[0]
    jdist = chains[:,ind]


    print "Determining maximum likelihood values"
    tlike = time.time()
    bestvals = np.zeros(len(variables))
    meds = np.zeros(len(variables))
    modes = np.zeros(len(variables))
    onesigs = np.zeros(len(variables))

    maxlike = np.max(lp)
    imax = np.array([i for i, j in enumerate(lp) if j == maxlike])
#    if imax.size > 1:
    imax = imax[0]
    for i in np.arange(len(variables)):
        bestvals[i] = chains[imax,i]

    esq = ecoswdist * ecoswdist + esinwdist * esinwdist
    roe = np.sqrt(1.0 - esq)
    sini = np.sqrt(1.0 - cosidist*cosidist)
    qpo = 1.0 + mratdist
    # Corrects for doppler shift of period
    omega = 2.0*np.pi*(1.0 + vsysdist*1000.0/eb.LIGHT) / (pdist*86400.0)
    tmp = ktotdist*1000.0 * roe
    sma = tmp / (eb.RSUN*omega*sini)
    mtotdist = tmp*tmp*tmp / (eb.GMSUN*omega*sini)
    m1dist = mtotdist / qpo
    m2dist = mratdist * m1dist

    r1dist = sma*rsumdist/(1+rratdist)
    r2dist = rratdist*r1dist

    edist  = np.sqrt(ecoswdist**2 + esinwdist**2)
 
    m1val = m1dist[imax]
    m2val = m2dist[imax]
    r1val = r1dist[imax]
    r2val = r2dist[imax]
    jval = jdist[imax]
    cosival = cosidist[imax]
    ecoswval = ecoswdist[imax]
    esinwval = esinwdist[imax]
    eval = np.sqrt(ecoswval**2 + esinwval**2)

    vals = []
    meds = []
    modes = []
    onesig = []
    med,mode,interval,lo,hi = distparams(m1dist)
    out = 'M1: max = {0:.5f}, med = {1:.5f}, mode = {2:.5f}, 1 sig int = {3:.5f}'
    print out.format(m1val,med,mode,interval)
    vals   = np.append(vals,m1val)
    meds   = np.append(meds,med)
    modes  = np.append(modes,mode)
    onesig = np.append(onesig,interval)
    
    med,mode,interval,lo,hi = distparams(m2dist)
    out = 'M2: max = {0:.5f}, med = {1:.5f}, mode = {2:.5f}, 1 sig int = {3:.5f}'
    print out.format(m2val,med,mode,interval)
    vals   = np.append(vals,m2val)
    meds   = np.append(meds,med)
    modes  = np.append(modes,mode)
    onesig = np.append(onesig,interval)

    med,mode,interval,lo,hi = distparams(r1dist)
    out = 'R1: max = {0:.5f}, med = {1:.5f}, mode = {2:.5f}, 1 sig int = {3:.5f}'
    print out.format(r1val,med,mode,interval)
    vals   = np.append(vals,r1val)
    meds   = np.append(meds,med)
    modes  = np.append(modes,mode)
    onesig = np.append(onesig,interval)

    med,mode,interval,lo,hi = distparams(r2dist)
    out = 'R2: max = {0:.5f}, med = {1:.5f}, mode = {2:.5f}, 1 sig int = {3:.5f}'
    print out.format(r2val,med,mode,interval)
    vals   = np.append(vals,r2val)
    meds   = np.append(meds,med)
    modes  = np.append(modes,mode)
    onesig = np.append(onesig,interval)

    med,mode,interval,lo,hi = distparams(edist)
    out = 'eccentricity: max = {0:.5f}, med = {1:.5f}, mode = {2:.5f}, 1 sig int = {3:.5f}'
    print out.format(eval,med,mode,interval)
    vals   = np.append(vals,eval)
    meds   = np.append(meds,med)
    modes  = np.append(modes,mode)
    onesig = np.append(onesig,interval)

    outstr = name+ ' %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f' % (vals[0],meds[0],modes[0],onesig[0],vals[1],meds[1],modes[1],onesig[1],vals[2],meds[2],modes[2],onesig[2],vals[3],meds[3],modes[3],onesig[3],vals[4],meds[4],modes[4],onesig[4])

    f = open(path+'MCMC/singlefits/E'+str(enum)+'/'+name+stag+thintag+
                '_bestparams_E'+str(enum)+'.txt','w')
    f.write(outstr+'\n')
    f.closed

    return 



def triangle_plot(chains=False,lp=False,thin=False,frac=0.001,sigfac=1.5):
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
    from copy import deepcopy as copy
    import matplotlib as mpl

    mpl.rc('axes', linewidth=1)
  
    bindiv = 10
    
    nsamp = nw*mcs
    
    tmaster = time.time()
    if chains == False:
        print "Reading in MCMC chains"        
        chains = np.zeros((nsamp,len(variables)))
        for i in np.arange(len(variables)):
            try:
                print "Reading MCMC chains for "+variables[i]
                tmp = np.loadtxt(path+'MCMC/'+name+stag+'_'+variables[i]+'chain.txt')
                chains[:,i] = tmp
            except:
                print name+stag+'_'+variables[i]+'chain.txt does not exist on disk !'

            print done_in(tmaster)


    if lp == False:
        try:
            print "Reading ln(prob) chain"
            lp = np.loadtxt(path+'MCMC/'+name+stag+'_lnprob.txt')
        except:
            print "lnprob chain does not exist. Exiting"
            return

    print "Converting posteriors into physical quantities"
    ind, = np.where(np.array(variables) == 'Rsum')[0]
    rsumdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'Rratio')[0]
    rratdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'cosi')[0]
    cosidist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'ecosw')[0]
    ecoswdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'esinw')[0]
    esinwdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'massratio')[0]
    mratdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'vsys')[0]
    vsysdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'period')[0]
    pdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'ktot')[0]
    ktotdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'J')[0]
    jdist = chains[:,ind]

    esq = ecoswdist * ecoswdist + esinwdist * esinwdist
    roe = np.sqrt(1.0 - esq)
    sini = np.sqrt(1.0 - cosidist*cosidist)
    qpo = 1.0 + mratdist
    # Corrects for doppler shift of period
    omega = 2.0*np.pi*(1.0 + vsysdist*1000.0/eb.LIGHT) / (pdist*86400.0)
    tmp = ktotdist*1000.0 * roe
    sma = tmp / (eb.RSUN*omega*sini)
    mtotdist = tmp*tmp*tmp / (eb.GMSUN*omega*sini)
    m1dist = mtotdist / qpo
    m2dist = mratdist * m1dist

    r1dist = sma*rsumdist/(1+rratdist)
    r2dist = rratdist*r1dist

    print "Determining maximum likelihood values"
    tlike = time.time()
    bestvals = np.zeros(len(variables))
    meds = np.zeros(len(variables))
    modes = np.zeros(len(variables))
    onesigs = np.zeros(len(variables))

    maxlike = np.max(lp)
    imax = np.array([i for i, j in enumerate(lp) if j == maxlike])
    if imax.size > 1:
        imax = imax[0]
    for i in np.arange(len(variables)):
        bestvals[i] = chains[imax,i]


    m1val = m1dist[imax]
    m2val = m2dist[imax]
    r1val = r1dist[imax]
    r2val = r2dist[imax]
    jval = jdist[imax]
    cosival = cosidist[imax]
    ecoswval = ecoswdist[imax]
    esinwval = esinwdist[imax]


    if thin:
        print "Thinning chains by a factor of "+str(thin)
        tthin = time.time()
        m1dist = m1dist[0::thin]
        m2dist = m2dist[0::thin]
        r1dist = r1dist[0::thin]
        r2dist = r2dist[0::thin]
        jdist = jdist[0::thin]
        cosidist = cosidist[0::thin]
        ecoswdist = ecoswdist[0::thin]
        esinwdist = esinwdist[0::thin]
        lp = lp[0::thin]
        nsamp /= thin


    print " "
    print "Starting grid of posteriors..."
    plt.figure(6,figsize=(8.5,8.5))
    nx = 8
    ny = 8

    gs = gridspec.GridSpec(nx,ny,wspace=0.1,hspace=0.1)
    print " "
    print "... top plot of first column"
    tcol = time.time()
    top_plot(r1dist,gs[0,0],val=r1val,sigfac=sigfac)
    print done_in(tcol)
    t = time.time()
    print "... first column"
    column_plot(r1dist,r2dist,gs[1,0],val1=r1val,val2=r2val,ylabel=r'$R_2$',sigfac=sigfac)
    print done_in(t)
    column_plot(r1dist,m1dist,gs[2,0],val1=r1val,val2=m1val,ylabel=r'$M_1$',sigfac=sigfac)
    column_plot(r1dist,m2dist,gs[3,0],val1=r1val,val2=m2val,ylabel=r'$M_2$',sigfac=sigfac)
    column_plot(r1dist,jdist,gs[4,0],val1=r1val,val2=jval,ylabel=r'$J$',sigfac=sigfac)
    column_plot(r1dist,cosidist,gs[5,0],val1=r1val,val2=cosival,ylabel=r'$\cos\,i$',sigfac=sigfac)
    column_plot(r1dist,ecoswdist,gs[6,0],val1=r1val,val2=ecoswval,ylabel=r'$e\cos\omega$',sigfac=sigfac)
    corner_plot(r1dist,esinwdist,gs[7,0],val1=r1val,val2=esinwval,\
                xlabel=r'$R_1$',ylabel=r'$e\sin\omega$',sigfac=sigfac)
    print "First column: "
    print done_in(tcol)

    print "... second column"
    t2 = time.time()
    top_plot(r2dist,gs[1,1],val=r2val,sigfac=sigfac)    
    middle_plot(r2dist,m1dist,gs[2,1],val1=r2val,val2=m1val,sigfac=sigfac)
    middle_plot(r2dist,m2dist,gs[3,1],val1=r2val,val2=m2val,sigfac=sigfac)
    middle_plot(r2dist,jdist,gs[4,1],val1=r2val,val2=jval,sigfac=sigfac)
    middle_plot(r2dist,cosidist,gs[5,1],val1=r2val,val2=cosival,sigfac=sigfac)
    middle_plot(r2dist,ecoswdist,gs[6,1],val1=r2val,val2=ecoswval,sigfac=sigfac)
    row_plot(r2dist,esinwdist,gs[7,1],val1=r2val,val2=esinwval,xlabel=r'$R_2$',sigfac=sigfac)
    print done_in(t2)

    print "... third column"
    t3 = time.time()
    top_plot(m1dist,gs[2,2],val=m1val,sigfac=sigfac)    
    middle_plot(m1dist,m2dist,gs[3,2],val1=m1val,val2=m2val,sigfac=sigfac)
    middle_plot(m1dist,jdist,gs[4,2],val1=m1val,val2=jval,sigfac=sigfac)
    middle_plot(m1dist,cosidist,gs[5,2],val1=m1val,val2=cosival,sigfac=sigfac)
    middle_plot(m1dist,ecoswdist,gs[6,2],val1=m1val,val2=ecoswval,sigfac=sigfac)
    row_plot(m1dist,esinwdist,gs[7,2],val1=m1val,val2=esinwval,xlabel=r'$M_1$',sigfac=sigfac)
    print done_in(t3)

    print "... fourth column"
    t4 = time.time()
    top_plot(m2dist,gs[3,3],val=m2val,sigfac=sigfac)    
    middle_plot(m2dist,jdist,gs[4,3],val1=m2val,val2=jval,sigfac=sigfac)
    middle_plot(m2dist,cosidist,gs[5,3],val1=m2val,val2=cosival,sigfac=sigfac)
    middle_plot(m2dist,ecoswdist,gs[6,3],val1=m2val,val2=ecoswval,sigfac=sigfac)
    row_plot(m2dist,esinwdist,gs[7,3],val1=m2val,val2=esinwval,xlabel=r'$M_2$',sigfac=sigfac)
    print done_in(t4)


    print "... fifth column"
    t5 = time.time()
    top_plot(jdist,gs[4,4],val=jval,sigfac=sigfac)    
    middle_plot(jdist,cosidist,gs[5,4],val1=jval,val2=cosival,sigfac=sigfac)
    middle_plot(jdist,ecoswdist,gs[6,4],val1=jval,val2=ecoswval,sigfac=sigfac)
    row_plot(jdist,esinwdist,gs[7,4],val1=jval,val2=esinwval,xlabel=r'$J$',sigfac=sigfac)
    print done_in(t5)

    print "... sixth column"
    t6 = time.time()
    top_plot(cosidist,gs[5,5],val=cosival,sigfac=sigfac)    
    middle_plot(cosidist,ecoswdist,gs[6,5],val1=cosival,val2=ecoswval,sigfac=sigfac)
    row_plot(cosidist,esinwdist,gs[7,5],val1=cosival,val2=esinwval,xlabel=r'$\cos\,i$',sigfac=sigfac)
    print done_in(t6)
  

    print "... seventh column"
    t7 = time.time()
    top_plot(ecoswdist,gs[6,6],val=ecoswval,sigfac=sigfac)    
    row_plot(ecoswdist,esinwdist,gs[7,6],val1=ecoswval,val2=esinwval,xlabel=r'$e\cos\omega$',sigfac=sigfac)
    print done_in(t7)

    print "... last column"
    t8 = time.time()
    top_plot(esinwdist,gs[7,7],val=esinwval,xlabel=r'$e\sin\omega$',sigfac=sigfac)    
    print done_in(t8)

    print "Saving output figures"
    plt.savefig(path+'MCMC/'+name+stag+'_triangle1.png', dpi=300)
    plt.savefig(path+'MCMC/'+name+stag+'_triangle1.eps', dpi=300)

    print "Procedure finished!"
    print done_in(tmaster)


    return



def top_plot(dist,position,val=False,sigfac=3.0,frac=0.001,bindiv=10,aspect=1,xlabel=False):
    from statsmodels.nonparametric.kernel_density import KDEMultivariate as KDE

#    pdb.set_trace()
#    sz = len(dist)
#    min = np.float(np.sort(dist)[np.round(frac*sz)])
#    max = np.float(np.sort(dist)[np.round((1.-frac)*sz)])
#    dists = np.linspace(np.min(dist)*0.5,np.max(dist)*1.5,1000)
#    kde = gaussian_kde(dist)
#    pdf = kde(dists)
#    cumdist = np.cumsum(pdf)/np.sum(pdf)
#    func = interp1d(cumdist,dists,kind='linear')
#    lo = np.float(func(math.erfc(1./np.sqrt(2))))
#    hi = np.float(func(math.erf(1./np.sqrt(2))))
    med = np.median(dist)
    sig = rb.std(dist)
    min = med - sigfac*sig
    max = med + sigfac*sig
#    print "Top plot min: %.5f" % min
#    print "Top plot max: %.5f" % max
    nb = np.round((max-min) / (sig/bindiv))
    ax = plt.subplot(position)
    inds, = np.where((dist >= min) & (dist <= max))
    plt.hist(dist[inds],bins=nb,normed=True,color='black')
    if not xlabel: 
        ax.set_xticklabels(())
    ax.set_yticklabels(())
    ax.set_xlim(min,max)
    xlimits = ax.get_xlim()
    ylimits = ax.get_ylim()
    ax.set_aspect(abs((xlimits[1]-xlimits[0])/(ylimits[1]-ylimits[0]))/aspect)
    if val:
        pass
#        plt.axvline(x=val,color='w',linestyle='--',linewidth=2)
    if xlabel:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(8) 
            tick.label.set_rotation('vertical')
        ax.set_xlabel(xlabel,fontsize=12)
 
    return



def column_plot(dist1,dist2,position,val1=False,val2=False,sigfac=3.0,sigsamp=3.0,frac=0.001,ylabel=None):
    from statsmodels.nonparametric.kernel_density import KDEMultivariate as KDE

    sz1 = len(dist1)
    med1 = np.median(dist1)
    sig1 = rb.std(dist1)
    min1 = med1 - sigfac*sig1
    max1 = med1 + sigfac*sig1
#    min1 = np.float(np.sort(dist1)[np.round(frac*sz1)])
#    max1 = np.float(np.sort(dist1)[np.round((1.-frac)*sz1)])

    sz2 = len(dist2)
    med2 = np.median(dist2)
    sig2 = rb.std(dist2)
    min2 = med2 - sigfac*sig2
    max2 = med2 + sigfac*sig2
#    min2 = np.float(np.sort(dist2)[np.round(frac*sz2)])
#    max2 = np.float(np.sort(dist2)[np.round((1.-frac)*sz2)])

#    inds1, = np.where((dist1 >= min1) & (dist1 <= max1))
#    inds2, = np.where((dist2 >= min2) & (dist2 <= max2))

    aspect = (max1-min1)/(max2-min2)
    X, Y = np.mgrid[min1:max1:100j, min2:max2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([dist1, dist2])

    kernel = KDE(values,var_type='cc',bw=[sig1/sigsamp,sig2/sigsamp])
    Z = np.reshape(kernel.pdf(positions).T, X.shape)

    ax = plt.subplot(position)
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,aspect=aspect,\
              extent=[min1, max1, min2, max2],origin='upper')
#    print "Column plot min1: %.5f" % min1
#    print "Column plot max1: %.5f" % max1
#    print "Column plot min2: %.5f" % min2
#    print "Column plot max2: %.5f" % max2
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



def row_plot(dist1,dist2,position,val1=False,val2=False,sigfac=3.0,sigsamp=3.0,xlabel=None):
    from statsmodels.nonparametric.kernel_density import KDEMultivariate as KDE


    sz1 = len(dist1)
    med1 = np.median(dist1)
    sig1 = rb.std(dist1)
    min1 = med1 - sigfac*sig1
    max1 = med1 + sigfac*sig1
#    min1 = np.float(np.sort(dist1)[np.round(frac*sz1)])
#    max1 = np.float(np.sort(dist1)[np.round((1.-frac)*sz1)])
    
    sz2 = len(dist2)
    med2 = np.median(dist2)
    sig2 = rb.std(dist2)
    min2 = med2 - sigfac*sig2
    max2 = med2 + sigfac*sig2
#    min2 = np.float(np.sort(dist2)[np.round(frac*sz2)])
#    max2 = np.float(np.sort(dist2)[np.round((1.-frac)*sz2)])
    
    inds1, = np.where((dist1 >= min1) & (dist1 <= max1))
    inds2, = np.where((dist2 >= min2) & (dist2 <= max2))

    aspect = (max1-min1)/(max2-min2)
    X, Y = np.mgrid[min1:max1:100j, min2:max2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([dist1, dist2])

    kernel = KDE(values,var_type='cc',bw=[sig1/sigsamp,sig2/sigsamp])
    Z = np.reshape(kernel.pdf(positions).T, X.shape)

    ax = plt.subplot(position)
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,aspect=aspect,\
              extent=[min1, max1, min2, max2],origin='upper')
#    print "Row plot min1: %.5f" % min1
#    print "Row plot max1: %.5f" % max1
#    print "Row plot min2: %.5f" % min2
#    print "Row plot max2: %.5f" % max2
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


def middle_plot(dist1,dist2,position,val1=False,val2=False,sigfac=3.0,sigsamp=3.0):
    from statsmodels.nonparametric.kernel_density import KDEMultivariate as KDE

    sz1 = len(dist1)
    med1 = np.median(dist1)
    sig1 = rb.std(dist1)
    min1 = med1 - sigfac*sig1
    max1 = med1 + sigfac*sig1
#    min1 = np.float(np.sort(dist1)[np.round(frac*sz1)])
#    max1 = np.float(np.sort(dist1)[np.round((1.-frac)*sz1)])

    sz2 = len(dist2)
    med2 = np.median(dist2)
    sig2 = rb.std(dist2)
    min2 = med2 - sigfac*sig2
    max2 = med2 + sigfac*sig2
#    min2 = np.float(np.sort(dist2)[np.round(frac*sz2)])
#    max2 = np.float(np.sort(dist2)[np.round((1.-frac)*sz2)])

    inds1, = np.where((dist1 >= min1) & (dist1 <= max1))
    inds2, = np.where((dist2 >= min2) & (dist2 <= max2))

    aspect = (max1-min1)/(max2-min2)
    X, Y = np.mgrid[min1:max1:100j, min2:max2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([dist1, dist2])

    kernel = KDE(values,var_type='cc',bw=[sig1/sigsamp,sig2/sigsamp])
    Z = np.reshape(kernel.pdf(positions).T, X.shape)

    ax = plt.subplot(position)
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,aspect=aspect,\
              extent=[min1, max1, min2, max2],origin='upper')
#    print "Middle plot min1: %.5f" % min1
#    print "Middle plot max1: %.5f" % max1
#    print "Middle plot min2: %.5f" % min2
#    print "Middle plot max2: %.5f" % max2
    clev = np.exp(np.log(np.max(Z))-0.5)
    cset = ax.contour(X,Y,Z,[clev],colors='w',linewidth=5,linestyles='dotted')
#    ax.plot(val1,val2, 'wx', markersize=3)
    ax.set_xlim(min1, max1)
    ax.set_ylim(min2, max2)
    ax.set_xticklabels(())
    ax.set_yticklabels(())
    return



def corner_plot(dist1,dist2,position,val1=False,val2=False,\
                sigfac=3.0,sigsamp=3.0,xlabel=None,ylabel=None):
    from statsmodels.nonparametric.kernel_density import KDEMultivariate as KDE

    sz1 = len(dist1)
    med1 = np.median(dist1)
    sig1 = rb.std(dist1)
    min1 = med1 - sigfac*sig1
    max1 = med1 + sigfac*sig1
#    min1 = np.float(np.sort(dist1)[np.round(frac*sz1)])
#    max1 = np.float(np.sort(dist1)[np.round((1.-frac)*sz1)])

    sz2 = len(dist2)
    med2 = np.median(dist2)
    sig2 = rb.std(dist2)
    min2 = med2 - sigfac*sig2
    max2 = med2 + sigfac*sig2
#    min2 = np.float(np.sort(dist2)[np.round(frac*sz2)])
#    max2 = np.float(np.sort(dist2)[np.round((1.-frac)*sz2)])

    inds1, = np.where((dist1 >= min1) & (dist1 <= max1))
    inds2, = np.where((dist2 >= min2) & (dist2 <= max2))

    aspect = (max1-min1)/(max2-min2)
    X, Y = np.mgrid[min1:max1:100j, min2:max2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([dist1, dist2])

    kernel = KDE(values,var_type='cc',bw=[sig1/sigsamp,sig2/sigsamp])
    Z = np.reshape(kernel.pdf(positions).T, X.shape)

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

   


def limbvals(angle,ldc,limbmodel='quad'):

    gamma = angle*np.pi/180.0
    mu = np.cos(gamma)

    if limb == 'nlin':
        c1 = ldc[0]
        c2 = ldc[1]
        c3 = ldc[2]
        c4 = ldc[3]
        Imu = 1.0 - c1*(1.0 - mu**0.5) - c2*(1.0 - mu) - \
              c3*(1.0 - mu**1.5) - c4*(1.0 - mu**2.0)
        
    elif limbmodel == 'quad':
        q1in = ldc[0]
        q2in = ldc[1]
        c1 =  2*np.sqrt(q1in)*q2in
        c2 =  np.sqrt(q1in)*(1-2*q2in)
        Imu = 1.0 - c1*(1.0 - mu) - c2*(1.0 - mu)**2.0
        
    elif limbmodel == 'sqrt':
        q1in = ldc[0]
        q2in = ldc[1]
        # Check the validity of this transformation
        c1 =  2*np.sqrt(q1in)*q2in
        c2 =  np.sqrt(q1in)*(1-2*q2in)
        Imu = 1.0 - c1*(1.0 - mu) - c2*(1.0 - mu**0.5)

    else: pass

    return Imu


def int_limb(ldc,limbmodel='quad'):

    """
    Integrate mu = cos(theta) from 1 to 0 for given limb model
    """
    
    if limb == 'nlin':
        c1 = ldc[0]
        c2 = ldc[1]
        c3 = ldc[2]
        c4 = ldc[3]
#        Imu = 1.0 - c1*(1.0 - mu**0.5) - c2*(1.0 - mu) - \
#              c3*(1.0 - mu**1.5) - c4*(1.0 - mu**2.0)
        intImu = c1/3.0 + c2/2.0 + 3*c3/5.0 + 2*c4/3.0 - 1.0
        
    elif limbmodel == 'quad':
        q1in = ldc[0]
        q2in = ldc[1]
        c1 =  2*np.sqrt(q1in)*q2in
        c2 =  np.sqrt(q1in)*(1-2*q2in)
#        Imu = 1.0 - c1*(1.0 - mu) - c2*(1.0 - mu)**2.0
        intImu = c1/2.0 + c2/3.0 - 1.0

    elif limbmodel == 'sqrt':
        q1in = ldc[0]
        q2in = ldc[1]
        # Check the validity of this transformation
        c1 =  2*np.sqrt(q1in)*q2in
        c2 =  np.sqrt(q1in)*(1-2*q2in)
#        Imu = 1.0 - c1*(1.0 - mu) - c2*(1.0 - mu**0.5)
        intImu = c1/2.0 + c2/3.0 - 1.0
    else: pass

    return intImu



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
        plt.savefig(path+'MCMC/'+str(koi)+stag+'_'+limbmodel+'.png')
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
            plt.figure(101,figsize=(11,8.5),dpi=300)
        else:
            plt.ion()
            plt.figure(123)
            plt.clf()

    sz = len(q1s)
    for i in range(sz):
        u1,u2 = qtou(q1s[i],q2s[i])
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
 


def foldtime(time,period=1.0,t0=0.0,phase=False):

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

    if phase:
        tfold = tfold/period

    return tfold


def foldtime_pos(time,t0=0.0,period=1.0,phase=False):
    """
    foldtime_pos:
    -------------
    Return folded time with all positive values
    """

# Number of transits before t0 in data
    npstart = np.round((t0 - np.min(time))/period)

# Time of first transit in data
    TT0 = t0 - npstart*period

# Let cycle start at first transit
    tcycle0 = TT0

# tcycle = 0 at first transit

    tcycle = (time - tcycle0) % period

# Time to mid transit is tcycle
    tfold  = tcycle

    if phase:
        tfold = tfold/period

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


def qtou(q1,q2,limb='quad'):
    if limb == 'quad':
        try:
            u1 =  2*np.sqrt(q1)*q2
            u2 =  np.sqrt(q1)*(1-2*q2)
        except:
            u1 = np.nan
            u2 = np.nan
    if limb == 'sqrt':
        try:
            u1 = np.sqrt(q1)*(1-2*q2)
            u2 = 2*np.sqrt(q1)*u2
        except:
            u1 = np.nan
            u2 = np.nan            
    return u1, u2


def utoq(u1,u2,limb='quad'):
    
    if limb == 'quad':
        try:
            q1 = (u1+u2)**2
            q2 = u1/(2.0*(u1+u2))
        except:
            q1 = np.nan
            q2 = np.nan
    if limb == 'sqrt':
        try:
            q1 = (u1+u2)**2
            q2 = u2/(2*(u1+u2))
        except:
            q1 = np.nan
            q2 = np.nan

    return q1, q2



def get_limb_coeff(Tstar,loggstar,filter='Kp',plot=False,network=None,limb='quad',interp='linear'):
    """
    Utility to look up the limb darkening coefficients given an effective temperature and log g.

    """
    from scipy.interpolate import griddata
    import pylab as pl
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    from scipy.interpolate import RectBivariateSpline as bspline

    
    global ldc1func,ldc2func,ldc3func,ldc4func

# Account for gap in look up tables between 4800 and 5000K
#    if (Tstar > 4800 and Tstar <= 4900):
#        Tstar = 4800
#    if (Tstar > 4900 and Tstar < 5000):
#        Tstar = 5000
    
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
    else:
        skiprows = 58
        filtcol = 4
        metcol = 5
        mercol = 6
        col1 = 9
        col2 = 10
        col3 = 11
        col4 = 12


    file1 = 'Claret_cool.dat'
    file2 = 'Claret_hot.dat'

    if network == 'gps':
        path = '/home/jswift/Mdwarfs/'
    if network == 'astro':
        path = '/home/jswift/Mdwarfs/'
    if network == None:
        path = '/Users/jonswift/Astronomy/Exoplanets/TransitFits/'

    # Get data from both the low and high temp files and then append
    limbdata1 = np.loadtxt(path+file1,dtype='string', delimiter='|',skiprows=skiprows)
    limbdata2 = np.loadtxt(path+file2,dtype='string', delimiter='|',skiprows=skiprows)

    logg = limbdata1[:,0].astype(np.float).flatten()
    logg = np.append(logg,limbdata2[:,0].astype(np.float).flatten())
    
    Teff = limbdata1[:,1].astype(np.float).flatten()
    Teff = np.append(Teff,limbdata2[:,1].astype(np.float).flatten())

    Z = limbdata1[:,2].astype(np.float).flatten()
    Z = np.append(Z,limbdata2[:,2].astype(np.float).flatten())

    xi = limbdata1[:,3].astype(np.float).flatten()
    xi = np.append(xi,limbdata2[:,3].astype(np.float).flatten())

    filt = np.char.strip(limbdata1[:,filtcol].flatten())
    filt = np.append(filt,np.char.strip(limbdata2[:,filtcol].flatten()))
    
    method = limbdata1[:,metcol].flatten()
    method = np.append(method,limbdata2[:,metcol].flatten())
    
    avec = limbdata1[:,col1].astype(np.float).flatten()
    avec = np.append(avec,limbdata2[:,col1].astype(np.float).flatten())

    bvec = limbdata1[:,col2].astype(np.float).flatten()
    bvec = np.append(bvec,limbdata2[:,col2].astype(np.float).flatten())

    cvec = limbdata1[:,col3].astype(np.float).flatten()
    cvec = np.append(cvec,limbdata2[:,col3].astype(np.float).flatten())

    dvec = limbdata1[:,col4].astype(np.float).flatten()
    dvec = np.append(dvec,limbdata2[:,col4].astype(np.float).flatten())

# Select out correct filter and method.

    idata, = np.where((filt == filter) & (method == 'L'))
    
    npts = idata.size

    uTeff = np.unique(Teff[idata])
    ulogg = np.unique(logg[idata])

    locs = np.zeros(2*npts).reshape(npts,2)
    locs[:,0] = Teff[idata].flatten()
    locs[:,1] = logg[idata].flatten()
    
    vals = np.zeros(npts)
    vals[:] = avec[idata]

    agrid = np.zeros((len(uTeff),len(ulogg)))
    for i in np.arange(len(uTeff)):
        for ii in np.arange(len(ulogg)):
            eval  = np.array([uTeff[i],ulogg[ii]]).reshape(1,2)
            val = griddata(locs,vals,eval,method='nearest')
            if len(val) > 0:
                agrid[i,ii] = val[0]
            else:
                pass 

    ldc1func = bspline(uTeff, ulogg, agrid, kx=1, ky=1, s=0)    
    aval = ldc1func(Tstar,loggstar)[0][0]

    if plot:      
        plt.figure(1)
        plt.clf()
        plt.imshow(agrid,interpolation='nearest',
                   extent=[np.min(ulogg),np.max(ulogg),np.min(uTeff),np.max(uTeff)],
                   aspect=1./3000,vmin=np.nanmin(agrid),vmax=np.nanmax(agrid))
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
            val = griddata(locs,vals,eval,method='nearest')
            if len(val) > 0:
                bgrid[i,ii] = val[0]
            else:
                pass 

    ldc2func = bspline(uTeff, ulogg, bgrid, kx=1, ky=1, s=0)
    bval = ldc2func(Tstar,loggstar)[0][0]

    if plot:      
        plt.figure(2)
        plt.clf()
        plt.imshow(bgrid,interpolation='nearest',
                   extent=[np.min(ulogg),np.max(ulogg),np.min(uTeff),np.max(uTeff)],
                   aspect=1./3000,vmin=np.nanmin(bgrid),vmax=np.nanmax(bgrid))
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
            val = griddata(locs,vals,eval,method='nearest')
            if len(val) > 0:
                cgrid[i,ii] = val[0]
            else:
                pass 

    ldc3func = bspline(uTeff, ulogg, cgrid, kx=1, ky=1, s=0)
    cval = ldc3func(Tstar,loggstar)[0][0]

    if plot:      
        plt.figure(3)
        plt.clf()
        plt.imshow(cgrid,interpolation='nearest',
                   extent=[np.min(ulogg),np.max(ulogg),np.min(uTeff),np.max(uTeff)],
                   aspect=1./3000,vmin=np.nanmin(cgrid),vmax=np.nanmax(cgrid))
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
            val = griddata(locs,vals,eval,method='nearest')
            if len(val) > 0:
                dgrid[i,ii] = val[0]
            else:
                pass 

    ldc4func = bspline(uTeff, ulogg, dgrid, kx=1, ky=1, s=0)
    dval = ldc4func(Tstar,loggstar)[0][0]

    if plot:      
        plt.figure(4)
        plt.clf()
        plt.imshow(dgrid,interpolation='nearest',
                   extent=[np.min(ulogg),np.max(ulogg),np.min(uTeff),np.max(uTeff)],
                   aspect=1./3000,vmin=np.nanmin(dgrid),vmax=np.nanmax(dgrid))
        plt.colorbar()

    if limb == 'quad':
        return aval, bval

    if limb == 'sqrt':
        return cval, dval

    if limb == 'nlin':
        return aval, bval, cval, dval


def fit_all_singles(kic,nwalkers=1000,burnsteps=1000,mcmcsteps=1000,clobber=False,
                    fit_limb=False,fit_L3=False,network=None,thin=1,clip=False,
                    limbmodel='quad',doplots=False,short=False,rprior=False,
                    lprior=False,getsamp=False,bindiv=10,L3=0.027,reduce=10,
                    start=0,stop=None,claret_limb=False):
    
    import numpy as np
    import time
    import os
    import constants as c

    print ""
    print "----------------------------------------------------------------------"
    print "Getting info on KIC "+str(kic)
    print "----------------------------------------------------------------------"
    info = get_eb_info(kic,L3=L3)

    print ""
    print "----------------------------------------------------------------------"
    print "Getting light curves..."
    print "----------------------------------------------------------------------"
    lcf,lcm = get_lc_data(kic,short=short)

    print ""
    print "----------------------------------------------------------------------"
    print "Solving for all mid-eclipse times..."
    print "----------------------------------------------------------------------"
    tpiter = eclipse_times(lcf,info)

    if not stop:
        stop = len(tpiter)

    print ""
    print "----------------------------------------------------------------------"
    print "Starting iteration over all eclipses"
    print "----------------------------------------------------------------------"
    for i in range(start,stop):
        print ""
        print "----------------------------------------------------------------------"
        print "Getting eclipse #"+str(i)
        print "----------------------------------------------------------------------"
        eclipse_ut = select_eclipse(i,info,thin=1,durfac=2.25,fbuf=1.2,order=3,plot=False)
        if eclipse_ut['status'] == 0:
            print "----------------------------------------------------------------------"
            print "Starting MCMC fitting for KIC "+str(kic)
            print "----------------------------------------------------------------------"
            eclipse = select_eclipse(i,info,thin=thin,durfac=2.25,fbuf=1.2,order=3,plot=True)
            lp,chains,variables = single_fit(eclipse,info,clobber=clobber,nwalkers=nwalkers, \
                                             burnsteps=burnsteps,mcmcsteps=mcmcsteps, \
                                             fit_limb=fit_limb,fit_L3=fit_L3,reduce=reduce, \
                                             claret_limb=claret_limb)
            bestvals = best_single_vals(eclipse,info,chains=chains,lp=lp)
            params_of_interest(eclipse,chains=chains,lp=lp)
        else:
            print "----------------------------------------------------------------------"
            print "Skipping MCMC fit on account of intractable spots"
            print "----------------------------------------------------------------------"

    else:
        print "----------------------------------------------------------------------"
        print "Skipping eclipse #"+str(i)+" due to poor extraction"
        print "----------------------------------------------------------------------"
        
    return
