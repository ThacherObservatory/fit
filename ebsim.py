# Updates
# -------
# Made all band specific quantities iterable (grav dark, reflection, ooe)
# Need to specify band even for a single photometric dataset

# TO DO:
# ------
#
# Compute limb darkening for multiple bands
# Realistic surface brightness ratio given main sequence radius and band
# (see MS_COLORS.csv)

# Be able to tie LDs to Teff and logg through Claret (return functions for
# each band and pass with the fitinfo dictionary)

# In process
# Update make_model_data to be able to produce multiple photometry datasets
# Need independent J, qs, L3, photnoise for each band.


import sys,math,ipdb,time,glob,re,os,eb,emcee,pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import constants as c
import scipy as sp
import robust as rb
from scipy.io.idl import readsav
from length import length
from statsmodels.nonparametric.kernel_density import KDEMultivariate as KDE
from stellar import rt_from_m, flux2mag, mag2flux
import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel, WhiteKernel
from scipy.interpolate import interp1d


################################################################################
# Find geometric base
################################################################################
def find_base(N):

    """
    Routine to interpolate the optimal geometric base for Eq. 5 from empirical data
    Saunders et al. (2006) Figure 16
    """
    
    if N < 10:
        print 'Only valid for N >= 10!'
        return None
    
    # Values from GraphClick
    npts = np.array([9.999, 12.440, 15.449, 19.863, 27.689, 38.222, 51.294, 64.578,	
                     80.080, 92.298, 103.495, 120.361, 140.106, 167.826, 192.838, 221.644,	
                     280.065, 309.063, 353.711, 420.382, 450.130, 498.680])
    
    base =np.array([1.164, 1.133, 1.111, 1.084, 1.059, 1.043, 1.032, 1.024, 1.021, 1.018,
                    1.016, 1.013, 1.012, 1.015, 1.013, 1.012, 1.009, 1.009, 1.008, 1.007,
                    1.006, 1.005])

    base_func = interp1d(npts,base,kind='cubic')

    return base_func(N)


################################################################################
# RV Sampling
################################################################################
def RV_sampling(N,T):
    """Creates a group of RV samples according to Saunders et al. (2006) Eq. 5"""
    
    x = find_base(N)
    
    k = np.arange(N)

    return (x**k - 1)/(x**(N-1) - 1) * T


################################################################################
# Conversion functions from Boyajian et al. (2012)
################################################################################

def r_to_l(r):
    """Converts radius to luminosity (solar units)
    Boyajian et. al 2012 equation 7"""
    
    if r < .1 or r > .9:
        return "Radius outside of suitable range"
    
    log_l = -3.5822 + 6.8639*r - 7.185*r**2 + 4.5169*r**3
    
    return 10**log_l


def r_to_m(r):
    """Converts radius to mass (solar units)
    Inverts Boyajian et. al 2012 equation 11"""
    
    if r < .1 or r > .9:
        return "Radius outside of suitable range"
        
    a = -0.1297
    b = 1.0718
    c = 0.0135-r
    
    return (-b + np.sqrt(b**2 - 4*a*c))/(2*a)


################################################################################
# EB Input Dictionary
################################################################################

def ebinput(m1=None,m2=None,                        # Stellar masses
            r1=0.5,r2=0.3,                          # Stellar radii
            l1=None,l2=None,                        # Stellar luminosity
            L3=0.0,                                 # Third light (fractional amount)
            ecc=0.0,omega=0.0,impact=0,             # Orbital shape and orientation
            period=5.0,t0=2457998.0,                # Ephemeris Sept 1, 2017 (~ TESS launch)
            vsys=10.0,                              # System velocity
            Prot1=None,Prot2=None):                 # Rotation periods for the two stars
    
    """
    This routine creates a dictionary of all the physical parameters of the EB in question. This
    dictionary can then be passed to other functions to create either simulated photometry or 
    radial velocity data.

    Output from this routine can also be used to create starting values for a 
    fit to real data.

    """

    # Set mass to be equal to radius in solar units if flag is set
    if not m1:
        m1 = r_to_m(r1)
        print '... estimating mass of primary star: %f' % m1
    if not m2:
        m2 = r_to_m(r2)
        print '...  estimating mass of secondary star: %f' % m2

    # Mass ratio is not used unless gravity darkening is considered.
    massratio = m2/m1 

    # Surface brightness ratio
    if not l1:
        l1 = r_to_l(r1)
        print '... estimating luminosity of primary star: %f Lsun' % l1
    if not l2:
        l2 = r_to_l(r2)
        print '... estimating luminosity of secondary star: %f Lsun' % l2

    # Effective temperatures
    Teff1 = (l1*c.Lsun/(4*np.pi*(r1*c.Rsun)**2*c.sb ))**(0.25)
    Teff2 = (l2*c.Lsun/(4*np.pi*(r2*c.Rsun)**2*c.sb ))**(0.25)

    # Reference Julian Day
    bjd = 2457998.0

    # For consistency with EB code (GMsun forced to be equal)
    NewtonG = eb.GMSUN*1e6/c.Msun

    # Compute additional orbital parameters from input
    # These calculations were lifted from eb-master code (for consistency).
    ecosw0 = ecc * np.cos(np.radians(omega))
    esinw0 = ecc * np.sin(np.radians(omega))
    ecc0 = ecc
    Mstar1 = m1*c.Msun
    Mstar2 = m2*c.Msun
    sma = ((period*24.0*3600.0)**2*NewtonG*(Mstar1+Mstar2)/(4.0*np.pi**2))**(1.0/3.0)

    # Use Winn (2010) to get inclination from impact parameter of the primary eclipse
    inc = np.arccos(impact*r1*c.Rsun*(1+esinw0)/(sma*(1-ecc0**2)))
    
    esq = ecosw0**2+esinw0**2
    roe = np.sqrt(1.0-esq)
    sini = np.sin(inc)
    qpo = 1.0+Mstar2/Mstar1
    gamma = vsys
    comega = 2.0*np.pi*(1.0 + gamma*1000/eb.LIGHT) / (period*86400.0)
    ktot = (NewtonG*(Mstar1+Mstar2) * comega * sini)**(1.0/3.0)*roe / 1e5


    # Create initial ebpar dictionary
    ebin = {'L1':l1, 'L2':l2, 'L3': L3,
            'Rsum_a':(r1*c.Rsun + r2*c.Rsun)/sma, 'Rratio':r2/r1,
            'Mratio':massratio, 'ecosw':ecosw0, 'esinw':esinw0,
            'Period':period, 't01':t0, 't02':None, 'dt12':None,
            'tdur1':None, 'tdur2':None,'bjd':bjd,
            'Mstar1':m1, 'Mstar2':m2, 'Vsys':vsys, 'Ktot':ktot,
            'Rstar1':r1, 'Rstar2':r2,'cosi':np.cos(inc), 'sma': sma,
            'Rot1':Prot1, 'Rot2':Prot2,
            'Teff1':Teff1, 'Teff2':Teff2}

    return ebin


######################################################################
# L ratio to surface brightness ratio in a given band
######################################################################
#!!! Under construction !!!
#!!! Needs some sanity checks!!!!

def teff_to_j(Teff1,Teff2,band,network=None):

    """
    Only valid for main sequence effective temperatures!
    """

    file = 'MS_Colors.csv'

    if network == None or network == 'bellerophon':
	path = '/home/administrator/python/fit/'
    elif network == 'doug':
        path = '/home/douglas/Astronomy/Resources/'
    elif network == 'astro':
        path = '/home/jswift/Mdwarfs/'
    elif network == 'swift':
        path = '/Users/jonswift/python/fit/'

    data = pd.read_csv(path+file)

    # Effective temperature
    Teff = np.array([np.float(val) for val in data['Teff']])
    # Absolute V band magnitude
    data['Mv'].replace(to_replace='...',value='0.0',inplace=True)
    Mv = np.array([np.float(val) for val in data['Mv']])
    
    # V band bolometric correction
    data['BCv'].replace(to_replace='...',value='0.0',inplace=True)
    BCv  = np.array([np.float(val) for val in data['BCv']])

    if band == 'V':
        color = np.zeros(len(Mv))
    
    if band == 'R' or band == 'Rc' or band == 'Kp':
        # color is V-R
        color = np.array([np.float(val) for val in data['V-Rc'].replace(to_replace='...',value='0.0')])

    if band == 'B':
        # color is V-B
        color = -np.array([np.float(val) for val in data['B-V'].replace(to_replace='...',value='0.0')])

    if band == 'U':
        # color is V-U
        color = -np.array([np.float(val) for val in data['U-B'].replace(to_replace='...',value='0.0')]) - \
                np.array([np.float(val) for val in data['B-V'].replace(to_replace='...',value='0.0')])

    if band == 'J':
        # color is V-J
        color = np.array([np.float(val) for val in data['V-Ks'].replace(to_replace='...',value='0.0')]) - \
                np.array([np.float(val) for val in data['H-K'].replace(to_replace='...',value='0.0')]) - \
                np.array([np.float(val) for val in data['J-H'].replace(to_replace='...',value='0.0')])

    if band == 'H':
        # color is V-H
        color = np.array([np.float(val) for val in data['V-Ks'].replace(to_replace='...',value='0.0')]) - \
                np.array([np.float(val) for val in data['H-K'].replace(to_replace='...',value='0.0')])

    if band == 'K' or band =='Ks':
        # color is V-K
        color = np.array([np.float(val) for val in data['V-Ks'].replace(to_replace='...',value='0.0')])

    
    func = interp1d(Teff,Mv-color,kind='linear')

    
    j = 10**(0.4*(func(Teff1)-func(Teff2)))

    return j
    



################################################################################
# Make photometry data
################################################################################
def make_phot_data(ebin,
                   band='Kp',                              # Photometric band
                   limb='quad',                            # LD type
                   q1a=None,q2a=None,q1b=None,q2b=None,    # LD params
                   J=None,                                 # Surface brightness ratio
                   L3=0.0,                                 # Third light
                   photnoise=0.0003,                       # Photometric noise
                   gravdark=False,reflection=False,        # Higher order effects
                   ellipsoidal=False,                      # Ellipsoidal variations (caution!)
                   tideang=False,                          # Tidal angle (deg)
                   lighttravel=True,                       # Roemer delay
                   TESSshort=False,TESSlong=False,         # Short or long cadence TESS data
                   Kepshort=False,Keplong=False,           # Short or long cadence Kepler data
                   obsdur=27.4,int=120.0,                  # Duration of obs, and int time
                   durfac=2.0,                             # Amount of data to keep around eclipses
                   modelfac=11.0,                          # Integration oversampling 
                   spotamp1=None,spotP1=0.0,P1double=False,# Spot amplitude and period frac for star 1
                   spotfrac1=0.0,spotbase1=0.0,            # Fraction of spots eclipsed, and base
                   spotamp2=None,spotP2=0.0,P2double=False,# Spot amplitude and period frac for star 2
                   spotfrac2=0.0,spotbase2=0.0,            # Fraction of spots eclipsed, and base
                   network=None,outpath='./',write=False): # Network info
    
    """ 
    Function to return light curve data for a given set of inputs and a physical model
    for an EB
    """
    # TESS or Kepler long or short keywords trump obsdur and int keywords
    if TESSshort:
        int = 120.0
        obsdur = 27.4
    if TESSlong:
        int = 1800.0
        obsdur = 27.4
    if Kepshort:
        int = 60.0
        obsdur = 90.0 
    if Keplong:
        int = 1800.0
        obsdur = 1400.0

    # Surface brightness ratio
    # Need to do something to estimate a realistic surface brightness ratio
    # ms = pd.read_csv('MS_Colors.csv',header=0) 
    J = teff_to_j(ebin['Teff1'],ebin['Teff2'],band,network=network)
    print '... estimating surface brightness ratio in '+band+' band: %.3f' % J
    
    l1 = ebin['L1']
    l2 = ebin['L2']

    #####################################################################
    # Input luminosity coversion thing here to get proper amplitudes spot
    # moduluation

    # Spot amplitudes with random phase
    if spotamp1:
        spotflag1 = True
        if spotP1 == 0.0:
            print "Spot Period 1 = 0: Spots on star 1 will not be implemented!"
        spa1 = l1*spotamp1
        spph1 = np.random.uniform(0,np.pi*2,1)[0]
        sinamp1 = spa1*np.cos(spph1)
        cosamp1 = np.sqrt(spa1**2-sinamp1**2)
        if P1double:
            p2 = np.random.uniform(0,2.*np.pi)
            sincosamp1  = P1double*sinamp1 * np.sin(p2)
            squaredamp1 = P1double*sinamp1 * np.cos(p2)
            sinamp1     = (1.0 - P1double)*sinamp1
        else:
            sincosamp1  = 0.0
            squaredamp1 = 0.0            
    else:
        spotP1 = 0.0 ; spotfrac1 = 0.0 ; spotbase1 = 0.0 ; sinamp1 = 0.0 ; cosamp1 = 0.0 ;
        sincosamp1 = 0.0 ; squaredamp1 = 0.0 ; spotflag1 = False

    if spotamp2:
        spotflag2 = True
        if spotP2 == 0.0:
            print "Spot Period 2 = 0: Spots on star 2 will not be implemented!"
        spa2 = l2*spotamp2
        spph2 = np.random.uniform(0,np.pi*2,1)[0]
        sinamp2 = spa2*np.cos(spph2)
        cosamp2 = np.sqrt(spa2**2-sinamp2**2)
        if P2double:
            p2 = np.random.uniform(0,2.*np.pi)
            sincosamp2  = P2double*sinamp2 * np.sin(p2)
            squaredamp2 = P2double*sinamp2 * np.cos(p2)
            sinamp2     = (1.0 - P2double)*sinamp2
        else:
            sincosamp2  = 0.0
            squaredamp2 = 0.0            
    else:
        spotP2 = 0.0 ; spotfrac2 = 0.0 ; spotbase2 = 0.0 ; sinamp2 = 0.0 ; cosamp2 = 0.0 ;
        sincosamp2 = 0.0 ; squaredamp2 = 0.0 ; spotflag2 = False
        
    spotflag = spotflag1 or spotflag2
    
    # Get limb darkening according to input stellar params
    if not q1a or not q2a:
        q1a,q2a = get_limb_qs(Mstar=ebin['Mstar1'],Rstar=ebin['Rstar1'],Tstar=ebin['Teff1'],
                              limb=limb,network=network,band=band)
    if not  q1b or not q2b:
        q1b,q2b = get_limb_qs(Mstar=ebin['Mstar2'],Rstar=ebin['Rstar2'],Tstar=ebin['Teff2'],
                              limb=limb,network=network,band=band)
             
    # Convert q's to u's
    u1a,u2a = qtou(q1a,q2a)
    u1b,u2b = qtou(q1b,q2b)

    # For consistency with EB code (GMsun forced to be equal)
    NewtonG = eb.GMSUN*1e6/c.Msun
    Mstar1 = ebin['Mstar1']*c.Msun
    Mstar2 = ebin['Mstar2']*c.Msun
    sma = ((ebin['Period']*24.0*3600.0)**2*NewtonG*(Mstar1+Mstar2)/(4.0*np.pi**2))**(1.0/3.0)

    rratio = ebin['Rstar2']/ebin['Rstar1']
    rsum = (ebin['Rstar1'] + ebin['Rstar2'])*c.Rsun/sma

    # Integration time and reference time (approximate date of TESS data as default)
    integration = int
    bjd = ebin['bjd']

    ktot = ebin['Ktot']
    
    if lighttravel:
        # double check that the units work out for this
        cltt = ktot / eb.LIGHT         
    else:
        cltt = 0.0

    if gravdark:
        # Lucy 1967, Claret 2000, Lara 2012
        # GD1 = 0.32, GD2 = 0.32 for low mass stars?
        pass
    else:
        GD1 = 0.0 ; GD2 = 0.0

    if reflection:
        #  'Ref1':0.4,  'Ref2':0.4
        pass
    else:
        Ref1 = 0.0 ; Ref2 = 0.0

    if not tideang:
        tideang = 0.0

    # eb.h has 37 parameters. One, "integ", is not used.
    # ktot and vsys are added in our dictionary for completeness.
    ebpar = {'J': J,                                   # surface brightness ratio
             'Rsum': rsum,                             # sum of radii / semimajor axis
             'Rratio': rratio,                         # radius ratio
             'cosi': ebin['cosi'],                     # cosine of inclination
             'ecosw': ebin['ecosw'],                   # ecosw
             'esinw': ebin['esinw'],                   # esini
             'LDlin1':u1a,                             # linear LD, star 1
             'LDnon1':u2a,                             # non-linear LD, star 1
             'LDlin2':u1b,                             # linear LD, star 2   
             'LDnon2':u2b,                             # non-linear LD, star 2
             'GD1': GD1,                               # gravity darkening parameter, star 1
             'GD2': GD2,                               # gravity darkening parameter, star 2
             'Ref1': Ref1,                             # albedo (default) or reflection, star 1
             'Ref2': Ref2,                             # albedo (default) or reflection, star 2
             'Mratio': ebin['Mratio'],                 # stellar mass ratio
             'TideAng': tideang,                       # tidal angle in (degrees)
             'L3': L3,                                 # third light
             'phi0': 0.0,                              # phase of inf. conj. (i think!)
             't0': 0.0, # t0 will be added in later    # epoch of inf. conj. (if phi0=0)
             'Period': ebin['Period'],                 # period of binary
             'Magoff':0.0,                             # magnitude zeropoint (if using mags)
             'Rot1': spotP1,                           # rotation parameter (frac. of period)
             'spFrac1': spotfrac1,                     # fraction of spots covered (prim. eclipse)
             'spBase1': spotbase1,                     # base spottedness, star 1
             'spSin1': sinamp1,                        # sine amp, star 1
             'spCos1': cosamp1,                        # cosine amp, star 1
             'spSinCos1': sincosamp1,                  # sinecosine amp, star 1
             'spSqSinCos1': squaredamp1,               # cos^2-sin^2 amp, star 1                     
             'Rot2': spotP2,                           # rotation parameter, star 2
             'spFrac2': spotfrac2,                     # fraction of spots covered (sec. eclipse) 
             'spBase2': spotbase2,                     # base spottedness, star 2                   
             'spSin2': sinamp2,                        # sine amp, star 2                    
             'spCos2': cosamp2,                        # cosine amp, star 2                          
             'spSinCos2': sincosamp2,                  # sinecosine amp, star 2                      
             'spSqSinCos2': squaredamp2,               # cos^2-sin^2 amp, star 2                     
             'Light_tt': cltt,                         # light travel time
             'Ktot': ktot,                             # total RV amplitude
             'Vsys': ebin['Vsys']}                     # system velocity

    parm, vder = dict_to_params(ebpar)
    
    debug = False
    if debug:
        print "Model parameters:"
        for nm, vl, unt in zip(eb.parnames, parm, eb.parunits):
            print "{0:<10} {1:14.6f} {2}".format(nm, vl, unt)

        vder = eb.getvder(parm, ebin['Vsys'], ktot)
        print "Derived parameters:"
        for nm, vl, unt in zip(eb.dernames, vder, eb.derunits):
            print "{0:<10} {1:14.6f} {2}".format(nm, vl, unt)
            
    period = ebin['Period']
    
    # Contact points of the eclipses
    (ps, pe, ss, se) = eb.phicont(parm)
    # Durations (in hours) and secondary timing
    tdur1 = (pe+1 - ps)*period*24.0
    tdur2 = (se - ss)*period*24.0
    t02   = ebpar['t0'] + (se+ss)/2*period 

    # Photometry sampling
    tstart = -pe*period * durfac
    tstop  = tstart + obsdur
    time  = np.arange(tstart,tstop,integration/86400.0)
    tfold = time % period
    phase = tfold/period
    p0sec = (se+ss)/2
    pprim = (pe-ps+1)*durfac # add one because start phase (ps) is positive and near 1, not negative
    psec  = (se-ss)*durfac
    pinds, = np.where((phase >= 1-pprim/2) | (phase <= pprim/2))
    sinds, = np.where((phase >= p0sec-psec/2) & (phase <= p0sec+psec/2))
    inds = np.append(pinds,sinds)

    s = np.argsort(time[inds])
    tfinal = time[inds][s]
    pfinal = phase[inds][s]

    if not ellipsoidal:
        parm[eb.PAR_Q] = 0

    # tref remains zero so that the ephemeris within parm manifests correctly
    lightmodel = compute_eclipse(tfinal,parm,integration=integration,modelfac=modelfac,
                                     fitrvs=False,tref=0.0,period=period,ooe1fit=None,ooe2fit=None,
                                     unsmooth=False,spotflag=spotflag)
    parm[eb.PAR_Q] = ebin['Mratio']
    
    # Out of eclipse light
    # Phase duration of integration time
    ptint = integration/(3600.*24 * period)

    # Indices of out-of-eclipse light
    ooeinds, = np.where(((pfinal > pe+ptint ) & (pfinal < ss-ptint)) |
                        ((pfinal > se+ptint) & (pfinal < ps-ptint)))

    # Add photometric noise
    if photnoise != None:
        n = len(lightmodel)
        lightmodel += np.random.normal(0,photnoise,n)
        lighterr = np.ones(len(lightmodel))*photnoise
    else:
        lighterr = np.zeros(len(lightmodel))

    lout = np.array([tfinal+bjd,lightmodel,lighterr])
    ooe = np.array([tfinal[ooeinds]+bjd,lightmodel[ooeinds],lighterr[ooeinds]])

    if write:
        np.savetxt('lightcurve_model_'+band+'.txt',lout.T)
        np.savetxt('ooe_model_'+band+'.txt',ooe.T)

    data = {'light':lout, 'ooe':ooe, 'band':band, 'integration':integration, 'L3': L3}
        
    return data



################################################################################
# Make RV data
################################################################################
def make_RV_data(ebin,
                 tRV=None,RVnoise=1.0,RVsamples=100,     # RV noise and sampling
                 lighttravel=True,                       # Roemer delay
                 tideang = None,                         # Don't think this is needed
                 network=None,outpath='./',write=False): # Network info
    
    """ 
    Function to return RV data for a given set of inputs and a physical model for an EB.

    If no "tRV" vector is given, then RVs are sampled geometrically

    """
    period =  ebin['Period']
    
    mratio = ebin['Mstar2']/ebin['Mstar1']

    rratio = ebin['Rstar2']/ebin['Rstar1']

    # For consistency with EB code (GMsun forced to be equal)
    NewtonG = eb.GMSUN*1e6/c.Msun
    Mstar1 = ebin['Mstar1']*c.Msun
    Mstar2 = ebin['Mstar2']*c.Msun
    sma = ((ebin['Period']*24.0*3600.0)**2*NewtonG*(Mstar1+Mstar2)/(4.0*np.pi**2))**(1.0/3.0)
    rsum = (ebin['Rstar1'] + ebin['Rstar2'])*c.Rsun/sma

    # Integration time and reference time (approximate date of TESS data as default)
    integration = int
    bjd = ebin['bjd']

    ktot = ebin['Ktot']
    vsys = ebin['Vsys']
    
    if lighttravel:
        # double check that the units work out for this
        cltt = ktot / eb.LIGHT         
    else:
        cltt = 0.0

    if not tideang:
        tideang = 0.0


    # eb.h has 37 parameters. One, integ, not used.
    # ktot and vsys are added in our dictionary for completeness.
    ebpar = {'J': 1,                                   # surface brightness ratio
             'Rsum': rsum,                             # sum of radii / semimajor axis
             'Rratio': rratio,                         # radius ratio
             'cosi': ebin['cosi'],                     # cosine of inclination
             'ecosw': ebin['ecosw'],                   # ecosw
             'esinw': ebin['esinw'],                   # esini
             'LDlin1':0.0,                             # linear LD, star 1
             'LDnon1':0.0,                             # non-linear LD, star 1
             'LDlin2':0.0,                             # linear LD, star 2   
             'LDnon2':0.0,                             # non-linear LD, star 2
             'GD1': 0.0,                               # gravity darkening parameter, star 1
             'GD2': 0.0,                               # gravity darkening parameter, star 2
             'Ref1': 0.0,                              # albedo (default) or reflection, star 1
             'Ref2': 0.0,                              # albedo (default) or reflection, star 2
             'Mratio': mratio,                         # stellar mass ratio
             'TideAng': tideang,                       # tidal angle in (degrees)
             'L3': 0.0,                                # third light
             'phi0': 0.0,                              # phase of inf. conj. (i think!)
             't0': 0.0, # t0 will be added in later    # epoch of inf. conj. (if phi0=0)
             'Period':period,                          # period of binary
             'Magoff':0.0,                             # magnitude zeropoint (if using mags)
             'Rot1': 0.0,                              # rotation parameter (frac. of period)
             'spFrac1': 0.0,                           # fraction of spots covered (prim. eclipse)
             'spBase1': 0.0,                           # base spottedness, star 1
             'spSin1': 0.0,                            # sine amp, star 1
             'spCos1': 0.0,                            # cosine amp, star 1
             'spSinCos1': 0.0,                         # sinecosine amp, star 1
             'spSqSinCos1': 0.0,                       # cos^2-sin^2 amp, star 1                     
             'Rot2': 0.0,                              # rotation parameter, star 2
             'spFrac2': 0.0,                           # fraction of spots covered (sec. eclipse) 
             'spBase2': 0.0,                           # base spottedness, star 2                   
             'spSin2': 0.0,                            # sine amp, star 2                    
             'spCos2': 0.0,                            # cosine amp, star 2                          
             'spSinCos2': 0.0,                         # sinecosine amp, star 2                      
             'spSqSinCos2': 0.0,                       # cos^2-sin^2 amp, star 2                     
             'light_tt': cltt,                         # light travel time
             'Ktot': ktot,                             # total RV amplitude
             'Vsys': vsys}                             # system velocity

    parm, vder = dict_to_params(ebpar)

    if tRV == None:
        tRV = RV_sampling(RVsamples,period)

    rvs = compute_eclipse(tRV,parm,fitrvs=True,tref=0.0,period=period)
    
    massratio = mratio
    k2 = ktot/(1+massratio)
    k1 = k2*massratio
    rv1 = rvs*k1 + vsys
    rv2 = -1.0*rvs*k2 + vsys

    # make this so that one can determine RV error for each RV point
    if RVnoise != None:
        n1 = len(rv1)
        rv1 += np.random.normal(0,RVnoise,n1)
        rv1_err = np.ones(len(rv1))*RVnoise
        n2 = len(rv2)
        rv2 += np.random.normal(0,RVnoise,n2)
        rv2_err = np.ones(len(rv2))*RVnoise

    r1out = np.array([tRV+bjd,rv1,rv1_err])
    r2out = np.array([tRV+bjd,rv2,rv2_err])
    
    data = {'rv1':r1out, 'rv2':r2out}


    if write:
        np.savetxt('rv1_model.txt',r1out.T)
        np.savetxt('rv2_model.txt',r2out.T)
    
    return data




################################################################################
# Make model data
################################################################################
#!!! Need to make L3 an option that can have different values in each band !!!
def make_model_data(ebin,
                    nphot=None,                             # Number of photometry datasets
                    band=None,                              # Photometric bands of each dataset
                    photnoise=None,                         # Noise of each photometric dataset
                    L3=None,                                # Third light in each band
                    q1a=None,q2a=None,q1b=None,q2b=None,    # LD params for each photometric band
                    limb='quad',                            # LD type                    
                    obsdur=None,int=None,                   # Duration of obs, and int time
                    durfac=None,                            # Amount of data to keep around eclipses
                    gravdark=False,reflection=False,        # Higher order effects
                    ellipsoidal=False,                      # Ellipsoidal variations (caution!)
                    lighttravel=True,                       # Roemer delay
                    tRV=None,RVnoise=1.0,RVsamples=None,    # RV noise and sampling
                    spotamp1=None,spotP1=0.0,P1double=False,# Spot amplitude and period frac for star 1
                    spotfrac1=0.0,spotbase1=0.0,            # Fraction of spots eclipsed, and base
                    spotamp2=None,spotP2=0.0,P2double=False,# Spot amplitude and period frac for star 2
                    spotfrac2=0.0,spotbase2=0.0,            # Fraction of spots eclipsed, and base
                    write=False,network=None,outpath='./'):    # Network info



    data_dict = {}

    for i in range(nphot):
        key = 'phot'+str(i)
        phot = make_phot_data(ebin,band=band[i],limb=limb,
                              photnoise=photnoise[i],
                              q1a=q1a[i],q2a=q2a[i],
                              q1b=q1b[i],q2b=q2b[i],
                              obsdur=obsdur[i],int=int[i],
                              durfac=durfac[i],
                              L3=L3[i],
                              gravdark=gravdark,reflection=reflection,
                              ellipsoidal=ellipsoidal,
                              lighttravel=lighttravel,
                              spotamp1=spotamp1[i],spotP1=spotP1,P1double=P1double,
                              spotfrac1=spotfrac1[i],spotbase1=spotbase1[i],
                              spotamp2=spotamp2[i],spotP2=spotP2,P2double=P2double,
                              spotfrac2=spotfrac2[i],spotbase2=spotbase2[i],
                              write=write,network=network,outpath=outpath)
                       
        data_dict[key] = phot

    if RVsamples:
        rvdata = make_RV_data(ebin,tRV=tRV,RVnoise=RVnoise,RVsamples=RVsamples,
                              lighttravel=lighttravel,network=network,outpath=outpath,
                              write=write)
        
        data_dict['RVdata'] = rvdata

    return data_dict


######################################################################
# EBPAR dictionary to parm array for eb input
######################################################################
def dict_to_params(ebpar):
    """
    ----------------------------------------------------------------------
    dict_to_params:
    --------------
    Function to convert a dictionary of input parameters into a parameter vector
    that is compatible with eb.model
    """

    parm = np.zeros(eb.NPAR, dtype=np.double)
    parm[eb.PAR_J]      = ebpar['J']
    parm[eb.PAR_RASUM]  = ebpar['Rsum']
    parm[eb.PAR_RR]     = ebpar['Rratio']
    parm[eb.PAR_COSI]   = ebpar['cosi']
    parm[eb.PAR_ECOSW]  = ebpar['ecosw']
    parm[eb.PAR_ESINW]  = ebpar['esinw']
    parm[eb.PAR_LDLIN1] = ebpar['LDlin1']
    parm[eb.PAR_LDNON1] = ebpar['LDnon1']
    parm[eb.PAR_LDLIN2] = ebpar['LDlin2']
    parm[eb.PAR_LDNON2] = ebpar['LDnon2']
    parm[eb.PAR_GD1]    = ebpar['GD1']
    parm[eb.PAR_GD2]    = ebpar['GD2']
    parm[eb.PAR_REFL1]  = ebpar['Ref1']
    parm[eb.PAR_REFL2]  = ebpar['Ref2']
    parm[eb.PAR_Q]      = ebpar['Mratio']
    parm[eb.PAR_TIDANG] = ebpar['TideAng']
    parm[eb.PAR_L3]     = ebpar['L3']
    parm[eb.PAR_PHI0]   = ebpar['phi0']
    parm[eb.PAR_T0]     = ebpar['t0']
    parm[eb.PAR_P]      = ebpar['Period']
    parm[eb.PAR_M0]     = ebpar['Magoff']
    parm[eb.PAR_ROT1]   = ebpar['Rot1']
    parm[eb.PAR_ROT2]   = ebpar['Rot2']
    parm[eb.PAR_FSPOT1] = ebpar['spFrac1']
    parm[eb.PAR_FSPOT2] = ebpar['spFrac2']
    parm[eb.PAR_OOE1O]  = ebpar['spBase1']
    parm[eb.PAR_OOE2O]  = ebpar['spBase2']
    parm[eb.PAR_OOE11A] = ebpar['spSin1']
    parm[eb.PAR_OOE11B] = ebpar['spCos1']
    parm[eb.PAR_OOE12A] = ebpar['spSinCos1']
    parm[eb.PAR_OOE12B] = ebpar['spSqSinCos1']
    parm[eb.PAR_OOE21A] = ebpar['spSin2']
    parm[eb.PAR_OOE21B] = ebpar['spCos2']
    parm[eb.PAR_OOE22A] = ebpar['spSinCos2']
    parm[eb.PAR_OOE22B] = ebpar['spSqSinCos2']

    vder = eb.getvder(parm,ebpar['Vsys'],ebpar['Ktot'])
    
    return parm,vder




################################################################################
# Check model data
################################################################################
def check_model(data_dict):
    """
    Produces a quick look plot of the light curve and RV data
    """
    plt.ion()
    
    for i in range(50):
        key = 'phot'+str(i)
        if data_dict.has_key(key):
            data = data_dict[key]
            phot = data['light']
            time = phot[0,:]
            light = phot[1,:]
            ooe = data['ooe']
            tooe = ooe[0,:]
            looe = ooe[1,:]

            plt.figure(i)
            plt.clf()
            plt.plot(time,light,'.',label='All data')
            plt.plot(tooe,looe,'.',label='Out-of-eclipse data')
            plt.title('Photometric Data for Band '+data['band'])
            plt.legend(loc='best')

    if data_dict.has_key('RVdata'):
        data = data_dict['RVdata']
        rvdata1 = data['rv1']
        t1 = rvdata1[0,:]
        rv1 = rvdata1[1,:]

        rvdata2 = data['rv2']
        t2 = rvdata2[0,:]
        rv2 = rvdata2[1,:]

        plt.figure(i+1)
        plt.clf()
        plt.plot(t1,rv1,'bo',label='Primary Star')
        plt.plot(t2,rv2,'go',label='Secondary Star')
        plt.title('Radial Velocity Data')
        plt.legend(loc='best')

    plt.ioff()
    return



######################################################################
# Create dictionary of fit parameters
######################################################################

def fit_params(nwalkers=1000,burnsteps=1000,mcmcsteps=1000,clobber=False,
               fit_lighttravel=True,tie_LD=False,fit_gravdark=False,
               fit_reflection=False,fit_period=True,fit_limb=True,
               fit_rvs=True,fit_ooe1=False,fit_ooe2=False,fit_L3=False,
               fit_tideang=False,
               fit_ellipsoidal=False,write=True,thin=1,outpath='./',network=None):
    """ 
    Generate a dictionary that contains all the information about the fit
    """
    
    fitinfo = {'fit_period':fit_period, 'thin':thin,
               'fit_rvs':fit_rvs, 'fit_limb':fit_limb, 'tie_LD':tie_LD,
               'fit_ooe1':fit_ooe1,'fit_ooe2':fit_ooe2,'fit_ellipsoidal':fit_ellipsoidal,
               'fit_lighttravel':fit_lighttravel,'fit_L3':fit_L3,
               'fit_gravdark':fit_gravdark,'fit_reflection':fit_reflection,
               'fit_tideang':fit_tideang,
               'nwalkers':nwalkers,'burnsteps':burnsteps,'mcmcsteps':mcmcsteps,
               'clobber':clobber,'write':write,'outpath': outpath,'network':network}

    return fitinfo


######################################################################
# Utilities to diagnose a given dataset
######################################################################

def numphot(data_dict):
    """
    Count how many photometry datasets are in 
    """
    try:
        return len([key for key in data_dict.keys() if key.startswith('phot')])
    except:
        print 'There are no photometry datasets in this dictionary!'
        return None

def numbands(data_dict):
    """
    Count how many unique photometry bands there are in data_dict.
    """
    photkeys = [key for key in data_dict.keys() if key.startswith('phot')]
    bands = [data_dict[key]['band'] for key in photkeys]
    return length(np.unique(np.array(bands)))


def uniquebands(data_dict):
    """
    Determine the number of unique photometry bands
    """
    photkeys = [key for key in data_dict.keys() if key.startswith('phot')]
    bands = [data_dict[key]['band'] for key in photkeys]
    _, idx = np.unique(bands, return_index=True)
    final = np.array(bands)[np.sort(idx)].tolist()
    print 'Unique bands are ',final
    return final


######################################################################
# Fit photometry/RV data using emcee
######################################################################

def ebsim_fit(data_dict,fitinfo,ebin,debug=False,threads=1):

    """
    Fit the simulated data using emcee with starting parameters based on the 
    ebin dictionary and according to the fitting parameters outlined in
    fitinfo

    """

    # Check photometry data
    nphot = numphot(data_dict)
    if not nphot:
        print 'Data dictionary does not contain photometry data!'
        return
  
    nbands = numbands(data_dict)
    ubands = uniquebands(data_dict)

    # Check for RVs
    if not data_dict.has_key('RVdata'):
        print 'Data dictionary does not contain RV data!'
        return

    # Check for output directory
    directory = fitinfo['outpath']
    if not os.path.exists(directory):
        os.makedirs(directory)

    print ""
    print "---Starting MCMC fitting routine---"

    twomin = 2./(24.*60.)
    onesec = 1./(24.*60.*60.)

    nw = fitinfo['nwalkers']
    
    ######################################################################
    # Initial chain values
    print "... deriving starting values for chains"

    # Fractional radius        
    variables = []
    p0_init = [np.random.uniform(ebin['Rsum_a']*0.9999,ebin['Rsum_a']*1.0001, nw)]
    variables = np.append(variables,'Rsum')

    # Surface brightness ratio for each band
    for band in ubands:
        J = teff_to_j(ebin['Teff1'],ebin['Teff2'],band,network=fitinfo['network'])    
        p0_init = np.append(p0_init,[np.random.uniform(J*0.9999,J*1.0001,nw)],axis=0) # surface brightness
        variables = np.append(variables,'J_'+band)

    # Radius ratio
    p0_init = np.append(p0_init,[np.random.uniform(ebin['Rratio']*0.9999,
                                                   ebin['Rratio']*1.0001, nw)],axis=0)
    variables = np.append(variables,'Rratio')

    # cos i
    if ebin['cosi'] == 0:
        p0_init = np.append(p0_init,[np.random.uniform(-.0001,.0001,nw)],axis=0)
    else:
        p0_init = np.append(p0_init,[np.random.uniform(ebin['cosi']*0.9999,
                                                       ebin['cosi']*1.0001, nw)],axis=0)
    variables = np.append(variables,'cosi')

    # ecosw
    if ebin['ecosw'] == 0:
        p0_init = np.append(p0_init,[np.random.uniform(-.00001,.00001, nw)],axis=0)
    else:
        p0_init = np.append(p0_init,[np.random.uniform(ebin['ecosw']*0.9999,
                                                       ebin['ecosw']*1.0001, nw)],axis=0)
    variables = np.append(variables,'ecosw')

    # esinw
    if ebin['esinw'] == 0:
        p0_init = np.append(p0_init,[np.random.uniform(-.00001,.00001, nw)],axis=0)
    else:
        p0_init = np.append(p0_init,[np.random.uniform(ebin['esinw']*0.9999,
                                                       ebin['esinw']*1.0001, nw)],axis=0)
    variables = np.append(variables,'esinw')

    # Mid-eclipse time
    p0_init = np.append(p0_init,[np.random.normal(ebin['t01'],onesec,nw)],axis=0)
    variables = np.append(variables,'t0')

    
    # Period
    p0_init = np.append(p0_init,[np.random.uniform(ebin['Period']-onesec,
                                                   ebin['Period']+onesec,nw)],axis=0)
    variables = np.append(variables,'Period')

    # Mass ratio
    p0_init = np.append(p0_init,[np.random.normal(ebin['Mratio'],ebin['Mratio']*0.05,nw)],axis=0)
    variables = np.append(variables,'Mratio')
    
    
    # Limb darkening, two parameters in each band for each star
    for i in range(nbands):
        band = ubands[i]
        try:
            do = fitinfo['fit_limb'][i]
        except:
            do = fitinfo['fit_limb']

        if do:
            # Star 1
            q1a,q2a = get_limb_qs(Mstar=ebin['Mstar1'],Rstar=ebin['Rstar1'],Tstar=ebin['Teff1'],
                                  limb='quad',network=fitinfo['network'],band=band)
            p0_init = np.append(p0_init,[np.random.uniform(q1a*.999,q1a*1.001,nw)],axis=0)
            variables = np.append(variables,'q1a_'+band)
            p0_init = np.append(p0_init,[np.random.uniform(q2a*.999,q2a*1.001,nw)],axis=0)
            variables = np.append(variables,'q2a_'+band)
            
            # Star 2
            q1b,q2b = get_limb_qs(Mstar=ebin['Mstar2'],Rstar=ebin['Rstar2'],Tstar=ebin['Teff2'],
                                  limb='quad',network=fitinfo['network'],band=band)
            p0_init = np.append(p0_init,[np.random.uniform(q1b*.999,q1b*1.001,nw)],axis=0)
            variables = np.append(variables,'q1b_'+band)
            p0_init = np.append(p0_init,[np.random.uniform(q2b*.999,q2b*1.001,nw)],axis=0)
            variables = np.append(variables,'q2b_'+band)
        else:
            print 'Using default limb darkening parameters calculated from input stellar parameters'
            q1a,q2a = get_limb_qs(Mstar=ebin['Mstar1'],Rstar=ebin['Rstar1'],Tstar=ebin['Teff1'],
                                  limb='quad',network=fitinfo['network'],band=band)
            q1b,q2b = get_limb_qs(Mstar=ebin['Mstar2'],Rstar=ebin['Rstar2'],Tstar=ebin['Teff2'],
                                  limb='quad',network=fitinfo['network'],band=band)
            fitinfo['limb_defaults1_'+band] = np.array([q1a,q2a])
            fitinfo['limb_defaults2_'+band] = np.array([q1b,q2b])

            
    # Gravity darkening
    for i in range(nbands):
        band = ubands[i]
        try:
            do = fitinfo['fit_gravdark'][i]
        except:
            do = fitinfo['fit_gravdark']
        if do:
            p0_init = np.append(p0_init,[np.random.uniform(0,1,nw)],axis=0)
            variables = np.append(variables,'GD1_'+band)
            p0_init = np.append(p0_init,[np.random.uniform(0,1,nw)],axis=0)
            variables = np.append(variables,'GD2_'+band)

    #  Reflection/albedo
    for i in range(nbands):
        band = ubands[i]
        try:
            do = fitinfo['fit_reflection'][i]
        except:
            do = fitinfo['fit_reflection']
        if do:
  
            p0_init = np.append(p0_init,[np.random.uniform(0,1,nw)],axis=0)
            variables = np.append(variables,'Ref1_'+band)
            p0_init = np.append(p0_init,[np.random.uniform(0,1,nw)],axis=0)
            variables = np.append(variables,'Ref2_'+band)

    # Tidal angle of primary 
    if fitinfo['fit_tideang']:
        p0_init = np.append(p0_init,[np.random.uniform(0,90,nw)],axis=0)
        variables = np.append(variables,'TideAng')

    # Third light
    for i in range(nbands):
        band = ubands[i]
        try:
            do = fitinfo['fit_L3'][i]
        except:
            do = fitinfo['fit_L3']
        if do:
            p0_init = np.append(p0_init,[np.random.uniform(0,0.25,nw)],axis=0)
            variables = np.append(variables,'L3_'+band)

    # Epoch of inferior conjunction
    p0_init = np.append(p0_init,[np.random.normal(ebin['t01'],onesec,nw)],axis=0)
    variables = np.append(variables,'t0')

    # Period
    if fitinfo['fit_period']:
        p0_init = np.append(p0_init,[np.random.uniform(ebin['Period']-onesec,ebin['Period']+onesec,nw)],axis=0)
        variables = np.append(variables,'Period')


    ##############################
    # Spot Modeling (using GP)
    ##############################
    for i in range(nbands):
        band = ubands[i]
        try:
            do = fitinfo['fit_ooe1'][i]
        except:
            do = fitinfo['fit_ooe1']
        if do:
            # Star 1
            # Quasi-Periodic Kernel for Out of Eclipse Variations

            # Amplitude for QP kernel 1: overall scale of variance and covariance.
            # Initial distribution informed from GP_Fitter example
            p0_init = np.append(p0_init,[np.random.lognormal(0.01,0.001,nw)],axis=0)
            variables = np.append(variables,'OOE_Amp1_'+band)
        
            # Taper on periodic peaks (given as a variance). Smaller numbers, more taper.
            # Initial distribution informed from GP_Fitter example
            p0_init = np.append(p0_init,[np.random.lognormal(5,1,nw)],axis=0)
            variables = np.append(variables,'OOE_SineAmp1_'+band)

            # Width of each periodic peak (gamma = 1/(2s^2))
            # Initial distribution informed from GP_Fitter example
            p0_init = np.append(p0_init,[np.random.lognormal(45,5,nw)],axis=0)
            variables = np.append(variables,'OOE_Decay1_'+band)

            # Period: Separation of peaks (should this be not specific to each band??)
            p0_init = np.append(p0_init,[np.random.lognormal(3.99,0.01,nw)],axis=0)
            variables = np.append(variables,'OOE_Per1_'+band)
            
    try:
        do = True if any(fitinfo['fit_ooe1']) else False
    except:
        if length(fitinfo['fit_ooe1']) == 1:
            do = True if fitinfo['fit_ooe1'] else False
    if do:
        ### Possible to improve treatment of these nuisance parameters.
        # Fraction of spots covered (not band specific!)
        p0_init = np.append(p0_init,[np.random.uniform(0,1,nw)],axis=0)
        variables = np.append(variables,'FSCAve')

        # Base spottedness  = zero (accounted for by GP)
        
    if fitinfo['fit_ooe2']:
        # Wait until fit_ooe1 works, then duplicate here
        pass
    
    # RV fitting
    if fitinfo['fit_rvs']:
        p0_init = np.append(p0_init,[np.random.normal(ebin['Ktot'],1,nw)],axis=0)
        variables = np.append(variables,'Ktot')
        p0_init = np.append(p0_init,[np.random.normal(ebin['Vsys'],5,nw)],axis=0)
        variables = np.append(variables,'Vsys')

    fitinfo['variables'] = variables
    

# Transpose array of initial guesses
    p0 = np.array(p0_init).T

# Number of dimensions in the fit.
    ndim = np.shape(p0)[1]
    print "... fitting data using "+str(ndim)+" free parameters"
    
# Do not redo MCMC unless clobber flag is set
    done = os.path.exists(directory+'Rsum_chain.txt')
    if done == True and fitinfo['clobber'] == False:
        print "MCMC run already completed"
        return False,False

# Set up MCMC sampler
    print "... initializing emcee sampler"
    tstart = time.time()
    sampler = emcee.EnsembleSampler(nw, ndim, lnprob, args=(data_dict,fitinfo),
                                    kwargs={'debug':debug,'ebin': ebin},
                                    threads=threads)
# Run burn-in
    print "... running burn-in with "+str(fitinfo['burnsteps'])+" steps and "+str(fitinfo['nwalkers'])+" walkers"
    pos, prob, state = sampler.run_mcmc(p0, fitinfo['burnsteps'])
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
    np.savetxt(directory+'burnstats.txt',burn)

    # Reset sampler and run MCMC for reals
    print "... resetting sampler and running MCMC with "+str(fitinfo['mcmcsteps'])+" steps"
    sampler.reset()
    posf, probf, statef = sampler.run_mcmc(pos, fitinfo['mcmcsteps'])
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
    np.savetxt(directory+'finalstats.txt',stats)

    # Dump the initial parameter dictionary, fit information dictionary, and the
    # data used in the fit into files for reproduceability.
    pickle.dump(ebpar,open(directory+"ebpar.p", "wb" ))
    pickle.dump(fitinfo,open(directory+"fitinfo.p", "wb" ))
    pickle.dump(data,open(directory+"data.p", "wb" ))

    # Write out chains to disk
    if fitinfo['write']:
        thin = fitinfo['thin']
        thinst = '_thin_'+str(thin) if thin > 1 else ''
        print "Writing MCMC chains to disk"
        lp = sampler.lnprobability.flatten()
        np.savetxt(directory+'lnprob'+thinst+'.txt',lp[0::thin])
        for i in np.arange(len(variables)):
            np.savetxt(directory+variables[i]+'_chain'+thinst+'.txt',sampler.flatchain[0::thin,i])

    return sampler.lnprobability.flatten(),sampler.flatchain


################################################################################
# Input vector to EB parameters
################################################################################
def vec_to_params(x,variables,band=None,ebin=None,fitinfo=None,verbose=True):

    """
    ----------------------------------------------------------------------
    vec_to_params:
    --------------
    Function to convert a vector of input parameters into a parameter vector
    that is compatible with eb.model.

    """

    # For band specific variables
    if band != None:
        btag = '_'+band
    else:
        btag = ''

    ##############################
    # Surface brightness ratio
    try:
        J =   x[variables == 'J'+btag][0] 
    except:
        print "WARNING: you're really NOT going to fit for surface brightness ratio ?! "
        J =  ebin['L2']/ebin['L1']

    ##############################
    # Scaled sum of stellar radii
    try:
        rratio =  x[variables == 'Rsum'][0] 
    except:
        rratio = ebin['Rsum_a']

    ##############################
    # Radius ratio
    try:
        rratio = x[variables == 'Rratio'][0]
    except:
        rratio = ebin['Rratio']

    ##############################
    # Cosine inclination
    try:
        cosi = x[variables == 'cosi'][0]
    except:
        cosi = ebin['cosi']
    
    ##############################
    # ecosw
    try:
        ecosw = x[variables == 'ecosw'][0]
    except:
        ecosw = ebin['ecosw']

    ##############################
    # esinw
    try:
        esinw = x[variables == 'esinw'][0]
    except:
        esinw = ebin['esinw']

    ##############################
    # Period
    try:
        ipdb.set_trace()
        period = x[variables == 'Period']
    except:
        period = ebin['Period']
        
    ##############################
    # t0
    try:
        t0 =  x[variables == 't0'][0]   # T0 (epoch of primary eclipse)
    except:
        t0 = ebin['t01']-ebin['bjd']

    ##############################
    # LD params for star 1
    try:
        q1a = x[variables == 'q1a_'+btag][0]  
        q2a = x[variables == 'q2a_'+btag][0]  
        u1a, u2a = qtou(q1a,q2a,limb=limb)
    except:
        try:
            q1a,q2a = fitinfo['limb_defaults1_'+band]
            u1a,u2a = qtou(q1a,q2a,limb=limb)
        except:
            print 'WARNING: using (unrealistic) default values for limb darkening!'
            u1a=0.5 ; u2a=0.5

    ##############################
    # LD params for star 2
    try:
        q1b = x[variables == 'q1b_'+btag][0]  
        q2b = x[variables == 'q2b_'+btag][0] 
        u1b, u2b = qtou(q1b,q2b)
    except:
        try:
            q1b,q2b = fitinfo['limb_defaults2_'+band]
            u1b,u2b = qtou(q1b,q2b,limb=limb)
        except:
            print 'WARNING: using (unrealistic) default values for limb darkening!'
            u1b=0.5 ; u2b=0.5
 

    ##############################
    # Mass ratio
    # Used for computing ellipsoidal variation and light travel
    # time.  Set to zero to disable ellipsoidal.

    try:
        massratio  = x[variables == 'Mratio'][0]
        ktot  = x[variables == 'Ktot'][0]
        vsys  = x[variables == 'Vsys'][0]
    except:
        massratio  = ebin['Mratio']
        ktot = ebin['Ktot']
        vsys = ebin['Vsys']


    ##############################
    # Third light
    try:
        L3 = x[variables == 'L3'+btag][0]
    except:
        L3 = ebin['L3']
    
    ##############################
    # Light travel time
    try:
        cltt = ktot / eb.LIGHT     # ktot / c
    except:
        print "Cannot perform light travel time correction (no masses)"
        ktot = 0.0
        cllt = 0

    ##############################
    # Gravity darkening
    try:
        GD1 = x[variables == 'GD1'][0]
    except:
        GD1 = ebin['GD1']   # gravity darkening, std. value

    try:
        GD2 = x[variables == 'GD2'][0]
    except:
        GD2 = ebin['GD2']   # gravity darkening, std. value

    ##############################
    # Reflection
    try:
        Ref1 = x[variables == 'Ref1'][0]
    except:
        Ref1 =  ebin['Ref1']  # albedo, std. value

    try:
        Ref2 = x[variables == 'Ref2'][0]
    except:
        Ref2 =  ebin['Ref2']  # albedo, std. value
        


    ##############################
    # Internal spot parameters
    try:
        spotP1      = x[variables == 'Rot1'][0]        # rotation parameter (1 = sync.) 
        spotfrac1   = x[variables == 'spFrac1'][0]     # fraction of spots eclipsed     
        spotbase1   = x[variables == 'spBase1'][0]     # base spottedness out of eclipse
        sinamp1     = x[variables == 'spSin1'][0]      # amplitude of sine component    
        cosamp1     = x[variables == 'spCos1'][0]      # amplitude of cosine component  
        sincosamp1  = x[variables == 'spSinCos1'][0]   # amplitude of sincos cross term 
        squaredamp1 = x[variables == 'spSqSinCos1'][0] # amplitude of sin^2 + cos^2 term
    except:
        spotP1      = 0.0 # rotation parameter (1 = sync.)
        spotfrac1   = 0.0 # fraction of spots eclipsed
        spotbase1   = 0.0 # base spottedness out of eclipse
        sinamp1     = 0.0 # amplitude of sine component
        cosamp1     = 0.0 # amplitude of cosine component
        sincosamp1  = 0.0 # amplitude of sincos cross term
        squaredamp1 = 0.0 # amplitude of sin^2 + cos^2 term

    try:
        spotP2      = x[variables == 'Rot2'][0]        # rotation parameter (1 = sync.) 
        spotfrac2   = x[variables == 'spFrac2'][0]     # fraction of spots eclipsed     
        spotbase2   = x[variables == 'spBase2'][0]     # base spottedness out of eclipse
        sinamp2     = x[variables == 'spSin2'][0]      # amplitude of sine component    
        cosamp2     = x[variables == 'spCos2'][0]      # amplitude of cosine component  
        sincosamp2  = x[variables == 'spSinCos2'][0]   # amplitude of sincos cross term 
        squaredamp2 = x[variables == 'spSqSinCos2'][0] # amplitude of sin^2 + cos^2 term
    except:
        spotP2      = 0.0 # rotation parameter (1 = sync.)
        spotfrac2   = 0.0 # fraction of spots eclipsed
        spotbase2   = 0.0 # base spottedness out of eclipse
        sinamp2     = 0.0 # amplitude of sine component
        cosamp2     = 0.0 # amplitude of cosine component
        sincosamp2  = 0.0 # amplitude of sincos cross term
        squaredamp2 = 0.0 # amplitude of sin^2 + cos^2 term
    
    
    # eb.h has 37 parameters. One, integ, not used.
    # ktot and vsys are added in our dictionary for completeness.
    ebpar = {'J': J,                                   # surface brightness ratio
             'Rsum': rsum,                             # sum of radii / semimajor axis
             'Rratio': rratio,                         # radius ratio
             'cosi': cosi,                             # cosine of inclination
             'ecosw': ecosw,                           # ecosw
             'esinw': esinw,                           # esini
             'LDlin1':u1a,                             # linear LD, star 1
             'LDnon1':u2a,                             # non-linear LD, star 1
             'LDlin2':u1b,                             # linear LD, star 2   
             'LDnon2':u2b,                             # non-linear LD, star 2
             'GD1': GD1,                               # gravity darkening parameter, star 1
             'GD2': GD2,                               # gravity darkening parameter, star 2
             'Ref1': Ref1,                             # albedo (default) or reflection, star 1
             'Ref2': Ref2,                             # albedo (default) or reflection, star 2
             'Mratio': massratio,                      # stellar mass ratio
             'TideAng': tideang,                       # tidal angle in (degrees)
             'L3': L3,                                 # third light
             'phi0': 0.0,                              # phase of inf. conj. (i think!)
             't0': 0.0, # t0 will be added in later    # epoch of inf. conj. (if phi0=0)
             'Period': period,                         # period of binary
             'Magoff':0.0,                             # magnitude zeropoint (if using mags)
             'Rot1': spotP1,                           # rotation parameter (frac. of period)
             'spFrac1': spotfrac1,                     # fraction of spots covered (prim. eclipse)
             'spBase1': spotbase1,                     # base spottedness, star 1
             'spSin1': sinamp1,                        # sine amp, star 1
             'spCos1': cosamp1,                        # cosine amp, star 1
             'spSinCos1': sincosamp1,                  # sinecosine amp, star 1
             'spSqSinCos1': squaredamp1,               # cos^2-sin^2 amp, star 1                     
             'Rot2': spotP2,                           # rotation parameter, star 2
             'spFrac2': spotfrac2,                     # fraction of spots covered (sec. eclipse) 
             'spBase2': spotbase2,                     # base spottedness, star 2                   
             'spSin2': sinamp2,                        # sine amp, star 2                    
             'spCos2': cosamp2,                        # cosine amp, star 2                          
             'spSinCos2': sincosamp2,                  # sinecosine amp, star 2                      
             'spSqSinCos2': squaredamp2,               # cos^2-sin^2 amp, star 2                     
             'light_tt': cltt,                         # light travel time
             'ktot': ktot,                             # total RV amplitude
             'Vsys': vsys}                             # system velocity

    parm, vder = dict_to_params(ebpar)

    

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
    try:
        vder = eb.getvder(parm, vsys, ktot)
    except:
        vder = None
        
#    print "stop in vec_to_parm"
#    pdb.set_trace()
#    print mass1, mass2, ktot, vsys

#    print "Derived parameters:"
#    for name, value, unit in zip(eb.dernames, vder, eb.derunits):
#        print "{0:<10} {1:14.6f} {2}".format(name, value, unit)

    return parm, vder




def compute_eclipse(t,parm,integration=None,modelfac=11.0,fitrvs=False,tref=None,
                    period=None,ooe1fit=None,ooe2fit=None,unsmooth=False,spotflag=False):

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
        if not integration:
            print 'You need to supply an integration time for light curve data!'
            sys.exit()
        if tref == None or period == None:
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
        
#        # Compute ol1 and ol2 vectors if needed
#        if np.shape(ooe1fit):
#            ol1 = np.polyval(ooe1fit,phiarr)
#        else:
#            ol1 = None
#
#        if np.shape(ooe2fit):
#            ol2 = np.polyval(ooe2fit,phiarr)
#        else:
#            ol2 = None


        # Will need to update this to encorporate GP predictions for OOE light
        ol1  = None ; ol2 = None

        typ = np.empty_like(tdarr, dtype=np.uint8)

        # can use eb.OBS_LIGHT to get light output or
        # eb.OBS_MAG to get mag output        
        typ.fill(eb.OBS_LIGHT)
        if spotflag:
            # To create spots modulations using sines and cosines built into eb code
            # need to use time array.
            yarr = eb.model(parm, tdarr, typ)
        else:
            if length(ol1) == length(phiarr) and length(ol2) == length(phiarr):
                yarr = eb.model(parm, phiarr, typ, eb.FLAG_PHI, ol1=ol1, ol2=ol2)
            if length(ol1) == length(phiarr) and not length(ol2) == length(phiarr):
                yarr = eb.model(parm, phiarr, typ, eb.FLAG_PHI, ol1=ol1)
            if length(ol2) == length(phiarr) and not length(ol1) == length(phiarr):
                yarr = eb.model(parm, phiarr, typ, eb.FLAG_PHI, ol2=ol2)
            if ol1 == None and ol2 == None:
                yarr = eb.model(parm, phiarr, typ, eb.FLAG_PHI)
                
        # Average over each integration time
        smoothmodel = np.sum(yarr,axis=0)/np.float(modelfac)
        model = yarr[(modelfac-1)/2,:]

        # Return unsmoothed model if requested
        if unsmooth:
            return model
        else:
            return smoothmodel



def lnprob(x,datadict,fitinfo,ebin=None,debug=False):

    """
    ----------------------------------------------------------------------
    lnprob:
    -------
    Function to compute logarithmic probability of data given model. This
    function sets prior constaints explicitly and calls compute_trans to
    compare the data with the model. 

    Loops through photometric data unique bands and integration times. 
    """

    variables = fitinfo['variables']

    ipdb.set_trace()
    key = datadict.keys()[1]
    # Loop through each dataset in the data dictionary.
    for key in datadict.keys():
        data = datadict[key]
        if key[0:4] == 'phot':
            int = data['integration']
            band = data['band']
            parm,vder = vec_to_params(x,variables,band=band,ebin=ebin,fitinfo=fitinfo)

            time_ooe = data['ooe'][0,:]
            flux_ooe = data['ooe'][1,:]
            err_ooe  = data['ooe'][2,:]
            
            
    nphot = numphot(data)
    photi = np.where(data.keys() == 'phot')
        
    # Get band name from the data dictionary   
    # Issue vec_to_params for each photometry data set
    parm,vder = vec_to_params(x,variables,ebin=ebin)
    # Impose priors on parameters
    # Spot model on OOE light
    # eb model
    # Create log likelihood

    
    # Model RV points

    try:
        vsys = x[variables == 'Vsys'][0]
        ktot = x[variables == 'Ktot'][0]
    except:
        vsys = 0.0
        ktot = 0.0

    #!!! Loop though all unique photometric bands and assign limb darkening
    # - Search variables for 'q*a*' and loop through each set of 4
    # - put q = 0 - 1 prior directly in loops
    # - Option Tie LDs to Claret using stellar parameters if desired
    # - else use input LDs
    
    if fitinfo['tie_LD']:
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

    elif fitinfo['fit_limb']:
        q1a = x[variables == 'q1a'][0]  
        q2a = x[variables == 'q2a'][0]  
        q1b = x[variables == 'q1b'][0]  
        q2b = x[variables == 'q2b'][0]  
        u1a,u2a = qtou(q1a,q2a,limb=ebin['limb'])        
        u1b,u2b = qtou(q1b,q2b,limb=ebin['limb'])
        parm[eb.PAR_LDLIN1] = u1a  # u1 star 1
        parm[eb.PAR_LDNON1] = u2a  # u2 star 1
        parm[eb.PAR_LDLIN2] = u1b  # u1 star 2
        parm[eb.PAR_LDNON2] = u2b  # u2 star 2
    else:
        q1a,q2a = utoq(ebin['LDlin1'],epbar['LDnon1'],limb=ebin['limb'])
        q1b,q2b = utoq(ebin['LDlin2'],epbar['LDnon2'],limb=ebin['limb'])
        

    ##############################
    # LD Priors
    # Exclude conditions that give unphysical limb darkening parameters
    if q1b > 1 or q1b < 0 or q2b > 1 or q2b < 0 or np.isnan(q1b) or np.isnan(q2b):
        return -np.inf        
    if q1a > 1 or q1a < 0 or q2a > 1 or q2a < 0 or np.isnan(q1a) or np.isnan(q2a):
        return -np.inf        

    
    if fitinfo['fit_L3']:
        if parm[eb.PAR_L3] > 1 or parm[eb.PAR_L3] < 0:
            return -np.inf

    # No gravitational lensing or other exotic effects allowed.
    if parm[eb.PAR_J] < 0:
        return -np.inf

    # Primary eclipse
    t0 = parm[eb.PAR_T0]
    if np.abs(t0) > 1800:
        return -np.inf
    
    period = parm[eb.PAR_P]
    time   = data['light'][0,:]-ebin['bjd']
    flux   = data['light'][1,:]
    eflux  = data['light'][2,:]

    # 
    if fitinfo['fit_ooe1'] or fitinfo['fit_ooe2']:
        # Loop through each primary and secondary eclipse and fit each event using
        # individual spot parameters.
        ttm1 = foldtime(time,t0=t0,period=period)
        
        # Phases of contact points
        (ps, pe, ss, se) = eb.phicont(parm)


### Compute eclipse model for given input parameters ###
    massratio = parm[eb.PAR_Q]
    if massratio < 0 or massratio > 10:
        return -np.inf

    if not fitinfo['fit_ellipsoidal']:
        parm[eb.PAR_Q] = 0.0

    ##############################
    # Spot modeling 
    ##############################
    # Spots on primary
    if fitinfo['fit_ooe1'] and not fitinfo['fit_ooe2']:
        res = flux-sm
        theta = np.exp(np.array([x[variables=='OOE_Amp1'],x[variables=='OOE_SineAmp1'],
                                x[variables=='OOE_Per1'],x['OOE_Decay1']]))
        k =  theta[0] * ExpSquaredKernel(theta[1]) * ExpSine2Kernel(theta[2],theta[3])
        gp = george.GP(k,mean=np.mean(res))
        try:
            gp.compute(time, 4,sort=True)
        except (ValueError, np.linalg.LinAlgError):
            return -np.inf
        lf = gp.lnlikelihood(res, quiet=True)
    
        
        sm  = compute_eclipse(time,parm,integration=ebin['integration'],
                              fitrvs=False,tref=t0,period=period,fitooe1=True)

        
    ################################
    # Spots on secondary
    if fitinfo['fit_ooe2'] and not fitinfo['fit_ooe1']:
        pass

    ################################
    # Spots on primary and secondary    
    if fitinfo['fit_ooe1'] and fitinfo['fit_ooe2']:
        pass
    
    ##############################
    # No spots
    if not fitinfo['fit_ooe2'] and not fitinfo['fit_ooe1']:
        sm  = compute_eclipse(time,parm,integration=ebin['integration'],fitrvs=False,tref=t0,period=period)
        # Log Likelihood
        lf = np.sum(-1.0*(sm - flux)**2/(2.0*eflux**2))
        
    # need this for the RVs!
    parm[eb.PAR_Q] = massratio

    rvdata1 = data['rv1']
    rvdata2 = data['rv2']

    if fitinfo['fit_rvs']:
        if (vsys > max(np.max(rvdata1[1,:]),np.max(rvdata2[1,:]))) or \
           (vsys < min(np.min(rvdata1[1,:]),np.min(rvdata2[1,:]))): 
            return -np.inf
        rvmodel1 = compute_eclipse(rvdata1[0,:]-ebin['bjd'],parm,fitrvs=True)
        k2 = ktot/(1+massratio)
        k1 = k2*massratio
        rv1 = rvmodel1*k1 + vsys
        rvmodel2 = compute_eclipse(rvdata2[0,:]-ebin['bjd'],parm,fitrvs=True)
        rv2 = -1.0*rvmodel2*k2 + vsys
        lfrv1 = -np.sum((rv1 - rvdata1[1,:])**2/(2.0*rvdata1[2,:]))
        lfrv2 = -np.sum((rv2 - rvdata2[1,:])**2/(2.0*rvdata2[2,:]))
        lfrv = lfrv1 + lfrv2
        lf  += lfrv

    if debug:
        print "Model parameters:"
        for nm, vl, unt in zip(eb.parnames, parm, eb.parunits):
            print "{0:<10} {1:14.6f} {2}".format(nm, vl, unt)

        vder = eb.getvder(parm, vsys, ktot)
        print "Derived parameters:"
        for nm, vl, unt in zip(eb.dernames, vder, eb.derunits):
            print "{0:<10} {1:14.6f} {2}".format(nm, vl, unt)

        tfold = foldtime(time,t0=t0,period=period)
        keep, = np.where((tfold >= -0.2) & (tfold <=0.2))
        inds = np.argsort(tfold[keep])
        tprim = tfold[keep][inds]
        xprim = flux[keep][inds]
        mprim = sm[keep][inds]
        plt.ion()
        plt.figure(91)
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.plot(tprim,xprim,'ko')
        plt.plot(tprim,mprim,'r-')
        chi1 = -1*lf1
        plt.annotate(r'$\chi^2$ = %.0f' % chi1, [0.05,0.87],
                     horizontalalignment='left',xycoords='axes fraction',fontsize='large')


        tfold_pos = foldtime_pos(time,t0=t0,period=period)
        ph_pos = tfold_pos/period
        keep, = np.where((ph_pos >= 0.3) & (ph_pos <=0.7))
        inds = np.argsort(tfold_pos[keep])
        tsec = tfold_pos[keep][inds]
        xsec = flux[keep][inds]
        msec = sm[keep][inds]
        plt.subplot(2, 2, 2)
        plt.plot(tsec,xsec,'ko')
        plt.plot(tsec,msec,'r-')

        plt.subplot(2, 1, 2)
        phi1 = foldtime(rvdata1[0,:]-ebin['bjd'],t0=t0,period=period)/period
        plt.plot(phi1,rvdata1[1,:],'ko')
        plt.plot(phi1,rv1,'kx')
        tcomp = np.linspace(-0.5,0.5,10000)*period+t0
        rvmodel1 = compute_eclipse(tcomp,parm,fitrvs=True)
        k2 = ktot/(1+massratio)
        k1 = k2*massratio
        rvcomp1 = rvmodel1*k1 + vsys
        plt.plot(np.linspace(-0.5,0.5,10000),rvcomp1,'k--')
        plt.annotate(r'$\chi^2$ = %.2f' % -lfrv, [0.05,0.85],horizontalalignment='left',
                     xycoords='axes fraction',fontsize='large')
  
        phi2 = foldtime(rvdata2[0,:]-ebin['bjd'],t0=t0,period=period)/period
        plt.plot(phi2,rvdata2[1,:],'ro')
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
        plt.title('Limb Darkening')
        
        plt.legend()
        pdb.set_trace()
        
    if np.isnan(lf):
        print 'ln(prob) is not a number!!!'
        lf = -np.inf

    return lf





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




# !!! Needs modification
def varnameconv(variables):

    varmatch = np.array(["J","Rsum","Rratio","cosi",
                         "ecosw","esinw",
                         "magoff","t0","Period",
                         "q1a", "q2a", "q1b", "q2b","u1a","u1b",
                         "MRatio","L3",
                         "Rot1","spFrac1","spBase1","spSin1","spCos1","spSinCos1","spSqSinCos1",
                         "Rot2","spFrac2","spBase2","spSin2","spCos2","spSinCos2","spSqSinCos2",
                         "c0_1","c1_1","c2_1","c3_1","c4_1","c5_1",
                         "c0_2","c1_2","c2_2","c3_2","c4_2","c5_2",
                         "Ktot","Vsys"])

    varnames = np.array(["Surf. Br. Ratio", r"$(R_1+R_2)/a$", r"$R_2/R_1$", r"$\cos i$", 
                         r"$e\cos\omega$",r"$e\sin\omega$",
                         r"$\Delta m_0$", r"$\Delta t_0$ (s)","$\Delta P$ (s)",
                         "$q_1^p$","$q_2^p$","$q_1^s$","$q_2^s$","$u_1^p$","$u_1^s$",
                         "$M_2/M_1$", "$L_3$",
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



def get_limb_qs(Mstar=0.5,Rstar=0.5,Tstar=3800.0,limb='quad',band='Kp',network=None):
    import constants as c

    Ms = Mstar*c.Msun
    Rs = Rstar*c.Rsun
    loggstar = np.log10( c.G * Ms / Rs**2. )
 
    
    if limb == 'nlin':
        a,b,c,d = get_limb_coeff(Tstar,loggstar,network=network,limb=limb,filter=band,interp='linear')
        return a,b,c,d
    else:
        a,b = get_limb_coeff(Tstar,loggstar,network=network,limb=limb,filter=band,interp='linear')
        q1,q2 = utoq(a,b,limb=limb)
        return q1, q2
 


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



def get_limb_coeff(Tstar,loggstar,filter='Kp',plot=False,network=None,limb='quad',interp='linear',
                   passfuncs=False):
    """
    Utility to look up the limb darkening coefficients given an effective temperature and log g.

    """
    from scipy.interpolate import griddata
    import pylab as pl
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    from scipy.interpolate import RectBivariateSpline as bspline

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


    file1 = 'Claret_cool_all_filters.dat'
    file2 = 'Claret_hot_all_filters.dat'

    if network == None or network == 'bellerophon':
	path = '/home/administrator/python/fit/'
    elif network == 'doug':
        path = '/home/douglas/Astronomy/Resources/'
    elif network == 'astro':
        path = '/home/jswift/Mdwarfs/'
    elif network == 'swift':
        path = '/Users/jonswift/python/fit/'

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
        plt.show()
    if passfuncs:
        return ldc1func,ldc2func,ldc3func,ldc4func

    else:
        if limb == 'quad':
            return aval, bval

        if limb == 'sqrt':
            return cval, dval

        if limb == 'nlin':
            return aval, bval, cval, dval











































################################################################################
# Make model data OLD
################################################################################
def make_model_data_orig(m1=None,m2=None,                        # Stellar masses
                         r1=0.5,r2=0.3,                          # Stellar radii
                         l1=None,l2=None,                        # Stellar luminosity
                         ecc=0.0,omega=0.0,impact=0,             # Orbital shape and orientation
                         period=5.0,t0=2457998.0,                # Ephemeris Sept 1, 2017 (~ TESS launch)
                         J=None,                                 # Surface brightness ratio
                         q1a=None,q2a=None,q1b=None,q2b=None,    # LD params
                         limb='quad',                            # LD type
                         L3=0.0,vsys=10.0,                       # Third light and system velocity
                         photnoise=0.0003,                       # Photometric noise
                         short=False,long=False,                 # Short or long cadence TESS data
                         obsdur=27.4,int=120.0,                  # Duration of obs, and int time
                         durfac=2.0,                             # Amount of data to keep around eclipses
                         tRV=None,RVnoise=1.0,RVsamples=100,     # RV noise and sampling
                         lighttravel=True,                       # Roemer delay
                         gravdark=False,reflection=False,        # Higher order effects
                         ellipsoidal=False,                      # Ellipsoidal variations (caution!)
                         tideang=False,                          # Tidal angle (deg)
                         spotamp1=None,spotP1=0.0,P1double=False,# Spot amplitude and period frac for star 1
                         spotfrac1=0.0,spotbase1=0.0,            # Fraction of spots eclipsed, and base
                         spotamp2=None,spotP2=0.0,P2double=False,# Spot amplitude and period frac for star 2
                         spotfrac2=0.0,spotbase2=0.0,            # Fraction of spots eclipsed, and base
                         write=False,network=None,path='./'):    # Network info
    
    """
    Generator of model EB data

    Light curve data and RV data are sampled independently
    
    """

    # If long or short is set, this trumps obsdur and int keywords
    if short:
        int = 120.0
        obsdur = 27.4
    if long:
        int = 1800.0
        obsdur = 27.4
            
    # Set mass to be equal to radius in solar units if flag is set
    if not m1:
        m1 = r_to_m(r1)
    if not m2:
        m2 = r_to_m(r2)

    # Mass ratio is not used unless gravity darkening is considered.
    massratio = m2/m1 #if gravdark else 0.0

    # Surface brightness ratio
    if not l1:
        l1 = r_to_l(r1)
    if not l2:
        l2 = r_to_l(r2)
    if not J:
        J = l2/l1    

    #####################################################################
    # Input luminosity coversion thing here to get proper amplitudes spot
    # moduluation
        
    # Spot amplitudes with random phase
    if spotamp1:
        if spotP1 == 0.0:
            print "Spot Period 1 = 0: Spots on star 1 will not be implemented!"
        spa1 = l1*spotamp1
        spph1 = np.random.uniform(0,np.pi*2,1)[0]
        sinamp1 = spa1*np.cos(spph1)
        cosamp1 = np.sqrt(spa1**2-sinamp1**2)
        if P1double:
            p2 = np.random.uniform(0,2.*np.pi)
            sincosamp1  = P1double*sinamp1 * np.sin(p2)
            squaredamp1 = P1double*sinamp1 * np.cos(p2)
            sinamp1     = (1.0 - P1double)*sinamp1
        else:
            sincosamp1  = 0.0
            squaredamp1 = 0.0            
    else:
        spotP1 = 0.0 ; spa1 = 0.0 ; spph1 = 0.0 ; sinamp1 = 0.0 ; cosamp1 = 0.0 ; sincosamp1  = 0.0
        squaredamp1 = 0.0 

    if spotamp2:
        if spotP2 == 0.0:
            print "Spot Period 2 = 0: Spots on star 2 will not be implemented!"
        spa2 = l2*spotamp2
        spph2 = np.random.uniform(0,np.pi*2,1)[0]
        sinamp2 = spa2*np.cos(spph2)
        cosamp2 = np.sqrt(spa2**2-sinamp2**2)
        if P2double:
            p2 = np.random.uniform(0,2.*np.pi)
            sincosamp2  = P2double*sinamp2 * np.sin(p2)
            squaredamp2 = P2double*sinamp2 * np.cos(p2)
            sinamp2     = (1.0 - P2double)*sinamp2
        else:
            sincosamp2  = 0.0
            squaredamp2 = 0.0            
    else:
        spotP2 = 0.0 ; spa2 = 0.0 ; spph2 = 0.0 ; sinamp2 = 0.0 ; cosamp2 = 0.0 ; sincosamp2  = 0.0
        squaredamp2 = 0.0 

    # Effective temperatures
    Teff1 = (l1*c.Lsun/(4*np.pi*(r1*c.Rsun)**2*c.sb ))**(0.25)
    Teff2 = (l2*c.Lsun/(4*np.pi*(r2*c.Rsun)**2*c.sb ))**(0.25)

    # Get limb darkening according to input stellar params
    if not q1a or not q1b or not q2a or not q2b:
             q1a,q2a = get_limb_qs(Mstar=m1,Rstar=r1,Tstar=Teff1,limb=limb,network=network)
             q1b,q2b = get_limb_qs(Mstar=m2,Rstar=r2,Tstar=Teff2,limb=limb,network=network)
             
    # Integration time and reference time (approximate date of TESS data)
    integration = int
    bjd = 2457998.0

    # For consistency with EB code (GMsun forced to be equal)
    NewtonG = eb.GMSUN*1e6/c.Msun

    # Compute additional orbital parameters from input
    # These calculations were lifted from eb-master code (for consistency).
    ecosw0 = ecc * np.cos(np.radians(omega))
    esinw0 = ecc * np.sin(np.radians(omega))
    ecc0 = ecc
    Mstar1 = m1*c.Msun
    Mstar2 = m2*c.Msun
    sma = ((period*24.0*3600.0)**2*NewtonG*(Mstar1+Mstar2)/(4.0*np.pi**2))**(1.0/3.0)

    # Use Winn (2010) to get inclination from impact parameter of the primary eclipse
    inc = np.arccos(impact*r1*c.Rsun*(1+esinw0)/(sma*(1-ecc0**2)))
    
    esq = ecosw0**2+esinw0**2
    roe = np.sqrt(1.0-esq)
    sini = np.sin(inc)
    qpo = 1.0+Mstar2/Mstar1
    gamma = vsys
    comega = 2.0*np.pi*(1.0 + gamma*1000/eb.LIGHT) / (period*86400.0)
    ktot = (NewtonG*(Mstar1+Mstar2) * comega * sini)**(1.0/3.0)*roe / 1e5


    # Starting parameters
    p0_0  = J                               # surface brightness
    p0_1  = (r1*c.Rsun + r2*c.Rsun)/sma     # (r1 + r2) / a = fractional radius
    p0_2  = r2/r1                           # radius ratio
    p0_3  = np.cos(inc)                     # cos i
    p0_4  = ecc * np.cos(np.radians(omega)) # ecosw
    p0_5  = ecc * np.sin(np.radians(omega)) # esinw
    p0_6  = 10.0                            # mag zpt
    p0_7  = 0.0                             # ephemeris (needs to be zero)
    p0_8  = period                          # Period
    p0_9  = q1a                             # Limb darkening
    p0_10 = q2a                             # Limb darkening
    p0_11 = q1b                             # Limb darkening
    p0_12 = q2b                             # Limb darkening
    p0_13 = m2/m1                           # Mass ratio
    p0_14 = L3                              # Third Light
    p0_15 = spotP1                          # Star 1 rotation
    p0_16 = spotfrac1                       # Fraction of spots eclipsed
    p0_17 = spotbase1                       # base spottedness
    p0_18 = sinamp1                         # Sin amplitude
    p0_19 = cosamp1                         # Cos amplitude
    p0_20 = sincosamp1                      # SinCos amplitude
    p0_21 = squaredamp1                     # Cos^2-Sin^2 amplitude
    p0_22 = spotP2                          # Star 2 rotation
    p0_23 = spotfrac2                       # Fraction of spots eclipsed
    p0_24 = spotbase2                       # base spottedness
    p0_25 = sinamp2                         # Sin amplitude
    p0_26 = cosamp2                         # Cos amplitude
    p0_27 = sincosamp2                      # SinCos amplitude
    p0_28 = squaredamp2                     # Cos^2-Sin^2 amplitude
    p0_29 = ktot                            # Total radial velocity amp
    p0_30 = vsys                            # System velocity

    # Third light (L3) at 14 ... 14 and beyond + 1

    p0_init = np.array([p0_0,p0_1,p0_2,p0_3,p0_4,p0_5,p0_7])
    variables =["J","Rsum","Rratio","cosi","ecosw","esinw","t0"]

    p0_init = np.append(p0_init,[p0_8],axis=0)
    variables.append("period")

    limb0 = np.array([p0_9,p0_10,p0_11,p0_12])
    lvars = ["q1a", "q2a", "q1b", "q2b"]

    p0_init = np.append(p0_init,limb0,axis=0)
    for var in lvars:
        variables.append(var)

    p0_init = np.append(p0_init,[p0_14],axis=0)
    variables.append('L3')


    # Create output path if not already there.
    directory = path
    if not os.path.exists(directory):
        print 'Making directory '+directory
        os.makedirs(directory)

    # Convert q's to u's
    u1a,u2a = qtou(q1a,q2a)
    u1b,u2b = qtou(q1b,q2b)

    # Create initial ebpar dictionary
    ebpar = {'J':J, 'Rsum_a':(r1*c.Rsun + r2*c.Rsun)/sma, 'Rratio':r2/r1,
             'Mratio':massratio, 'LDlin1':u1a, 'LDnon1':u2a, 'LDlin2':u1b, 'LDnon2':u2b,
             'GD1':0.0, 'Ref1':0.0, 'GD2':0.0, 'Ref2':0.0,
             'ecosw':ecosw0, 'esinw':esinw0, 'Period':period, 't01':t0, 't02':None, 
             'et01':0.0, 'et02':0.0, 'dt12':None, 'tdur1':None, 'tdur2':None, 
             'mag0':10.0,'Vsys':vsys, 'Mstar1':m1, 'Mstar2':m2,
             'ktot':ktot, 'L3':L3,'Period':period, 'ePeriod':0.1,
             'integration':int,'obsdur':obsdur,'bjd':bjd,
             'variables':variables,'ninput':0, 'p_init':None,
             'lighttravel':lighttravel,'gravdark':gravdark,
             'reflection':reflection,'path':path,'limb':limb,
             'Rstar1':r1, 'Rstar2':r2,'cosi':np.cos(inc),
             'Rot1':None, 'Rot2':None}
              
    #              'GD1':0.32, 'Ref1':0.4, 'GD2':0.32, 'Ref2':0.4, 'Rot1':0.0,

    # Spots
    spotflag=False
    if spotP1 != 0.0:
        p0_init = np.append(p0_init,[p0_15],axis=0)
        variables.append("Rot1") ; ebpar['Rot1'] = p0_15
        p0_init = np.append(p0_init,[p0_16],axis=0)
        variables.append("spFrac1") ; ebpar['spFrac1'] = p0_16
        p0_init = np.append(p0_init,[p0_17],axis=0)
        variables.append("spBase1") ; ebpar['spBase1'] = p0_17
        p0_init = np.append(p0_init,[p0_18],axis=0)
        variables.append("spSin1") ; ebpar['spSin1'] = p0_18
        p0_init = np.append(p0_init,[p0_19],axis=0)
        variables.append("spCos1") ; ebpar['spCos1'] = p0_19
        p0_init = np.append(p0_init,[p0_20],axis=0)
        variables.append("spSinCos1") ; ebpar['spSinCos1'] = p0_20
        p0_init = np.append(p0_init,[p0_21],axis=0)
        variables.append("spSqSinCos1") ; ebpar['spSqSinCos1'] = p0_21
#        ebpar['spSinCos1'] = 0.0 ; ebpar['spSqSinCos1'] = 0.0
        spotflag=True
    if spotP2 != 0.0:
        p0_init = np.append(p0_init,[p0_22],axis=0)
        variables.append("Rot2") ; ebpar['Rot2'] = p0_22
        p0_init = np.append(p0_init,[p0_23],axis=0)
        variables.append("spFrac2") ; ebpar['spFrac2'] = p0_23
        p0_init = np.append(p0_init,[p0_24],axis=0)
        variables.append("spBase2") ; ebpar['spBase2'] = p0_24
        p0_init = np.append(p0_init,[p0_25],axis=0)
        variables.append("spSin2") ; ebpar['spSin2'] = p0_25
        p0_init = np.append(p0_init,[p0_26],axis=0)
        variables.append("spCos2") ; ebpar['spCos2'] = p0_26
        p0_init = np.append(p0_init,[p0_27],axis=0)
        variables.append("spSinCos2") ; ebpar['spSinCos2'] = p0_27
        p0_init = np.append(p0_init,[p0_28],axis=0)
        variables.append("spSqSinCos2") ; ebpar['spSqSinCos2'] = p0_28
#        ebpar['spSinCos2'] = 0.0 ; ebpar['spSqSinCos2'] = 0.0
        spotflag=True

    # Make "parm" vector
    parm,vder = vec_to_params(p0_init,ebpar,verbose=False)

    # Time of mid-eclipse will be added later.
    parm[eb.PAR_T0] = 0.0

    debug = True
    if debug:
        print "Model parameters:"
        for nm, vl, unt in zip(eb.parnames, parm, eb.parunits):
            print "{0:<10} {1:14.6f} {2}".format(nm, vl, unt)

        vder = eb.getvder(parm, vsys, ktot)
        print "Derived parameters:"
        for nm, vl, unt in zip(eb.dernames, vder, eb.derunits):
            print "{0:<10} {1:14.6f} {2}".format(nm, vl, unt)
            
    # Contact points of the eclipses
    (ps, pe, ss, se) = eb.phicont(parm)

    # Durations (in hours) and secondary timing
    ebpar['tdur1'] = (pe+1 - ps)*period*24.0
    ebpar['tdur2'] = (se - ss)*period*24.0
    ebpar['t02'] = ebpar['t01'] + (se+ss)/2*period 
    
    # Photometry sampling
    tstart = -pe*durfac*period # ?
    tstop  = tstart + obsdur
    time  = np.arange(tstart,tstop,integration/86400.0)
    tfold = time % period
    phase = tfold/period
    p0sec = (se+ss)/2
    pprim = (pe-ps+1)*durfac # add one because start phase (ps) is positive and near 1, not negative
    psec  = (se-ss)*durfac
    pinds, = np.where((phase >= 1-pprim/2) | (phase <= pprim/2))
    sinds, = np.where((phase >= p0sec-psec/2) & (phase <= p0sec+psec/2))
    inds = np.append(pinds,sinds)


    s = np.argsort(time[inds])
    tfinal = time[inds][s]
    pfinal = phase[inds][s]

    
    # This is only for no ellipsoidal variations and gravity darkening!
    mr = parm[eb.PAR_Q]

    # Do other higher order effects depend on the mass ratio?
    if not ellipsoidal:
        parm[eb.PAR_Q] = 0.0

    # tref remains zero so that the ephemeris within parm manifests correctly
    lightmodel = compute_eclipse(tfinal,parm,integration=ebpar['integration'],modelfac=11.0,
                                     fitrvs=False,tref=0.0,period=period,ooe1fit=None,ooe2fit=None,
                                     unsmooth=False,spotflag=spotflag)

    # Out of eclipse light
    # Phase duration of integration time
    ptint = ebpar['integration']/(3600.*24 * period)
    # indices of out of eclipse light
    ooeinds, = np.where(((pfinal > pe+ptint ) & (pfinal < ss-ptint)) |
                        ((pfinal > se+ptint) & (pfinal < ps-ptint)))
                        
    parm[eb.PAR_Q] = mr
    
    if photnoise != None:
        n = len(lightmodel)
        lightmodel += np.random.normal(0,photnoise,n)
        lighterr = np.ones(len(lightmodel))*photnoise
    else:
        lighterr = np.zeros(len(lightmodel))
    
    p0_init = np.append(p0_init,[p0_13],axis=0)
    variables.append('massratio')
    p0_init = np.append(p0_init,[p0_29],axis=0)
    variables.append('ktot')
    p0_init = np.append(p0_init,[p0_30],axis=0)
    variables.append("vsys")

    variables = np.array(variables)
    ebpar['variables'] = variables

    ninput = len(p0_init)
    ebpar['ninput'] = ninput

    ebpar['p_init'] = p0_init

    parm,vder = vec_to_params(p0_init,ebpar,verbose=False)

    # RV sampling
    #    tRV = np.random.uniform(0,1,RVsamples)*period
    if tRV == None:
        tRV = RV_sampling(RVsamples,period)

    rvs = compute_eclipse(tRV,parm,modelfac=11.0,fitrvs=True,tref=0.0,
                          period=period,ooe1fit=None,ooe2fit=None,unsmooth=False)
    
    massratio = m2/m1
    k2 = ktot/(1+massratio)
    k1 = k2*massratio
    rv1 = rvs*k1 + vsys
    rv2 = -1.0*rvs*k2 + vsys

    # make this so that one can determine RV error for each RV point
    if RVnoise != None:
        n1 = len(rv1)
        rv1 += np.random.normal(0,RVnoise,n1)
        rv1_err = np.ones(len(rv1))*RVnoise
        n2 = len(rv2)
        rv2 += np.random.normal(0,RVnoise,n2)
        rv2_err = np.ones(len(rv2))*RVnoise

    lout = np.array([tfinal+bjd,lightmodel,lighterr])
    r1out = np.array([tRV+bjd,rv1,rv1_err])
    r2out = np.array([tRV+bjd,rv2,rv2_err])
    ooe = np.array([tfinal[ooeinds]+bjd,lightmodel[ooeinds],lighterr[ooeinds]])
    
    data = {'light':lout, 'ooe':ooe, 'rv1':r1out, 'rv2':r2out}


    if write:
        np.savetxt('lightcurve_model.txt',lout.T)
        np.savetxt('ooe_model.txt',ooe.T)
        np.savetxt('rv1_model.txt',r1out.T)
        np.savetxt('rv2_model.txt',r2out.T)

    return ebpar, data


def fit_params_orig(ebpar,nwalkers=1000,burnsteps=1000,mcmcsteps=1000,clobber=False,
              fit_period=True,fit_limb=True,claret=False,fit_rvs=True,fit_ooe1=False,fit_ooe2=False,
              fit_L3=False,fit_sp2=False,full_spot=False,fit_ellipsoidal=False,write=True,order=3,
              thin=1):
    """ 
    Generate a dictionary that contains all the information about the fit

    fit_ooe1 and fit_ooe2 if not False should be the order of the polynomial that you would like
    to fit the out of eclipse data with.

    """

    fitinfo = {'ooe_order':order, 'fit_period':fit_period, 'thin':thin,
              'fit_rvs':fit_rvs, 'fit_limb':fit_limb,'claret':claret,
              'fit_ooe1':fit_ooe1,'fit_ooe2':fit_ooe2,'fit_ellipsoidal':fit_ellipsoidal,
              'fit_lighttravel':ebpar['lighttravel'],'fit_L3':fit_L3,
              'fit_gravdark':ebpar['gravdark'],'fit_reflection':ebpar['reflection'],
              'nwalkers':nwalkers,'burnsteps':burnsteps,'mcmcsteps':mcmcsteps,
               'clobber':clobber,'write':write,'variables':None}

    return fitinfo




def ebsim_fit_orig(data,ebpar,fitinfo,debug=False):
    """
    Fit the simulated data using emcee with starting parameters based on the 
    ebpar dictionary and according to the fitting parameters outlined in
    fitinfo
    """

    time   = data['light'][0,:]-ebpar['bjd']
    flux   = data['light'][1,:]
    eflux  = data['light'][2,:]
    sigflux = rb.std(flux)
    deltat = np.max(time)-np.min(time)

    directory = ebpar['path']
    if not os.path.exists(directory):
        os.makedirs(directory)

    print ""
    print "Starting MCMC fitting routine"

    twomin = 2./(24.*60.)
    onesec = 1./(24.*60.*60.)

    nw = fitinfo['nwalkers']
    bjd = ebpar['bjd']
    
# Initial chain values
    print ""
    print "Deriving starting values for chains"
    p0_0  = np.random.uniform(ebpar['J']*0.9999,ebpar['J']*1.0001,nw)             # surface brightness
    p0_1  = np.random.uniform(ebpar['Rsum_a']*0.9999,ebpar['Rsum_a']*1.0001, nw)  # fractional radius
    p0_2  = np.random.uniform(ebpar['Rratio']*0.9999,ebpar['Rratio']*1.0001, nw)  # radius ratio
    if ebpar['cosi'] == 0:                                                        # cos i
        p0_3 = np.random.uniform(-.0001,.0001,nw)
    else:
        p0_3  = np.random.uniform(ebpar['cosi']*0.9999,ebpar['cosi']*1.0001, nw)
    p0_4  = np.random.uniform(-.00001,.00001, nw)                                 # ecosw
    p0_5  = np.random.uniform(-.00001,.00001, nw)                                 # esinw
    p0_6  = np.random.normal(ebpar['mag0'],0.1, nw)                               # mag zpt
    p0_7  = np.random.normal(ebpar['t01']-bjd,onesec,nw)                          # mid-eclipse time
    p0_8  = np.random.uniform(ebpar['Period']-onesec,ebpar['Period']+onesec,nw)   # period

    q1a,q2a = utoq(ebpar['LDlin1'],ebpar['LDnon1'])
    q1b,q2b = utoq(ebpar['LDlin2'],ebpar['LDnon2'])
    p0_9 = np.random.uniform(q1a*.999,q1a*1.001,nw)                               # Limb darkening q1a
    p0_10 = np.random.uniform(q2a*.999,q2a*1.001,nw)                              # Limb darkening q2a
    p0_11 = np.random.uniform(q1b*.999,q1b*1.001,nw)                              # Limb darkening q1b
    p0_12 = np.random.uniform(q2b*.999,q2b*1.001,nw)                              # Limb darkening q2b

    p0_13 = np.abs(np.random.uniform(ebpar['Mratio']*0.9999,                      # Mass ratio
                                     ebpar['Mratio']*1.0001,nw))
    p0_14 = np.random.uniform(0,0.1,nw)                                           # Third Light
    

    ##############################
    # Spot Modeling (use GP)
    ##############################
    # Star 1
    # Quasi-Periodic Kernel for Out of Eclipse Variations
    p0_15 = np.log(np.random.normal(0.05,0.2,nw))                                 # Amplitude for QP kernel 1
    p0_16 = np.log(np.random.uniform(0,10,nw))                                    # Sine Amplitude for QP kernel 1
    p0_17 = np.log(np.random.normal(ebpar['Rot1'],0.1*ebpar['Rot1'],nw))          # Period for QP kernel 1 (star rotation)
    p0_18 = np.log(np.random.uniform(0.1,2*ebpar['Rot1'],nw))                     # Decay of QP kernel 1

    # Exponential Kernel for Fraction of Spots Covered
    p0_19 = np.log(np.random.uniform(ebpar['Rot1']*0.1,ebpar['Rot1']*1.1,nw))     # FSC Amplitude for E kernel 1
    p0_20 = np.log(np.random.uniform(0.1,0.2,nw))                                 # FSC Width for E kernel 1
    p0_21 = np.log(np.random.uniform(0.1,0.2,nw))                                 # FSC Width for E kernel 1

    # Exponential Kernel for Base Spottedness
    p0_22 = np.log(np.random.uniform(ebpar['Rot1']*0.1,ebpar['Rot1']*1.1,nw))     # BS Amplitude for E kernel 1
    p0_23 = np.log(np.random.uniform(0.1,0.2,nw))                                 # BS Width for E kernel 1

    ##############################
    # Star 2
    # Quasi-Periodic Kernel for Out of Eclipse Variations
    p0_24 = np.log(np.random.normal(0.05,0.2,nw))                                 # Amplitude for QP kernel 1
    p0_25 = np.log(np.random.uniform(0,10,nw))                                    # Sine Amplitude for QP kernel 1
    p0_26 = np.log(np.random.normal(ebpar['Rot1'],0.1*ebpar['Rot1'],nw))          # Period for QP kernel 1 (star rotation)
    p0_27 = np.log(np.random.uniform(0.1,2*ebpar['Rot1'],nw))                     # Decay of QP kernel 1

    # Fraction of Spots Covered
    # a + b sin(ct): a > b, a+b <= 1, c positive
    p0_28 = np.log(np.random.uniform(ebpar['Rot1']*0.1,ebpar['Rot1']*1.1,nw))     # FSC Amplitude for E kernel 1
    p0_29 = np.log(np.random.uniform(0.1,0.2,nw))                                 # FSC Width for E kernel 1
    p0_30 = np.log(np.random.uniform(0.1,0.2,nw))                                 # FSC Width for E kernel 1

    # Base Spottedness
    # a sin(bt): a must be restricted, b must be small
    p0_31 = np.log(np.random.uniform(ebpar['Rot1']*0.1,ebpar['Rot1']*1.1,nw))     # BS Amplitude for E kernel 1
    p0_32 = np.log(np.random.uniform(0.1,0.2,nw))                                 # BS Width for E kernel 1

    # System velocity
    p0_33 = np.abs(np.random.uniform(ebpar['ktot']*0.999,ebpar['ktot']*1.001,nw)) # Total radial velocity amp
    p0_34 = np.random.uniform(ebpar['Vsys']*.999,ebpar['Vsys']*1.001,nw)          # System velocity   


# L3 at 14 ... 14 and beyond + 1

    p0_init = np.array([p0_0,p0_1,p0_2,p0_3,p0_4,p0_5,p0_7])
    variables =["J","Rsum","Rratio","cosi","ecosw","esinw","t0"]

    if fitinfo['fit_period']:
        p0_init = np.append(p0_init,[p0_8],axis=0)
        variables.append("period")

    if fitinfo['fit_limb'] and fitinfo['tie_LD']:
        sys.exit('Cannot fit for LD parameters and constrain them according to the other fit parameters!')

    if fitinfo['fit_limb']:
        limb0 = np.array([p0_9,p0_10,p0_11,p0_12])
        lvars = ["q1a", "q2a", "q1b", "q2b"]
        p0_init = np.append(p0_init,limb0,axis=0)
        for var in lvars:
            variables.append(var)

    if fitinfo['fit_L3']:
        p0_init = np.append(p0_init,[p0_14],axis=0)
        variables.append('L3')


    #######################################################################
    # Spot modeling
    #######################################################################

    # Fraction of spots covered could vary from eclipse to eclipse.
    # Need GP kernel for each star
    # Use exponential squared kernel
    # Params = a and l

    
    if fitinfo['fit_ooe1']:
        # Out of eclipse variations = Quasi-periodic kernel
        # Amplitude for the out of eclipse flux
        p0_init = np.append(p0_init,[p0_15],axis=0)
        variables.append('OOE_Amp1')
        # Sine amplitude, makes periodic peaks sharper for higher numbers.
        p0_init = np.append(p0_init,[p0_16],axis=0)
        variables.append('OOE_SineAmp1')
        # Period, separation of peaks.
        p0_init = np.append(p0_init,[p0_17],axis=0)
        variables.append('OOE_Per1')
        # Decay, how significant is the next peak in the kernel
        p0_init = np.append(p0_init,[p0_18],axis=0)
        variables.append('OOE_Decay1')

        # Fraction of spots covered
        p0_init = np.append(p0_init,[p0_19],axis=0)
        variables.append('FSCOff1')
        p0_init = np.append(p0_init,[p0_20],axis=0)
        variables.append('FSCAmp1')
        p0_init = np.append(p0_init,[p0_21],axis=0)
        variables.append('FSCPer1')

        # Base spottedness  = exponential squared kernel
        p0_init = np.append(p0_init,[p0_21],axis=0)
        variables.append('BSAmp1')
        p0_init = np.append(p0_init,[p0_22],axis=0)
        variables.append('BSWid1')

        
    if fitinfo['fit_ooe2']:
        pass
        
    if fitinfo['fit_rvs']:
        p0_init = np.append(p0_init,[p0_13],axis=0)
        variables.append('massratio')
        p0_init = np.append(p0_init,[p0_33],axis=0)
        variables.append('ktot')
        p0_init = np.append(p0_init,[p0_34],axis=0)
        variables.append("vsys")

    variables = np.array(variables)

    fitinfo['variables'] = variables

    
# Transpose array of initial guesses
    p0 = np.array(p0_init).T

# Number of dimensions in the fit.
    ndim = np.shape(p0)[1]

# Do not redo MCMC unless clobber flag is set
    done = os.path.exists(directory+'Jchain.txt')
    if done == True and fitinfo['clobber'] == False:
        print "MCMC run already completed"
        return False,False


# Set up MCMC sampler
    print "... initializing emcee sampler"
    tstart = time.time()
    sampler = emcee.EnsembleSampler(nw, ndim, lnprob, args=(data,ebpar,fitinfo),kwargs={'debug':debug})

# Run burn-in
    print ""
    print "Running burn-in with "+str(fitinfo['burnsteps'])+" steps and "+str(fitinfo['nwalkers'])+" walkers"
    pos, prob, state = sampler.run_mcmc(p0, fitinfo['burnsteps'])
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
    np.savetxt(directory+'burnstats.txt',burn)

    # Reset sampler and run MCMC for reals
    print "... resetting sampler and running MCMC with "+str(fitinfo['mcmcsteps'])+" steps"
    sampler.reset()
    posf, probf, statef = sampler.run_mcmc(pos, fitinfo['mcmcsteps'])
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
    np.savetxt(directory+'finalstats.txt',stats)

    # Dump the initial parameter dictionary, fit information dictionary, and the
    # data used in the fit into files for reproduceability.
    pickle.dump(ebpar,open(directory+"ebpar.p", "wb" ))
    pickle.dump(fitinfo,open(directory+"fitinfo.p", "wb" ))
    pickle.dump(data,open(directory+"data.p", "wb" ))

    # Write out chains to disk
    if fitinfo['write']:
        thin = fitinfo['thin']
        thinst = '_thin_'+str(thin) if thin > 1 else ''
        print "Writing MCMC chains to disk"
        lp = sampler.lnprobability.flatten()
        np.savetxt(directory+'lnprob'+thinst+'.txt',lp[0::thin])
        for i in np.arange(len(variables)):
            np.savetxt(directory+variables[i]+'_chain'+thinst+'.txt',sampler.flatchain[0::thin,i])

    return sampler.lnprobability.flatten(),sampler.flatchain



################################################################################
# Input vector to EB parameters
################################################################################
def vec_to_params_orig(x,ebpar,fitinfo=None,verbose=True):
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
    !!! To Be Updated !!!
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
    x[30] = total radial velocity amplitude
    x[31] = system velocity



    """
    
    if fitinfo != None:
        variables = fitinfo['variables']
        fitooe1 = fitinfo['fit_ooe1']
        fitooe2 = fitinfo['fit_ooe2']
        if len(variables) != len(x):
            print 'Length of variables not equal to length of input vector'
    else:
        if verbose:
            print 'vec_to_params: Using default values from ebpar'
        variables = None
        fitooe1 = None
        fitooe2 = None

        parm = np.zeros(eb.NPAR, dtype=np.double)
    # These are the basic parameters of the model.
    try:
        parm[eb.PAR_J]      =  x[variables == 'J'][0]  # J surface brightness ratio
    except:
        parm[eb.PAR_J]      =  ebpar['J']
    try:
        parm[eb.PAR_RASUM]  =  x[variables == 'Rsum'][0] # (R_1+R_2)/a
    except:
        parm[eb.PAR_RASUM]  = ebpar['Rsum_a']
        
    try:
        parm[eb.PAR_RR]     = x[variables == 'Rratio'][0]   # R_2/R_1
    except:
        parm[eb.PAR_RR]     = ebpar['Rratio']
    try:
        parm[eb.PAR_COSI]   =  x[variables == 'cosi'][0]  # cos i
    except:
        parm[eb.PAR_COSI]   = ebpar['cosi']
        
    # Orbital parameters.
    try:
        parm[eb.PAR_ECOSW]  = x[variables == 'ecosw'][0]    # ecosw
    except:
        parm[eb.PAR_ECOSW]  = ebpar['ecosw']
    try:
        parm[eb.PAR_ESINW]  = x[variables == 'esinw'][0]    # esinw
    except:
        parm[eb.PAR_ESINW]  = ebpar['esinw']
        
    # Period
    try:
        parm[eb.PAR_P] = x[variables == 'Period'][0]
    except:
        parm[eb.PAR_P] = ebpar["Period"]
        
    # T0
    try:
        parm[eb.PAR_T0] =  x[variables == 't0'][0]   # T0 (epoch of primary eclipse)
    except:
        parm[eb.PAR_T0] = ebpar['t01']-ebpar['bjd']

    # offset magnitude
    try:
        parm[eb.PAR_M0] =  x[variables == 'Magoff'][0]  
    except:
        parm[eb.PAR_M0] = ebpar['mag0']


    # Limb darkening paramters for star 1
    try:
        q1a = x[variables == 'q1a'][0]  
        q2a = x[variables == 'q2a'][0]  
        a1, b1 = qtou(q1a,q2a,limb=limb)
        parm[eb.PAR_LDLIN1] = a1
        parm[eb.PAR_LDNON1] = b1
    except:
        parm[eb.PAR_LDLIN1] = ebpar["LDlin1"]   # u1 star 1
        parm[eb.PAR_LDNON1] = ebpar["LDnon1"]   # u2 star 1


    # Limb darkening paramters for star 2
    try:
        q1b = x[variables == 'q1b'][0]  
        q2b = x[variables == 'q2b'][0] 
        a2, b2 = qtou(q1b,q2b)
        parm[eb.PAR_LDLIN2] = a2
        parm[eb.PAR_LDNON2] = b2
    except:
        parm[eb.PAR_LDLIN2] = ebpar["LDlin2"]   # u1 star 2
        parm[eb.PAR_LDNON2] = ebpar["LDnon2"]   # u2 star 2


    # Mass ratio is used only for computing ellipsoidal variation and
    # light travel time.  Set to zero to disable ellipsoidal.

    try:
        parm[eb.PAR_Q]  = x[variables == 'MRatio'][0]
        ktot  = x[variables == 'Ktot'][0]
        vsys  = x[variables == 'Vsys'][0]
    except:
        parm[eb.PAR_Q]  = ebpar['Mratio']
        ktot = ebpar['Ktot']
        vsys = ebpar['Vsys']

    try:
        parm[eb.PAR_L3] = x[variables == 'L3'][0]
    except:
        parm[eb.PAR_L3] = ebpar["L3"]
    
    # Light travel time coefficient.
    if ebpar['lighttravel']:        
        try:
            cltt = ktot / eb.LIGHT
            parm[eb.PAR_CLTT]   =  cltt      # ktot / c
        except:
            print "Cannot perform light travel time correction (no masses)"
            ktot = 0.0


    if ebpar['gravdark']:
        parm[eb.PAR_GD1]    = ebpar['GD1']   # gravity darkening, std. value
        parm[eb.PAR_GD2]    = ebpar['GD2']   # gravity darkening, std. value

    if ebpar['reflection']:
        parm[eb.PAR_REFL1]  = ebpar['Ref1']  # albedo, std. value
        parm[eb.PAR_REFL2]  = ebpar['Ref2']  # albedo, std. value



    if ebpar['Rot1'] and fitinfo == None:
        parm[eb.PAR_ROT1]   = ebpar['Rot1']        # rotation parameter (1 = sync.)
        parm[eb.PAR_FSPOT1] = ebpar['spFrac1']     # fraction of spots eclipsed
        parm[eb.PAR_OOE1O]  = ebpar['spBase1']     # base spottedness out of eclipse
        parm[eb.PAR_OOE11A] = ebpar['spSin1']      # amplitude of sine component
        parm[eb.PAR_OOE11B] = ebpar['spCos1']      # amplitude of cosine component
        parm[eb.PAR_OOE12A] = ebpar['spSinCos1']   # amplitude of sincos cross term
        parm[eb.PAR_OOE12B] = ebpar['spSqSinCos1'] # amplitude of sin^2 + cos^2 term
            
    if ebpar['Rot2'] and fitinfo == None:
        parm[eb.PAR_ROT2]   = ebpar['Rot2']        # rotation parameter (1 = sync.)
        parm[eb.PAR_FSPOT2] = ebpar['spFrac2']     # fraction of spots eclipsed
        parm[eb.PAR_OOE2O]  = ebpar['spBase2']     # base spottedness out of eclipse
        parm[eb.PAR_OOE21A] = ebpar['spSin2']      # amplitude of sine component
        parm[eb.PAR_OOE21B] = ebpar['spCos2']      # amplitude of cosine component
        parm[eb.PAR_OOE22A] = ebpar['spSinCos2']   # amplitude of sincos cross term
        parm[eb.PAR_OOE22B] = ebpar['spSqSinCos2'] # amplitude of sin^2+cos^2 term


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
    try:
        vder = eb.getvder(parm, vsys, ktot)
    except:
        vder = None
        
#    print "stop in vec_to_parm"
#    pdb.set_trace()
#    print mass1, mass2, ktot, vsys

#    print "Derived parameters:"
#    for name, value, unit in zip(eb.dernames, vder, eb.derunits):
#        print "{0:<10} {1:14.6f} {2}".format(name, value, unit)

    return parm, vder

