# TO DO:
#-------

# High priority
#--------------
# Update bestvals, plot_model, and triangle to be compatible with
# output of simulation fits
#
# Low priority:
#--------------
# Reconfigure code for parallelization.


import sys,math,pdb,time,glob,re,os,eb,emcee,pickle
import numpy as np
import matplotlib.pyplot as plt
import constants as c
import scipy as sp
import robust as rb
from scipy.io.idl import readsav
from length import length
#from kepler_tools import *
import pyfits as pf
from statsmodels.nonparametric.kernel_density import KDEMultivariate as KDE
from stellar import rt_from_m, flux2mag, mag2flux


def find_base(N):
    """
    Routine to interpolate the optimal geometric base for Eq. 5 from empirical data
    Saunders et al. (2006) Figure 16
    """

    from scipy.interpolate import interp1d

    if N <10:
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



def RV_sampling(N,T):
    """Creates a group of RV samples according to Saunders et al. (2006) Eq. 5"""
    
    x = find_base(N)
    
    k = np.arange(N)
    return (x**k - 1)/(x**(N-1) - 1) * T


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


def make_model_data(m1=None,m2=None,r1=0.7,r2=0.5,ecc=0.0,omega=0.0,impact=0,
                    period=4.123456789, t0=2454833.0,int=60.0,
                    q1a=None,q2a=None,q1b=None,q2b=None,J=None,L3=0.0,vsys=10.0,
                    gravdark=False,reflection=False,ellipsoidal=False,
                    photnoise=0.0003,RVnoise=0.1,RVsamples=100,
                    lighttravel=True,durfac=1.5,ncycles=1,durpoints=None,
                    write=False,path='./',limb='quad'):


    """
    Generator of model EB data

    Light curve data and RV data are sampled independently
    
    """

    # Set mass to be equal to radius in solar units if flag is set
    if not m1 and not m2:
        m1 = r_to_m(r1)
        m2 = r_to_m(r2)

    # Mass ratio is not used unless gravity darkening is considered.
    massratio = m2/m1 #if gravdark else 0.0

    # Surface brightness ratio
    if not J:
        l1 = r_to_l(r1)
        l2 = r_to_l(r2)
        J = l2/l1

    Teff1 = (l1*c.Lsun/(4*np.pi*(r1*c.Rsun)**2*c.sb ))**(0.25)
    Teff2 = (l2*c.Lsun/(4*np.pi*(r2*c.Rsun)**2*c.sb ))**(0.25)

    # Get limb darkening according to input stellar params
    if not q1a or not q1b or not q2a or not q2b:
             q1a,q2a = get_limb_qs(Mstar=m1,Rstar=r1,Tstar=Teff1,limb=limb)
             q1b,q2b = get_limb_qs(Mstar=m2,Rstar=r2,Tstar=Teff2,limb=limb)
        
    # Integration time and reference time
    integration = int
    bjd = 2454833.0

    # For consistency with EB code
    NewtonG = eb.GMSUN*1e6/c.Msun

    # Compute additional orbital parameters from input
    # These calculations were lifted from eb-master code (for consistency).
    ecosw0 = ecc * np.cos(np.radians(omega))
    esinw0 = ecc * np.sin(np.radians(omega))
    ecc0 = ecc
    Mstar1 = m1*c.Msun
    Mstar2 = m2*c.Msun
    sma = ((period*24.0*3600.0)**2*NewtonG*(Mstar1+Mstar2)/(4.0*np.pi**2))**(1.0/3.0)

    # Winn 2010 to get inclination from impact parameter of the primary eclipse
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
    p0_7  = 0.0                             # ephemeris
    p0_8  = period                          # Period
    p0_9  = q1a                             # Limb darkening
    p0_10 = q2a                             # Limb darkening
    p0_11 = q1b                             # Limb darkening
    p0_12 = q2b                             # Limb darkening
    p0_13 = m2/m1                           # Mass ratio
    p0_14 = L3                              # Third Light
    p0_15 = 0.0                             # Star 1 rotation
    p0_16 = 0.0                             # Fraction of spots eclipsed
    p0_17 = 0.0                             # base spottedness
    p0_18 = 0.0                             # Sin amplitude
    p0_19 = 0.0                             # Cos amplitude
    p0_20 = 0.0                             # SinCos amplitude
    p0_21 = 0.0                             # Cos^2-Sin^2 amplitude
    p0_22 = 0.0                             # Star 2 rotation
    p0_23 = 0.0                             # Fraction of spots eclipsed
    p0_24 = 0.0                             # base spottedness
    p0_25 = 0.0                             # Sin amplitude
    p0_26 = 0.0                             # Cos amplitude
    p0_27 = 0.0                             # SinCos amplitude
    p0_28 = 0.0                             # Cos^2-Sin^2 amplitude
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

    
    # Create initial ebpar dictionary
    ebpar = {'J':J, 'Rsum_a':(r1*c.Rsun + r2*c.Rsun)/sma, 'Rratio':r2/r1,
              'Mratio':massratio, 'LDlin1':q1a, 'LDnon1':q2a, 'LDlin2':q1b, 'LDnon2':q2b,
              'GD1':0.0, 'Ref1':0.0, 'GD2':0.0, 'Ref2':0.0, 'Rot1':0.0,
              'ecosw':ecosw0, 'esinw':esinw0, 'Period':period, 't01':bjd, 't02':None, 
              'et01':0.0, 'et02':0.0, 'dt12':None, 'tdur1':None, 'tdur2':None, 
              'mag0':10.0,'vsys':vsys, 'Mstar1':m1, 'Mstar2':m2,
              'ktot':ktot, 'L3':L3,'Period':period, 'ePeriod':0.1,
              'integration':int,'bjd':bjd,'variables':variables,'ninput':0,
              'lighttravel':lighttravel,'gravdark':gravdark,
              'reflection':reflection,'path':path,'limb':limb,
              'Rstar1':r1, 'Rstar2':r2}
              
    #              'GD1':0.32, 'Ref1':0.4, 'GD2':0.32, 'Ref2':0.4, 'Rot1':0.0,


    # Make "parm" vector
    parm,vder = vec_to_params(p0_init,ebpar)

    debug = False
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

    # Durations (in hourse) and secondary timing
    ebpar['tdur1'] = (pe+1 - ps)*period*24.0
    ebpar['tdur2'] = (se - ss)*period*24
    ebpar['t02'] = ebpar['t01'] + (se+ss)/2*period 

    # Photometry sampling
    tdprim = (pe+1 - ps)*period * durfac
    tdsec  = (se - ss)*period * durfac
    t0sec = (se+ss)/2*period 

    # Sample primary eclipse with the specified number of points, else use integration time as cadence
    if durpoints:
        tphot = np.append(np.linspace(-tdprim/2,tdprim/2,durpoints),
                          np.linspace(-tdsec/2+t0sec,tdsec/2+t0sec,durpoints*tdsec/tdprim))
    else:
        tphot = np.append(np.arange(-tdprim/2,tdprim/2,integration/86400.0),
                          np.arange(-tdsec/2+t0sec,tdsec/2+t0sec,integration/86400.0))
        
    mr = parm[eb.PAR_Q]
    parm[eb.PAR_Q] = 0.0

    lightmodel = compute_eclipse(tphot,parm,integration=ebpar['integration'],modelfac=11.0,
                                 fitrvs=False,tref=0.0,period=period,ooe1fit=None,ooe2fit=None,
                                 unsmooth=False)
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
    
    parm,vder = vec_to_params(p0_init,ebpar)

    # RV sampling
#    tRV = np.random.uniform(0,1,RVsamples)*period

    tRV = RV_sampling(RVsamples,period)

    rvs = compute_eclipse(tRV,parm,modelfac=11.0,fitrvs=True,tref=0.0,
                          period=period,ooe1fit=None,ooe2fit=None,unsmooth=False)
    
    massratio = m2/m1
    k2 = ktot/(1+massratio)
    k1 = k2*massratio
    rv1 = rvs*k1 + vsys
    rv2 = -1.0*rvs*k2 + vsys

    if RVnoise != None:
        n1 = len(rv1)
        rv1 += np.random.normal(0,RVnoise,n1)
        rv1_err = np.ones(len(rv1))*RVnoise
        n2 = len(rv2)
        rv2 += np.random.normal(0,RVnoise,n2)
        rv2_err = np.ones(len(rv2))*RVnoise
    
    lout = np.array([tphot+2454833.0,lightmodel,lighterr])
    r1out = np.array([tRV+2454833.0,rv1,rv1_err])
    r2out = np.array([tRV+2454833.0,rv2,rv2_err])

    data = {'light':lout,'rv1':r1out,'rv2':r2out}


    if write:
        np.savetxt('lightcurve_model.txt',lout.T)
        np.savetxt('rv1_model.txt',r1out.T)
        np.savetxt('rv2_model.txt',r2out.T)

    return ebpar, data


def check_model(data):
    """
    Produces a quick look plot of the light curve and RV data
    """

    phot = data['light']
    time = phot[0,:]
    light = phot[1,:]

    rvdata1 = data['rv1']
    t1 = rvdata1[0,:]
    rv1 = rvdata1[1,:]

    rvdata2 = data['rv2']
    t2 = rvdata2[0,:]
    rv2 = rvdata2[1,:]

    plt.figure(1)
    plt.clf()
    plt.plot(time,light,'.')
    
    plt.figure(2)
    plt.clf()
    plt.plot(t1,rv1,'b.')
    plt.plot(t2,rv2,'g.')
    plt.show()
    
    return


def fit_params(ebpar,nwalkers=1000,burnsteps=1000,mcmcsteps=1000,clobber=False,
              fit_period=False,fit_limb=True,claret=False,fit_rvs=True,fit_ooe1=False,fit_ooe2=False,
              fit_L3=False,fit_sp2=False,full_spot=False,fit_ellipsoidal=False,write=True,order=3,
              thin=1):
    """ 
    Generate a dictionary that contains all the information about the fit
    """

    fitinfo = {'ooe_order':order, 'fit_period':fit_period, 'thin':thin,
              'fit_rvs':fit_rvs, 'fit_limb':fit_limb,'claret':claret,
              'fit_ooe1':fit_ooe1,'fit_ooe2':fit_ooe2,'fit_ellipsoidal':fit_ellipsoidal,
              'fit_lighttravel':ebpar['lighttravel'],'fit_L3':fit_L3,
              'fit_gravdark':ebpar['gravdark'],'fit_reflection':ebpar['reflection'],
              'nwalkers':nwalkers,'burnsteps':burnsteps,'mcmcsteps':mcmcsteps,
               'clobber':clobber,'write':write,'variables':None}

    return fitinfo


def ebsim_fit(data,ebpar,fitinfo):
    """
    Fit the simulated data using emcee with starting parameters based on the 
    ebpar dictionary and according to the fitting parameters outlined in
    fitinfo
    """
    
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
    p0_0  = np.random.uniform(ebpar['J']*0.9,ebpar['J']*1.1,nw)             # surface brightness
    p0_1  = np.random.uniform(ebpar['Rsum_a']*0.9,ebpar['Rsum_a']*1.1, nw)    # fractional radius
    p0_2  = np.random.uniform(ebpar['Rratio']*0.9,ebpar['Rratio']*1.1, nw)    # radius ratio
    p0_3  = np.random.uniform(0,ebpar['Rsum_a'], nw)                          # cos i
#    p0_3  = np.random.uniform(0,0.01, nw)                                    # cos i
    p0_4  = np.random.uniform(0,0.01, nw)                                      # ecosw
    p0_5  = np.random.uniform(0,0.01, nw)                                      # esinw
    p0_6  = np.random.normal(ebpar['mag0'],0.1, nw)                          # mag zpt
    p0_7  = np.random.normal(ebpar['t01']-bjd,onesec,nw)                     # ephemeris
    p0_8  = np.random.normal(ebpar['Period'],onesec,nw    )                  # Period
    p0_9  = np.random.uniform(0,1,nw)                                         # Limb darkening
    p0_10 = np.random.uniform(0,1,nw)                                         # Limb darkening
    p0_11 = np.random.uniform(0,1,nw)                                         # Limb darkening
    p0_12 = np.random.uniform(0,1,nw)                                         # Limb darkening
    p0_13 = np.abs(np.random.normal(ebpar['Mratio'],0.001,nw))               # Mass ratio
    p0_14 = np.random.uniform(0,0.1,nw)                                       # Third Light
    p0_15 = np.random.normal(ebpar['Rot1'],0.001,nw)                         # Star 1 rotation
    p0_16 = np.random.uniform(0,1,nw)                                         # Fraction of spots eclipsed
    p0_17 = np.random.normal(0,0.001,nw)                                      # base spottedness
    p0_18 = np.random.normal(0,0.0001,nw)                                     # Sin amplitude
    p0_19 = np.random.normal(0,0.0001,nw)                                     # Cos amplitude
    p0_20 = np.random.normal(0,0.0001,nw)                                     # SinCos amplitude
    p0_21 = np.random.normal(0,0.0001,nw)                                     # Cos^2-Sin^2 amplitude
    p0_22 = np.random.uniform(ebpar['Rot1'],0.001,nw)                        # Star 2 rotation
    p0_23 = np.random.uniform(0,1,nw)                                         # Fraction of spots eclipsed
    p0_24 = np.random.normal(0,0.001,nw)                                      # base spottedness
    p0_25 = np.random.normal(0,0.001,nw)                                      # Sin amplitude
    p0_26 = np.random.normal(0,0.001,nw)                                      # Cos amplitude
    p0_27 = np.random.normal(0,0.001,nw)                                      # SinCos amplitude
    p0_28 = np.random.normal(0,0.001,nw)                                      # Cos^2-Sin^2 amplitude
    p0_29 = np.abs(np.random.normal(ebpar['ktot'],ebpar['ktot']*0.2,nw))    # Total radial velocity amp
    p0_30 = np.random.normal(ebpar['vsys'],5.0,nw)                           # System velocity


# L3 at 14 ... 14 and beyond + 1

    p0_init = np.array([p0_0,p0_1,p0_2,p0_3,p0_4,p0_5,p0_7])
    variables =["J","Rsum","Rratio","cosi","ecosw","esinw","t0"]

    if fitinfo['fit_period']:
        p0_init = np.append(p0_init,[p0_8],axis=0)
        variables.append("period")

# Start from here 10/22/15

    if fitinfo['fit_limb'] and fitinfo['claret']:
        sys.exit('Cannot fit for LD parameters and constrain them according to the other fit parameters!')
    if fitinfo['fit_limb']:
        limb0 = np.array([p0_9,p0_10,p0_11,p0_12])
        lvars = ["q1a", "q2a", "q1b", "q2b"]
#        limb0 = np.array([p0_9,p0_11])
#        lvars = ["u1a", "u1b"]
        p0_init = np.append(p0_init,limb0,axis=0)
        for var in lvars:
            variables.append(var)

    if fitinfo['fit_L3']:
        p0_init = np.append(p0_init,[p0_14],axis=0)
        variables.append('L3')

    if fitinfo['fit_ooe1']:
        p0_init = np.append(p0_init,[p0_16],axis=0)
        variables.append('spFrac1')
        for i in range(fitorder+1):
            p0_init = np.append(p0_init,[np.random.normal(0,0.05,nw)],axis=0)
            variables.append('c'+str(i)+'_1')
        for i in range(fitorder+1):
            p0_init = np.append(p0_init,[np.random.normal(0,0.05,nw)],axis=0)
            variables.append('c'+str(i)+'_2')

    if fitinfo['fit_ooe2']:
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

    if fitinfo['fit_rvs']:
        p0_init = np.append(p0_init,[p0_13],axis=0)
        variables.append('massratio')
        p0_init = np.append(p0_init,[p0_29],axis=0)
        variables.append('ktot')
        p0_init = np.append(p0_init,[p0_30],axis=0)
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
    sampler = emcee.EnsembleSampler(nw, ndim, lnprob, args=(data,ebpar,fitinfo))

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
    burn = np.append(burn,sampler.acceptance_fraction)
    np.savetxt(directory+'burnstats.txt',burn)

    # Reset sampler and run MCMC for reals
    print "getting pdfs for LD coefficients"
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
    stats = np.append(stats,sampler.acceptance_fraction)
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
            np.savetxt(directory+variables[i]+'chain'+thinst+'.txt',sampler.flatchain[0::thin,i])

    return sampler.lnprobability.flatten(),sampler.flatchain



def vec_to_params(x,ebpar):
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
        parm[eb.PAR_P] = ebpar["Period"]
        
    # T0
    try:
        parm[eb.PAR_T0] =  x[variables == 't0'][0]   # T0 (epoch of primary eclipse)
    except:
        parm[eb.PAR_T0] = ebpar['t01']-ebpar['bjd']

    # offset magnitude
    try:
        parm[eb.PAR_M0] =  x[variables == 'magoff'][0]  
    except:
        parm[eb.PAR_M0] = ebpar['mag0']

        
    # Limb darkening paramters for star 1
    try:
#        q1a = x[variables == 'q1a'][0]  
#        q2a = x[variables == 'q2a'][0]  
#        a1, b1 = qtou(q1a,q2a,limb=limb)
        parm[eb.PAR_LDLIN1] = x[variables == 'u1a']  # u1 star 1
        parm[eb.PAR_LDNON1] = 0  # u2 star 1
    except:
        parm[eb.PAR_LDLIN1] = ebpar["LDlin1"]   # u1 star 1
        parm[eb.PAR_LDNON1] = ebpar["LDnon1"]   # u2 star 1


    # Limb darkening paramters for star 2
    try:
#        q1b = x[variables == 'q1b'][0]  
#        q2b = x[variables == 'q2b'][0]  
#        a2, b2 = qtou(q1b,q2b)
        parm[eb.PAR_LDLIN2] = x[variables == 'u1b']  # u1 star 2
        parm[eb.PAR_LDNON2] = 0  # u2 star 2
    except:
        parm[eb.PAR_LDLIN2] = ebpar["LDlin2"]   # u1 star 2
        parm[eb.PAR_LDNON2] = ebpar["LDnon2"]   # u2 star 2


    # Mass ratio is used only for computing ellipsoidal variation and
    # light travel time.  Set to zero to disable ellipsoidal.
    try:
        parm[eb.PAR_Q]  = x[variables == 'massratio'][0]
        ktot  = x[variables == 'ktot'][0]
        vsys  = x[variables == 'vsys'][0]
    except:
        parm[eb.PAR_Q]  = ebpar['Mratio']
        ktot = ebpar['ktot']
        vsys = ebpar['vsys']

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
#    else:
#        ktot = 0.0


    if ebpar['gravdark']:
        parm[eb.PAR_GD1]    = ebpar['GD1']   # gravity darkening, std. value
        parm[eb.PAR_GD2]    = ebpar['GD2']   # gravity darkening, std. value

    if ebpar['reflection']:
        parm[eb.PAR_REFL1]  = ebpar['Ref1']  # albedo, std. value
        parm[eb.PAR_REFL2]  = ebpar['Ref2']  # albedo, std. value


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



def compute_eclipse(t,parm,integration=None,modelfac=11.0,fitrvs=False,tref=None,
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




def lnprob(x,data,ebpar,fitinfo):

    """
    ----------------------------------------------------------------------
    lnprob:
    -------
    Function to compute logarithmic probability of data given model. This
    function sets prior constaints explicitly and calls compute_trans to
    compare the data with the model. Only data within the smoothed transit
    curve is compared to model. 

    """

    parm,vder = vec_to_params(x,ebpar)
    
    vsys = x[-1]
    ktot = x[-2]

    variables = fitinfo['variables']
    
    if fitinfo['claret']:
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
        u1a,u2a = qtou(q1a,q2a,limb=ebpar['limb'])        
        u1b,u2b = qtou(q1b,q2b,limb=ebpar['limb'])
        parm[eb.PAR_LDLIN1] = u1a  # u1 star 1
        parm[eb.PAR_LDNON1] = u2a  # u2 star 1
        parm[eb.PAR_LDLIN2] = u1b  # u1 star 2
        parm[eb.PAR_LDNON2] = u2b  # u2 star 2
      
#        u1a = x[variables == 'u1a']
#        u2a = 0
#        u1b = x[variables == 'u1b']
#        u2b = 0
    else:
        q1a = 0.5 ; q2a = 0.5 ; q1b = 0.5 ; q2b = 0.5
#        u1a = 0 ; u2a = 0 ; u1b = 0 ; u2b = 0
        
    # Exclude conditions that give unphysical limb darkening parameters
    if q1b > 1 or q1b < 0 or q2b > 1 or q2b < 0 or np.isnan(q1b) or np.isnan(q2b):
        return -np.inf        
    if q1a > 1 or q1a < 0 or q2a > 1 or q2a < 0 or np.isnan(q1a) or np.isnan(q2a):
        return -np.inf        
#    if u1a < 0 or u1b < 0 or  u1a > 1 or u1b > 1 or np.isnan(u1a) or np.isnan(u1b):
#        return -np.inf        
    
    # Priors
    if fitinfo['fit_L3']:
        if parm[eb.PAR_L3] > 1 or parm[eb.PAR_L3] < 0:
            return -np.inf

    # Need to understand exactly what this parameter is!!
    if fitinfo['fit_ooe1']:
        if parm[eb.PAR_FSPOT1] < 0 or parm[eb.PAR_FSPOT1] > 1:
            return -np.inf
        coeff1 = []
        for i in range(fitorder+1):
            coeff1 = np.append(coeff1,x[variables == 'c'+str(i)+'_1'])
        
### Compute eclipse model for given input parameters ###
    massratio = parm[eb.PAR_Q]
    if massratio < 0 or massratio > 1:
        return -np.inf

    if not fitinfo['fit_ellipsoidal']:
        parm[eb.PAR_Q] = 0.0

    # Primary eclipse
    t0 = parm[eb.PAR_T0]
    if np.abs(t0) > 1800:
        return -np.inf
    
    period = parm[eb.PAR_P]
    time   = data['light'][0,:]-ebpar['bjd']
    flux   = data['light'][1,:]
    eflux  = data['light'][2,:]

    sm  = compute_eclipse(time,parm,integration=ebpar['integration'],fitrvs=False,tref=t0,period=period)

    # Log Likelihood Vector
    lfi = -1.0*(sm - flux)**2/(2.0*eflux**2)

    # Log likelihood
    lf1 = np.sum(lfi)

#    # Secondary eclipse
#    tsec = fitdict['tsec']
#    xsec = fitdict['xsec']/norm
#    esec = fitdict['esec']/norm
#
#    if fitsp1:
#        coeff2 = []
#        for i in range(fitorder+1):
#            coeff2 = np.append(coeff2,x[variables == 'c'+str(i)+'_2'])
#
#    sm2  = compute_eclipse(tsec,parm,fitrvs=False,tref=t0,period=period,ooe1fit=coeff2)
#    
#    # Log Likelihood Vector
#    lfi2 = -1.0*(sm2 - xsec)**2/(2.0*esec**2)
#    
#    # Log likelihood
#    lf2 = np.sum(lfi2)
#
    lf = lf1 #+lf2

    # need this for the RVs!
    parm[eb.PAR_Q] = massratio

    rvdata1 = data['rv1']
    rvdata2 = data['rv2']

    if fitinfo['fit_rvs']:
        if (vsys > max(np.max(rvdata1[1,:]),np.max(rvdata2[1,:]))) or \
           (vsys < min(np.min(rvdata1[1,:]),np.min(rvdata2[1,:]))): 
            return -np.inf
        rvmodel1 = compute_eclipse(rvdata1[0,:]-ebpar['bjd'],parm,fitrvs=True)
        k2 = ktot/(1+massratio)
        k1 = k2*massratio
        rv1 = rvmodel1*k1 + vsys
        rvmodel2 = compute_eclipse(rvdata2[0,:]-ebpar['bjd'],parm,fitrvs=True)
        rv2 = -1.0*rvmodel2*k2 + vsys
        lfrv1 = -np.sum((rv1 - rvdata1[1,:])**2/(2.0*rvdata1[2,:]))
        lfrv2 = -np.sum((rv2 - rvdata2[1,:])**2/(2.0*rvdata2[2,:]))
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
#        chi2 = -1*lf2
#        plt.annotate(r'$\chi^2$ = %.0f' % chi2, [0.1,0.1],horizontalalignment='left',
#                     xycoords='axes fraction',fontsize='large')


        plt.subplot(2, 1, 2)
        phi1 = foldtime(rvdata1[0,:]-ebpar['bjd'],t0=t0,period=period)/period
        plt.plot(phi1,rvdata1[1,:],'ko')
        plt.plot(phi1,rv1,'kx')
        tcomp = np.linspace(-0.5,0.5,10000)*period+t0
        rvmodel1 = compute_eclipse(tcomp,parm,fitrvs=True)
        k2 = ktot/(1+massratio)
        k1 = k2*massratio
        rvcomp1 = rvmodel1*k1 + vsys
        plt.plot(np.linspace(-0.5,0.5,10000),rvcomp1,'k--')
        plt.annotate(r'$\chi^2$ = %.0f' % -lfrv, [0.05,0.85],horizontalalignment='left',
                     xycoords='axes fraction',fontsize='large')
  
        phi2 = foldtime(rvdata2[0,:]-ebpar['bjd'],t0=t0,period=period)/period
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



def varnameconv(variables):

    varmatch = np.array(["J","Rsum","Rratio","cosi",
                         "ecosw","esinw",
                         "magoff","t0","period",
                         "q1a", "q2a", "q1b", "q2b","u1a","u1b",
                         "massratio","L3",
                         "Rot1","spFrac1","spBase1","spSin1","spCos1","spSinCos1","spSqSinCos1",
                         "Rot2","spFrac2","spBase2","spSin2","spCos2","spSinCos2","spSqSinCos2",
                         "c0_1","c1_1","c2_1","c3_1","c4_1","c5_1",
                         "c0_2","c1_2","c2_2","c3_2","c4_2","c5_2",
                         "ktot","vsys"])

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


    file1 = 'Claret_cool.dat'
    file2 = 'Claret_hot.dat'

    if network == None or network == 'bellerophon':
	path = '/home/administrator/python/fit/'
    elif network == 'doug':
        path = '/home/douglas/Astronomy/Resources/'
    elif network == 'astro':
        path = '/home/jswift/Mdwarfs/'
    elif network == 'swift':
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

    if passfuncs:
        return ldc1func,ldc2func,ldc3func,ldc4func
    else:
        if limb == 'quad':
            return aval, bval

        if limb == 'sqrt':
            return cval, dval

        if limb == 'nlin':
            return aval, bval, cval, dval


