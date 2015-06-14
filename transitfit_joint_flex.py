import numpy as np
import matplotlib.pyplot as plt
import emcee
import occultquad as oq
import transit as trans
import sys
import math
import robust as rb
import pdb
import time
import scipy as sp
from scipy.stats.kde import gaussian_kde
import os    
import constants as c
import glob

#----------------------------------------------------------------------
# foldtime
#     take a time series and fold it on a period and ephemeris
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
    n, _ = np.histogram(x, bins=nbins)
    sy, _ = np.histogram(x, bins=nbins, weights=y)
    sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)

    binvals = (_[1:] + _[:-1])/2
    yvals = mean
    yerr = std/np.sqrt(len(std))
    
    return binvals,yvals,yerr



#----------------------------------------------------------------------
# numplanets:
#    compute number of planets in koi system
#    does not depend on any global variables
#----------------------------------------------------------------------

def numplanet(koiin,network=None):
    
# Import modules
    print " "
    print "... importing necessary packages"
    
    info1 = np.loadtxt(path+'Refine/'+str(koiin)+tag+'.out',delimiter=',',ndmin=2)
    npl = info1.shape[0]
    names = []
    for i in xrange(0,npl):
        names.append("%.2f" % (info1[i,0]))

    return npl,names



#------------------------------------------------------------------------
# get_limb_curves:
#------------------------------------------------------------------------

def get_limb_curve(ldc,limb='quad'):

    gamma = np.linspace(0,np.pi/2.0,1000,endpoint=True)
    theta = gamma*180.0/np.pi
    mu = np.cos(gamma)
 
    if limb == 'nlin':
        c1 = ldc[0]
        c2 = ldc[1]
        c3 = ldc[2]
        c4 = ldc[3]
        Imu = 1.0 - c1*(1.0 - mu**0.5) - c2*(1.0 - mu) - \
            c3*(1.0 - mu**1.5) - c4*(1.0 - mu**2.0)
    elif limb == 'quad':
        c1 = ldc[0]
        c2 = ldc[1]
        Imu = 1.0 - c1*(1.0 - mu) - c2*(1.0 - mu)**2.0
    elif limb == 'sqrt':
        c1 = ldc[0]
        c2 = ldc[1]
        Imu = 1.0 - c1*(1.0 - mu) - c2*(1.0 - mu**0.5)
    else: pass

    return theta, Imu


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


#------------------------------------------------------------------------
# compute_trans
#------------------------------------------------------------------------

# Compute transit curve for given transit parameters
def compute_trans(rprs,impact,t0,period,rhostar,ldc,plot=0):
    from scipy.ndimage.filters import uniform_filter, uniform_filter1d

    if limb == 'nlin':
        a1 = ldc[0]
        a2 = ldc[1]
        a3 = ldc[2]
        a4 = ldc[3]

    else:        
        q1in = ldc[0]
        q2in = ldc[1]
        u1, u2 = qtou(q1in,q2in,limb=limb)

        
 # Create transit
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

 
# Convert rhostar, period, and stellar parameters to transit duration
    rs_a = (3.0 * np.pi/((period*24.0*3600.0)**2 * c.G * rhostar))**(1.0/3.0)
    aors = 1.0/rs_a
    
#!!! continue from here !!!

    arg1 = 1.0 - (impact*rs_a)**2
    arg = max( ((1 + rl)**2 - impact**2), 0.0)
    if arg == 0.0 or arg1 <= 0:
        duration = 0.0
        sztrans = 10
        modelt = sp.linspace(-1*period/2,period/2,sztrans*2)
        smoothmodel = np.ones(sztrans*2)
        return modelt+t0,smoothmodel

    sini = np.sqrt(arg1)
    asval = rs_a * np.sqrt(arg) / sini
    if asval > 1.0:
        print "asval > 1!"
        asval = 1.0

    # duration  = Ttot
    duration = period / np.pi * np.arcsin(asval)
    # Number of integrations in 1/2 full duration
    nint = duration*24.*3600./(2.0*integration)
    # factor beyond full duration that an integer number of integrations extends
    ifac = np.ceil(nint)/nint

    # Number of transit samples = integer number of integrations * modelfac
    # extra 1 to account for one step = 2 values
    sztrans = np.ceil(nint)*modelfac + 1

    # Compute transit to fourth contact point
    xtot  = np.sqrt((1+rl)**2 - impact**2)
    btmp  = np.linspace(0,xtot*ifac,sztrans)

    # impact parameter from min to 1
    b0 = np.sqrt(impact**2 + btmp**2)

    # Time vector (linear with btmp = circular orbit approximation)
    dbtime = integration/(modelfac*24.0*3600.)
    b0time = np.arange(len(btmp))*dbtime
    btime = np.arange(len(btmp))*dbtime

    if limb == 'quad':
        model = np.array(oq.occultquad(b0,u1,u2,rl))[0]
    if limb == 'sqrt':
        # a1 = u2 and a2 = u1
        model = np.array(trans.occultnonlin(b0,rl,np.array([u1,u2,0.0,0.0])))
    if limb == 'nlin':
        model = np.array(trans.occultnonlin(b0,rl,np.array([a1,a2,a3,a4])))


# Append additional ones 2 integration times outside of transit with same sample 
# rate as the rest of the transit (for smoothing)
    nadd = 2*modelfac
    addtime = np.linspace(dbtime,nadd*dbtime,nadd)+np.max(btime)  
    btime = np.append(btime,addtime)
    model = np.append(model,np.ones(nadd))

# Append 1's far from transit (1/2 period away for linear interpolation
    btime = np.append(btime,period/2)
    model = np.append(model,1.0)

# Final model time and flux
    tmodel = np.append(-1*btime[::-1][:-1],btime)
    fmodel  = np.append(model[::-1][:-1],model)

# Smooth to integration time
    # just a quick check to make sure all things add up 
    smoothlength = integration/(dbtime*24.0*3600.0)
    sl = round(smoothlength)
    if sl != modelfac:
        print "smoothing length not equal to sampling rate!!!"
        pdb.set_trace()

    smoothmodel = uniform_filter1d(fmodel, np.int(sl))

    return tmodel+t0,smoothmodel



def lnprob(x):

    rhostar = x[0]
    qf1     = x[1]
    qf2     = x[2]
    ldc = [qf1,qf2]

    lf = []

    for i in np.arange(nplanets):
        rprs = x[i*4+3]
        impact = x[i*4+4]
        period = x[i*4+5]
        ephem = x[i*4+6]
        
        # impose priors here
        if rhostar < 0 or rhostar > 200:
            return -np.inf

        if qf1 < 0 or qf1 > 1:
            return -np.inf

        if qf2 < 0 or qf2 > 1:
            return -np.inf

        if impact < 0 or impact > (1+rprs):
            return -np.inf

        if rprs < 0 or rprs > 1:
            return -np.inf

        # approximately 30 minutes out of 1 month
        if np.abs(period-periods[i]) >= 0.0007:
            return -np.inf
            
        if np.abs(ephem) >= duration0s[i]:
            return -np.inf


        modelt,smoothmodel = compute_trans(rprs,impact,ephem,period,rhostar,ldc)

        # compare data to model in only transit region
        tlo = np.min(modelt[1:-1])
        thi = np.max(modelt[1:-1])
        
        ttm = foldtime(times[i],period=period,t0=t0s[i])
        ins  = np.zeros(len(ttm),dtype=bool)
        ininds = np.where((ttm >= tlo) & (ttm <= thi))
        ins[ininds] = True

        cfunc = sp.interpolate.interp1d(modelt,smoothmodel,kind='linear')
        mfit = np.ones(len(ttm),dtype=float)
        mfit[ins] = cfunc(ttm[ins])

        #    lfi = -np.log((2*np.pi*e_flux**2)**0.5) -1*(mfit - ffits)**2/(2.0*e_flux**2)

        lfi = np.sum(-1*(mfit - fluxes[i])**2/(2.0*errors[i]**2))
        if lfi == np.nan:
            print "NaN for lnprob!"
            pdb.set_trace()

        lf = np.append(lf,lfi)
    
        if jeffries == True:
            lf = np.append(lf,-2*np.log(rprs))

    return np.sum(lf)



# rhostar = p0[0]
# qf1     = p0[1]
# qf2     = p0[2]
# rprs1   = p0[3]
# impact1 = p0[4]
# period1 = p0[5]
# ephem1  = p0[6]
# and so on for all planets...


def get_koi_info(koiin,sc=True,network=None,clip=True,limbmodel='quad',
                 rprior=True,bin=False,debug=False,keep=[1,2,3]):

    import get_limb_coeff as gl

# Global variables
    global short
    global binfac
    global tag
    global btag
    global ctag
    global ltag
    global koi
    global name
    global u1o
    global u2o
    global u10s
    global u20s
    global q1o
    global q2o
    global q10s
    global q20s
    global ndim
    global rhostar0
    global periods
    global t0s
    global e_periods
    global rprs0s
    global duration0s
    global impact0s
    global times
    global fluxes
    global errors
    global nplanets
    global integration
    global binned
    global limb
    global db
    global path
    global doplanets
    global jtag
    global jeffries, rtag

    koi = koiin

    bjd = 2454833.0

    jtag = ''
    for val in keep:
        jtag += str(val)

    binfac = bin
    short = sc
    doplanets = keep

    if debug:
        db = True
    else:
        db = False


    if short:
        tag = '_short'
        integration = 54.2
    else:
        tag = '_long'
        integration = 1626.0
        

    if binfac:
        print "binning data"
        btag = '_bin'
        binned = 1
    else:
        btag = ''
        binned = 0

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

    limb = limbmodel
    if limb == 'quad':
        ltag = '_quad'
    if limb == 'sqrt':
        ltag = '_sqrt'
    if limb == 'nlin':
        ltag = '_nlin'

# Star properties
    koi = koiin
    name = str(koi)

# Setup path and info specific to KOI
    if network == 'astro':
        path = '/scr2/jswift/Mdwarfs/outdata/'+str(koi)+'/'
    if network == 'koi':
        path = '/Users/jonswift/Astronomy/Exoplanets/KOI'+str(koi)+'/lc_fit/outdata/'+str(koi)+'/'
    if network == 'gps':
        path = '/home/jswift/Mdwarfs/outdata/'+str(koi)+'/'
    if network == None:        
        path = '/Users/jonswift/Astronomy/Exoplanets/Mdwarfs/All/outdata/'+str(koi)+'/'

    npl,names = numplanet(koi,network=network)

    if npl < 2:
        sys.exit("Number of planets for KOI-"+name+" = "+str(npl))

    nplanets = len(keep)

# MCMC parameters
    ndim = 3+4*nplanets

    print "... getting star properties"
    info2 = np.loadtxt(path+'Fits/'+names[0]+tag+'_fixlimb_fitparams.dat',dtype='string')
    Mstar = np.float(info2[1])*c.Msun
    eMstar = np.float(info2[2])*c.Msun
    Rstar = np.float(info2[3])*c.Rsun
    eRstar = np.float(info2[4])*c.Rsun
    Tstar = np.float(info2[5])
    eTstar = np.float(info2[6])
    loggstar = np.log10( c.G * Mstar / Rstar**2. )


    print " "
    print "... getting quadratic limb darkening coefficients"
    u1v,u2v = gl.get_limb_coeff(Tstar,loggstar,limb=limb)
#print  "Claret et al. limb darkening coefficients:"
    if limb == 'quad':
        u1o = u1v
        u2o = u2v
    if limb == 'sqrt':
        u1o = u1v
        u2o = u2v

#    aout = '     u1 = {0:.4f}'
#    print aout.format(u1o)
#    bout = '     u2 = {0:.4f}'
#    print bout.format(u2o)

    q1o = (u1o + u2o)**2
    q2o = u1o/(2*(u1o+u2o))
 


# Intial guesses for MCMC 
    dpath = path+'MCMC/'

# First planet:
    pst = str(keep[0])
    file = name+'.0'+pst+tag+ctag+rtag+ltag+'fit_fitparams.txt'
    match = len(glob.glob(dpath+file))
    if match == 0:
        file = name+'.0'+pst+tag+ctag+ltag+'fit_fitparams.txt'
        match2 = len(glob.glob(dpath+file))
        if match2 == 0:
            sys.exit(file+" not found !")
    info = np.loadtxt(dpath+file)
    periods = info[17]
    t0s     = info[13]
# size of occulting body (in units of the primary)
    rprs0s = info[1]
    impact0s = info[9]
# duration in hours
    duration0s = midtotot(info[5],rprs0s,impact0s)[0]
#    q10s = info[16]
#    q20s = info[19]
    q10s = q1o
    q20s = q2o
    u10s = 2.*np.sqrt(q10s)*q20s
    u20s = np.sqrt(q10s)*(1-2*q20s)

    # The rest of the planets...
    for planet in np.arange(1,nplanets):
        pst = str(keep[planet])
        file = name+'.0'+pst+tag+ctag+rtag+ltag+'fit_fitparams.txt'
        match = len(glob.glob(dpath+file))
        if match == 0:
            file = name+'.0'+pst+tag+ctag+ltag+'fit_fitparams.txt'
            match2 = len(glob.glob(dpath+file))
            if match2 == 0:
                sys.exit(file+" not found !")
        info = np.loadtxt(dpath+file)
        periods = np.append(periods,info[17])
        t0s = np.append(t0s,info[13])
        # size of occulting body (in units of the primary)
        rprs0s = np.append(rprs0s, info[1])
        # duration in hours
        impact0s = np.append(impact0s,info[9])
        duration0s = np.append(duration0s,midtotot(info[5],info[1],info[9])[0] )


# Get the data...
    # Planet 1    
    pst = str(keep[0])
    dpath = path+'Refine/'  
    filename = name+'.0'+pst+tag+ctag+'.dat'
    data = np.loadtxt(dpath+filename)
    t = data[:,0]
    flux = data[:,2]
    e_flux = data[:,3]
    times = {0:t}
    fluxes = {0:flux}
    errors = {0:e_flux}

    # other planets
    for planet in np.arange(1,nplanets): 
        pst = str(keep[planet])
        filename = name+'.0'+pst+tag+ctag+'.dat'
        data = np.loadtxt(dpath+filename)
        t = data[:,0]
        flux = data[:,2]
        e_flux = data[:,3]
        ttemp = {planet:t}
        ftemp = {planet:flux}
        etemp = {planet:e_flux}
        times.update(ttemp)
        fluxes.update(ftemp)
        errors.update(etemp)
        ttemp = []
        ftemp = []
        etemp = []
        
    return



#----------------------------------------------------------------------
# do_mcmc
#----------------------------------------------------------------------

def do_mcmc(nwalkers=200,burnsteps=500,mcmcsteps=500):

    global nw
    global mcs
    global bs

    directory = path+'MCMC/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    nw  = nwalkers
    mcs = mcmcsteps
    bs  = burnsteps

    print " "
    print "... initializing emcee sampler with "+str(nw)+" walkers"
    tstart = time.time()
    sampler = emcee.EnsembleSampler(nw, ndim, lnprob)
    
# Create overdispersed starting values
    # stellar density
    p0 = np.random.uniform(10,80,nw)
    # Kipping limb darkening parameters
    for qs in np.arange(2):
        pnext = np.random.uniform(0,1,nw)
        p0 = np.append(p0,pnext).reshape(qs+2,nw)

    twomin = 2./(24.*60.)
    onesec = 1./(24.*60.*60.)

    # Add planets...
    for pl in np.arange(nplanets):
        pnext = rprs0s[pl] + np.random.normal(0,0.3*max(rprs0s[pl],0.001),nw)
        p0 = np.append(p0,pnext).reshape(pl*4+4,nw)
        pnext = np.random.uniform(0,0.99,nw)
        p0 = np.append(p0,pnext).reshape(pl*4+5,nw)
        pnext = periods[pl] + np.random.uniform(-0.5*onesec,0.5*onesec,nw)
        p0 = np.append(p0,pnext).reshape(pl*4+6,nw)
        pnext = np.random.uniform(-0.5*twomin, 0.5*twomin, nw)
        p0 = np.append(p0,pnext).reshape(pl*4+7,nw)


    p0 = p0.T

    variables =["rhostar","q1","q2"]
    for planet in doplanets:
        variables.append("Rp"+str(planet))
        variables.append("b"+str(planet))
        variables.append("P"+str(planet))
        variables.append("t"+str(planet))
    
    print "... running burn-in with "+str(bs)+" steps"
    pos, prob, state = sampler.run_mcmc(p0, bs)
    done_in(tstart)

    # Calculate G-R scale factor for each variable
    Rs = GR_test(sampler.flatchain,variables=variables)

    for var in np.arange(ndim):
        acout = "Autocorrelation time for "+variables[var]+" = {0:0.3f}"
        print acout.format(sampler.acor[var])

    afout = "Mean acceptance fraction: {0:.3f}"
    print afout.format(np.mean(sampler.acceptance_fraction))
    print ""
 
# Save burn in stats
    print "Writing out burn-in MCMC stats"
    burn = np.append(Rs,sampler.acor)
    burn = np.append(burn,sampler.acceptance_fraction)
    np.savetxt(path+'MCMC/'+name+tag+ctag+rtag+ltag+'_joint'+jtag+'_burnstats.txt',burn)

    print "... resetting sampler and running MCMC with "+str(mcs)+" steps"
    
    sampler.reset()
    posf, probf, statef = sampler.run_mcmc(pos, mcs)
    done_in(tstart)

# Calculate G-R scale factor for each variable
    print ""
    Rf = GR_test(sampler.flatchain,variables=variables)
    for var in np.arange(ndim):
        acout = "Autocorrelation time for "+variables[var]+" = {0:0.3f}"
        print acout.format(sampler.acor[var])

    afout = "Mean acceptance fraction: {0:.3f}"
    print afout.format(np.mean(sampler.acceptance_fraction))
    print ""

 # Save final stats
    print "Writing out final MCMC stats"
    stats = np.append(Rf,sampler.acor)
    stats = np.append(stats,sampler.acceptance_fraction)
    np.savetxt(path+'MCMC/'+name+tag+ctag+rtag+ltag+'_joint'+jtag+'_finalstats.txt',stats)


    print "Writing MCMC chains to disk"
    # Stellar density
    rhodist = sampler.flatchain[:,0]    
    np.savetxt(path+'MCMC/'+name+tag+ctag+rtag+ltag+btag+'_rhochain_joint'+jtag+'.txt',rhodist)
    # Kipping qs
    q1dist = sampler.flatchain[:,1]
    np.savetxt(path+'MCMC/'+name+tag+ctag+rtag+ltag+btag+'_q1chain_joint'+jtag+'.txt',q1dist)
    q2dist = sampler.flatchain[:,2]
    np.savetxt(path+'MCMC/'+name+tag+ctag+rtag+ltag+btag+'_q2chain_joint'+jtag+'.txt',q2dist)
    # Planet chains
    for pl in np.arange(nplanets):
        rprsdist = sampler.flatchain[:,pl*4+3]
        plnum = str(doplanets[pl])
        np.savetxt(path+'MCMC/'+name+tag+ctag+rtag+ltag+btag+'_r'+plnum+'chain_joint'+jtag+'.txt',rprsdist)
        bdist = sampler.flatchain[:,pl*4+4]
        np.savetxt(path+'MCMC/'+name+tag+ctag+rtag+ltag+btag+'_b'+plnum+'chain_joint'+jtag+'.txt',bdist)
        pdist = sampler.flatchain[:,pl*4+5]
        np.savetxt(path+'MCMC/'+name+tag+ctag+rtag+ltag+btag+'_p'+plnum+'chain_joint'+jtag+'.txt',pdist)
        tdist = sampler.flatchain[:,pl*4+6]
        np.savetxt(path+'MCMC/'+name+tag+ctag+rtag+ltag+btag+'_t'+plnum+'chain_joint'+jtag+'.txt',tdist)

    lp = sampler.lnprobability.flatten()
    np.savetxt(path+'MCMC/'+name+tag+ctag+rtag+ltag+btag+'_lnprob_joint'+jtag+'.txt',lp)
    
    return


#----------------------------------------------------------------------
# GR_test:
#    Compute the Gelman-Rubin scale factor for each variable given input
#    flat chains
#----------------------------------------------------------------------


def GR_test(chains,variables=False):
    import numpy as np

    nels = bs/2.
    Rs = np.zeros(ndim)
    for var in np.arange(ndim):
        distvec = chains[:,var].reshape(nw,bs)
        
        mean_total = np.mean(distvec[:,bs/2:])
        means = np.zeros(nw)
        vars = np.zeros(nw)
        for i in np.arange(nw):
            means[i] = np.mean(distvec[i,bs/2:])
            vars[i] = 1./(nels-1) * np.sum((distvec[i,bs/2.:] - means[i])**2)

        B = nels/(nw-1) * np.sum((means - mean_total)**2)
        W = 1./nw * np.sum(vars)

        V_hat = (nels-1.0)/nels * W + B/nels

        R =  np.sqrt(V_hat/W)

        if len(variables) == ndim:
            out = "Gelman Rubin scale factor for "+variables[var]+" = {0:0.3f}"
            print out.format(R)
            
        Rs[var] = R

    return Rs


def get_chains():

    tstart = time.time()
    print "Importing MCMC chains"
    ddir = path+'MCMC/'
    pdir = ddir

    # Stellar density
    chains = np.loadtxt(ddir+name+tag+ctag+rtag+ltag+btag+'_rhochain_joint'+jtag+'.txt')
    mcsteps = len(chains)

    # Kipping qs
    for qs in np.arange(2):
        cnext = np.loadtxt(ddir+name+tag+ctag+rtag+ltag+btag+'_q'+str(qs+1)+'chain_joint'+jtag+'.txt')
        chains = np.append(chains,cnext).reshape(qs+2,mcsteps)
  
    for pl in np.arange(nplanets):
        plnum = str(doplanets[pl])
        cnext = np.loadtxt(ddir+name+tag+ctag+rtag+ltag+btag+'_r'+plnum+'chain_joint'+jtag+'.txt')
        chains = np.append(chains,cnext).reshape(pl*4+4,mcsteps)
        cnext = np.loadtxt(ddir+name+tag+ctag+rtag+ltag+btag+'_b'+plnum+'chain_joint'+jtag+'.txt')
        chains = np.append(chains,cnext).reshape(pl*4+5,mcsteps)        
        cnext = (np.loadtxt(ddir+name+tag+ctag+rtag+ltag+btag+'_p'+plnum+'chain_joint'+jtag+'.txt')-\
                 periods[pl])*24.*3600.
        chains = np.append(chains,cnext).reshape(pl*4+6,mcsteps)        
        cnext = (np.loadtxt(ddir+name+tag+ctag+rtag+ltag+btag+'_t'+plnum+'chain_joint'+jtag+'.txt'))*24*60*60
        chains = np.append(chains,cnext).reshape(pl*4+7,mcsteps)        

    lnp = np.loadtxt(ddir+name+tag+ctag+rtag+ltag+btag+'_lnprob_joint'+jtag+'.txt')
    chains = np.append(chains,cnext).reshape(nplanets*4+4,mcsteps)
    done_in(tstart)

    # Get maximum likelihood values
    maxlike = np.max(lnp)
    imax = np.array([i for i, j in enumerate(lnp) if j == maxlike])
    if imax.size > 1:
        imax = imax[0]

    vals = chains[:,imax]

    q1val = vals[1]
    q2val = vals[2]
    u1val =  2*np.sqrt(q1val)*q2val
    u2val =  np.sqrt(q1val)*(1-2*q2val)


    return chains,vals



def plot_star(chains=False,write=False,frac=0.001):
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
    import robust as rb
    import sys
    import math
    from scipy.ndimage.filters import uniform_filter
    from scipy.stats.kde import gaussian_kde
    from mpl_toolkits.mplot3d import Axes3D
    import pdb
    import time

    # Astronomical Constants
    import constants as c

    ddir = path+'MCMC/'

    if chains:
        rhodist = chains[0,:]
        q1dist  = chains[1,:]
        q2dist  = chains[2,:]
        lnp     = chains[-1,:]
    else:
        rhodist = np.loadtxt(ddir+name+tag+ctag+rtag+ltag+btag+'_rhochain_joint'+jtag+'.txt')
        q1dist  = np.loadtxt(ddir+name+tag+ctag+rtag+ltag+btag+'_q1chain_joint'+jtag+'.txt')
        q2dist  = np.loadtxt(ddir+name+tag+ctag+rtag+ltag+btag+'_q2chain_joint'+jtag+'.txt')
        lnp     = np.loadtxt(ddir+name+tag+ctag+rtag+ltag+btag+'_lnprob_joint'+jtag+'.txt')

    # Get maximum likelihood values
    maxlike = np.max(lnp)
    imax = np.array([i for i, j in enumerate(lnp) if j == maxlike])
    if imax.size > 1:
        imax = imax[0]
    rhomax = np.float(rhodist[imax])
    q1max = np.float(q1dist[imax])
    q2max = np.float(q2dist[imax])

    q1 = (u1o + u2o)**2
    q2 = u1o/(2*(u1o+u2o))

    bindiv = 10.

    nsamp = len(rhodist)

#----------------------------------------------------------------------
# Plot stellar parameters

# Stellar density
    print ''
    print 'Best fit parameters for '+name
    if write:
        plt.figure(1,figsize=(8.5,11),dpi=300)
    else:
        plt.figure(1)
        plt.ion()
    rhos = np.linspace(np.min(rhodist)*0.5,np.max(rhodist)*1.5,1000)
    rho_kde = gaussian_kde(rhodist)
    rhopdf = rho_kde(rhos)
    rhodist_c = np.cumsum(rhopdf)/np.sum(rhopdf)
    rhofunc = sp.interpolate.interp1d(rhodist_c,rhos,kind='linear')
    rholo = np.float(rhofunc(math.erfc(1./np.sqrt(2))))
    rhohi = np.float(rhofunc(math.erf(1./np.sqrt(2))))
    nbins = np.ceil((np.max(rhodist)-np.min(rhodist)) / ((rhohi-rholo)/bindiv))

    rhoout = 'Stellar density = {0:.4f} + {1:.4f} - {2:.4f} g/cc'
    print rhoout.format(rhomax,rhohi-rhomax,rhomax-rholo)
    
    ax1 = plt.subplot(3,1,1)
    
    ax1.hist(rhodist,bins=nbins,normed=True)
#    ax1.plot(rhos,rhopdf,color='c')
#    plt.axvline(x=rhomax,color='r')
#    plt.axvline(x=rhostar0,color='r',linestyle='--')
    xmin = np.sort(rhodist)[np.round(frac*nsamp)]
    xmax = np.sort(rhodist)[np.round((1-frac)*nsamp)]
    ax1.set_xlim([xmin, xmax])
    ax1.set_xlabel(r'$\rho$'+'$_{*}$ (g cm$^{-3}$)')
    ax1.set_ylabel(r'$dP/d\rho_{*}$')
    ax1.set_title('Stellar Parameter Distributions for KOI-'+name)

# Limb darkening
    q1s = np.linspace(np.min(q1dist)*0.5,np.max(q1dist)*1.5,1000)
    q1_kde = gaussian_kde(q1dist)
    q1pdf = q1_kde(q1s)
    q1dist_c = np.cumsum(q1pdf)/np.sum(q1pdf)
    q1func = sp.interpolate.interp1d(q1dist_c,q1s,kind='linear')
    q1lo = np.float(q1func(math.erfc(1./np.sqrt(2))))
    q1hi = np.float(q1func(math.erf(1./np.sqrt(2))))
    nbins = np.ceil((np.max(q1dist)-np.min(q1dist)) / ((q1hi-q1lo)/bindiv))
    
    q1out = 'Q1 Limb Darkenining Parameter  = {0:.4f} + {1:.4f} - {2:.4f}'
    print q1out.format(q1max,q1hi-q1max,q1max-q1lo)
    ax2 = plt.subplot(3,1,2)
    ax2.hist(q1dist,bins=nbins,normed=True)
#    ax2.plot(q1s,q1pdf,color='c')
#    plt.axvline(x=q1max,color='r')
#    plt.axvline(x=q1,color='r',linestyle='--')
    ax2.set_xlim([0,1])
    ax2.set_xlabel(r'$q_1$ (Limb darkening)')
    ax2.set_ylabel(r'$dP/dq_1$')


    q2s = np.linspace(np.min(q2dist)*0.5,np.max(q2dist)*1.5,1000)
    q2_kde = gaussian_kde(q2dist)
    q2pdf = q2_kde(q2s)
    q2dist_c = np.cumsum(q2pdf)/np.sum(q2pdf)
    q2func = sp.interpolate.interp1d(q2dist_c,q2s,kind='linear')
    q2lo = np.float(q2func(math.erfc(1./np.sqrt(2))))
    q2hi = np.float(q2func(math.erf(1./np.sqrt(2))))
    nbins = np.ceil((np.max(q2dist)-np.min(q2dist)) / ((q2hi-q2lo)/bindiv))
    
    q2out = 'Q2 Limb Darkenining Parameter  = {0:.4f} + {1:.4f} - {2:.4f}'
    print q2out.format(q2max,q2hi-q2max,q2max-q2lo)
    ax3 = plt.subplot(3,1,3)
    ax3.hist(q2dist,bins=nbins,normed=True)
#    ax3.plot(q2s,q2pdf,color='c')
#    plt.axvline(x=q2max,color='r')
#    plt.axvline(x=q2,color='r',linestyle='--')
    ax3.set_xlim(xmin=0,xmax=1)
    ax3.set_xlabel(r'$q_2$ (Limb darkening)')
    ax3.set_ylabel(r'$dP/dq_2$')
    
    if write:
        plt.savefig(ddir+name+tag+ctag+rtag+ltag+btag+'_starparams_joint'+jtag+'.png')
        print '... created '+ddir+name+tag+ctag+rtag+ltag+btag+'_starparams_joint'+jtag+'.png'
    else:
        plt.show()

    plt.clf()


    if write:
        plt.figure(2,figsize=(11,8.5),dpi=300)
    else:
        plt.figure(2)

    u1max =  2*np.sqrt(q1max)*q2max
    u2max =  np.sqrt(q1max)*(1-2*q2max)

    theta0,Imu = get_limb_curve([u1o,u2o],limb=limb)
    thetaf,Imuf = get_limb_curve([u1max,u2max],limb=limb)


    if limb == 'quad':
        title = ' Quadratic '
    if limb == 'sqrt':
        title = ' Square-Root '

    label1 = r"Best fit $u_1$, $u_2$ = {0:0.2f}, {1:0.2f}"
    plt.plot(thetaf,Imuf,label=label1.format(u1max,u2max))
    label2 = r"Claret et al. $u_1$, $u_2$ = {0:0.2f}, {1:0.2f}"
    plt.plot(theta0,Imu,'--',label=label2.format(u1o,u2o))
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel(r"$\theta$ (degrees)",fontsize=18)
    plt.ylabel(r"$I(\theta)/I(0)$",fontsize=18)
#    plt.annotate(r'$u_1$ = %.2f (%.2f)' % (np.float(u1max),np.float(u1o)), \
#                     [0.7,0.85],horizontalalignment='left',\
#                     xycoords='figure fraction',fontsize='large')
#    plt.annotate(r'$u_2$ = %.2f (%.2f)' % (np.float(u2max),np.float(u2o)), \
#                     [0.7,0.82],horizontalalignment='left', \
#                     xycoords='figure fraction',fontsize='large')
    plt.title(name+title+'Limb Darkening')
    plt.legend(loc=3)
    plt.axis([0.0,90.0,0.0,1.1])
#    plt.ylim=([0,1.2])
#    plt.xlim=([0,90])
    
    if write:        
        plt.savefig(ddir+name+tag+ctag+rtag+ltag+btag+'_limbdark_joint'+jtag+'.png')
        print '... created '+ddir+name+tag+ctag+rtag+ltag+btag+'_limbdark_joint'+jtag+'.png'
    else:
        plt.show()

    plt.clf()


    return


def plot_planet(planet=1,chains=False,write=False,frac=0.001,thin=False):
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
    import robust as rb
    import sys
    import math
    from scipy.ndimage.filters import uniform_filter
    from scipy.stats.kde import gaussian_kde
    from mpl_toolkits.mplot3d import Axes3D
    import pdb
    import time
    
    bindiv = 10.

    ddir = path+'MCMC/'
    pdir = ddir

    pind, = np.where(np.array(doplanets) == planet)[0]

    if chains:
        rprsdist = chains[4*pind+3,:] 
        bdist    = chains[4*pind+4,:] 
        pdist    = (chains[4*pind+5,:]-periods[pind])*24.*3600.
        tdist    = chains[4*pind+6,:]*24.0*3600.0
        lnp      = chains[-1,:]
    else:
        print "Importing MCMC chains..."
        rprsdist = np.loadtxt(ddir+name+tag+ctag+rtag+ltag+btag+'_r'+str(planet)+'chain_joint'+jtag+'.txt')
        bdist = np.loadtxt(ddir+name+tag+ctag+rtag+ltag+btag+'_b'+str(planet)+'chain_joint'+jtag+'.txt')
        pdist = (np.loadtxt(ddir+name+tag+ctag+rtag+ltag+btag+'_p'+str(planet)+'chain_joint'+jtag+'.txt')-periods[pind])*24.*3600.
        tdist = (np.loadtxt(ddir+name+tag+ctag+rtag+ltag+btag+'_t'+str(planet)+'chain_joint'+jtag+'.txt'))*24.0*3600.0
        lnp = np.loadtxt(ddir+name+tag+ctag+rtag+ltag+btag+'_lnprob_joint'+jtag+'.txt')

        
    # Get maximum likelihood values
    maxlike = np.max(lnp)
    imax = np.array([i for i, j in enumerate(lnp) if j == maxlike])
    if imax.size > 1:
        imax = imax[0]
    rprsval = np.float(rprsdist[imax])
    bval = np.float(bdist[imax])
    pval = np.float(pdist[imax])
    tval = np.float(tdist[imax])

    if thin:
        rprsdist=rprsdist[0::thin]
        bdist=bdist[0::thin]
        pdist=pdist[0::thin]
        tdist=tdist[0::thin]
    else:
        thin = 1.0

    nsamp = len(bdist)

#----------------------------------------------------------------------
# Plot planet parameters

    if write:
        plt.figure(3,figsize=(8.5,11),dpi=300)
    else:
        plt.figure(3)

# Rp/R* distribution
    rprss = np.linspace(np.min(rprsdist)*0.5,np.max(rprsdist)*1.5,1000)
    rprs_kde = gaussian_kde(rprsdist)
    rprspdf = rprs_kde(rprss)
    rprsdist_c = np.cumsum(rprspdf)/np.sum(rprspdf)
    rprsfunc = sp.interpolate.interp1d(rprsdist_c,rprss,kind='linear')
    rprslo = np.float(rprsfunc(math.erfc(1./np.sqrt(2))))
    rprshi = np.float(rprsfunc(math.erf(1./np.sqrt(2))))
    nbins = np.ceil((np.max(rprsdist)-np.min(rprsdist)) / ((rprshi-rprslo)/bindiv))
    
    rprsout = 'Rp/Rstar = {0:.4f} + {1:.4f} - {2:.4f}'
    print rprsout.format(rprsval,rprshi-rprsval,rprsval-rprslo)
    
    ax1 = plt.subplot(4,1,1)
    ax1.hist(rprsdist,bins=nbins,normed=True)
#    ax1.plot(rprss,rprspdf,color='c')
#    ax1.axvline(x=rprsval,color='r')
    rprsmin = np.float(np.sort(rprsdist)[np.round(frac*nsamp)])
    rprsmax = np.float(np.sort(rprsdist)[np.round((1-frac)*nsamp)])
    ax1.set_xlim([rprsmin,rprsmax])
    ax1.set_xlabel(r'$R_p/R_{*}$')
    ax1.set_ylabel(r'$dP/d(R_p/R_{*})$')
    ax1.set_title('KOI-'+name+'.0'+str(planet))
    
# impact parameter distribution
    bs = np.linspace(np.min(bdist)*0.5,np.max(bdist)*1.5,1000)
    b_kde = gaussian_kde(bdist)
    bpdf = b_kde(bs)
    bdist_c = np.cumsum(bpdf)/np.sum(bpdf)
    bfunc = sp.interpolate.interp1d(bdist_c,bs,kind='linear')
    blo = np.float(bfunc(math.erfc(1./np.sqrt(2))))
    bhi = np.float(bfunc(math.erf(1./np.sqrt(2))))
    nbins = np.ceil((np.max(bdist)-np.min(bdist)) / ((bhi-blo)/bindiv))
    
    bout = 'b = {0:.4f} + {1:.4f} - {2:.4f}'
    print bout.format(bval,bhi-bval,bval-blo)

    ax2 = plt.subplot(4,1,2)
    ax2.hist(bdist,bins=nbins,normed=True)
#    ax2.plot(bs,bpdf,color='c')
#    ax2.axvline(x=bval,color='r')
    bmin = np.float(np.sort(bdist)[np.round(frac*nsamp)])
    bmax = np.float(np.sort(bdist)[np.round((1-frac)*nsamp)])
    ax2.set_xlim([bmin,bmax])
    ax2.set_xlabel(r'$b$')
    ax2.set_ylabel(r'$dP/db$')

# Period distribution
    ps = np.linspace(np.min(pdist)*0.5,np.max(pdist)*1.5,1000)
    p_kde = gaussian_kde(pdist)
    ppdf = p_kde(ps)
    pdist_c = np.cumsum(ppdf)/np.sum(ppdf)
    pfunc = sp.interpolate.interp1d(pdist_c,ps,kind='linear')
    plo = np.float(pfunc(math.erfc(1./np.sqrt(2))))
    phi = np.float(pfunc(math.erf(1./np.sqrt(2))))
    nbins = np.ceil((np.max(pdist)-np.min(pdist)) / ((phi-plo)/bindiv))
    pout = 'Period = {0:.3f} + {1:.3f} - {2:.3f}'
    print pout.format(pval,phi-pval,pval-plo)

    ax3 = plt.subplot(4,1,3)
    ax3.hist(pdist,bins=nbins,normed=True)
#    ax3.plot(ps,ppdf,color='c')
#    ax3.axvline(x=pval,color='r')
    pmin = np.float(np.sort(pdist)[np.round(frac*nsamp)])
    pmax = np.float(np.sort(pdist)[np.round((1-frac)*nsamp)])
    ax3.set_xlim([pmin,pmax])
    ax3.set_xlabel(r'$\Delta Period$ (s)')
    ax3.set_ylabel(r'$dP/dPeriod$')

# Ephemeris distribution
    ts = np.linspace(np.min(tdist)*0.5,np.max(tdist)*1.5,1000)
    t_kde = gaussian_kde(tdist)
    tpdf = t_kde(ts)
    tdist_c = np.cumsum(tpdf)/np.sum(tpdf)
    tfunc = sp.interpolate.interp1d(tdist_c,ts,kind='linear')
    tlo = np.float(tfunc(math.erfc(1./np.sqrt(2))))
    thi = np.float(tfunc(math.erf(1./np.sqrt(2))))
    nbins = np.ceil((np.max(tdist)-np.min(tdist)) / ((thi-tlo)/bindiv))
    
    tout = 'Ephem = {0:.3f} + {1:.3f} - {2:.3f}'
    print tout.format(tval,thi-tval,tval-tlo)

    ax4 = plt.subplot(4,1,4)
    ax4.hist(tdist,bins=nbins,normed=True)
#    ax4.plot(ts,tpdf,color='c')
#    ax4.axvline(x=tval,color='r')
    tmin = np.float(np.sort(tdist)[np.round(frac*nsamp)])
    tmax = np.float(np.sort(tdist)[np.round((1-frac)*nsamp)])
#    print tmin,tmax
    ax4.set_xlim([tmin,tmax])
    ax4.set_xlabel(r'$\Delta t_0$ (s)')
    ax4.set_ylabel(r'$dP/dt_0$')

    plt.subplots_adjust(hspace=0.3)

    if write:
        plt.savefig(pdir+name+'.0'+str(planet)+tag+ctag+rtag+ltag+btag+'_params_joint'+jtag+'.png')
        print '... created '+pdir+name+'.0'+str(planet)+tag+ctag+rtag+ltag+btag+'_params_joint'+jtag+'.png'
        plt.clf()
    else:
        plt.show()

    return



def plot_transits(vals=False,markersize=5,submarkersize=3,nbins=100,fixwid=False):
    
    tstart = time.time()
    ddir = path+'MCMC/'
    pdir = ddir

    try:
        len(vals)
    except:
        chains,vals = get_chains()

    rhoval   = vals[0]
    q1val    = vals[1]
    q2val    = vals[2]

    u1val =  2*np.sqrt(q1val)*q2val
    u2val =  np.sqrt(q1val)*(1-2*q2val)

    plt.figure(4,figsize=(8.5,11),dpi=300)


    # Do plots
    chi = 0.0
    factor = 0.0
    for pl in np.arange(nplanets):
        rprsval = vals[pl*4+3]
        bval = vals[pl*4+4]
        ax = plt.subplot(nplanets,1,pl+1)
        ttm = foldtime(times[pl],period=vals[pl*4+5]/(24.*3600.0)+periods[pl],\
                       t0=t0s[pl]+vals[pl*4+6]/(24.*3600))
        mt,model = compute_trans(vals[pl*4+3],vals[pl*4+4],0.0,\
                                 vals[pl*4+5]/(24.*3600.0)+periods[pl],rhoval,[q1val,q2val])

        s = np.argsort(ttm)
        tfits = ttm[s]
        ffits = fluxes[pl][s]
        efits = errors[pl][s]
        cfunc = sp.interpolate.interp1d(mt,model,kind='linear')
        mfit = cfunc(tfits)
        resid = ffits - mfit

        chi += np.sum((resid/efits)**2)
        factor += len(ffits)

        rms = rb.std(resid)
        ymin = ((np.min(model) - 5.*rms)-1)*1e6
        ymax = ((1.+7*rms)-1)*1e6
        ax.plot(tfits,(ffits-1)*1e6,'.',color='gray',markersize=submarkersize)
        tbin, ybin, ebin = bin_lc(tfits,ffits,nbins=nbins)
        ax.plot(tbin,(ybin-1)*1e6,'bo',markersize=markersize)
        ax.plot(mt,(model-1)*1e6,color='r')
        minval = min(np.max(ttm),np.max(np.abs(ttm)))
        minval = np.min(ttm)
        maxval = np.max(ttm)
        if fixwid != False:
            ax.set_xlim([-1*fixwid,fixwid])
        else:
            ax.set_xlim([minval,maxval])
        ax.set_ylim([ymin,ymax])
        locs = ax.get_yticks()
        ax.set_yticks(locs, map(lambda x: "%.4f" % x, locs))

        ax.set_ylabel('ppm')
        #    plt.xlabel('Time From Mid Transit (days)')
        plname = str(doplanets[pl])
        ax.annotate('KOI-'+name+'.0'+plname,[0.03,0.9],horizontalalignment='left', \
                    xycoords='axes fraction',fontsize='medium')
        pval = vals[pl*4+5]/(24.*3600.0)+periods[pl]
        ax.annotate('Period = %.6f days' % pval,[0.03,0.8],horizontalalignment='left', \
                    xycoords='axes fraction',fontsize='medium')
        ax.annotate(r'$R_p/R_{*}$ = %.4f' % rprsval,[0.97,0.9],horizontalalignment='right', \
                    xycoords='axes fraction',fontsize='medium')
        ax.annotate('b = %.2f' % bval,[0.97,0.8],horizontalalignment='right', \
        xycoords='axes fraction',fontsize='medium')
        
        

    factor -= (ndim+1)

    chisq = chi/factor

    ax = plt.subplot(nplanets,1,1)
    ax.annotate(r'$\chi_{r}^2$ = %.5f' % chisq,[0.97,0.06],horizontalalignment='right', \
                    xycoords='axes fraction',fontsize='medium')
    ax.annotate(r'$\rho_{*}$ = %.1f g cm$^{-3}$' % rhoval,[0.5,0.9],horizontalalignment='center', \
                    xycoords='axes fraction',fontsize='medium')
    ax.annotate(r'$u_1$ = %.2f' % u1val,[0.03,0.16],horizontalalignment='left', \
                    xycoords='axes fraction',fontsize='medium')
    ax.annotate(r'$u_2$ = %.2f' % u2val,[0.03,0.06],horizontalalignment='left', \
                    xycoords='axes fraction',fontsize='medium')

    ax.set_title('KOI-'+name+' Joint Transit Fits')

    plt.tight_layout()

    plt.savefig(pdir+name+tag+ctag+rtag+ltag+btag+'_transits_joint'+jtag+'.png')
    plt.clf()


#----------------------------------------------------------------------    
# Residuals
#----------------------------------------------------------------------

    plt.figure(5,figsize=(8.5,11),dpi=300)
    for pl in np.arange(nplanets):
        ax = plt.subplot(nplanets,1,pl+1)
        ttm = foldtime(times[pl],period=vals[pl*4+5]/(24.*3600.0)+periods[pl],\
                       t0=t0s[pl]+vals[pl*4+6]/(24.*3600))
        mt,model = compute_trans(vals[pl*4+3],vals[pl*4+4],0.0,\
                                 vals[pl*4+5]/(24.*3600.0)+periods[pl],rhoval,[q1val,q2val])

        s = np.argsort(ttm)
        tfits = ttm[s]
        ffits = fluxes[pl][s]
        efits = errors[pl][s]
        cfunc = sp.interpolate.interp1d(mt,model,kind='linear')
        mfit = cfunc(tfits)
        resid = ffits - mfit
        rms = rb.std(resid)
        ymin = (-5.*rms)*1e6
        ymax = (7*rms)*1e6
        ax = plt.subplot(nplanets,1,pl+1)
        ax.plot(tfits,resid*1e6,'.',color='gray',markersize=submarkersize)
        trbin, yrbin, erbin = bin_lc(tfits,resid,nbins=nbins)
        ax.plot(trbin,yrbin*1e6,'bo',markersize=markersize)
        ax.axhline(y=0,color='r')
        minval = min(np.max(tfits),np.max(np.abs(tfits)))
        minval = np.min(tfits)
        maxval = np.max(tfits)
        if fixwid != False:
            ax.set_xlim([-1*fixwid,fixwid])
        else:
            ax.set_xlim([minval,maxval])
        ax.set_ylim([ymin,ymax])
        locs = ax.get_yticks()
        ax.set_yticks(locs, map(lambda x: "%.4f" % x, locs))
        ax.set_ylabel('ppm')
        plname = str(doplanets[pl])
        ax.annotate('KOI-'+name+'.0'+plname,[0.03,0.9],horizontalalignment='left', \
                        xycoords='axes fraction',fontsize='medium')

        RMSE = np.sum((resid/efits)**2)/len(resid)
        ax.annotate(r'$RMSE_{\rm norm}$ = %.5f' % RMSE,[0.97,0.9],horizontalalignment='right', \
                    xycoords='axes fraction',fontsize='medium')

    ax = plt.subplot(nplanets,1,1)
    ax.set_title('KOI-'+name+' Transit Fit Residuals')
    plt.tight_layout()

    plt.savefig(pdir+name+tag+ctag+rtag+ltag+btag+'_residuals_joint'+jtag+'.png')
    plt.clf()

    return



def single_triangle_plot(planet=1,chains=False,vals=False,thin=False,frac=0.001):
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

    tstart = time.time()
    
    bindiv = 10

    ddir = path+'MCMC/'
    pdir = ddir


    try:
        len(chains)
    except:
        chains,vals = get_chains()
 
    pind, = np.where(np.array(doplanets) == planet)[0]

    rhoval   = vals[0]
    q1val    = vals[1]
    q2val    = vals[2]
    rprsval = vals[4*pind+3]
    bval    = vals[4*pind+4]
    pval    = vals[4*pind+5]
    tval    = vals[4*pind+6]
    u1val =  2*np.sqrt(q1val)*q2val
    u2val =  np.sqrt(q1val)*(1-2*q2val)


    rhodist   = chains[0,:]
    q1dist    = chains[1,:]
    q2dist    = chains[2,:]
    rprsdist = chains[4*pind+3,:] 
    bdist    = chains[4*pind+4,:] 
    pdist    = chains[4*pind+5,:]
    tdist    = chains[4*pind+6,:]
    lnp      = chains[-1,:]


    if thin:
        rhodist = rhodist[0::thin]
        q1dist = q1dist[0::thin]
        q2dist = q2dist[0::thin]
        rprsdist = rprsdist[0::thin]
        bdist = bdist[0::thin]
        pdist = pdist[0::thin]
        tdist = tdist[0::thin]


    print " "
    print "Starting grid of posteriors..."
    plt.figure(88,figsize=(8.5,8.5))
    nx = 7
    ny = 7
    
    pst = str(planet)

    gs = gridspec.GridSpec(nx,ny,wspace=0.1,hspace=0.1)
    dist = rprsdist
    val = rprsval
    print " "
    print "Beginning first column"
    print " ... top plot of first column"
    tcol = time.time()
    top_plot(rprsdist,gs[0,0],val=rprsval,frac=frac)
    done_in(tcol)
    t = time.time()
    print " ... first column plot"
    column_plot(rprsdist,bdist,gs[1,0],val1=rprsval,val2=bval,ylabel=r'$b_'+pst+'$',frac=frac)
    done_in(t)
    column_plot(rprsdist,pdist,gs[2,0],val1=rprsval,val2=pval,ylabel=r'$P_'+pst+'$',frac=frac)
    column_plot(rprsdist,tdist,gs[3,0],val1=rprsval,val2=tval,ylabel=r'$t_'+pst+'$',frac=frac)
    column_plot(rprsdist,q1dist,gs[4,0],val1=rprsval,val2=q1val,ylabel=r'$q_1$',frac=frac)
    column_plot(rprsdist,q2dist,gs[5,0],val1=rprsval,val2=q2val,ylabel=r'$q_2$',frac=frac)
    corner_plot(rprsdist,rhodist,gs[6,0],val1=rprsval,val2=rhoval,\
                xlabel=r'$R_'+pst+'/R_{*}$',ylabel=r'$\rho_{*}$',frac=frac)
    print "First column: "
    done_in(tcol)
    
    print " "
    print "Beginning second column"
    t2 = time.time()
    top_plot(bdist,gs[1,1],val=bval,frac=frac)    
    middle_plot(bdist,pdist,gs[2,1],val1=bval,val2=pval,frac=frac)
    middle_plot(bdist,tdist,gs[3,1],val1=bval,val2=tval,frac=frac)
    middle_plot(bdist,q1dist,gs[4,1],val1=bval,val2=q1val,frac=frac)
    middle_plot(bdist,q2dist,gs[5,1],val1=bval,val2=q2val,frac=frac)
    row_plot(bdist,rhodist,gs[6,1],val1=bval,val2=rhoval,xlabel=r'$b_'+pst+'$',frac=frac)
    print "Second column: "
    done_in(t2)

    print " "
    print "Beginning third column"
    t3 = time.time()
    top_plot(pdist,gs[2,2],val=pval,frac=frac)    
    middle_plot(pdist,tdist,gs[3,2],val1=pval,val2=tval,frac=frac)
    middle_plot(pdist,q1dist,gs[4,2],val1=pval,val2=q1val,frac=frac)
    middle_plot(pdist,q2dist,gs[5,2],val1=pval,val2=q2val,frac=frac)
    row_plot(pdist,rhodist,gs[6,2],val1=pval,val2=rhoval,xlabel=r'$\Delta P_'+pst+'$ (s)',frac=frac)
    print "Third column: "
    done_in(t3)

    
    print ""
    print "Beginning fourth column"
    t4 = time.time()
    top_plot(tdist,gs[3,3],val=tval,frac=frac)    
    middle_plot(tdist,q1dist,gs[4,3],val1=tval,val2=q1val,frac=frac)
    middle_plot(tdist,q2dist,gs[5,3],val1=tval,val2=q2val,frac=frac)
    row_plot(tdist,rhodist,gs[6,3],val1=tval,val2=rhoval,xlabel=r'$\Delta t_'+pst+'$ (s)',frac=frac)
    print "Fourth column:"
    done_in(t4)


    print ""
    print "Beginning fifth column"
    t5 = time.time()
    top_plot(q1dist,gs[4,4],val=q1val,frac=frac)    
    middle_plot(q1dist,q2dist,gs[5,4],val1=q1val,val2=q2val,frac=frac)
    row_plot(q1dist,rhodist,gs[6,4],val1=q1val,val2=rhoval,xlabel=r'$q_1$',frac=frac)
    print "Fifth column:"
    done_in(t5)

    print ""
    print "Beginning sixth column"
    t6 = time.time()
    top_plot(q2dist,gs[5,5],val=q2val,frac=frac)    
    row_plot(q2dist,rhodist,gs[6,5],val1=q2val,val2=rhoval,xlabel=r'$q_2$',frac=frac)
    print "Sixth column:"
    done_in(t6)


    print "Last plot!"
    t7 = time.time()
    top_plot(rhodist,gs[6,6],val=rhoval,xlabel=r'$\rho_{*}$',frac=frac)    
    done_in(t7)
    
    print "Saving output figures"
    plt.savefig(pdir+name+tag+ctag+rtag+ltag+btag+'_joint'+jtag+'_triangle'+pst+'.png', dpi=300)
    plt.savefig(pdir+name+tag+ctag+rtag+ltag+btag+'_joint'+jtag+'_triangle'+pst+'.eps', format='eps', dpi=600)

    print "Procedure finished!"
    done_in(tstart)
    return


def triangle_plot(thin=10,frac=0.001):
    for planet in doplanets:
        single_triangle_plot(planet=planet,thin=thin,frac=frac)
    return


def top_plot(dist,position,val=False,frac=0.001,bindiv=10,aspect=1,xlabel=False):
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.stats.kde import gaussian_kde
    from scipy.interpolate import interp1d
    import pdb
    import time

    len = np.size(dist)
    min = np.float(np.sort(dist)[np.round(frac*len)+1])
    max = np.float(np.sort(dist)[np.round((1.-frac)*len)-1])
    dists = np.linspace(np.min(dist)*0.5,np.max(dist)*1.5,1000)
    kde = gaussian_kde(dist)
    pdf = kde(dists)
    cumdist = np.cumsum(pdf)/np.sum(pdf)
    func = interp1d(cumdist,dists,kind='linear')
    lo = np.float(func(math.erfc(1./np.sqrt(2))))
    hi = np.float(func(math.erf(1./np.sqrt(2))))
    nbins = np.ceil((np.max(dist)-np.min(dist)) / ((hi-lo)/bindiv))
    ax = plt.subplot(position)
    plt.hist(dist,bins=nbins,normed=True,color='black')
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
            tick.label.set_fontsize(6) 
            tick.label.set_rotation('vertical')
        ax.set_xlabel(xlabel,fontsize=10)
 

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
    min1 = np.float(np.sort(dist1)[np.round(frac*len1)+1])
    max1 = np.float(np.sort(dist1)[np.round((1.-frac)*len1)-1])

    len2 = np.size(dist2)
    min2 = np.float(np.sort(dist2)[np.round(frac*len2)+1])
    max2 = np.float(np.sort(dist2)[np.round((1.-frac)*len2)-1])

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
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(6) 
    ax.set_ylabel(ylabel,fontsize=10)

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
    min1 = np.float(np.sort(dist1)[np.round(frac*len1)+1])
    max1 = np.float(np.sort(dist1)[np.round((1.-frac)*len1)-1])
    
    len2 = np.size(dist2)
    min2 = np.float(np.sort(dist2)[np.round(frac*len2)+1])
    max2 = np.float(np.sort(dist2)[np.round((1.-frac)*len2)-1])
    
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
        tick.label.set_fontsize(6) 
        tick.label.set_rotation('vertical')
    ax.set_xlabel(xlabel,fontsize=10)
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
    min1 = np.float(np.sort(dist1)[np.round(frac*len1)+1])
    max1 = np.float(np.sort(dist1)[np.round((1.-frac)*len1)-1])

    len2 = np.size(dist2)
    min2 = np.float(np.sort(dist2)[np.round(frac*len2)+1])
    max2 = np.float(np.sort(dist2)[np.round((1.-frac)*len2)-1])

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
    min1 = np.float(np.sort(dist1)[np.round(frac*len1)+1])
    max1 = np.float(np.sort(dist1)[np.round((1.-frac)*len1)-1])

    len2 = np.size(dist2)
    min2 = np.float(np.sort(dist2)[np.round(frac*len2)+1])
    max2 = np.float(np.sort(dist2)[np.round((1.-frac)*len2)-1])

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
        tick.label.set_fontsize(6) 
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(6) 
        tick.label.set_rotation('vertical')
    ax.set_xlabel(xlabel,fontsize=10)
    ax.set_ylabel(ylabel,fontsize=10)

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


    return


def get_rho(final=False):
    from scipy.stats.kde import gaussian_kde
    import scipy as sp
    
    ddir = path+'MCMC/'
    pdir = ddir

     
    tmaster = time.time()
    print "Reading in stellar density and lnprob chains"
    rhodist   = np.loadtxt(ddir+name+tag+ctag+rtag+ltag+btag+'_rhochain_joint'+jtag+'.txt')
    lnp       = np.loadtxt(ddir+name+tag+ctag+rtag+ltag+btag+'_lnprob_joint'+jtag+'.txt')
    done_in(tmaster)

    print "Determining maximum likelihood values"
    maxlike = np.max(lnp)
    imax = np.array([i for i, j in enumerate(lnp) if j == maxlike])
    if imax.size > 1:
        imax = imax[0]
    rhoval   = np.float(rhodist[imax])

    rhos = np.linspace(np.min(rhodist)*0.5,np.max(rhodist)*1.5,1000)
    rho_kde = gaussian_kde(rhodist)
    rhopdf = rho_kde(rhos)
    rhopeak = rhos[np.argmax(rhopdf)]
    rhodist_c = np.cumsum(rhopdf)/np.sum(rhopdf)
    rhofunc = sp.interpolate.interp1d(rhodist_c,rhos,kind='linear')

    rhothi = np.linspace(.684,.999,100)
    rhotlo = rhothi-0.6827
    rhohis = rhofunc(rhothi)
    rholos = rhofunc(rhotlo)
    
    interval = np.min(rhohis-rholos)
    
    return rhoval,rhopeak,interval


def do_fit(koi,nwalkers=1000,burnsteps=1000,mcmcsteps=1000,limbmodel='quad',bin=False,thin=10,frac=0.001,network=None,keep=[1,2],short=True,rprior=True,clip=True):
    get_koi_info(koi,network=network,keep=keep,sc=short,rprior=rprior,clip=clip)
    do_mcmc(nwalkers=nwalkers,burnsteps=burnsteps,mcmcsteps=mcmcsteps)
#    plot_star(write=True,frac=frac)
#    for planet in keep:
#        plot_planet(planet=planet,frac=frac,write=True)
#    plot_transits()
#    triangle_plot(thin=thin)

def do_plots(koi,limbmodel='quad',bin=False,thin=10,frac=0.001,network=None,keep=[1,2],short=True,fixwid=False,rprior=True):
    get_koi_info(koi,network=network,keep=keep,sc=short,rprior=rprior)
    plot_star(write=True,frac=frac)
    for planet in keep:
        plot_planet(planet=planet,frac=frac,thin=thin,write=True)
    plot_transits(fixwid=fixwid)
    triangle_plot(thin=thin,frac=frac)


def do_conf(koi,limbmodel='quad',bin=False,network=None,keep=[1,2],short=True):
    get_koi_info(koi,network=network,keep=keep,sc=short)
    rhoval,rhopeak,interval = get_rho()
    print ""
    print "Best fit stellar density: %.2f g/cc" % rhoval
    print "Most likely stellar density: %.2f g/cc" % rhopeak
    print "Full width of 68 percent interval %.2f g/cc" % interval

