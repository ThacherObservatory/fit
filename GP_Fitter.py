import emcee, george, corner
from george.kernels import ExpSine2Kernel, ExpSquaredKernel
import numpy as np
import matplotlib.pyplot as plt
import sys,pdb
import time as tm
from done_in import done_in

tmaster = tm.time()
time,flux,err = np.loadtxt("10935310_ooe_real.txt",unpack=True)

s = np.argsort(time)
time0 = time[s] ; flux0=flux[s] ; err0=err[s]


a = np.array([4697,6918,9139,11360,13581,15802])
b = np.array([6918,9139,11360,13581,15802,18026])
#for testing 4-6
#a = a[3:]
#b = b[3:]


a1 = np.array([0,21,0])
b1 = np.array([31,50,35])

dir = np.array(['first_interval','second_interval','third_interval','fourth_interval','fifth_interval','sixth_interval'])
#dir = dir[3:]


def lnprob(theta,time,flux,err):
    """log likelihood function for MCMC"""
    
    #theta[0] = amplitude
    #theta[1] = variance of the envelope of periodic correlations
    #theta[2] = inverse variance of the periodic peaks 
    #theta[3] = period

    
    # NOTE
    # theta is actually the natural log of the input parameters
    # (amp) 

    gp.kernel[:] = theta 

    # We know the period is 3.99 from autocorrelation function
    pprior =  -1.0*(np.exp(theta[3])-3.99)**4/(2.0*0.2**4)

    # Don't let period get much less than half of known period
    if np.exp(theta[3]) < 1.9:
        return -np.inf
    
    # To prevent overfitting of data, Gamma should be smaller than about 
    # 50 (or 5 data points)
    if np.exp(theta[2]) > 45:
        gprior = -1.0*(np.exp(theta[2])-45.0)**4/(2.0*1.0**4)
    else:
        gprior = 0.0

    # Do not allow the "envelope" to be smaller than the variance of each peak
    if np.exp(theta[1]) < 1.0/(2*np.exp(theta[2])**2):
        return -np.inf

    # Don't let taper get smaller than about 4 (period)
    if np.exp(theta[1]) < 4:
        tprior = -1.0*(np.exp(theta[1])-4)**2/(2.0*0.1**2)
    else:
        tprior = 0.0

#    if np.exp(theta[0]/2.0) < 0.0001  or np.exp(theta[0]/2.0) > 100.0:
#        return -np.inf

    try:
        gp.compute(time,yerr=err,sort=True)
    except (ValueError, np.linalg.LinAlgError):
        print 'Oh, no!'
        return -np.inf
    
    loglike = gp.lnlikelihood(flux, quiet=True) + pprior + gprior + tprior

    return loglike


for i in range(len(a)):
    print("starting interval " +str(i+1))
    if i < 3:
	continue
    time = time0[a[i]:b[i]]
    flux = flux0[a[i]:b[i]]
    err = err0[a[i]:b[i]]
    
    # Notes:
    ##############################
    # Amp
    # Overall amplitude of the correlations
    # 0.01 to 10
    
    # ExpSq:
    # Taper on semi-periodic envelope
    # 1-100
    
    # Gamma:
    # Width of each semi-periodic peak (c3 = 1/(2 sig^2))
    # Can't be too big, else will over fit noise.
    # Limit width to be biggger than  ~2 data points (approx 50)
    
    # Period:
    # Not surprisingly, it overfits for small period and underfits for
    # long period. This parameter can probably be obtained from the
    # autocorrelation of the light curve and then held fixed.
    # Start = 3.99 from Rotation.png
    ##############################
    

    # Define kernel 
    # Quasi periodic variations
    k1 = 0.01**2 * ExpSquaredKernel(10.0) * ExpSine2Kernel(20.0,3.99)
    # "base spottedness": long period trend of unknown shape
    gp = george.GP(k1,mean=np.mean(flux),solver=george.HODLRSolver)

    gp.compute(time,yerr=err,sort=True)
    flux_fit, cov_fit = gp.predict(flux, time)


    # Plot data
    ##############################
    plt.figure(1)
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(time,flux,'ko',markersize=5)
    plt.xlabel('Time (BKJD)')
    plt.ylabel('Flux (ADU)')
    plt.figure(1)
    plt.plot(time,flux_fit,'c.')
    
    plt.subplot(2,1,2)
    plt.plot(time,flux-flux_fit,'ko')
    plt.axhline(y=0,linestyle='-',color='red',lw=3)
    plt.xlabel('Time (BKJD)')
    plt.ylabel('Residuals (ADU)')
    ##############################
    plt.savefig(dir[i]+'/Initial_fit.png',dpi=300)
    """
    tmodel = np.linspace(time[a1[i]],time[b1[i]],300)
    flux_model, cov_model = gp.predict(flux, tmodel)

    plt.figure(2)
    plt.clf()
    plt.plot(time[a1[i]:b1[i]],flux[a1[i]:b1[i]],'ko',markersize=5)
    plt.xlabel('Time (BKJD)')
    plt.ylabel('Flux (ADU)')
    plt.plot(tmodel,flux_model,'r-')
    ##############################
    
    plt.savefig(dir[i]+'/Initial_zoom.png',dpi=300)
    
    """
  
    # Display initial log probability before MCMC  
    gp.compute(time,yerr=err,sort=True)
    print 'Max ln prob before fitting = %f.5' % gp.lnlikelihood(flux)

    # Initialize the MCMC Hammer
    p0 = gp.kernel.vector
    nwalkers = 20
    burnsteps = 1000
    mcmcsteps = 1000
    ndim = len(p0)
    p0_vec = [p0[j]+1e-2*np.random.randn(nwalkers) for j in range(ndim)]
    p0_init = np.array(p0_vec).T
    
    print done_in(tmaster)
    
    # Drop the MCMC Hammer
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time,flux,err), threads=32)
    print("Starting Burn-in")
    
    tburn = tm.time()
    pos,prob,state = sampler.run_mcmc(p0_init, burnsteps)
    print done_in(tburn)
    
    # Pick up the MCMC Hammer
    sampler.reset()
    
    # Drop the MCMC Hammer again, for real this time
    tmcmc = tm.time()
    print("Starting MCMC for reals")
    pos, prob, state = sampler.run_mcmc(pos,mcmcsteps)
    print done_in(tmcmc)
    
    # Extract maximum likelihood value
    maxlnprob = np.max(sampler.flatlnprobability)
    maxind, = np.where(sampler.flatlnprobability == maxlnprob)
    print 'Max ln prob after fitting = %f.5' % maxlnprob
    maxind = maxind[0]
    
    # Extract max like values from chains for use in GP visualization
    amp = np.exp(sampler.flatchain[maxind,0]/2.0)
    expsq = np.exp(sampler.flatchain[maxind,1])
    gamma = np.exp(sampler.flatchain[maxind,2])
    period = np.exp(sampler.flatchain[maxind,3])
    
    # Apply maximum likelihood values to GP
    k = amp**2 * ExpSquaredKernel(expsq) * ExpSine2Kernel(gamma,period)
    gp = george.GP(k,mean=np.mean(flux),solver=george.HODLRSolver)
    
    gp.compute(time,yerr=err,sort=True)
    flux_final, cov_final = gp.predict(flux, time)
    
    
    # Plot GP prediction
    plt.figure(3)
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(time,flux,'k.')
    plt.plot(time,flux_final,'c.')
    
    #plot residuals
    plt.subplot(2,1,2)
    plt.plot(time,flux_final-flux,'ko')
    plt.axhline(y=0,linestyle='-',color='red',lw=3)
    plt.xlabel('Time (BKJD)')
    plt.ylabel('Residuals (ADU)')
    plt.savefig(dir[i]+"/Final_fit.png",dpi=300)
    
    # Generate corner plot of gp parameters
    samples = sampler.chain.reshape((-1, ndim))
    samples[:,0] = np.exp(samples[:,0]/2)
    samples[:,1:] = np.exp(samples[:,1:])
    figure = corner.corner(samples, labels=["Amplitude","Envelope Variance","Inverse Peak Variance","Period (days)"])
    figure.savefig(dir[i]+"/Final_corner.png",dpi=300)
    """
    tmodel = np.linspace(time[a1[i]],time[b1[i]],300)
    flux_model, cov_model = gp.predict(flux, tmodel)
    plt.figure(5)
    plt.clf()
    plt.plot(time[a1[i]:b1[i]],flux[a1[i]:b1[i]],'ko',markersize=5)
    plt.xlabel('Time (BKJD)')
    plt.ylabel('Flux (ADU)')
    plt.plot(tmodel,flux_model,'r-')
    plt.savefig(dir[i]+'/Final_zoom.png',dpi=300)
    """
    print done_in(tmaster)
    print ''
    print ''
    
    
