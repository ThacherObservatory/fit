"""
@author: Doug Klink
GP fitter for out-of-eclipse light variations for EB system kepler 10935310
"""

import emcee, george, corner
from george.kernels import ExpSine2Kernel, ExpSquaredKernel
import numpy as np
import matplotlib.pyplot as plt

time,flux,err = np.loadtxt("10935310_ooe_real.txt",unpack=True)

#chunk of data to analyze for testing
time = time[8000:10000]
flux = flux[8000:10000]
err = err[8000:10000]

print("starting...")

#plot data
plt.ion()
plt.figure(1)
plt.clf()
plt.subplot(2,1,1)
plt.plot(time,flux,'ko',markersize=2)
plt.xlabel('Time (BKJD)')
plt.ylabel('Flux (ADU)')


#create GP kernel to model flux variation due to starspots
k =  .02**2 * ExpSquaredKernel(1) * ExpSine2Kernel(4,.002)
gp = george.GP(k,mean=np.mean(flux),solver=george.HODLRSolver)


#test: effects of combining two identical kernels?
#note: gp.kernel.vector has length 8
#k1 =  .02**2 * ExpSquaredKernel(1) * ExpSine2Kernel(4,.002)
#k2 =  .01**2 * ExpSquaredKernel(0.5) * ExpSine2Kernel(2,.001)
#k = k1 + k2
#print(gp.kernel.vector)

def lnprob(theta,time,flux,err):
    """log likelihood function for MCMC"""
    #theta[0] = amplitude
    #theta[1] = width
    #theta[2] = width of semi periodic kernel
    #theta[3] = period

    #theta is actually the natural log of the input parameters!

    #Note to Dr. Swift: what's going on here with theta 2 and 3?
    gp.kernel[:] = np.array([theta[0],theta[1],0.5,theta[2]])

    if np.exp(theta[1]) < 0.75  or np.exp(theta[1]) > 1.25:
        return -np.inf
    if np.exp(theta[2]) < 3 or np.exp(theta[2]) > 5:
        return -np.inf
    if np.exp(theta[3]) < .0005 or np.exp(theta[3]) > .005:
        return -np.inf
   
    try:
        gp.compute(time,4,sort=True)
    except (ValueError, np.linalg.LinAlgError):
        return -np.inf

    loglike = gp.lnlikelihood(flux, quiet=True)

    
    return loglike
  
#display initial log probability before MCMC  
gp.compute(time,4,sort=True)
print(gp.lnlikelihood(flux))

#Initialize the MCMC Hammer
p0 = gp.kernel.vector
nwalkers = 100
burnsteps = 2000
mcmcsteps = 2000
ndim = len(p0)
p0_vec = [np.abs(p0[i])+1e-3*np.random.randn(nwalkers) for i in range(ndim)]
p0_init = np.array(p0_vec).T

#Drop the MCMC Hammer
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time,flux,err))
print("Starting Burn-in")
pos,prob,state = sampler.run_mcmc(p0_init, burnsteps)

#Pick up the MCMC Hammer
sampler.reset()

#Drop the MCMC Hammer again, for real this time
print("Starting MCMC")
pos, prob, state = sampler.run_mcmc(pos,mcmcsteps)

#extract median values from chains for use in GP visualization
logamp = np.median(sampler.flatchain[:,0])
logg = np.median(sampler.flatchain[:,1])
logp = np.median(sampler.flatchain[:,2])
logw = np.median(sampler.flatchain[:,3])

#apply median values to GP
fit =  np.array([logamp,logg,logp,logw])
gp.kernel[:] = fit

#report lnlikelihood post-MCMC
print(gp.lnlikelihood(flux))


#calculate gp prediction for each data point
gp.compute(time,4,sort=True)
flux_fit, cov_fit = gp.predict(flux, time)

#plot gp prediction
plt.figure(1)
plt.plot(time,flux_fit,'r.')

#plot residuals
plt.figure(1)
plt.subplot(2,1,2)
plt.plot(time,flux_fit-flux,'ko')
plt.axhline(y=0,linestyle='-',color='red',lw=3)
plt.xlabel('Time (BKJD)')
plt.ylabel('Residuals (ADU)')
plt.savefig("10935310_ooe_fit.png",dpi=300)

#generate corner plot of gp parameters
samples = sampler.chain.reshape((-1, ndim))
figure = corner.corner(samples, labels=["$T1$","$T2$","$T3$","$T4$","$T5$"])
figure.savefig("gp_test_corner.png",dpi=300)