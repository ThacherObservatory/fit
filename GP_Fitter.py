"""
@author: Doug Klink
GP fitter for out-of-eclipse light variations for EB system kepler 10935310
"""

import emcee, george, corner
from george.kernels import ExpSine2Kernel, ExpSquaredKernel
import numpy as np
import matplotlib.pyplot as plt
import sys

time,flux,err = np.loadtxt("10935310_ooe_real.txt",unpack=True)

time = time[9000:10000]
flux = flux[9000:10000]
err = err[9000:10000]

print("starting...")

#plot data
plt.ion()
plt.figure(1)
plt.clf()
plt.subplot(2,1,1)
plt.plot(time,flux,'ko',markersize=5)
plt.xlabel('Time (BKJD)')
plt.ylabel('Flux (ADU)')

#test: effects of combining two identical kernels?
# Yes, one can be attributed to one star, and one from the other.
# Don't know if it will work, but it is similar to Ben's choice to
# make the amplitude of the semi-periodic kernel semi-periodic.
#note: gp.kernel.vector has length 8
# I haven't tested it yet. Note, I'm only using k1 for now.
k1 =  20**2 * ExpSquaredKernel(10) * ExpSine2Kernel(1.0,4)
k2 =  .01**2 * ExpSquaredKernel(0.5) * ExpSine2Kernel(2,.001)
k = k1 + k2
gp = george.GP(k1,mean=np.mean(flux),solver=george.HODLRSolver)
print(gp.kernel.vector)

gp.compute(time,4,sort=True)
flux_fit, cov_fit = gp.predict(flux, time)

#plot gp prediction
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(time,flux_fit,'g.')

#sys.exit()

def lnprob(theta,time,flux,err):
    """log likelihood function for MCMC"""
    #theta[0] = amplitude
    #theta[1] = width
    #theta[2] = width of semi periodic kernel
    #theta[3] = period

    #theta is actually the natural log of the input parameters!

    #Note to Dr. Swift: what's going on here with theta 2 and 3?
    # I was experimenting with holding some parameters constant.
    gp.kernel[:] = np.array([theta[0],theta[1],theta[2],theta[3]])

    if np.exp(theta[0]) < 0.0  or np.exp(theta[0]) > 1000.0:
        return -np.inf
#    if np.exp(theta[1]) < 0.01  or np.exp(theta[1]) > 5:
#        return -np.inf
#    if np.exp(theta[2]) < 2 or np.exp(theta[2]) > 7:
#        return -np.inf
#    if np.exp(theta[3]) < 3 or np.exp(theta[3]) > 5:
#        return -np.inf
#    print gp.kernel.vector
   
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
nwalkers = 20
burnsteps = 100
mcmcsteps = 100
ndim = len(p0)
p0_vec = [p0[i]+1e-3*np.random.randn(nwalkers) for i in range(ndim)]
p0_init = np.array(p0_vec).T


#Drop the MCMC Hammer
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time,flux,err))
print("Starting Burn-in")

# Put in progress bar for emcee!

#for i, (pos, prob, state) in enumerate(sampler.run_mcmc(p0_init, burnsteps)):
#    if (i+1) % 1 == 0:
#        print("{0:.1f}%".format(100 * float(i) / burnsteps))
#sys.exit()

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

print(gp.kernel.vector)


#report lnlikelihood post-MCMC
print(gp.lnlikelihood(flux))


#calculate gp prediction for each data point
gp.compute(time,4,sort=True)
flux_fit, cov_fit = gp.predict(flux, time)

#plot gp prediction
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(time,flux_fit,'r.')

#plot residuals
plt.figure(1)
plt.subplot(2,1,2)
plt.plot(time,flux_fit-flux,'ko')
plt.axhline(y=0,linestyle='-',color='red',lw=3)
plt.xlabel('Time (BKJD)')
plt.ylabel('Residuals (ADU)')
#plt.savefig("10935310_ooe_fit.png",dpi=300)

#generate corner plot of gp parameters
samples = sampler.chain.reshape((-1, ndim))
figure = corner.corner(samples, labels=["$T_1$","$T_2$","$T_3$","$T_4$"])
#figure.savefig("gp_test_corner.png",dpi=300)
