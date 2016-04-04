"""
@author: Doug Klink
GP fitter for out-of-eclipse light variations for EB system kepler 10935310
"""

import emcee, george, corner
from george.kernels import ExpSine2Kernel, ExpSquaredKernel
import numpy as np
import matplotlib.pyplot as plt
import sys,pdb
import time as tm

time,flux,err = np.loadtxt("10935310_ooe_real.txt",unpack=True)

time = time[9000:10000]
flux = flux[9000:10000]
err = err[9000:10000]

print("starting...")

plt.ion()
plt.figure(1)

def test_params():
    ampvec   = np.array([1,10,100,1000,10000])
    expsqvec = np.array([1,3,5,10,30,50,100])
    gammavec = np.array([0.3,1,3,10,30])
    pvec     = np.array([1,3,7,10,30,70,100])
    
#    for i in range(len(ampvec)):
#    for i in range(len(expsqvec)):
#    for i in range(len(gammavec)):
    for i in range(len(pvec)):
        #plot data
        plt.clf()
        plt.plot(time,flux,'ko',markersize=5)
        plt.xlabel('Time (BKJD)')
        plt.ylabel('Flux (ADU)')

#        k =  ampvec[i]**2 * ExpSquaredKernel(10) * ExpSine2Kernel(1.0,4)
#        k =  500**2 * ExpSquaredKernel(expsqvec[i]) * ExpSine2Kernel(1.0,4)
#        k =  500**2 * ExpSquaredKernel(10) * ExpSine2Kernel(gammavec[i],4)
        k =  500**2 * ExpSquaredKernel(10) * ExpSine2Kernel(1.0,pvec[i])
        gp = george.GP(k,mean=np.mean(flux),solver=george.HODLRSolver)
        v = gp.kernel.vector
        print(v)

        gp.compute(time,sort=True)
        flux_fit, cov_fit = gp.predict(flux, time)

        #plot gp prediction
        plt.plot(time,flux_fit,'r.')
        plt.xlim(790.5,792)
        plt.ylim(0.985,1.00)
#        plt.title('Amp = '+str(ampvec[i]))
#        plt.title('ExpSq = '+str(expsqvec[i]))
#        plt.title('Gamma = '+str(gammavec[i]))
        plt.title('Period = '+str(pvec[i]))
#        plt.savefig('Amp='+str(ampvec[i])+'.png',dpi=150)
#        plt.savefig('ExpSq='+str(expsqvec[i])+'.png',dpi=150)
        plt.savefig('Period='+str(pvec[i])+'.png',dpi=150)
    return
#test_params()
#sys.exit()

# Notes:
##############################
# Amp:
# Might be able to vary freely
# Start at 500

# ExpSq:
# Doesn't change much until it gets near 100, then seems like it
# misses the mark
# Start at 10

# Gamma:
# Doesn't change much until it gets near 30, then starts overfitting
# Start = 5

# Period:
# Not surprisingly, it overfits for small period and underfits for
# long period. This parameter can probably be obtained from the
# autocorrelation of the light curve and then held fixed.
# Start = 3.99 from Rotation.png


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
k1 = 0.01**2 * ExpSquaredKernel(0.1)# * ExpSine2Kernel(0.01,4)
k2 =  .01**2 * ExpSquaredKernel(0.5) * ExpSine2Kernel(2,.001)
k = k1 + k2
#gp = george.GP(k1,mean=np.mean(flux),solver=george.HODLRSolver)
gp = george.GP(k1,solver=george.HODLRSolver)
print(gp.kernel.vector)

gp.compute(time,sort=True)
flux_fit, cov_fit = gp.predict(flux, time)

#plot gp prediction
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(time,flux_fit,'g.')

sys.exit()

def lnprob(theta,time,flux,err):
    """log likelihood function for MCMC"""
    #theta[0] = amplitude
    #theta[1] = width
    #theta[2] = width of semi periodic kernel
    #theta[3] = period

    # theta is actually the natural log of the input parameters
    # (amp) 

#    gp.kernel[:] = np.array([theta[0],theta[1],theta[2],theta[3]])
    gp.kernel[:] = theta #np.array([theta[0],theta[1],theta[2],theta[3]])

#    if np.exp(theta[0]/2) < 10.0  or np.exp(theta[0]/2) > 1000.0:
#        return -np.inf
#    if np.exp(theta[1]) < 1  or np.exp(theta[1]) > 100:
#        return -np.inf
#    if np.exp(theta[2]) < 1 or np.exp(theta[2]) > 50:
#        return -np.inf
#    if np.exp(theta[3]) < 3 or np.exp(theta[3]) > 5:
#        return -np.inf
#    print gp.kernel.vector

    try:
        gp.compute(time,sort=True)
#        gp.compute(time,4,sort=True)
    except (ValueError, np.linalg.LinAlgError):
        return -np.inf

    flux_fit, cov_fit = gp.predict(flux, time)
    loglike = (flux_fit-flux)**2/(2.0*err**2) - (np.exp(theta[3]) - 3.99)**2/(2*0.2**2)
#    loglike = gp.lnlikelihood(flux, quiet=True) - (np.exp(theta[3]) - 3.99)**2/(2*0.2**2)

    return loglike
  
#display initial log probability before MCMC  
gp.compute(time,sort=True)
print(gp.lnlikelihood(flux))

#Initialize the MCMC Hammer
p0 = gp.kernel.vector
nwalkers = 20
burnsteps = 20
mcmcsteps = 20
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
loge = np.median(sampler.flatchain[:,1])
logg = np.median(sampler.flatchain[:,2])
logp = np.median(sampler.flatchain[:,3])

# extract maximum likelihood value
maxlnprob = np.max(sampler.flatlnprobability)
maxind, = np.where(sampler.flatlnprobability == maxlnprob)

maxind = maxind[0]

amp = np.exp(sampler.flatchain[maxind,0]/2.0)
expsq = np.exp(sampler.flatchain[maxind,1])
gamma = np.exp(sampler.flatchain[maxind,2])
period = np.exp(sampler.flatchain[maxind,3])


#apply median values to GP
fit =  np.array([logamp,loge,logg,logp])
gp.kernel[:] = fit

#print(gp.kernel.vector)

kfinal =  amp**2 * ExpSquaredKernel(expsq) * ExpSine2Kernel(gamma,period)
gp = george.GP(kfinal,mean=np.mean(flux),solver=george.HODLRSolver)
print(gp.kernel.vector)

#calculate gp prediction for each data point
gp.compute(time)
flux_fit, cov_fit = gp.predict(flux, time)


#report lnlikelihood post-MCMC
print(gp.lnlikelihood(flux))

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
