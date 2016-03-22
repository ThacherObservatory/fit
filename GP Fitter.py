"""
@author: Doug Klink
GP fitter for out-of-eclipse light variations for EB system kepler 10935310
"""

import kplr, emcee, george, corner
from george.kernels import ExpSine2Kernel, ExpSquaredKernel
import pdb,sys, pickle
import numpy as np
import matplotlib.pyplot as plt

time,flux,err = np.loadtxt("10935310_sim_ooe.txt",unpack=True)


plt.ion()
plt.figure(1)
plt.clf()
plt.subplot(2,1,1)
plt.plot(time,flux,'ko',markersize=2)
plt.xlabel('Time (BKJD)')
plt.ylabel('Flux (ADU)')

k =  .02**2 * ExpSquaredKernel(1) * ExpSine2Kernel(4,.002)
gp = george.GP(k,mean=np.mean(flux))
#use solver=george.HODLRSolver for O(nlog^2n) instead of O(n^3)

def lnprob(theta,time,flux,err):
    #theta[0] = amplitude
    #theta[1] = width
    #theta[2] = width of semi periodic kernel
    #theta[3] = period

#    theta is actually the natural log of the input parameters!
    gp.kernel[:] = np.array([theta[0],theta[1],0.5,theta[2]])

#    print(np.exp(theta))
#    if np.exp(theta[0]) <= 0 or np.exp(theta[0]) > 1000:
#        return -np.inf
#
    if np.exp(theta[1]) < 0.75  or np.exp(theta[1]) > 1.25:
        return -np.inf
#
    if np.exp(theta[2]) < 3 or np.exp(theta[2]) > 5:
        return -np.inf
#
    if np.exp(theta[3]) < .0005 or np.exp(theta[3]) > .005:
        return -np.inf
   
    try:
        gp.compute(time,4,sort=True)
#        gp.compute(time,2,sort=True)
    except (ValueError, np.linalg.LinAlgError):
#        print('WTF!')
        return -np.inf

    loglike = gp.lnlikelihood(flux, quiet=True)
#    loglike -= 0.5*((np.exp(theta[1])-2)/0.01)**2
#    if np.exp(theta[1]) <= 0.3:
#        return -np.inf
#    loglike -= 0.5*((np.exp(theta[2])-2)/0.1)**2
#    loglike -= 0.5*((np.exp(theta[3])-0.5)/0.1)**4
    
    return loglike #gp.lnlikelihood(flux, quiet=True)
    
gp.compute(time,4,sort=True)
#gp.compute(time,2,sort=True)
print(gp.lnlikelihood(flux))

p0 = gp.kernel.vector

nwalkers = 20
burnsteps = 100
mcmcsteps = 100
ndim = len(p0)

# drop the MCMC hammer, yo.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time,flux,err))#,threads=3)

p0_vec = [np.abs(p0[i])+1e-3*np.random.randn(nwalkers) for i in range(ndim)]
p0_init = np.array(p0_vec).T

pos,prob,state = sampler.run_mcmc(p0_init, burnsteps)

sampler.reset()
pos, prob, state = sampler.run_mcmc(pos,mcmcsteps)

#plt.figure(2)
#plt.clf()

logamp = np.median(sampler.flatchain[:,0])
logg = np.median(sampler.flatchain[:,1])
logp = np.median(sampler.flatchain[:,2])
logw = np.median(sampler.flatchain[:,3])

fit =  np.array([logamp,logg,logp,logw])
#fit =  np.array([logamp,logg])
gp.kernel[:] = fit
print(gp.lnlikelihood(flux))

x = np.linspace(np.min(time), np.max(time), 5000)
gp.compute(time,4,sort=True)
#gp.compute(time,2,sort=True)
mu, cov = gp.predict(flux, x)
plt.figure(1)
plt.plot(x,mu,'r-')

flux_fit, cov_fit = gp.predict(flux,time)
#plt.plot(time,flux_fit,'r.-')

#plt.xlim(355.7,355.8)
#plt.ylim(14500,14600)

plt.figure(1)
plt.subplot(2,1,2)
plt.plot(time,flux_fit-flux,'ko')
plt.axhline(y=0,linestyle='-',color='red',lw=3)
plt.xlabel('Time (BKJD)')
plt.ylabel('Residuals (ADU)')

pickle.dump( sampler, open( "george_test.pkl", "wb" ) )
sys.exit()
sampler = pickle.load( open( "george_test.pkl", "rb" ) )

#corner plot
#samples = sampler.flatchain.reshape([-1, ndim])
samples = sampler.chain.reshape((-1, ndim))

figure = corner.corner(samples, labels=["$T1$","$T2$","$T3$","$T4$","$T5$"])
figure.savefig("gp_test_corner.png",dpi=300)
