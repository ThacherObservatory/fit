import scipy.optimize as op
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import pdb

"""
MCMC fitter to test the criteria for paralellizing the emcee package
"""

#############################################################
# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294
f_true = 0.534

#############################################################
# Generate some synthetic data from the model.
#############################################################
N = 200
x = np.sort(10*np.random.rand(N))
yerr = 0.1+0.5*np.random.rand(N)
y = m_true*x+b_true
y += np.abs(f_true) * np.random.randn(N)
y += yerr * np.random.randn(N)

plt.errorbar(x,y,yerr=yerr,fmt='o',color='k')


#############################################################
# Maximum Likelihood Fitting
#############################################################
def lnlike(theta, x, y, yerr):
    m, b = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2)
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))
    
nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [m_true, b_true], args=(x, y, yerr))

#############################################################
# MCMC Fitting
#############################################################

def lnprior(theta):
    m, b = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0:
        return 0.0
    return -np.inf

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

ntemps,ndim, nwalkers = 5, 2, 100
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
pos = np.reshape(np.tile(pos,ntemps),(ntemps,nwalkers,ndim))

#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr),threads=7)
nthreads = 2
sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnlike, lnprior,loglargs=[x,y,yerr])

sampler.run_mcmc(pos, 100)

samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
chains = sampler.flatchain


mmed = np.median(chains[:,:,0])
bmed = np.median(chains[:,:,1])

xmc = np.linspace(0,10,1000)
ymc = mmed*xmc + bmed

plt.plot(xmc,ymc)

fig = corner.corner(samples, labels=["$m$", "$b$"],
                      truths=[m_true, b_true])