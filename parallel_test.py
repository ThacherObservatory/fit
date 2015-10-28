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
N = 50
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

def run_mcmc(ntemps=3,nwalkers=100,nsteps=100, graph=False):
    ndim = 2
    pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    pos = np.reshape(np.tile(pos,ntemps),(ntemps,nwalkers,ndim))

    sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnlike, lnprior,loglargs=[x,y,yerr])
    sampler.run_mcmc(pos,nsteps)
    sampler.reset()
    sampler.run_mcmc(pos, nsteps)

    chains = sampler.flatchain
    samples = sampler.chain.reshape(ntemps*10000,2)
    
    
    mmed = np.median(chains[:,:,0])
    mstd = np.std(chains[:,:,0])
    bmed = np.median(chains[:,:,1])
    bstd = np.std(chains[:,:,1])
    
    if graph:
        xmc = np.linspace(0,10,1000)
        ymc = mmed*xmc + bmed
        
        plt.plot(xmc,ymc)
        
        corner.corner(samples, labels=["$m$", "$b$"],
                              truths=[m_true, b_true])
    
    return mstd, bstd, mmed, bmed

def test_temps():
    temps = range(1,10)
    mstds = []
    mmeds = []
    bstds = []
    bmeds = []
    for n in temps:
        ms,bs,mm,bm = run_mcmc(ntemps=n)
        mstds.append(ms)
        bstds.append(bs)
        mmeds.append(mm)
        bmeds.append(bm)
    plt.plot(temps,mstds)
    plt.plot(temps,bstds)
    plt.figure()
    plt.plot(temps,mmeds)
    plt.plot(temps,bmeds)
    
def test_walkers():
    walkers = range(4,1000,10)
    mstds = []
    mmeds = []
    bstds = []
    bmeds = []
    for n in walkers:
        ms,bs,mm,bm = run_mcmc(nwalkers=n)
        mstds.append(ms)
        bstds.append(bs)
        mmeds.append(mm)
        bmeds.append(bm)
    plt.plot(walkers,mstds)
    plt.plot(walkers,bstds)
    plt.figure()
    plt.plot(walkers,mmeds)
    plt.plot(walkers,bmeds)

def test_steps():
    steps = range(1,1000,100)
    mstds = []
    mmeds = []
    bstds = []
    bmeds = []
    for n in steps:
        ms,bs,mm,bm = run_mcmc(nsteps=n)
        mstds.append(ms)
        bstds.append(bs)
        mmeds.append(mm)
        bmeds.append(bm)
    plt.plot(steps,mstds)
    plt.plot(steps,bstds)
    plt.figure()
    plt.plot(steps,mmeds)
    plt.plot(steps,bmeds)
