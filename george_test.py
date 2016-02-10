import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import kplr
import emcee

client = kplr.API()

#star = client.star(10935310)
star = client.star(4175707)

lcs = star.get_light_curves(fetch=True)

times = []
fluxes = []
errs = []

for lc in lcs:
    with lc.open() as f:
        data = f[1].data
        t = data['time']
        f = data['PDCSAP_FLUX']
        e = data['PDCSAP_FLUX_ERR']
    
    times = np.append(times,t)
    fluxes = np.append(fluxes,f)
    errs =  np.append(errs,e)
    
# Use quasi-periodic kernel -- follow example
f = (np.isfinite(times)) & (np.isfinite(fluxes)) & (np.isfinite(errs))
#i = np.where((times[f] > 540) & (times[f] < 570))
#i = np.where((times[f] > 353) & (times[f] < 360))
i = np.where((times[f] > 325) & (times[f] < 330))
time = times[f][i] ; flux = fluxes[f][i] ; err = errs[f][i]

plt.ion()
plt.figure(1)
plt.clf()
plt.subplot(2,1,1)
plt.plot(time,flux,'ko-',markersize=8)
#plt.xlabel('Time (BKJD)')
plt.ylabel('Flux (ADU)')

k =  9.8 * ExpSquaredKernel(1) * ExpSine2Kernel(1,.5)

gp = george.GP(k,mean=np.mean(flux))

def lnprob(theta):
    #theta[0] = amplitude
    #theta[1] = width
    #theta[2] = width of semi periodic kernel
    #theta[3] = period

    gp.kernel[:] = theta

    
    try:
        gp.compute(time,4,sort=True)
    except (ValueError, np.linalg.LinAlgError):
        print('WTF!')
        return np.inf

    return gp.lnlikelihood(flux, quiet=True)

gp.compute(time,4,sort=True)
print(gp.lnlikelihood(flux))

p0 = gp.kernel.vector
nwalkers = 100
nsteps = 1000
ndim = len(p0)
#drop the MCMC hammer

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

pos,prob,state = sampler.run_mcmc(p0, nsteps)

sampler.reset()
pos, prob, state = sampler.run_mcmc(pos,1000)

print(gp.lnlikelihood(flux))

x = np.linspace(np.min(time), np.max(time), 5000)
gp.compute(time,4,sort=True)
mu, cov = gp.predict(flux, x)
plt.plot(x,mu,'r-')

flux_fit, cov_fit = gp.predict(flux,time)
#plt.plot(time,flux_fit,'r.-')

#plt.xlim(355.7,355.8)
#plt.ylim(14500,14600)

plt.subplot(2,1,2)
plt.plot(time,flux_fit-flux,'ko')
plt.axhline(y=0,linestyle='-',color='red',lw=3)
plt.xlabel('Time (BKJD)')
plt.ylabel('Residuals (ADU)')

plt.savefig('GP_4175707.png',dpi=300)
