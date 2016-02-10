import george
from george.kernels import ExpSine2Kernel, ExpSquaredKernel
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import kplr

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

k =  10.0**2 * ExpSquaredKernel(2.0) * ExpSine2Kernel(0.5,1)

gp = george.GP(k,mean=np.mean(flux))

def nll(theta):
    #theta[0] = amplitude
    #theta[1] = width
    #theta[2] = width of semi periodic kernel
    #theta[3] = period
    #theta[4] = noise term

    gp.kernel[:] = theta

    if np.any(np.isnan(theta)) or np.any(theta < 0):
        return np.inf
    
    try:
        gp.compute(time,4,sort=True)
    except (ValueError, np.linalg.LinAlgError):
        print('WTF!')
        return np.inf

    neglike = -gp.lnlikelihood(flux, quiet=True)
    return neglike


def grad_nll(theta):
    # Update the kernel parameters and compute the likelihood.
    gp.kernel[:] = theta
    return -gp.grad_lnlikelihood(flux, quiet=True)

gp.compute(time,4,sort=True)
print(gp.lnlikelihood(flux))

p0 = gp.kernel.vector
results = op.minimize(nll, p0,jac=grad_nll)

gp.kernel[:] = np.abs(results.x)
print(results.x)
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
