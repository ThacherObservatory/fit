import kplr, emcee, george, corner
from george.kernels import ExpSine2Kernel, ExpSquaredKernel
import pdb,sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

debug = False

client = kplr.API()

star = client.star(10935310)
#star = client.star(4175707)
#star = client.star(11913210)

t0 = 2454957.32211 - 2454833.0
period =4.1287977964 

print('Get LCs...')
lcs = star.get_light_curves(fetch=True)

times = []
fluxes = []
errs = []

print('Extract data...')
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
i = np.where((times[f] > 540) & (times[f] < 620))
#i = np.where((times[f] > 353) & (times[f] < 360))
#i = np.where((times[f] > 292) & (times[f] < 300))
time = times[f][i] ; flux = fluxes[f][i] ; err = errs[f][i]
bad, = np.where((time>295.5) & (time<295.6))
bad.tolist()
time = np.delete(time,bad)
flux = np.delete(flux,bad)
err = np.delete(err,bad)

phase = ((time - t0) % period)/period
ooei1, = np.where((phase > 0.02) & (phase < 0.48))
ooei2, = np.where((phase > 0.52) & (phase < 0.98))
ooei = np.sort(np.append(ooei1,ooei2))
time = time[ooei] ; flux = flux[ooei]; err=err[ooei]


plt.ion()
plt.figure(1)
plt.clf()
plt.subplot(2,1,1)
plt.plot(time,flux,'ko',markersize=2)
plt.xlabel('Time (BKJD)')
plt.ylabel('Flux (ADU)')

k =  100**2 * ExpSquaredKernel(1) * ExpSine2Kernel(4,4)
#k =  300**2 * ExpSquaredKernel(2)
gp = george.GP(k,mean=np.mean(flux))

if debug:
    gp.compute(time,2,sort=True)
    x = np.linspace(np.min(time), np.max(time), 5000)
    mu, cov = gp.predict(flux, x)
    plt.figure(1)
    plt.plot(x,mu,'r-')
    flux_fit, cov_fit = gp.predict(flux,time)
    plt.subplot(2,1,2)
    plt.plot(time,flux_fit-flux,'ko')
    plt.axhline(y=0,linestyle='-',color='red',lw=3)
    plt.xlabel('Time (BKJD)')
    plt.ylabel('Residuals (ADU)')
    plt.suptitle('KIC 10935310 (eclipses blanked)')
    plt.savefig('KIC10935310_oee.png',dpi=300)
    sys.exit()
    


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
    if np.exp(theta[3]) < 2 or np.exp(theta[3]) > 6:
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
#p0 = p0[0:3]
nwalkers = 20
burnsteps = 100
mcmcsteps = 100
ndim = len(p0)

# drop the MCMC hammer, yo.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time,flux,err))#,threads=3)

p0_vec = [np.abs(p0[i])+1e-3*np.random.randn(nwalkers) for i in range(ndim)]
p0_init = np.array(p0_vec).T

pos,prob,state = sampler.run_mcmc(p0_init, burnsteps)

#plt.figure(2)
#plt.clf()
"""
for i in range(nwalkers):
    plt.subplot(2,2,1)
    plt.plot(sampler.chain[i,:,0])
    plt.title('Log Amp')
for i in range(nwalkers):
    plt.subplot(2,2,2)
    plt.plot(np.exp(sampler.chain[i,:,1]))
    plt.title('Sine Amp')
for i in range(nwalkers):
    plt.subplot(2,2,3)
    plt.plot(np.exp(sampler.chain[i,:,2]))
    plt.title('Period')
for i in range(nwalkers):
    plt.subplot(2,2,4)
    plt.plot(sampler.chain[i,:,3])
    plt.title('Decay')
plt.suptitle('Burn-in',fontsize=18)
"""
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos,mcmcsteps)
"""
plt.figure(3)
plt.clf()
for i in range(nwalkers):
    plt.subplot(2,2,1)
    plt.plot(sampler.chain[i,:,0])
    plt.title('Log Amp')
for i in range(nwalkers):
    plt.subplot(2,2,2)
    plt.plot(np.exp(sampler.chain[i,:,1]))
    plt.title('Sine Amp')
for i in range(nwalkers):
    plt.subplot(2,2,3)
    plt.plot(np.exp(sampler.chain[i,:,2]))
    plt.title('Period')
for i in range(nwalkers):
    plt.subplot(2,2,4)
    plt.plot(sampler.chain[i,:,3])
    plt.title('Decay')
plt.suptitle('Final',fontsize=18)
"""
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
