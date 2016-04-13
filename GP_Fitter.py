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

time = time[8000:8800]
flux = flux[8000:8800]
err = err[8000:8800]

print("starting...")

plt.ion()
plt.figure(1)

def test_params(amp=False,exp=False,gamma=False,period=False):

    if amp:
        ampvec   = np.array([0.01,0.1,1,10,100,1000])
        vec = ampvec
    elif exp:
        expsqvec = np.array([1,10,50,100,500,1000])
        vec = expsqvec

    elif gamma:
        gammavec = np.array([0.1,1,10,50,100,500])
        vec = gammavec

    elif period:
        pvec     = np.array([1,3,7,10,30,70,100])
        vec = pvec
    
    for i in range(len(vec)):

        if amp:
            k =  ampvec[i]**2 * ExpSquaredKernel(10) * ExpSine2Kernel(1.0,4) 
        elif exp:
            k =  1**2 * ExpSquaredKernel(expsqvec[i]) * ExpSine2Kernel(1.0,4)
        elif gamma:
            k =  1**2 * ExpSquaredKernel(10) * ExpSine2Kernel(gammavec[i],4)
        elif period:
            k =  1**2 * ExpSquaredKernel(10) * ExpSine2Kernel(1.0,pvec[i])

        gp = george.GP(k,mean=np.mean(flux),solver=george.HODLRSolver)
        v = gp.kernel.vector
        print(v)

        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(time,flux,'ko',markersize=5)
        plt.xlabel('Time (BKJD)')
        plt.ylabel('Flux (Normalized)')

        #plot gp prediction
        gp.compute(time,yerr=err,sort=True)        
        flux_fit, cov_fit = gp.predict(flux, time)
        plt.plot(time,flux_fit,'c.')
        t = np.linspace(np.min(time),np.max(time),10000)
        f_f, c_f = gp.predict(flux,t)
        plt.plot(t,f_f,'r.')
        
        plt.subplot(2,1,2)
        plt.plot(time,flux-flux_fit,'ko')
        plt.axhline(y=0,linestyle='-',color='red',lw=3)
        plt.xlabel('Time (BKJD)')
        plt.ylabel('Residuals')
        lnlike = -gp.lnlikelihood(flux)

        if amp:
            plt.suptitle('Amp = '+str(ampvec[i])+'.   ln(prob) = '+str(lnlike))
            plt.savefig('Amp='+str(ampvec[i])+'.png',dpi=150)
        elif exp:
            plt.suptitle('ExpSq = '+str(expsqvec[i])+'.   ln(prob) = '+str(lnlike))
            plt.savefig('ExpSq='+str(expsqvec[i])+'.png',dpi=150)
        elif gamma:
            plt.suptitle('Gamma = '+str(gammavec[i])+'.   ln(prob) = '+str(lnlike))
            plt.savefig('Gamma='+str(gammavec[i])+'.png',dpi=150)
        elif period:
            plt.suptitle('Period = '+str(pvec[i])+'.   ln(prob) = '+str(lnlike))
            plt.savefig('Period='+str(pvec[i])+'.png',dpi=150)
    return

#test_params(gamma=True)
#sys.exit()

# Notes:
##############################
# Amp
# Overall amplitude of the correlations
# 0.01 to 10

# ExpSq:
# Taper on semi-periodic envelope
# 1-100

# Gamma:
# Width of each semi-periodic peak. Can't be too big, else will over fit noise.
# 

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
k1 = 0.1**2 * ExpSquaredKernel(5) * ExpSine2Kernel(1,4)
k2 = 0.01**2 * ExpSquaredKernel(0.5) * ExpSine2Kernel(2,.001)
k = k1 + k2
#gp = george.GP(k1,mean=np.mean(flux))
gp = george.GP(k1,mean=np.mean(flux),solver=george.HODLRSolver)
print(gp.kernel.vector)

gp.compute(time,yerr=err,sort=True)
flux_fit, cov_fit = gp.predict(flux, time)

#plot gp prediction
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(time,flux_fit,'c.')

plt.subplot(2,1,2)
plt.plot(time,flux-flux_fit,'ko')
plt.axhline(y=0,linestyle='-',color='red',lw=3)
plt.xlabel('Time (BKJD)')
plt.ylabel('Residuals (ADU)')


def lnprob(theta,time,flux,err):
    """log likelihood function for MCMC"""
    #theta[0] = amplitude
    #theta[1] = width
    #theta[2] = width of semi periodic kernel
    #theta[3] = period

    # theta is actually the natural log of the input parameters
    # (amp) 

#    gp.kernel[:] = np.array([theta[0],theta[1],theta[2],theta[3]])
    gp.kernel[:] = theta #np.arrinstallay([theta[0],theta[1],theta[2],theta[3]])

    pprior =  (np.exp(theta[3]) - 3.99)**2/(2*0.2**2)


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
#        gp.compute(time,yerr=err,sort=True)
        gp.compute(time,sort=True)
    except (ValueError, np.linalg.LinAlgError):
        print 'Oh, no!'
        return -np.inf
    
#    flux_fit, cov_fit = gp.predict(flux, time)
#    print  -np.sum((flux_fit-flux)**2/(2.0*err**2)) # - (np.exp(theta[3]) - 3.99)**2/(2*0.2**2))
    loglike = -gp.lnlikelihood(flux, quiet=True) - pprior

    return loglike
  
#display initial log probability before MCMC  
gp.compute(time,yerr=err,sort=True)
print(-gp.lnlikelihood(flux))

#Initialize the MCMC Hammer
p0 = gp.kernel.vector
nwalkers = 50
burnsteps = 500
mcmcsteps = 500
ndim = len(p0)
p0_vec = [p0[i]+1*np.random.randn(nwalkers) for i in range(ndim)]
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

# extract maximum likelihood value
maxlnprob = np.max(sampler.flatlnprobability)
maxind, = np.where(sampler.flatlnprobability == maxlnprob)
print 'Max ln prob = %f.5' % maxlnprob
maxind = maxind[0]

#extract max like values from chains for use in GP visualization
amp = np.exp(sampler.flatchain[maxind,0]/2.0)
expsq = np.exp(sampler.flatchain[maxind,1])
gamma = np.exp(sampler.flatchain[maxind,2])
period = np.exp(sampler.flatchain[maxind,3])


#apply maximum likelihood values to GP
fit =  np.array( [sampler.flatchain[maxind,i] for i in range(len(p0))] )
gp.kernel[:] = fit
gp.compute(time,yerr=err,sort=True)
flux_fit, cov_fit = gp.predict(flux, time)

#report lnlikelihood post-MCMC
checkln = -gp.lnlikelihood(flux)
print 'Check ln prob = %f.5' % checkln


#plot gp prediction
plt.figure(2)
plt.clf()
plt.subplot(2,1,1)
plt.plot(time,flux,'k.')
plt.plot(time,flux_fit,'c.')

#plot residuals
plt.subplot(2,1,2)
plt.plot(time,flux_fit-flux,'ko')
plt.axhline(y=0,linestyle='-',color='red',lw=3)
plt.xlabel('Time (BKJD)')
plt.ylabel('Residuals (ADU)')
plt.savefig("10935310_ooe_fit.png",dpi=300)

#generate corner plot of gp parameters
plt.figure(3)
plt.clf()
samples = sampler.chain.reshape((-1, ndim))
figure = corner.corner(samples, labels=["$T_1$","$T_2$","$T_3$","$T_4$"])
figure.savefig("gp_test_corner.png",dpi=300)






# Put in progress bar for emcee!

#for i, (pos, prob, state) in enumerate(sampler.run_mcmc(p0_init, burnsteps)):
#    if (i+1) % 1 == 0:
#        print("{0:.1f}%".format(100 * float(i) / burnsteps))
#sys.exit()
#print(gp.kernel.vector)

#kfinal =  amp**2 * ExpSquaredKernel(expsq) * ExpSine2Kernel(gamma,period)
#gp = george.GP(kfinal,mean=np.mean(flux),solver=george.HODLRSolver)
#print(gp.kernel.vector)

#calculate gp prediction for each data point
#gp.compute(time,yerr=err,sort=True)
#flux_fit, cov_fit = gp.predict(flux, time)

