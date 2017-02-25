import kplr, emcee, george, corner, pickle
from george.kernels import ExpSine2Kernel, ExpSquaredKernel
import ebsim as ebs
import ebsim_results as ebr
import numpy as np
import robust as rb
import constants as c
import matplotlib.pyplot as plt
import collections as col
from astropy.io.fits import getdata
from plot_params import *
import sys
import sigmaRejection as sr

# From: /Users/jonswift/Astronomy/EBs/outdata/9821078/Refine/9821078_short.out
period = 8.429434002
t1 = 2454838.85109817
dur1 = 3.46998 / 24.0
t2 = 2454843.06965465
dur2 = 3.41664 / 24.0
durfac = 1.3
keepfac = 5.0

# Rotation period from 9821078_rot.png
rotp = 9.87


######################################################################
# Fit out-of-eclipse light first using the entire dataset
######################################################################
client = kplr.API()

star = client.star(9821078)

print 'Getting light curves...'
lcs = star.get_light_curves(fetch=True)  
alltimes = []
allfluxes = []
allerrs = []

print 'Extracting data...'
for lc in lcs:
    if lc.params['ktc_target_type'] == 'SC':
        with lc.open() as f:
            data = f[1].data
            t = data['time']
            f = data['PDCSAP_FLUX']
            e = data['PDCSAP_FLUX_ERR']
            alltimes = np.append(alltimes,t)
            allfluxes = np.append(allfluxes,f)
            allerrs =  np.append(allerrs,e)

alltimes += 2454833.0
real, = np.where((np.isfinite(alltimes)) & (np.isfinite(allfluxes)) & (np.isfinite(allerrs)))
alltimes = alltimes[real] ; allfluxes = allfluxes[real] ; allerrs = allerrs[real]

# Some SC data have long term trends likely due to the target moving across the CCD,
# There are otherwise some gaps and jumps sprinkled throughout the SC dataset.
# For now, choose relatively clean region of 2 eclipse pairs and work from there. 
inds, = np.where((alltimes >  2455570.0) & (alltimes < 2455586.0))

times = alltimes[inds][::30] ; fluxes = allfluxes[inds][::30] ; errs = allerrs[inds][::30]

plt.ion()
plt.figure(1)
plt.clf()
plt.plot(alltimes,allfluxes,'bo',alpha=0.1,label='All Data')
plt.plot(times,fluxes,'r.',label='Selected Data')
plt.legend(loc='best',numpoints=1)

ph1 = (times-t1) % period
ph2 = (times-t2) % period

plt.figure(2)
plt.clf()
plt.plot(ph1,fluxes,'k.',label='Phase Folded Flux')

# This will only work if t2 is greater than t1
dt12 = (t2-t1) % period
dt21 = (t1+period-t2) % period

allooei1, = np.where( (ph1 > dur1/2.0*durfac) & (ph1 < (dt12 - dur2/2.0*durfac)))
allooei2, = np.where( (ph2 > dur2/2.0*durfac) & (ph2 < (dt21 - dur1/2.0*durfac)))

plt.plot(ph1[allooei1],fluxes[allooei1],'r.',label='OOE1')
plt.plot(ph1[allooei2],fluxes[allooei2],'g.',label='OOE2')

allooei = np.append(allooei1,allooei2)

plt.plot(ph1[allooei],fluxes[allooei],'bo',alpha=0.05,label='OOE')
plt.legend(loc='best',numpoints=1)



med = np.median(fluxes[allooei])
plt.figure(3)
plt.clf()
plt.plot(times[allooei],fluxes[allooei]/med,'.k',label='Normalized OOE Flux')
plt.axhline(y=1,ls='--',color='r')
plt.legend(loc='best',numpoints=1)

k =  100**2 * ExpSquaredKernel(20) * ExpSine2Kernel(rotp,20)
tooe = times[allooei]
fooe = fluxes[allooei]/med
gp = george.GP(k,mean=np.mean(fooe))
gp.compute(tooe,2,sort=True)
mu, cov = gp.predict(fooe, times)

plt.figure(4)
plt.clf()
plt.plot(tooe,fooe,'ko',label='Normalized OOE Flux')
plt.plot(times,mu,'r.',label='GP Model')
plt.legend(loc='best',numpoints=1)

muooe, covooe = gp.predict(fooe, tooe)
plt.figure(5)
plt.clf()
res = fooe-muooe
plt.plot(tooe,res,'ko',label='Residuals: Flux - OOE Model')
sig = np.std(res,ddof=1)
plt.legend(loc='best',numpoints=1)

plt.figure(6)
plt.clf()
plt.hist(res,bins=50)
plt.title('Histogram of Residuals')
from statsmodels.stats.diagnostic import normal_ad
d,p = normal_ad(res)
print 'P-value for Anderson-Darling test on residuals = %.3f%%' %  (p*100.0)
print 'Std. of Histogram = %.2f ppm' % (sig*1e6)
print 'Median of reported errors = %.2f ppm' % (np.median(allerrs)/med*1e6)


# Fit to OOE looks good. Use these parameters for GP kernel!
# Now select out data within 2.5 durations of each eclipse.

finaltimes = alltimes[inds] ; finalfluxes = allfluxes[inds]/med ; finalerrs = allerrs[inds]/med
plt.figure(7)
plt.clf()
plt.plot(finaltimes,finalfluxes,'ko',label='Selected Data')
ph1 = (finaltimes-t1) % period
ph2 = (finaltimes-t2) % period
primi1, =  np.where( ph1 < (dur1*keepfac/2.0) )
primi2, =  np.where( ph1 > (period - dur1*keepfac/2.0) )
primi = np.append(primi1,primi2)
plt.plot(finaltimes[primi],finalfluxes[primi],'r.',label='Primary Eclipses')

seci1, =  np.where( ph2 < (dur2*keepfac/2.0) )
seci2, =  np.where( ph2 > (period - dur2*keepfac/2.0) )
seci   = np.append(seci1,seci2)

plt.plot(finaltimes[seci],finalfluxes[seci],'g.',label='Secondary Eclipses')
plt.legend(loc='best',numpoints=1)

ooe1i, = np.where( (ph1 < (dur1*keepfac/2.0)) & (ph1 > (dur1/2.0*durfac)) )
ooe2i, = np.where( (ph1 < (period - dur1/2.0*durfac)) & (ph1 > (period - dur1*keepfac/2.0)) )
ooei1 = np.append(ooe1i,ooe2i)
ooe3i, = np.where( (ph2 < (dur2*keepfac/2.0)) & (ph2 > (dur2/2.0*durfac)) )
ooe4i, = np.where( (ph2 < (period - dur2/2.0*durfac)) & (ph2 > (period - dur2*keepfac/2.0)) )
ooei2 = np.append(ooe3i,ooe4i)

ooei = np.append(ooei1,ooei2)
ooetimes = finaltimes[ooei] ; ooefluxes = finalfluxes[ooei] ; ooeerrs = finalerrs[ooei]
ots = np.argsort(ooetimes)
ooetimes = ooetimes[ots] ; ooefluxes = ooefluxes[ots] ; ooeerrs = ooeerrs[ots]

alli = np.append(primi,seci)
times = finaltimes[alli] ; fluxes = finalfluxes[alli]; errs = finalerrs[alli]
ts = np.argsort(times)
times = times[ts] ; fluxes = fluxes[ts] ; errs = errs[ts]

plt.figure(7)
plt.clf()
plt.plot(times,fluxes,'ko',label='Selected Data')
plt.plot(ooetimes,ooefluxes,'r.',label='OOE Data')
plt.legend(loc='best',numpoints=1)


# Reject outliers (flares) in one iteration of sigma rejection
gp = george.GP(k,mean=np.mean(ooefluxes))
gp.compute(ooetimes,2,sort=True)
mu, cov = gp.predict(ooefluxes, times)
plt.figure(8)
plt.clf()
plt.plot(times,fluxes,'.k',label='Final Data')
plt.plot(times,mu,'r.',label='GP Prediction for OOE Flux')
plt.legend(loc='best',numpoints=1)


muooe, covooe = gp.predict(ooefluxes, ooetimes)
plt.figure(9)
plt.clf()
res = ooefluxes-muooe
plt.plot(ooetimes,res,'r.',label='Outliers')
#good = sr.sigmaRejection(res,indices=True,m=2.0)
good, = np.where(res < np.abs(np.min(res)))
plt.plot(ooetimes[good],res[good],'k.',label='Good Data')
plt.legend(loc='best',numpoints=1)


gp = george.GP(k,mean=np.mean(ooefluxes[good]))
gp.compute(ooetimes[good],2,sort=True)
mu, cov = gp.predict(ooefluxes[good], times)
plt.figure(10)
plt.clf()
plt.plot(times,fluxes,'.k',label='Final Data')
plt.plot(times,mu,'r.',label='GP Prediction for OOE Flux')
plt.legend(loc='best',numpoints=1)
plt.title('After Sigma Rejection')


# Create vector of times that correspond to the proper time sampling
# used by compute_eclipse
int = 6.019802903270
read = 0.518948526144
integration = int*9.0 + read*8.0

modelfac = 5

tdarr = ebs.get_time_stack(times,integration=integration,modelfac=modelfac)
ooe1_model = np.zeros_like(tdarr)
for i in range(np.shape(tdarr)[0]):
    ooe1_model[i,:],cov = gp.predict(ooefluxes[good],tdarr[i,:])

plt.figure(11)
plt.clf()
plt.plot(times,fluxes,'.k',label='Final Data')
plt.plot(tdarr,ooe1_model,'.')
plt.legend(loc='best',numpoints=1)
plt.title('Final OOE Prediction')

ooedata = np.append([ooetimes],[ooefluxes],axis=0)
ooedata = np.append(ooedata,[ooeerrs],axis=0)

GP_dict = {'ooe':ooedata,'ooe_predict':(tdarr,ooe1_model)}

pickle.dump( GP_dict, open( "9821078_GP.p", mode="wb" ) )

