import ebsim as ebs
import numpy as np
import constants as c
import matplotlib.pyplot as plt
import collections as col
import sys

# From Cakirli 2013
m1 = 0.680 * c.Msun ; r1 = 0.613 * c.Rsun
m2 = 0.341 * c.Msun ; r2 = 0.897 * c.Rsun
ecc = 0.0 ; omega = 0.0
period = 4.12879779 * 86400.0
t0 = 2454957.3221
sma = (period**2 * c.G * (m1 + m2) / (4 * np.pi**2))**(1.0/3.0)
impact = sma/r1 * np.cos(np.radians(83.84))
vsys=-4.764
RVnoise = 5.0
RVsamples= 10
T1 = 4320.0
l1 = 4*np.pi*r1**2*c.sb*T1**4
T2 = 2750.0
l2 = 4*np.pi*r2**2*c.sb*T2**4
J  = l2/l1
network = 'swift'

nphot = 4
band = ['Kp','J','H','K']
photnoise = [0.0003,0.002,0.005,0.01]
q1a = [0.5,0.4,0.3,0.2]
q2a = [0.1,0.2,0.3,0.4]
q1b = [0.7,0.6,0.5,0.4]
q2b = [0.1,0.2,0.3,0.4]
L3  = [0.0,0.0,0.0,0.0] 
obsdur = [90.0,10.0,10.0,10.0]
inttime = [1800.0,300.,300.,300.]
durfac = [3.0,2,2,2]
RVsamples = 20
spotamp1 = [1.0, 0., 0., 0.]
spotP1 = np.pi*86400/period
spotfrac1 = [0.5,0.0,0.0,0.0]
spotbase1 = np.zeros(4)
spotamp2 = np.zeros(4)
spotfrac2 = np.zeros(4)
spotbase2 = np.zeros(4)
P1double = 0.8


ebin = ebs.ebinput(m1=m1/c.Msun, m2=m2/c.Msun, r1=r1/c.Rsun, r2=r2/c.Rsun,
                   vsys=vsys, period=period/86400.0, t0=t0, ecc=ecc,
                   omega=omega, impact=impact)
                   
                   

datadict = ebs.make_model_data(ebin,nphot=nphot,band=band,photnoise=photnoise,
                                 q1a=q1a,q2a=q2a,q1b=q1b,q2b=q2b,L3=L3,
                                 obsdur=obsdur,int=inttime,durfac=durfac,
                                 RVsamples=RVsamples,
                                 spotamp1=spotamp1, spotP1=spotP1, spotfrac1=spotfrac1,
                                 P1double=P1double, spotbase1=spotbase1,
                                 spotamp2=spotamp2, spotfrac2=spotfrac2,
                                 spotbase2=spotbase2,
                                 network=network,write=False)
                                 

datadict = col.OrderedDict(sorted(datadict.items()))

ebs.check_model(datadict)

ubands = ebs.uniquebands(datadict,quiet=True)

fitinfo = ebs.fit_params(nwalkers=100,burnsteps=100,mcmcsteps=100,clobber=True,
                         fit_ooe1=[True,False,False,False],network=network)

ebs.ebsim_fit(datadict,fitinfo,ebin,debug=True)



import sys
sys.exit()




plt.ion()
ebs.check_model(data)
time = data['light'][0,:]
flux = data['light'][1,:]
ttm1 = 86400.0*ebs.foldtime(time,t0=t0,period=period/86400.0)/period
dtb = np.diff(np.roll(ttm1,-1))
dtm = np.diff(ttm1)
dte = np.diff(np.roll(ttm1,1))
ei  = np.where((dtb < 0) & (dtm < 0))
bi  = np.where((dtm < 0) & (dte < 0))

plt.figure(3)
plt.clf()
#phase = 86400.0*ebs.foldtime(time,t0=t0,period=period/86400.0)/period
endpoint = 100
plt.plot(time[0:endpoint],flux[0:endpoint],'k.')
plt.plot(time[0:20],flux[0:20],'r-')

#calculate number of eclipses via number of big gaps in data + 1
num_eclipses = len(np.where(np.diff(time[0:endpoint]) > 1)[0]) + 1
print "there were {} eclipses in this graph".format(num_eclipses)

#problematic: breaks if data begins and stops before eclipse: for example, set endpoint to 200
