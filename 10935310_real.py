import ebsim as ebs
import numpy as np
import robust as rb
import constants as c
import matplotlib.pyplot as plt
import collections as col
from astropy.io.fits import open, getdata
import sys

######################################################################
# Photometry data
# datadict['phot0'].keys()
# ['ooe', 'light', 'integration', 'band', 'L3', 'limb']
# np.shape(ooe) = (3,npts) => JD, normphot, err (just ooe data)
# np.shape(light) = (3,npts) => JD, normphot, err (all data)
# 'integration' = integration time in seconds
# 'band' = band name (string)
# 'L3' = third light fraction
# 'limb' = 'quad'

######################################################################
# Kepler Data (optical)
######################################################################
dpath = '/Users/jonswift/Astronomy/EBs/outdata/10935310/Refine/'
dpath = '/home/administrator/Astronomy/EBs/KIC10935310/'
file1 = '10935310_1_norm.dat'
file2 = '10935310_2_norm.dat'

data1 = np.loadtxt(dpath+file1)
data2 = np.loadtxt(dpath+file2)

kpdata = np.append(data1,data2,axis=0)
i = np.argsort(kpdata[:,0])
data = kpdata[i,:].T
data[0,:] += 2454833.0

plt.ion()
plt.figure(0)
plt.clf()
plt.plot(data[0,:],data[1,:],'.k')

phot0 = {'light':data}

# From: /Users/jonswift/Astronomy/EBs/outdata/10935310/Refine/10935310.out
period = 4.128795073
t1 = 2454957.32213430
dur1 = 2.62899 / 24.0
t2 = 2454959.38180266
dur2 = 2.78933 / 24.0
durfac = 1.1

ph1 = (data[0,:]-t1) % period
ph2 = (data[0,:]-t2) % period

ooei1, = np.where( (ph1 > dur1/2.0*durfac) & (ph1 < 1.0))
ooei2, = np.where( (ph1 < period-(dur1/2.0*durfac)) & (ph1 > 3.0))
ooei3, = np.where( (ph2 > dur2/2.0*durfac) & (ph2 < 1.0))
ooei4, = np.where( (ph2 < period-(dur2/2.0*durfac)) & (ph2 > 3.0))

ooeiprim = np.append(ooei1,ooei2)
ooeisec  = np.append(ooei3,ooei4)

ooei = np.append(ooeiprim,ooeisec)
plt.figure(0)
plt.clf()
plt.plot(ph1,data[1,:],'.k')
plt.plot(ph1[ooei],data[1,ooei],'.r')

phot0['ooe'] = data[:,ooei]

int = 6.019802903270
read = 0.518948526144

phot0['integration'] =  integration = int*270.0 + read*269.0

phot0['L3'] = 0.02729

phot0['limb'] = 'quad'

phot0['band'] = 'Kp'


######################################################################
# Mimir Data (near infrared)
######################################################################

dpath = '/Users/jonswift/Astronomy/EBs/KIC10935310/'
dpath = '/home/administrator/Astronomy/EBs/KIC10935310/'
file = 'KIC 10935310_UT2014Jul30.fits'
data, header = getdata(dpath+file, 0, header=True)

# From header
jd = data[0][0]
airmass = data[0][1]
band = data[0][2]
ref_flux = data[0][3]
ref_flux_tot = data[0][4]
target_flux = data[0][5]
cal_flux = data[0][6]

# Exclude weird data in one integration sequence
jind, = np.where((band == 'J') & (np.isfinite(cal_flux) == True) &
                 (np.isfinite(jd) == True) &
                 ((jd < 2456868.874) ^ (jd > 2456868.876)))
hind, = np.where((band == 'H') & (np.isfinite(cal_flux) == True) &
                 (np.isfinite(jd) == True))
kind, = np.where((band == 'Ks') & (np.isfinite(cal_flux) == True) &
                 (np.isfinite(jd) == True))

##############################
# J band
jdj = jd[jind]
lightj = cal_flux[jind]

ooeindj, = np.where(jdj < 2456868.892)
ooej = lightj[ooeindj]
tooej = jdj[ooeindj]

norm = np.nanmedian(ooej)

lightj /= norm
ooej /= norm
errj = np.zeros(len(lightj))+rb.std(ooej)
errooej = np.zeros(len(ooej))+rb.std(ooej)

plt.ion()
plt.figure(1)
plt.clf()
plt.plot(jdj,lightj,'.k')
plt.plot(tooej,ooej,'.r')

light = np.array([jdj,lightj,errj])

phot1 = {'light':light}

ooe = np.array([tooej,ooej,errooej])

phot1['ooe'] = ooe

phot1['integration'] = 8.0

phot1['band'] = 'J'

phot1['limb'] = 'quad'

phot1['L3'] = 0.0802

##############################
# H band
jdh = jd[hind]
lighth = cal_flux[hind]

ooeindh, = np.where(jdh < 2456868.897)
ooeh = lighth[ooeindh]
tooeh = jdh[ooeindh]

norm = np.nanmedian(ooeh)

lighth /= norm
ooeh /= norm
errh = np.zeros(len(lighth))+rb.std(ooeh)
errooeh = np.zeros(len(ooeh))+rb.std(ooeh)

plt.ion()
plt.figure(2)
plt.clf()
plt.plot(jdh,lighth,'.k')
plt.plot(tooeh,ooeh,'.r')

light = np.array([jdh,lighth,errh])

phot2 = {'light':light}

ooe = np.array([tooeh,ooeh,errooeh])

phot2['ooe'] = ooe

phot2['integration'] = 8.0

phot2['band'] = 'H'

phot2['limb'] = 'quad'

phot2['L3'] = 0.0802

##############################
# K band
jdk = jd[kind]
lightk = cal_flux[kind]

ooeindk, = np.where(jdk < 2456868.899)
ooek = lightk[ooeindk]
tooek = jdk[ooeindk]

norm = np.nanmedian(ooek)

lightk /= norm
ooek /= norm
errk = np.zeros(len(lightk))+rb.std(ooek)
errooek = np.zeros(len(ooek))+rb.std(ooek)

plt.figure(3)
plt.clf()
plt.plot(jdk,lightk,'.k')
plt.plot(tooek,ooek,'.r')

light = np.array([jdk,lightk,errk])

phot3 = {'light':light}

ooe = np.array([tooek,ooek,errooek])

phot3['ooe'] = ooe

phot3['integration'] = 8.0

phot3['band'] = 'K'

phot3['limb'] = 'quad'

phot3['L3'] = 0.0802


######################################################################
# RV data
# datadict.keys()
# ['RVdata', 'phot0', 'phot1', 'phot2', 'phot3']
# datadict['RVdata'].keys()
# ['rv2', 'rv1']
# Each rv1 and rv2 has dimensions (3,npts)
# JD, RV, RVerr

dpath = '/Users/jonswift/Astronomy/EBs/outdata/10935310/RVs/'
dpath = '/home/administrator/Astronomy/EBs/KIC10935310/'
file1 = 'KIC10935310_comp1_BJD.dat'
file2 = 'KIC10935310_comp2_BJD.dat'

rv1 = np.loadtxt(dpath+file1).T
rv2 = np.loadtxt(dpath+file2).T

plt.figure(4)
plt.clf()
plt.errorbar(rv1[0,:]%period,rv1[1,:],rv1[2,:],fmt='o',color='k')
plt.errorbar(rv2[0,:]%period,rv2[1,:],rv2[2,:],fmt='o',color='r')

RVdata = {'rv1':rv1,'rv2':rv2}

datadict = {'RVdata':RVdata,'phot0':phot0,
            'phot1':phot1,'phot2':phot2,
            'phot3':phot3}

######################################################################

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
network = 'bellerophon'

#nphot = 4
#band = ['Kp','J','H','K']
#photnoise = [0.0003,0.002,0.005,0.01]
#q1a = [0.5,0.4,0.3,0.2]
#q2a = [0.1,0.2,0.3,0.4]
#q1b = [0.7,0.6,0.5,0.4]
#q2b = [0.1,0.2,0.3,0.4]
#L3  = [0.0,0.0,0.0,0.0] 
#obsdur = [90.0,10.0,10.0,10.0]
#inttime = [1800.0,300.,300.,300.]
#durfac = [3.0,2,2,2]
#RVsamples = 20
#spotamp1 = [1.0, 0., 0., 0.]
#spotP1 = np.pi*86400/period
#spotfrac1 = [0.5,0.0,0.0,0.0]
#spotbase1 = [-0.1,0.0,0.0,0.0]
#spotamp2 = np.zeros(4)
#spotfrac2 = np.zeros(4)
#spotbase2 = np.zeros(4)
#P1double = 0.8


ebin = ebs.ebinput(m1=m1/c.Msun, m2=m2/c.Msun, r1=r1/c.Rsun, r2=r2/c.Rsun,
                   vsys=vsys, period=period/86400.0, t0=t0, ecc=ecc,
                   omega=omega, impact=impact)
                   
                   

datadict = col.OrderedDict(sorted(datadict.items()))

#ebs.check_model(datadict)

ubands = ebs.uniquebands(datadict,quiet=True)

fitinfo = ebs.fit_params(nwalkers=500,burnsteps=500,mcmcsteps=1000,clobber=True,
                         fit_ooe1=[True,False,False,False],network=network,
                         outpath=dpath)

ebs.ebsim_fit(datadict,fitinfo,ebin,debug=False,threads=30)

sys.exit()
plt.ion()
ebs.check_model(datadict)

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
