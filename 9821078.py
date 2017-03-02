import kplr, emcee, george, corner, sys, pickle
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

plot = False
bellerophon = True
debug = False
threads = 32
do_ooe = True
over_disperse = False
clobber = False
nw = 500
bs = 10000
mcs = 10000

# Create vector of times that correspond to the proper time sampling
# used by compute_eclipse
int = 6.019802903270
read = 0.518948526144
integration = int*9.0 + read*8.0
modelfac = 5

keepfac = 5.0
durfac = 1.3

# From: /Users/jonswift/Astronomy/EBs/outdata/9821078/Refine/9821078_short.out
# Refined from 2017Feb26 run
#period = 8.429434002
period = 8.428971039
#t1 = 2454838.85109817
t1 = 2454838.89275878
dur1 = 3.46998 / 24.0
#t2 = 2454843.06965465
t2 = 2454843.084555870
dur2 = 3.41664 / 24.0

# Rotation period from 9821078_rot.png
rotp = 9.87

######################################################################
# Photometry data
# datadict['phot0'].keys()# ['ooe', 'light', 'integration', 'band', 'L3', 'limb']
# np.shape(ooe) = (3,npts) => JD, normphot, err (just ooe data)
# np.shape(light) = (3,npts) => JD, normphot, err (all data)
# 'integration' = integration time in seconds
# 'band' = band name (string)
# 'L3' = third light fraction
# 'limb' = 'quad'

######################################################################
# Kepler Data (optical)
######################################################################
if bellerophon:
    dpath = '/home/administrator/Astronomy/EBs/KIC9821078/data/'
    outpath = '/home/administrator/Astronomy/EBs/KIC9821078/'
else:
    dpath = '/Users/jonswift/Astronomy/EBs/outdata/9821078/Refine/'
    outpath = '/Users/jonswift/Astronomy/EBs/outdata/9821078/MCMC/2017Mar01_short/'

phot0 = pickle.load( open( "9821078_GP.p", "rb" ) )

phot0['integration'] =  integration

phot0['limb'] = 'quad'

phot0['band'] = 'Kp'


######################################################################
# RV data
# datadict.keys()
# ['RVdata', 'phot0', 'phot1', 'phot2', 'phot3']
# datadict['RVdata'].keys()
# ['rv2', 'rv1']
# Each rv1 and rv2 has dimensions (3,npts)
# JD, RV, RVerr

if bellerophon:
    dpath = '/home/administrator/Astronomy/EBs/KIC9821078/data/'
else:
    dpath = '/Users/jonswift/Astronomy/EBs/outdata/9821078/RVs/'

file1 = '9821078_comp1.dat'
file2 = '9821078_comp2.dat'

rv1 = np.loadtxt(dpath+file1,usecols=[0,1,2]).T
rv2 = np.loadtxt(dpath+file2,usecols=[0,1,2]).T

# If errors are nan, replace with largest error.
if np.sum(np.isnan(rv2[2,:])) >= 1:
    rv2[2,np.isnan(rv2[2,:])] =  np.nanmax(rv2[2,:])

if np.sum(np.isnan(rv1[2,:])) >= 1:
    rv1[2,np.isnan(rv1[2,:])] =  np.nanmax(rv1[2,:])

phrv1 = (rv1[0,:]-t1)%period
phrv2 = (rv2[0,:]-t1)%period

if plot:
    plt.figure(1)
    plt.clf()
    plt.errorbar(phrv1,rv1[1,:],rv1[2,:],fmt='o',color='k',linewidth=1.5,label='Primary')
    plt.errorbar(phrv2,rv2[1,:],rv2[2,:],fmt='o',color='r',linewidth=1.5,label='Secondary')
    plt.xlim(0,period)
    plt.legend(loc='best',numpoints=1)
    
RVdata = {'rv1':rv1,'rv2':rv2}

datadict = {'RVdata':RVdata,'phot0':phot0}#,
#            'phot1':phot1,'phot2':phot2,
#            'phot3':phot3}

######################################################################


# Initial guesses
m1 = 0.6 * c.Msun ; r1 = 0.6 * c.Rsun
m2 = 0.5 * c.Msun ; r2 = 0.5 * c.Rsun
ecc = 0.0 ; omega = 0.0
period = period * 86400.0
t0=t1
sma = (period**2 * c.G * (m1 + m2) / (4 * np.pi**2))**(1.0/3.0)
impact = sma/r1 * np.cos(np.radians(89.0))
vsys=-22.0
T1 = 4000.0
l1 = 4*np.pi*r1**2*c.sb*T1**4
T2 = 3800.0
l2 = 4*np.pi*r2**2*c.sb*T2**4
J  = l2/l1


if bellerophon:
    network = 'bellerophon'
else:
    network = 'swift'

ebin = ebs.ebinput(m1=m1/c.Msun, m2=m2/c.Msun, r1=r1/c.Rsun, r2=r2/c.Rsun,
                   vsys=vsys, period=period/86400.0, t0=t0, ecc=ecc,
                   omega=omega, impact=impact)
                   

datadict = col.OrderedDict(sorted(datadict.items()))
#ebs.check_model(datadict)

ubands = ebs.uniquebands(datadict,quiet=True)

fitinfo = ebs.fit_params(nwalkers=nw,burnsteps=bs,mcmcsteps=mcs,
                         data_dict=datadict,do_ooe=[1],
                         clobber=clobber,fit_ooe1=[False],fit_L3=[True],
                         network=network,outpath=outpath,modelfac=modelfac)

ebs.ebsim_fit(datadict,fitinfo,ebin,debug=debug,threads=threads,over_disperse=over_disperse)

chains,lp = ebr.get_chains(path=outpath)
bestvals = ebr.best_vals(path=outpath,chains=chains,lp=lp)
datadict,fitinfo,ebin = ebr.get_pickles(path=outpath)
#ebr.plot_model_compare(bestvals,datadict,fitinfo,ebin,write=True,outpath=outpath)
ebr.plot_model(bestvals,datadict,fitinfo,ebin,write=True,outpath=outpath)
ebr.params_of_interest(chains=chains,lp=lp,outpath=outpath)
