import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import constants as c
import pdb
import scipy as sp
import time
import glob
import re    
import os
import robust as rb
from scipy.io.idl import readsav
import eb
from length import *
import pyfits as pf
from statsmodels.nonparametric.kernel_density import KDEMultivariate as KDE

def get_limb_coeff(Tstar,loggstar,filter='Kp',plot=False,network=None,limb='quad',interp='linear'):
    from scipy.interpolate import griddata
    import pylab as pl
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    from scipy.interpolate import RectBivariateSpline as bspline

    global ldc1func,ldc2func,ldc3func,ldc4func

# Account for gap in look up tables between 4800 and 5000K
    if (Tstar > 4800 and Tstar <= 4900):
        Tstar = 4800
    if (Tstar > 4900 and Tstar < 5000):
        Tstar = 5000
    
# Choose proper file to read
    if limb == 'nlin':
        skiprows = 49
        filtcol = 8
        metcol = 9
        mercol = 10
        col1 = 4
        col2 = 5
        col3 = 6
        col4 = 7
        if (Tstar <= 4800):
            file = 'Claret_cool_nlin.dat'
        if (Tstar >= 5000):
            file = 'Claret_hot_nlin.dat'
    else:
        skiprows = 58
        filtcol = 4
        metcol = 5
        mercol = 6
        col1 = 9
        col2 = 10
        col3 = 11
        col4 = 12
        if (Tstar <= 4800):
            file = 'Claret_cool.dat'
        if (Tstar >= 5000):
            file = 'Claret_hot.dat'
                
    if network == 'doug':
        path = '/home/douglas/Astronomy/Resources/'
    if network == 'astro':
        path = '/home/jswift/Mdwarfs/'
    if network == None:
        path = '/Users/jonswift/Astronomy/Exoplanets/TransitFits/'

    limbdata = np.loadtxt(path+file,dtype='string', delimiter='|',skiprows=skiprows)

    logg = limbdata[:,0].astype(np.float).flatten()
    Teff = limbdata[:,1].astype(np.float).flatten()
    Z = limbdata[:,2].astype(np.float).flatten()
    xi = limbdata[:,3].astype(np.float).flatten()
    filt = np.char.strip(limbdata[:,filtcol].flatten())
    method = limbdata[:,metcol].flatten()
    avec = limbdata[:,col1].astype(np.float).flatten()
    bvec = limbdata[:,col2].astype(np.float).flatten()
    cvec = limbdata[:,col3].astype(np.float).flatten()
    dvec = limbdata[:,col4].astype(np.float).flatten()

# Select out the limb darkening coefficients
#    inds = np.where((filt == 'Kp') & (Teff == 3000) & (logg == 5.0) & (method == 'L'))

    idata, = np.where((filt == filter) & (method == 'L'))
    
    npts = idata.size

    uTeff = np.unique(Teff[idata])
    ulogg = np.unique(logg[idata])
    
#    agrid0 = np.zeros((len(uTeff),len(ulogg)))
#    for i in np.arange(len(uTeff)):
#        for ii in np.arange(len(ulogg)):
#            ind, = np.where((Teff[idata] == uTeff[i]) & (logg[idata] == ulogg[ii]))
#            val = avec[idata[ind]]
#            if len(val) > 0:
#                agrid0[i,ii] = val[0]
#            else:
#                pass #pdb.set_trace()


    locs = np.zeros(2*npts).reshape(npts,2)
    locs[:,0] = Teff[idata].flatten()
    locs[:,1] = logg[idata].flatten()
    
    vals = np.zeros(npts)
    vals[:] = avec[idata]

    agrid = np.zeros((len(uTeff),len(ulogg)))
    for i in np.arange(len(uTeff)):
        for ii in np.arange(len(ulogg)):
            eval  = np.array([uTeff[i],ulogg[ii]]).reshape(1,2)
            val = griddata(locs,vals,eval,method='cubic')
            if len(val) > 0:
                agrid[i,ii] = val[0]
            else:
                pass #pdb.set_trace()

    ldc1func = bspline(uTeff, ulogg, agrid, kx=1, ky=1, s=0)    
    aval = ldc1func(Tstar,loggstar)[0][0]

    if plot:      
        plt.figure(1)
        plt.clf()
        plt.imshow(agrid,interpolation='none',
                   extent=[np.min(ulogg),np.max(ulogg),np.min(uTeff),np.max(uTeff)],
                   aspect=1./1000,vmin=np.min(agrid),vmax=np.max(agrid))
        plt.colorbar()

#------------------------------
# Second coefficient
#------------------------------
    vals = np.zeros(npts)
    vals[:] = bvec[idata]

    bgrid = np.zeros((len(uTeff),len(ulogg)))
    for i in np.arange(len(uTeff)):
        for ii in np.arange(len(ulogg)):
            eval  = np.array([uTeff[i],ulogg[ii]]).reshape(1,2)
            val = griddata(locs,vals,eval,method='cubic')
            if len(val) > 0:
                bgrid[i,ii] = val[0]
            else:
                pass 

    ldc2func = bspline(uTeff, ulogg, bgrid, kx=1, ky=1, s=0)
    bval = ldc2func(Tstar,loggstar)[0][0]

    if plot:      
        plt.figure(2)
        plt.clf()
        plt.imshow(bgrid,interpolation='none',
                   extent=[np.min(ulogg),np.max(ulogg),np.min(uTeff),np.max(uTeff)],
                   aspect=1./1000,vmin=np.min(bgrid),vmax=np.max(bgrid))
        plt.colorbar()

#------------------------------
# Third coefficient
#------------------------------
    vals = np.zeros(npts)
    vals[:] = cvec[idata]

    cgrid = np.zeros((len(uTeff),len(ulogg)))
    for i in np.arange(len(uTeff)):
        for ii in np.arange(len(ulogg)):
            eval  = np.array([uTeff[i],ulogg[ii]]).reshape(1,2)
            val = griddata(locs,vals,eval,method='cubic')
            if len(val) > 0:
                cgrid[i,ii] = val[0]
            else:
                pass 

    ldc3func = bspline(uTeff, ulogg, cgrid, kx=1, ky=1, s=0)
    cval = ldc3func(Tstar,loggstar)[0][0]

    if plot:      
        plt.figure(3)
        plt.clf()
        plt.imshow(cgrid,interpolation='none',
                   extent=[np.min(ulogg),np.max(ulogg),np.min(uTeff),np.max(uTeff)],
                   aspect=1./1000,vmin=np.min(cgrid),vmax=np.max(cgrid))
        plt.colorbar()


#------------------------------
# Fourth coefficient
#------------------------------

    vals = np.zeros(npts)
    vals[:] = dvec[idata]

    dgrid = np.zeros((len(uTeff),len(ulogg)))
    for i in np.arange(len(uTeff)):
        for ii in np.arange(len(ulogg)):
            eval  = np.array([uTeff[i],ulogg[ii]]).reshape(1,2)
            val = griddata(locs,vals,eval,method='cubic')
            if len(val) > 0:
                dgrid[i,ii] = val[0]
            else:
                pass 

    ldc4func = bspline(uTeff, ulogg, dgrid, kx=1, ky=1, s=0)
    dval = ldc4func(Tstar,loggstar)[0][0]

    if plot:      
        plt.figure(4)
        plt.clf()
        plt.imshow(dgrid,interpolation='none',
                   extent=[np.min(ulogg),np.max(ulogg),np.min(uTeff),np.max(uTeff)],
                   aspect=1./1000,vmin=np.min(dgrid),vmax=np.max(dgrid))
        plt.colorbar()

    if limb == 'quad':
        return aval, bval

    if limb == 'sqrt':
        return cval, dval

    if limb == 'nlin':
        return aval, bval, cval, dval



def read_iso(feh=0,alpha=0,age=5.0):

    agestr = str(np.int(np.round(age*1e3)))
    alen = len(agestr)
    if alen == 4:
        agestr = '0'+agestr

    if feh < 0:
        fstr = 'm'
    else:
        fstr = 'p'
    fval = str(np.round(np.abs(feh)*10))
    if len(fval) == 1:
        fval ='0'+fval

    if alpha < 0:
        astr = 'm'
    else:
        astr = 'p'
    aval = str(np.int(np.round(alpha)))
    
    #hardwired because of laziness
    #badboyz
    path = '/home/douglas/Astronomy/Resources/DSEP/'
    file = 'a'+agestr+'feh'+fstr+fval+'afe'+astr+aval+'.UBVRIJHKsKp'

    data = np.loadtxt(path+file)
    isodict = {'mass': data[:,1], 'teff': data[:,2], 'logg': data[:,3], 'umag': data[:,4],
               'bmag': data[:,5], 'vmag': data[:,6], 'rmag': data[:,7], 'imag': data[:,8],
               'jmag': data[:,9], 'hmag': data[:,10], 'kmag': data[:,11], 'kpmag': data[:,12],
               'd51': data[:,13]}

    return isodict



def rt_from_m(mass=0.5,feh=0,alpha=0,age=5.0):
    """
    rt_from_m:
    ----------
    Returns the radius and effective temperature for a star 
    given its mass, iron index, alpha enhancement, and age

    """

    from scipy.interpolate import interp1d
    isodict = read_iso(feh=feh,alpha=alpha,age=age)
    tfunc = interp1d(isodict["mass"],isodict["teff"],kind='linear')  
    lgfunc = interp1d(isodict["mass"],isodict["logg"],kind='linear')  

    Teff = 10.0**tfunc(mass)
    g = 10.0**lgfunc(mass)

    radius = np.sqrt(c.G*mass*c.Msun/g)/c.Rsun

    return radius,Teff


def flux2mag(flux,eflux,refmag,refflux):
    """
    flux2mag:
    ---------
    Return magnitude and error in magnitude given flux and flux error. Needs a
    zeropoint
    """

    mag  = refmag - 2.5*np.log10(flux/refflux)
    emag = abs(2.5*eflux/(refflux*np.log(10)))
               
    return mag,emag
    
    
def mag2flux(mag,emag,refmag,refflux):

    """
    mag2flux:
    ---------
    Return scaled flux and flux error given magnitude and magnitude error.
    Implicit in this routine is that a magnitude 0 source has flux of 1
    """
    flux = refflux*10.0**((refmag-mag)/2.5)
    eflux = flux*np.log(10)*emag/2.5

    return flux,eflux

