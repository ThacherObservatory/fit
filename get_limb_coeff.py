import numpy as np
import math
import constants as c
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import pylab as pl



def get_limb_coeff(Tstar,loggstar,filter='Kp',plot=0,network=None,limb='quad',interp='linear'):


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
                
    if network == 'gps':
        path = '/home/jswift/Mdwarfs/'
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

    idata = np.array(np.where((filt == filter) & (method == 'L')))
    
    npts = idata.size

    locs = np.zeros(2*npts).reshape(npts,2)
    locs[:,0] = Teff[idata].flatten()
    locs[:,1] = logg[idata].flatten()
    
    vals = np.zeros(npts)
    vals[:] = avec[idata]

    eval = np.array([Tstar,loggstar]).reshape(1,2)

    aval = griddata(locs,vals,eval,method=interp)
    

    if plot == 1:
#        pl.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Teff[idata], logg[idata], avec[idata], color='b')
        ax.plot(np.array([Tstar]),np.array([loggstar]),aval,'o',color='r')
        plt.show()

    vals = np.zeros(npts)
    vals[:] = bvec[idata]

    bval = griddata(locs,vals,eval,method=interp)


    if plot == 1:
        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')
        ax.scatter(Teff[idata], logg[idata], bvec[idata], color='b')
        ax.plot(np.array([Tstar]),np.array([loggstar]),bval,'o',color='r')
        plt.show()


    vals = np.zeros(npts)
    vals[:] = cvec[idata]

    cval = griddata(locs,vals,eval,method=interp)


    if plot == 1:
        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')
        ax.scatter(Teff[idata], logg[idata], cvec[idata], color='b')
        ax.plot(np.array([Tstar]),np.array([loggstar]),cval,'o',color='r')
        plt.show()

    vals = np.zeros(npts)
    vals[:] = dvec[idata]

    dval = griddata(locs,vals,eval,method=interp)


    if plot == 1:
        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')
        ax.scatter(Teff[idata], logg[idata], dvec[idata], color='b')
        ax.plot(np.array([Tstar]),np.array([loggstar]),dval,'o',color='r')
        plt.show()

    if limb == 'quad':
        return aval[0], bval[0]

    if limb == 'sqrt':
        return cval[0], dval[0]

    if limb == 'nlin':
        return aval[0], bval[0], cval[0], dval[0]
