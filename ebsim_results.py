import sys,math,pdb,time,glob,re,os,eb,emcee,pickle
import numpy as np
import matplotlib.pyplot as plt
import constants as c
import scipy as sp
import robust as rb
from scipy.io.idl import readsav
from length import length
from statsmodels.nonparametric.kernel_density import KDEMultivariate as KDE
from stellar import rt_from_m, flux2mag, mag2flux
import matplotlib as mpl
from scipy.stats.kde import gaussian_kde
from plot_params import plot_params, plot_defaults
import run_ebsim as reb
import ebsim as ebs
from done_in import done_in


def analyze_run(run,network=None,thin=10,full=False):

    chains,lp = get_chains(run,network=network)
    
    best_vals(run,chains=chains,lp=lp,network=network,thin=thin)

    if full:
        params_of_interest(run,chains=chains,lp=lp,network=network)
        triangle_plot(run,chains=chains,lp=lp,network=network,thin=thin)
    
    return
    
def plot_chains(seq_num,network=None):
    chains,lp = get_chains(seq_num, network=network)
    path = reb.get_path(network=network)+str(seq_num)+'/'
    fitinfo = pickle.load( open( path+'fitinfo.p', 'rb' ) )
    nwalkers = fitinfo['nwalkers']
    nsteps = fitinfo['mcmcsteps']
    variables = fitinfo['variables'] 

    for i in range(len(variables)):
        var = variables[i]
        print 'plotting ' + var
        var_chains = np.reshape(chains[:,i],[nwalkers,nsteps])
        plt.ioff()
        plt.figure()
        plt.title(var + ' chains, run ' + str(seq_num))
        for chain in var_chains:
            plt.plot(chain)
        plt.savefig(path+var+'Chains.png')
    


def get_chains(seq_num,network=None):
    path = reb.get_path(network=network)+str(seq_num)+'/'
    ebpar   = pickle.load( open( path+'ebpar.p', 'rb' ) )
    data    = pickle.load( open( path+'data.p', 'rb' ) )
    fitinfo = pickle.load( open( path+'fitinfo.p', 'rb' ) )
    
    nsamp = fitinfo['nwalkers']*fitinfo['mcmcsteps']

    variables = fitinfo['variables']
    for i in np.arange(len(variables)):
        try:
            print "Reading MCMC chains for "+variables[i]
            tmp = np.loadtxt(path+variables[i]+'chain.txt')
            if i == 0:
                chains = np.zeros((len(tmp),len(variables)))
                
            chains[:,i] = tmp
        except:
            print variables[i]+'chain.txt does not exist on disk !'

    try:
        print "Reading ln(prob) chain"
        lp = np.loadtxt(path+'lnprob.txt')
    except:
        print 'lnprob.txt does not exist. Exiting'
        return

    return chains, lp



def best_vals(seq_num,chains=False,lp=False,network=None,bindiv=20.0,
                thin=False,frac=0.001,nbins=100,rpmax=1,durmax=10,sigrange=5.0):


    """
    ----------------------------------------------------------------------
    best_vals:
    ---------
    Find the best values from the 1-d posterior pdfs of a fit to a single
    primary and secondary eclipse pair
    ----------------------------------------------------------------------
    """

    plot_params(linewidth=1.5,fontsize=12)

    path = reb.get_path(network=network)+str(seq_num)+'/'
    
    ebpar   = pickle.load( open( path+'ebpar.p', 'rb' ) )
    data    = pickle.load( open( path+'data.p', 'rb' ) )
    fitinfo = pickle.load( open( path+'fitinfo.p', 'rb' ) )
    
    nsamp = fitinfo['nwalkers']*fitinfo['mcmcsteps']

    variables = fitinfo['variables']

    var_init = ebpar['variables']
    p_init = ebpar['p_init']
    
    # Use supplied chains or read from disk
    if not np.shape(chains):
        for i in np.arange(len(variables)):
            try:
                print "Reading MCMC chains for "+variables[i]
                tmp = np.loadtxt(path+variables[i]+'chain.txt')
                if i == 0:
                    chains = np.zeros((len(tmp),len(variables)))

                chains[:,i] = tmp
            except:
                print variables[i]+'chain.txt does not exist on disk !'

    if not np.shape(lp):
        try:
            print "Reading ln(prob) chain"
            lp = np.loadtxt(path+'lnprob.txt')
        except:
            print 'lnprob.txt does not exist. Exiting'
            return

#  Get maximum likelihood values
    bestvals = np.zeros(len(variables))
    meds = np.zeros(len(variables))
    modes = np.zeros(len(variables))
    onesigs = np.zeros(len(variables))

    maxlike = np.max(lp)
    imax = np.array([i for i, j in enumerate(lp) if j == maxlike])
    if imax.size > 1:
        print 'Multiple maximum likelihood values!'
        imax = imax[0]
    for i in np.arange(len(variables)):
        bestvals[i] = chains[imax,i]

    print 'Maximum likelihood = '+str(maxlike)
        
    if thin:
        print "Thinning chains by a factor of "+str(thin)
        nsamp /= thin
        thinchains = np.zeros((nsamp,len(variables)))
        for i in np.arange(len(variables)):
            thinchains[:,i] = chains[0::thin,i]
        lp = lp[0::thin]
        chains = thinchains 

    varnames = ebs.varnameconv(variables)


    # Primary Variables
    priminds, = np.where((np.array(variables) == 'J') ^ (np.array(variables) =='Rsum') ^ 
                         (np.array(variables) == 'Rratio') ^ (np.array(variables) == 'ecosw') ^ 
                         (np.array(variables) == 'esinw') ^ (np.array(variables) == 'cosi'))

    plt.ioff()
    plt.figure(4,figsize=(8.5,11),dpi=300)    
    plt.clf()
    plotnum = 0
    for i in priminds:
        print ''
        dist = chains[:,i]
        med,mode,interval,lo,hi = distparams(dist)
        meds[i] = med
        modes[i] = mode
        onesigs[i] = interval
        minval = np.min(dist)
        maxval = np.max(dist)
        sigval = rb.std(dist)
        maxval = med + sigrange*np.abs(hi-med)
        minval = med - sigrange*np.abs(med-lo)
        nb = min(np.ceil((maxval-minval) / (interval/bindiv)),500)
        print 'Best fit parameters for '+variables[i]        
        out = variables[i]+': max = {0:.5f}, med = {1:.5f}, mode = {2:.5f}, 1 sig int = {3:.5f}'
        print out.format(bestvals[i], med, mode, interval)
        
        # actual value
        ind, = np.where(np.array(var_init) == np.array(variables)[i])
        val = p_init[ind]
        
        # do plot
        plotnum += 1
        plt.subplot(len(priminds),1,plotnum)
        print "Computing histogram of data"
        pinds, = np.where((dist >= minval) & (dist <= maxval))
        try:
            plt.hist(dist[pinds],bins=nb,normed=True)
        except:
            plt.hist(dist,bins=nb,normed=True)
            
        #    plt.xlim([minval,maxval])
        plt.axvline(x=bestvals[i],color='g',linestyle='--',linewidth=2)
        plt.axvline(x=med,color='c',linestyle='--',linewidth=2)
        plt.axvline(x=val,color='r',linestyle='--',linewidth=2)
        plt.xlabel(varnames[i])
        plt.ylabel(r'$dP$')
        if plotnum == 1:
            plt.title('Parameter Distributions for Run '+str(seq_num))

    plt.subplots_adjust(hspace=0.55)
    plt.savefig(path+'params1.png', dpi=300)
    plt.clf()




# Second set of parameters
    secinds, = np.where((np.array(variables) == 't0') ^ (np.array(variables) == 'massratio') ^ 
                        (np.array(variables) == 'ktot') ^ (np.array(variables) == 'vsys'))

    plt.figure(5,figsize=(8.5,11),dpi=300)    
    plt.clf()
    plotnum = 0
    for i in secinds:
        print ''
        if variables[i] == 't0':
            dist   = (chains[:,i] - (ebpar["t01"] - ebpar['bjd']))*24.*3600.0
            t0val = bestvals[i]
            bestvals[i] = (t0val -(ebpar["t01"] - ebpar['bjd']))*24.*3600.0
        elif variables[i] == 'period':
            dist   = (chains[:,i] - ebpar["Period"])*24.*3600.0
            pval  =  bestvals[i]
            bestvals[i] = (pval - ebpar["Period"])*24.*3600.0
        else:
            dist = chains[:,i]

        med,mode,interval,lo,hi = distparams(dist)
        meds[i] = med
        modes[i] = mode
        onesigs[i] = interval
        minval = np.min(dist)
        maxval = np.max(dist)
        sigval = rb.std(dist)
        maxval = med + sigrange*np.abs(hi-med)
        minval = med - sigrange*np.abs(med-lo)
        nb = min(np.ceil((maxval-minval) / (interval/bindiv)),500)
        print 'Best fit parameters for '+variables[i]        
        out = variables[i]+': max = {0:.5f}, med = {1:.5f}, mode = {2:.5f}, 1 sig int = {3:.5f}'
        print out.format(bestvals[i], med, mode, interval)
        
        # actual value
        ind, = np.where(np.array(var_init) == np.array(variables)[i])
        
        val = p_init[ind]
        
        # do plot
        plotnum += 1
        plt.subplot(len(secinds),1,plotnum)
        print "Computing histogram of data"
        pinds, = np.where((dist >= minval) & (dist <= maxval))
        try:
            plt.hist(dist[pinds],bins=nb,normed=True)
        except:
            plt.hist(dist,bins=nb,normed=True)
        plt.xlim([minval,maxval])
        plt.axvline(x=bestvals[i],color='g',linestyle='--',linewidth=2)
        plt.axvline(x=med,color='c',linestyle='--',linewidth=2)
        plt.axvline(x=val,color='r',linestyle='--',linewidth=2)
        plt.xlabel(varnames[i])
        plt.ylabel(r'$dP$')
        if plotnum == 1:
            plt.title('Parameter Distributions for Run '+str(seq_num))
        if variables[i] == 't0':
            plt.annotate(r'$t_0$ = %.6f BJD' % ebpar["t01"], xy=(0.96,0.8),
                         ha="right",xycoords='axes fraction',fontsize='large')
            bestvals[i] = t0val
        if variables[i] == 'period':
            plt.annotate(r'$P$ = %.6f d' % ebpar["Period"], xy=(0.96,0.8),
                         ha="right",xycoords='axes fraction',fontsize='large')
            bestvals[i] = pval

    plt.subplots_adjust(hspace=0.55)
    plt.savefig(path+'params2.png', dpi=300)
    plt.clf()



    plotnum = 1
    # For the remaining variables
    allinds = np.array(list(priminds) + list(secinds))
    allinds = np.sort(allinds)
    if len(allinds) < len(variables):
        print ''
        print "Starting plots for remaining variables"
        plt.figure(6,figsize=(8.5,11),dpi=300)
        vinds = np.arange(len(variables))
        missedi = []
        for vi in vinds:
            try:
                leni = len(np.where(vi == allinds)[0])
                if leni == 0:
                    missedi.append(vi)
            except:
                pass

        for i in missedi:
            print ''
            dist   = chains[:,i]
            
            med,mode,interval,lo,hi = distparams(dist)
            meds[i] = med
            modes[i] = mode
            onesigs[i] = interval
            minval = np.min(dist)
            maxval = np.max(dist)
            sigval = rb.std(dist)
            maxval = med + sigrange*np.abs(hi-med)
            minval = med - sigrange*np.abs(med-lo)
            if variables[i][0] == 'q':
                minval = 0.0
                maxval = 1.0
            else:
                sigval = rb.std(dist)
                minval = np.min(dist) - sigval
                maxval = np.max(dist) + sigval
            nb = min(np.ceil((maxval-minval) / (interval/bindiv)),500)
            print 'Best fit parameters for '+variables[i]        
            out = variables[i]+': max = {0:.5f}, med = {1:.5f}, mode = {2:.5f}, 1 sig int = {3:.5f}'
            print out.format(bestvals[i], med, mode, interval)
            
            # actual value
            ind, = np.where(np.array(var_init) == np.array(variables)[i])
            val = p_init[ind]
        
             # do plot
            plt.subplot(len(missedi),1,plotnum)
            print "Computing histogram of data"
            pinds, = np.where((dist >= minval) & (dist <= maxval))
            try:
                plt.hist(dist[pinds],bins=nb,normed=True)
            except:
                plt.hist(dist,bins=nb,normed=True)
            plt.xlim([minval,maxval])
            plt.axvline(x=bestvals[i],color='g',linestyle='--',linewidth=2)
            plt.axvline(x=med,color='c',linestyle='--',linewidth=2)
            plt.axvline(x=val,color='r',linestyle='--',linewidth=2)
            plt.xlabel(varnames[i])
            plt.ylabel(r'$dP$')
            if plotnum == 1:
                plt.title('Parameter Distributions for Run '+str(seq_num))
            plotnum += 1
            
        plt.subplots_adjust(hspace=0.55)
        plt.savefig(path+'params3.png', dpi=300)
        plt.clf()
 
   # calculate limb darkening parameters
#    if limb == 'quad':
#        ind1, = np.where(variables == 'q1a')
#        ind2, = np.where(variables == 'q2a')        
#        u1a,u2a =  qtou(bestvals[ind1],bestvals[ind2],limb=limb)
#        ldc = [u1a,u2a]
#        plot_limb_curves(ldc=ldc,limbmodel=limb,write=True,network=network)
#        ind1, = np.where(variables == 'q1b')
#        ind2, = np.where(variables == 'q2b')        
#        u1a,u2a =  qtou(bestvals[ind1],bestvals[ind2],limb=limb)
#        ldc = [u1a,u2a]
#        plot_limb_curves(ldc=ldc,limbmodel=limb,write=True,network=network)
#
#
#
#    vals = [[bestvals],[meds],[modes],[onesigs]]

#    ebs.lnprob(bestvals,data,ebpar,fitinfo,debug=True)
    
    plot_model(bestvals,data,ebpar,fitinfo,data,tag='_fit',network=network)

    f = open(path+'fitparams.txt','w')
    for i in np.arange(len(variables)):
        outstr = []
        outstr.append(variables[i])
        outstr.append("{:.8f}".format(bestvals[i]))
        outstr.append("{:.8f}".format(meds[i]))
        outstr.append("{:.8f}".format(modes[i]))
        outstr.append("{:.8f}".format(onesigs[i]))
        f.write(', '.join(outstr)+'\n')
        
    f.closed

    plot_defaults()

    return bestvals



def plot_model(x,data,ebpar,fitinfo,markersize=5,smallmark=2,nbins=100,
               errorbars=False,durfac=5,enum=1,tag='',network=None):

    """
    ----------------------------------------------------------------------
    plot_model_single:
    ------------------
    Plot transit model given model params.

    ----------------------------------------------------------------------
    """
    plot_params(fontsize=10,linewidth=1.2)

    # Check for output directory   
    path = ebpar['path']
    run_num = int(path.split('/')[-2])
    directory = reb.get_path(network=network)+str(run_num)+'/'
    
    variables = fitinfo['variables']

    parm,vder = ebs.vec_to_params(x,ebpar,fitinfo=fitinfo)

    # Phases of contact points
    (ps, pe, ss, se) = eb.phicont(parm)

    vsys = x[variables == 'vsys'][0]
    ktot = x[variables == 'ktot'][0]

    massratio = parm[eb.PAR_Q]


    if fitinfo['claret']:
        T1,logg1,T2,logg2 = get_teffs_loggs(parm,vsys,ktot)

        u1a = ldc1func(T1,logg1)[0][0]
        u2a = ldc2func(T1,logg1)[0][0]
        
        u1b = ldc1func(T2,logg2)[0][0]
        u2b = ldc2func(T2,logg2)[0][0]
        
        q1a,q2a = ebs.utoq(u1a,u2a,limb=limb)        
        q1b,q2b = ebs.utoq(u1b,u2b,limb=limb)
        
        parm[eb.PAR_LDLIN1] = u1a  # u1 star 1
        parm[eb.PAR_LDNON1] = u2a  # u2 star 1
        parm[eb.PAR_LDLIN2] = u1b  # u1 star 2
        parm[eb.PAR_LDNON2] = u2b  # u2 star 2

    elif fitinfo['fit_limb']:
        q1a = x[variables == 'q1a'][0]  
        q2a = x[variables == 'q2a'][0]  
        q1b = x[variables == 'q1b'][0]  
        q2b = x[variables == 'q2b'][0]  
        u1a,u2a = ebs.qtou(q1a,q2a,limb=ebpar['limb'])        
        u1b,u2b = ebs.qtou(q1b,q2b,limb=ebpar['limb'])
        parm[eb.PAR_LDLIN1] = u1a  # u1 star 1
        parm[eb.PAR_LDNON1] = u2a  # u2 star 1
        parm[eb.PAR_LDLIN2] = u1b  # u1 star 2
        parm[eb.PAR_LDNON2] = u2b  # u2 star 2
    else:
        q1a,q2a = ebs.utoq(ebpar['LDlin1'],epbar['LDnon1'],limb=ebpar['limb'])
        q1b,q2b = ebs.utoq(ebpar['LDlin2'],epbar['LDnon2'],limb=ebpar['limb'])


    # Need to understand exactly what this parameter is!!
    if fitinfo['fit_ooe1']:
        if parm[eb.PAR_FSPOT1] < 0 or parm[eb.PAR_FSPOT1] > 1:
            return -np.inf
        coeff1 = []
        for i in range(fitorder+1):
            coeff1 = np.append(coeff1,x[variables == 'c'+str(i)+'_1'])
        
### Compute eclipse model for given input parameters ###
    massratio = parm[eb.PAR_Q]
    if massratio < 0 or massratio > 10:
        return -np.inf

    if not fitinfo['fit_ellipsoidal']:
        parm[eb.PAR_Q] = 0.0

    # Primary eclipse
    t0 = parm[eb.PAR_T0]
    
    period = parm[eb.PAR_P]
    time   = data['light'][0,:]-ebpar['bjd']
    flux   = data['light'][1,:]
    eflux  = data['light'][2,:]

    sm  = ebs.compute_eclipse(time,parm,integration=ebpar['integration'],fitrvs=False,tref=t0,period=period)

    tfold = ebs.foldtime(time,t0=t0,period=period)
    keep, = np.where((tfold >= -0.2) & (tfold <=0.2))
    inds = np.argsort(tfold[keep])
    tprim = tfold[keep][inds]
    xprim = flux[keep][inds]
    mprim = sm[keep][inds]

    tcomp1 = np.linspace(np.min(tprim),np.max(tprim),10000)
    lcomp1  = ebs.compute_eclipse(tcomp1,parm,integration=ebpar['integration'],fitrvs=False,
                              tref=t0,period=period)
    
    tfold_pos = ebs.foldtime_pos(time,t0=t0,period=period)
    ph_pos = tfold_pos/period
    keep, = np.where((ph_pos >= 0.3) & (ph_pos <=0.7))
    inds = np.argsort(tfold_pos[keep])
    tsec = tfold_pos[keep][inds]
    xsec = flux[keep][inds]
    msec = sm[keep][inds]

    tcomp2 = np.linspace(np.min(tsec),np.max(tsec),10000)
    lcomp2  = ebs.compute_eclipse(tcomp2,parm,integration=ebpar['integration'],fitrvs=False,
                              tref=t0,period=period)


    # Log Likelihood Vector
    lfi = -1.0*(sm - flux)**2/(2.0*eflux**2)

    # Log likelihood
    lf1 = np.sum(lfi)

    lf = lf1

    # need this for the RVs!
    parm[eb.PAR_Q] = massratio

    rvdata1 = data['rv1']
    rvdata2 = data['rv2']

    if fitinfo['fit_rvs']:
        if (vsys > max(np.max(rvdata1[1,:]),np.max(rvdata2[1,:]))) or \
           (vsys < min(np.min(rvdata1[1,:]),np.min(rvdata2[1,:]))): 
            return -np.inf
        rvmodel1 = ebs.compute_eclipse(rvdata1[0,:]-ebpar['bjd'],parm,fitrvs=True)
        k2 = ktot/(1+massratio)
        k1 = k2*massratio
        rv1 = rvmodel1*k1 + vsys
        rvmodel2 = ebs.compute_eclipse(rvdata2[0,:]-ebpar['bjd'],parm,fitrvs=True)
        rv2 = -1.0*rvmodel2*k2 + vsys
        lfrv1 = -np.sum((rv1 - rvdata1[1,:])**2/(2.0*rvdata1[2,:]))
        lfrv2 = -np.sum((rv2 - rvdata2[1,:])**2/(2.0*rvdata2[2,:]))
        lfrv = lfrv1 + lfrv2
        lf  += lfrv

    print "Model parameters:"
    for nm, vl, unt in zip(eb.parnames, parm, eb.parunits):
        print "{0:<10} {1:14.6f} {2}".format(nm, vl, unt)

    vder = eb.getvder(parm, vsys, ktot)
    print "Derived parameters:"
    for nm, vl, unt in zip(eb.dernames, vder, eb.derunits):
        print "{0:<10} {1:14.6f} {2}".format(nm, vl, unt)


    ################################################
    # Plotting

    # Primary eclipse
    fig = plt.figure(109,dpi=300)
    plt.clf()
    plt.subplot(2, 2, 1)
    phiprim = tprim/period
    phicomp1 = tcomp1/period
    plt.plot(phiprim,xprim,'ko')
#    plt.plot(phiprim,mprim,'gx')
    plt.plot(phicomp1,lcomp1,'r-')

    plt.axvline(x=ps-1.0,color='b',linestyle='--')
    plt.axvline(x=pe,color='b',linestyle='--')
    ymax = np.max(np.array(list(xprim)+list(mprim)))
    ymin = np.min(np.array(list(xprim)+list(mprim)))
    ytop = ymax + (ymax-ymin)*0.1
    ybot = ymin - (ymax-ymin)*0.1
    plt.ylim(ybot,ytop)
    plt.ylabel('Flux (normalized)')
    plt.xlabel('Phase')
    plt.title('Primary Eclipse',fontsize=12)

#    chi1 = -1*lf1
#    plt.annotate(r'$\chi^2$ = %.0f' % chi1, [0.05,0.87],
#                 horizontalalignment='left',xycoords='axes fraction',fontsize='large')


    # Secondary eclipse
    phisec = tsec/period
    phicomp2 = tcomp2/period
    plt.subplot(2, 2, 2)
    plt.plot(phisec,xsec,'ko')
#    plt.plot(phisec,msec,'gx')
    plt.plot(phicomp2,lcomp2,'r-')
    plt.axvline(x=ss,color='b',linestyle='--')
    plt.axvline(x=se,color='b',linestyle='--')
    ymax = np.max(np.array(list(xsec)+list(msec)))
    ymin = np.min(np.array(list(xsec)+list(msec)))
    ytop = ymax + (ymax-ymin)*0.1
    ybot = ymin - (ymax-ymin)*0.1
    plt.ylim(ybot,ytop)
    plt.xlabel('Phase')
    plt.title('Secondary Eclipse',fontsize=12)

    plt.subplot(2, 1, 2)
    phi1 = ebs.foldtime(rvdata1[0,:]-ebpar['bjd'],t0=t0,period=period)/period
    plt.plot(phi1,rvdata1[1,:],'ko')
#    plt.plot(phi1,rv1,'gx')
    tcomp = np.linspace(-0.5,0.5,10000)*period+t0
    rvmodel1 = ebs.compute_eclipse(tcomp,parm,fitrvs=True)
    k2 = ktot/(1+massratio)
    k1 = k2*massratio
    rvcomp1 = rvmodel1*k1 + vsys
    plt.plot(np.linspace(-0.5,0.5,10000),rvcomp1,'b-')

    #plt.annotate(r'$\chi^2$ = %.2f' % -lfrv, [0.05,0.85],horizontalalignment='left',
    #             xycoords='axes fraction',fontsize='large')
  
    phi2 = ebs.foldtime(rvdata2[0,:]-ebpar['bjd'],t0=t0,period=period)/period
    plt.plot(phi2,rvdata2[1,:],'ro')
#    plt.plot(phi2,rv2,'gx')
    tcomp = np.linspace(-0.5,0.5,10000)*period+t0
    rvmodel2 = ebs.compute_eclipse(tcomp,parm,fitrvs=True)
    rvcomp2 = -1.0*rvmodel2*k2 + vsys
    plt.plot(np.linspace(-0.5,0.5,10000),rvcomp2,'c-')
    plt.xlim(-0.5,0.5)
    plt.ylabel('Radial Velocity (km/s)')
    plt.xlabel('Phase')

    plt.suptitle('Fitting Results for Run ' + str(run_num))

    plt.savefig(directory+'MCMCfit.png',dpi=300)

    # Limb Darkening
    gamma = np.linspace(0,np.pi/2.0,1000,endpoint=True)
    theta = gamma*180.0/np.pi
    mu = np.cos(gamma)
    Imu1 = 1.0 - u1a*(1.0 - mu) - u2a*(1.0 - mu)**2.0
    Imu2 = 1.0 - u1b*(1.0 - mu) - u2b*(1.0 - mu)**2.0

    
    fig = plt.figure(110,dpi=300)
    plt.clf()
    label1 = '%.2f, ' % u1a + '%.2f' % u2a +' (Primary)'
    label2 = '%.2f, ' % u1b + '%.2f' % u2b +' (Secondary)'
    plt.plot(theta,Imu1,label=label1)
    plt.plot(theta,Imu2,label=label2)
    plt.ylim([0,1.0])
    plt.xlabel(r"$\theta$ (degrees)",fontsize=18)
    plt.ylabel(r"$I(\theta)/I(0)$",fontsize=18)
    plt.title('Limb Darkening')
    
    plt.legend()
    plt.savefig(directory+'Limbfit.png',dpi=300)


    ################################################
    # Residuals

    # Primary eclipse
    fig = plt.figure(111,dpi=300)
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.plot(phiprim,(xprim-mprim)*100,'ko')
    plt.axhline(y=0,color='r',linestyle='-')
    plt.axvline(x=ps-1.0,color='b',linestyle='--')
    plt.axvline(x=pe,color='b',linestyle='--')
    plt.ylabel('Residual Flux (percent)')
    plt.xlabel('Phase')
    plt.title('Primary Eclipse',fontsize=12)

    chi1 = -1*lf1
    plt.annotate(r'$\chi^2$ = %.0f' % chi1, [0.05,0.87],
                 horizontalalignment='left',xycoords='axes fraction',fontsize='large')


    # Secondary eclipse
    plt.subplot(2, 2, 2)
    plt.plot(phisec,(xsec-msec)*100,'ko')
    plt.axhline(y=0,color='r',linestyle='-')
    plt.axvline(x=ss,color='b',linestyle='--')
    plt.axvline(x=se,color='b',linestyle='--')
    plt.xlabel('Phase')
    plt.title('Secondary Eclipse',fontsize=12)

    plt.subplot(2, 1, 2)
    plt.plot(phi1,rvdata1[1,:]-rv1,'ko')

    plt.annotate(r'$\chi^2$ = %.2f' % -lfrv, [0.05,0.85],horizontalalignment='left',
                 xycoords='axes fraction',fontsize='large')
  
    plt.plot(phi2,rvdata2[1,:]-rv2,'ro')
    plt.axhline(y=0,color='k',linestyle='--')
    plt.xlim(-0.5,0.5)
    plt.ylabel('RV Residuals (km/s)')
    plt.xlabel('Phase')

    plt.suptitle('Fitting Residuals for Run ' + str(run_num))

    plt.savefig(directory+'MCMCres.png',dpi=300)

    return






def old_crap():

    if fitinfo['claret']:
        T1,logg1,T2,logg2 = get_teffs_loggs(parm,vsys,ktot)
        
        u1a = ldc1func(T1,logg1)[0][0]
        u2a = ldc2func(T1,logg1)[0][0]
        
        u1b = ldc1func(T2,logg2)[0][0]
        u2b = ldc2func(T2,logg2)[0][0]
        
        q1a,q2a = ebs.utoq(u1a,u2a,limb=limb)        
        q1b,q2b = ebs.utoq(u1b,u2b,limb=limb)
        
        parm[eb.PAR_LDLIN1] = u1a  # u1 star 1
        parm[eb.PAR_LDNON1] = u2a  # u2 star 1
        parm[eb.PAR_LDLIN2] = u1b  # u1 star 2
        parm[eb.PAR_LDNON2] = u2b  # u2 star 2

    elif fitinfo['fit_limb']:
        q1a = x[variables == 'q1a'][0]  
        q2a = x[variables == 'q2a'][0]  
        q1b = x[variables == 'q1b'][0]  
        q2b = x[variables == 'q2b'][0]  
    else:
        pass

    print "Model parameters:"
    for vname, value, unit in zip(eb.parnames, parm, eb.parunits):
        print "{0:<10} {1:14.6f} {2}".format(vname, value, unit)

    print "Derived parameters:"
    for vname, value, unit in zip(eb.dernames, vder, eb.derunits):
        print "{0:<10} {1:14.6f} {2}".format(vname, value, unit)


######################################################################
# Light curve model
######################################################################

    time = data["light"][0,:]
    flux = data["light"][1,:]
    
    if not fitinfo['fit_ellipsoidal']:
        parm[eb.PAR_Q] = 0.0        

    # Phases of contact points
    (ps, pe, ss, se) = eb.phicont(parm)

    # Primary eclipse
    t0 = parm[eb.PAR_T0]
    period = parm[eb.PAR_P]

    # This could be done better
    tfold = ebs.foldtime(time-ebpar['bjd'],t0=t0,period=period)
    keep, = np.where((tfold >= -0.2) & (tfold <=0.2))
    inds = np.argsort(tfold[keep])
    tprim = tfold[keep][inds]
    phiprim = tprim/period
    xprim = flux[keep][inds]

    if fitinfo['fit_ooe1']:
        coeff1 = []
        for i in range(fitorder+1):
            coeff1 = np.append(coeff1,x[variables == 'c'+str(i)+'_1'])

    # erased tprim+t0
    model1  = ebs.compute_eclipse(tprim,parm,integration=ebpar['integration'],
                                  fitrvs=False,tref=t0,period=period)

    tcomp1 = np.linspace(np.min(tprim),np.max(tprim),10000) #+t0
    compmodel1  = ebs.compute_eclipse(tcomp1,parm,integration=ebpar['integration'],
                                  fitrvs=False,tref=t0,period=period)

    phicomp1 = tcomp1/period


    # Secondary eclipse
    tfold_pos = ebs.foldtime_pos(time-ebpar['bjd'],t0=t0,period=period)
    ph_pos = tfold_pos/period
    keep, = np.where((ph_pos >= 0.3) & (ph_pos <=0.7))
    inds = np.argsort(tfold_pos[keep])
    tsec = tfold_pos[keep][inds]
    phisec = tsec/period
    xsec = flux[keep][inds]

    if fitinfo['fit_ooe1']:
        coeff2 = []
        for i in range(fitorder+1):
            coeff2 = np.append(coeff2,x[variables == 'c'+str(i)+'_2'])

    # erased tsec+t0
    model2  = ebs.compute_eclipse(tsec,parm,integration=ebpar['integration'],
                              fitrvs=False,tref=t0,period=period)
    
    tcomp2 = np.linspace(np.min(tsec),np.max(tsec),10000) # +t0
    compmodel2 = ebs.compute_eclipse(tcomp2,parm,integration=ebpar['integration'],
                                 fitrvs=False,tref=t0,period=period)
        
    phicomp2 = tcomp2/period
    
    parm[eb.PAR_Q] = massratio

    if fitinfo['fit_rvs']:
        rvdata1 = data['rv1']
        rvdata2 = data['rv2']
        rvmodel1 = ebs.compute_eclipse(rvdata1[0,:],parm,fitrvs=True)
        k2 = ktot/(1+massratio)
        k1 = k2*massratio
        rv1 = rvmodel1*k1 + vsys
        rvmodel2 = ebs.compute_eclipse(rvdata2[0,:],parm,fitrvs=True)

# Does dof need another - 1 ???
#    dof = np.float(len(tfit)) - np.float(len(variables))
#    chisquare = np.sum((res/e_ffit)**2)/dof
    

#------------------------------
# PLOT

# Primary eclipse
    fig = plt.figure(109,dpi=300)
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.plot(phiprim,xprim,'ko',markersize=3)
#    plt.plot(phiprim,model1,'rx')
    plt.plot(phicomp1,compmodel1,'r-')
    plt.axvline(x=ps-1.0,color='b',linestyle='--')
    plt.axvline(x=pe,color='b',linestyle='--')
    ymax = np.max(np.array(list(xprim)+list(compmodel1)))
    ymin = np.min(np.array(list(xprim)+list(compmodel1)))
    ytop = ymax + (ymax-ymin)*0.1
    ybot = ymin - (ymax-ymin)*0.1
    plt.ylim(ybot,ytop)
    plt.ylabel('Flux (normalized)')
    plt.xlabel('Phase')
    plt.title('Primary Eclipse',fontsize=12)

    plt.subplot(2, 2, 2)
    plt.plot(phisec,xsec,'ko',markersize=3)
#    plt.plot(phisec,model2,'xo')
    plt.plot(phicomp2,compmodel2,'r-')
    plt.axvline(x=ss,color='b',linestyle='--')
    plt.axvline(x=se,color='b',linestyle='--')
    ymax = np.max(np.array(list(xsec)+list(compmodel2)))
    ymin = np.min(np.array(list(xsec)+list(compmodel2)))
    ytop = ymax + (ymax-ymin)*0.1
    ybot = ymin - (ymax-ymin)*0.1
    plt.ylim(ybot,ytop)
    plt.xlabel('Phase')
    plt.title('Secondary Eclipse',fontsize=12)

    
    plt.subplot(2, 1, 2)
    phi1 = ebs.foldtime_pos(rvdata1[0,:]-ebpar['bjd'],t0=t0,period=period)/period
    plt.plot(phi1,rvdata1[1,:],'ko',markersize=3)
#    plt.plot(phi1,rv1,'kx')
    tcomp = np.linspace(0,1,10000)*period+t0
    rvmodel1 = ebs.compute_eclipse(tcomp,parm,fitrvs=True)
    k2 = ktot/(1+massratio)
    k1 = k2*massratio
    rvcomp1 = rvmodel1*k1 + vsys
    plt.plot(np.linspace(0,1,10000),rvcomp1,'g--')
    
    phi2 = ebs.foldtime_pos(rvdata2[0,:]-ebpar['bjd'],t0=t0,period=period)/period
    plt.plot(phi2,rvdata2[1,:],'ro',markersize=3)
#    plt.plot(phi2,rv2,'rx')
    tcomp = np.linspace(0,1,10000)*period+t0
    rvmodel2 = ebs.compute_eclipse(tcomp,parm,fitrvs=True)
    rvcomp2 = -1.0*rvmodel2*k2 + vsys
    plt.plot(np.linspace(0,1,10000),rvcomp2,'b--')
    plt.xlim(0,1)
    plt.ylabel('Radial Velocity (km/s)')
    plt.xlabel('Phase')

    plt.suptitle('Fitting Results for Run ' + str(run_number))
    
    plt.savefig(directory+'MCMCfit.png')

    plot_defaults()

    return



def params_of_interest(seq_num,chains=False,lp=False,network=None):

    path = reb.get_path(network=network)+str(seq_num)+'/'
    ebpar   = pickle.load( open( path+'ebpar.p', 'rb' ) )
    data    = pickle.load( open( path+'data.p', 'rb' ) )
    fitinfo = pickle.load( open( path+'fitinfo.p', 'rb' ) )
    
    run_num = seq_num
    
    nsamp = fitinfo['nwalkers']*fitinfo['mcmcsteps']

    variables = fitinfo['variables']
    
    # Use supplied chains or read from disk
    if not np.shape(chains):
        for i in np.arange(len(variables)):
            try:
                print "Reading MCMC chains for "+variables[i]
                tmp = np.loadtxt(path+variables[i]+'chain.txt')
                if i == 0:
                    chains = np.zeros((len(tmp),len(variables)))

                chains[:,i] = tmp
            except:
                print variables[i]+'chain.txt does not exist on disk !'

    if not np.shape(lp):
        try:
            print "Reading ln(prob) chain"
            lp = np.loadtxt(path+'lnprob.txt')
        except:
            print 'lnprob.txt does not exist. Exiting'
            return

    variables = fitinfo['variables']
    
    print "Deriving values for parameters of interest"


    print "Converting posteriors into physical quantities"
    ind, = np.where(np.array(variables) == 'Rsum')[0]
    rsumdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'Rratio')[0]
    rratdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'cosi')[0]
    cosidist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'ecosw')[0]
    ecoswdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'esinw')[0]
    esinwdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'massratio')[0]
    mratdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'vsys')[0]
    vsysdist = chains[:,ind]

    try:
        ind, = np.where(np.array(variables) == 'period')[0]
        pdist = chains[:,ind]
    except:
        pdist = ebpar['Period']

    ind, = np.where(np.array(variables) == 'ktot')[0]
    ktotdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'J')[0]
    jdist = chains[:,ind]


    print "Determining maximum likelihood values"
    tlike = time.time()
    bestvals = np.zeros(len(variables))
    meds = np.zeros(len(variables))
    modes = np.zeros(len(variables))
    onesigs = np.zeros(len(variables))

    maxlike = np.max(lp)
    imax = np.array([i for i, j in enumerate(lp) if j == maxlike])
#    if imax.size > 1:
    imax = imax[0]
    for i in np.arange(len(variables)):
        bestvals[i] = chains[imax,i]

    esq = ecoswdist * ecoswdist + esinwdist * esinwdist
    roe = np.sqrt(1.0 - esq)
    sini = np.sqrt(1.0 - cosidist*cosidist)
    qpo = 1.0 + mratdist
    # Corrects for doppler shift of period
    omega = 2.0*np.pi*(1.0 + vsysdist*1000.0/eb.LIGHT) / (pdist*86400.0)
    tmp = ktotdist*1000.0 * roe
    sma = tmp / (eb.RSUN*omega*sini)
    mtotdist = tmp*tmp*tmp / (eb.GMSUN*omega*sini)
    m1dist = mtotdist / qpo
    m2dist = mratdist * m1dist

    r1dist = sma*rsumdist/(1+rratdist)
    r2dist = rratdist*r1dist

    edist  = np.sqrt(ecoswdist**2 + esinwdist**2)
 
    m1val = m1dist[imax]
    m2val = m2dist[imax]
    r1val = r1dist[imax]
    r2val = r2dist[imax]
    jval = jdist[imax]
    cosival = cosidist[imax]
    ecoswval = ecoswdist[imax]
    esinwval = esinwdist[imax]
    eval = np.sqrt(ecoswval**2 + esinwval**2)

    vals = []
    meds = []
    modes = []
    onesig = []
    med,mode,interval,lo,hi = distparams(m1dist)
    out = 'M1: max = {0:.5f}, med = {1:.5f}, mode = {2:.5f}, 1 sig int = {3:.5f}'
    print out.format(m1val,med,mode,interval)
    vals   = np.append(vals,m1val)
    meds   = np.append(meds,med)
    modes  = np.append(modes,mode)
    onesig = np.append(onesig,interval)
    
    med,mode,interval,lo,hi = distparams(m2dist)
    out = 'M2: max = {0:.5f}, med = {1:.5f}, mode = {2:.5f}, 1 sig int = {3:.5f}'
    print out.format(m2val,med,mode,interval)
    vals   = np.append(vals,m2val)
    meds   = np.append(meds,med)
    modes  = np.append(modes,mode)
    onesig = np.append(onesig,interval)

    med,mode,interval,lo,hi = distparams(r1dist)
    out = 'R1: max = {0:.5f}, med = {1:.5f}, mode = {2:.5f}, 1 sig int = {3:.5f}'
    print out.format(r1val,med,mode,interval)
    vals   = np.append(vals,r1val)
    meds   = np.append(meds,med)
    modes  = np.append(modes,mode)
    onesig = np.append(onesig,interval)

    med,mode,interval,lo,hi = distparams(r2dist)
    out = 'R2: max = {0:.5f}, med = {1:.5f}, mode = {2:.5f}, 1 sig int = {3:.5f}'
    print out.format(r2val,med,mode,interval)
    vals   = np.append(vals,r2val)
    meds   = np.append(meds,med)
    modes  = np.append(modes,mode)
    onesig = np.append(onesig,interval)

    med,mode,interval,lo,hi = distparams(edist)
    out = 'eccentricity: max = {0:.5f}, med = {1:.5f}, mode = {2:.5f}, 1 sig int = {3:.5f}'
    print out.format(eval,med,mode,interval)
    vals   = np.append(vals,eval)
    meds   = np.append(meds,med)
    modes  = np.append(modes,mode)
    onesig = np.append(onesig,interval)

    outstr = 'Run'+str(seq_num)+ ' %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f' % (vals[0],meds[0],modes[0],onesig[0],vals[1],meds[1],modes[1],onesig[1],vals[2],meds[2],modes[2],onesig[2],vals[3],meds[3],modes[3],onesig[3],vals[4],meds[4],modes[4],onesig[4])

    f = open(path+'bestparams.txt','w')
    f.write(outstr+'\n')
    f.closed

    return 



def triangle_plot(seq_num,chains=False,lp=False,thin=False,frac=0.001,sigfac=4.0,network=None):
    import matplotlib.gridspec as gridspec

    tmaster = time.time()

    mpl.rc('axes', linewidth=1)
  
    bindiv = 10
    
    path = reb.get_path(network=network)+str(seq_num)+'/'

    ebpar   = pickle.load( open( path+'ebpar.p', 'rb' ) )
    data    = pickle.load( open( path+'data.p', 'rb' ) )
    fitinfo = pickle.load( open( path+'fitinfo.p', 'rb' ) )

    run_num = seq_num
    
    nsamp = fitinfo['nwalkers']*fitinfo['mcmcsteps']

    variables = fitinfo['variables']
    
    # Use supplied chains or read from disk
    if not np.shape(chains):
        for i in np.arange(len(variables)):
            try:
                print "Reading MCMC chains for "+variables[i]
                tmp = np.loadtxt(path+variables[i]+'chain.txt')
                if i == 0:
                    chains = np.zeros((len(tmp),len(variables)))

                chains[:,i] = tmp
            except:
                print variables[i]+'chain.txt does not exist on disk !'

    if not np.shape(lp):
        try:
            print "Reading ln(prob) chain"
            lp = np.loadtxt(path+'lnprob.txt')
        except:
            print 'lnprob.txt does not exist. Exiting'
            return

    variables = fitinfo['variables']
    

    print "Converting posteriors into physical quantities"
    ind, = np.where(np.array(variables) == 'Rsum')[0]
    rsumdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'Rratio')[0]
    rratdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'cosi')[0]
    cosidist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'ecosw')[0]
    ecoswdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'esinw')[0]
    esinwdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'massratio')[0]
    mratdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'vsys')[0]
    vsysdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'ktot')[0]
    ktotdist = chains[:,ind]
    ind, = np.where(np.array(variables) == 'J')[0]
    jdist = chains[:,ind]

    try:
        ind, = np.where(np.array(variables) == 'period')[0]
        pdist = chains[:,ind]
    except:
        pdist = ebpar['Period']

    esq = ecoswdist * ecoswdist + esinwdist * esinwdist
    roe = np.sqrt(1.0 - esq)
    sini = np.sqrt(1.0 - cosidist*cosidist)
    qpo = 1.0 + mratdist
    # Corrects for doppler shift of period
    omega = 2.0*np.pi*(1.0 + vsysdist*1000.0/eb.LIGHT) / (pdist*86400.0)
    tmp = ktotdist*1000.0 * roe
    sma = tmp / (eb.RSUN*omega*sini)
    mtotdist = tmp*tmp*tmp / (eb.GMSUN*omega*sini)
    m1dist = mtotdist / qpo
    m2dist = mratdist * m1dist

    r1dist = sma*rsumdist/(1+rratdist)
    r2dist = rratdist*r1dist

    print "Determining maximum likelihood values"
    tlike = time.time()
    bestvals = np.zeros(len(variables))
    meds = np.zeros(len(variables))
    modes = np.zeros(len(variables))
    onesigs = np.zeros(len(variables))

    maxlike = np.max(lp)
    imax = np.array([i for i, j in enumerate(lp) if j == maxlike])
    if imax.size > 1:
        imax = imax[0]
    for i in np.arange(len(variables)):
        bestvals[i] = chains[imax,i]


    m1val = m1dist[imax]
    m2val = m2dist[imax]
    r1val = r1dist[imax]
    r2val = r2dist[imax]
    jval = jdist[imax]
    cosival = cosidist[imax]
    ecoswval = ecoswdist[imax]
    esinwval = esinwdist[imax]


    if thin:
        print "Thinning chains by a factor of "+str(thin)
        tthin = time.time()
        m1dist = m1dist[0::thin]
        m2dist = m2dist[0::thin]
        r1dist = r1dist[0::thin]
        r2dist = r2dist[0::thin]
        jdist = jdist[0::thin]
        cosidist = cosidist[0::thin]
        ecoswdist = ecoswdist[0::thin]
        esinwdist = esinwdist[0::thin]
        lp = lp[0::thin]
        nsamp /= thin


    print " "
    print "Starting grid of posteriors..."
    plt.figure(6,figsize=(8.5,8.5))
    nx = 8
    ny = 8

    gs = gridspec.GridSpec(nx,ny,wspace=0.1,hspace=0.1)
    print " "
    print "... top plot of first column"
    tcol = time.time()
    top_plot(r1dist,gs[0,0],val=r1val,sigfac=sigfac)
    print done_in(tcol)
    t = time.time()
    print "... first column"
    column_plot(r1dist,r2dist,gs[1,0],val1=r1val,val2=r2val,ylabel=r'$R_2$',sigfac=sigfac)
    print done_in(t)
    column_plot(r1dist,m1dist,gs[2,0],val1=r1val,val2=m1val,ylabel=r'$M_1$',sigfac=sigfac)
    column_plot(r1dist,m2dist,gs[3,0],val1=r1val,val2=m2val,ylabel=r'$M_2$',sigfac=sigfac)
    column_plot(r1dist,jdist,gs[4,0],val1=r1val,val2=jval,ylabel=r'$J$',sigfac=sigfac)
    column_plot(r1dist,cosidist,gs[5,0],val1=r1val,val2=cosival,ylabel=r'$\cos\,i$',sigfac=sigfac)
    column_plot(r1dist,ecoswdist,gs[6,0],val1=r1val,val2=ecoswval,ylabel=r'$e\cos\omega$',sigfac=sigfac)
    corner_plot(r1dist,esinwdist,gs[7,0],val1=r1val,val2=esinwval,\
                xlabel=r'$R_1$',ylabel=r'$e\sin\omega$',sigfac=sigfac)
    print "First column: "
    print done_in(tcol)

    print "... second column"
    t2 = time.time()
    top_plot(r2dist,gs[1,1],val=r2val,sigfac=sigfac)    
    middle_plot(r2dist,m1dist,gs[2,1],val1=r2val,val2=m1val,sigfac=sigfac)
    middle_plot(r2dist,m2dist,gs[3,1],val1=r2val,val2=m2val,sigfac=sigfac)
    middle_plot(r2dist,jdist,gs[4,1],val1=r2val,val2=jval,sigfac=sigfac)
    middle_plot(r2dist,cosidist,gs[5,1],val1=r2val,val2=cosival,sigfac=sigfac)
    middle_plot(r2dist,ecoswdist,gs[6,1],val1=r2val,val2=ecoswval,sigfac=sigfac)
    row_plot(r2dist,esinwdist,gs[7,1],val1=r2val,val2=esinwval,xlabel=r'$R_2$',sigfac=sigfac)
    print done_in(t2)

    print "... third column"
    t3 = time.time()
    top_plot(m1dist,gs[2,2],val=m1val,sigfac=sigfac)    
    middle_plot(m1dist,m2dist,gs[3,2],val1=m1val,val2=m2val,sigfac=sigfac)
    middle_plot(m1dist,jdist,gs[4,2],val1=m1val,val2=jval,sigfac=sigfac)
    middle_plot(m1dist,cosidist,gs[5,2],val1=m1val,val2=cosival,sigfac=sigfac)
    middle_plot(m1dist,ecoswdist,gs[6,2],val1=m1val,val2=ecoswval,sigfac=sigfac)
    row_plot(m1dist,esinwdist,gs[7,2],val1=m1val,val2=esinwval,xlabel=r'$M_1$',sigfac=sigfac)
    print done_in(t3)

    print "... fourth column"
    t4 = time.time()
    top_plot(m2dist,gs[3,3],val=m2val,sigfac=sigfac)    
    middle_plot(m2dist,jdist,gs[4,3],val1=m2val,val2=jval,sigfac=sigfac)
    middle_plot(m2dist,cosidist,gs[5,3],val1=m2val,val2=cosival,sigfac=sigfac)
    middle_plot(m2dist,ecoswdist,gs[6,3],val1=m2val,val2=ecoswval,sigfac=sigfac)
    row_plot(m2dist,esinwdist,gs[7,3],val1=m2val,val2=esinwval,xlabel=r'$M_2$',sigfac=sigfac)
    print done_in(t4)


    print "... fifth column"
    t5 = time.time()
    top_plot(jdist,gs[4,4],val=jval,sigfac=sigfac)    
    middle_plot(jdist,cosidist,gs[5,4],val1=jval,val2=cosival,sigfac=sigfac)
    middle_plot(jdist,ecoswdist,gs[6,4],val1=jval,val2=ecoswval,sigfac=sigfac)
    row_plot(jdist,esinwdist,gs[7,4],val1=jval,val2=esinwval,xlabel=r'$J$',sigfac=sigfac)
    print done_in(t5)

    print "... sixth column"
    t6 = time.time()
    top_plot(cosidist,gs[5,5],val=cosival,sigfac=sigfac)    
    middle_plot(cosidist,ecoswdist,gs[6,5],val1=cosival,val2=ecoswval,sigfac=sigfac)
    row_plot(cosidist,esinwdist,gs[7,5],val1=cosival,val2=esinwval,xlabel=r'$\cos\,i$',sigfac=sigfac)
    print done_in(t6)
  

    print "... seventh column"
    t7 = time.time()
    top_plot(ecoswdist,gs[6,6],val=ecoswval,sigfac=sigfac)    
    row_plot(ecoswdist,esinwdist,gs[7,6],val1=ecoswval,val2=esinwval,xlabel=r'$e\cos\omega$',sigfac=sigfac)
    print done_in(t7)

    print "... last column"
    t8 = time.time()
    top_plot(esinwdist,gs[7,7],val=esinwval,xlabel=r'$e\sin\omega$',sigfac=sigfac)    
    print done_in(t8)

    plt.suptitle('2D posteriors for Run '+str(run_num))
    
    print "Saving output figures"
    plt.savefig(path+'triangle1.png', dpi=300)
    plt.savefig(path+'triangle1.eps', dpi=300)

    print "Procedure finished!"
    print done_in(tmaster)


    return



def top_plot(dist,position,val=False,sigfac=3.0,frac=0.001,bindiv=10,aspect=1,xlabel=False):

#    pdb.set_trace()
#    sz = len(dist)
#    min = np.float(np.sort(dist)[np.round(frac*sz)])
#    max = np.float(np.sort(dist)[np.round((1.-frac)*sz)])
#    dists = np.linspace(np.min(dist)*0.5,np.max(dist)*1.5,1000)
#    kde = gaussian_kde(dist)
#    pdf = kde(dists)
#    cumdist = np.cumsum(pdf)/np.sum(pdf)
#    func = interp1d(cumdist,dists,kind='linear')
#    lo = np.float(func(math.erfc(1./np.sqrt(2))))
#    hi = np.float(func(math.erf(1./np.sqrt(2))))
    med = np.median(dist)
    sig = rb.std(dist)
    min = med - sigfac*sig
    max = med + sigfac*sig
#    print "Top plot min: %.5f" % min
#    print "Top plot max: %.5f" % max
    nb = np.round((max-min) / (sig/bindiv))
    ax = plt.subplot(position)
    inds, = np.where((dist >= min) & (dist <= max))
    plt.hist(dist[inds],bins=nb,normed=True,color='black')
    if not xlabel: 
        ax.set_xticklabels(())
    ax.set_yticklabels(())
    ax.set_xlim(min,max)
    xlimits = ax.get_xlim()
    ylimits = ax.get_ylim()
    ax.set_aspect(abs((xlimits[1]-xlimits[0])/(ylimits[1]-ylimits[0]))/aspect)
    if val:
        pass
#        plt.axvline(x=val,color='w',linestyle='--',linewidth=2)
    if xlabel:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(8) 
            tick.label.set_rotation('vertical')
        ax.set_xlabel(xlabel,fontsize=12)
 
    return



def column_plot(dist1,dist2,position,val1=False,val2=False,sigfac=3.0,sigsamp=3.0,frac=0.001,ylabel=None):

    sz1 = len(dist1)
    med1 = np.median(dist1)
    sig1 = rb.std(dist1)
    min1 = med1 - sigfac*sig1
    max1 = med1 + sigfac*sig1
#    min1 = np.float(np.sort(dist1)[np.round(frac*sz1)])
#    max1 = np.float(np.sort(dist1)[np.round((1.-frac)*sz1)])

    sz2 = len(dist2)
    med2 = np.median(dist2)
    sig2 = rb.std(dist2)
    min2 = med2 - sigfac*sig2
    max2 = med2 + sigfac*sig2
#    min2 = np.float(np.sort(dist2)[np.round(frac*sz2)])
#    max2 = np.float(np.sort(dist2)[np.round((1.-frac)*sz2)])

#    inds1, = np.where((dist1 >= min1) & (dist1 <= max1))
#    inds2, = np.where((dist2 >= min2) & (dist2 <= max2))

    aspect = (max1-min1)/(max2-min2)
    X, Y = np.mgrid[min1:max1:100j, min2:max2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([dist1, dist2])

    kernel = KDE(values,var_type='cc',bw=[sig1/sigsamp,sig2/sigsamp])
    Z = np.reshape(kernel.pdf(positions).T, X.shape)

    ax = plt.subplot(position)
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,aspect=aspect,\
              extent=[min1, max1, min2, max2],origin='upper')
#    print "Column plot min1: %.5f" % min1
#    print "Column plot max1: %.5f" % max1
#    print "Column plot min2: %.5f" % min2
#    print "Column plot max2: %.5f" % max2
    clev = np.exp(np.log(np.max(Z))-0.5)
    cset = ax.contour(X,Y,Z,[clev],colors='w',linewidth=5,linestyles='dotted')
#    ax.plot(val1,val2, 'wx', markersize=3)
#    ax.set_xlim(min1, max1)
#    ax.set_ylim(min2, max2)
    ax.set_xticklabels(())
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
    ax.set_ylabel(ylabel,fontsize=12)

    return



def row_plot(dist1,dist2,position,val1=False,val2=False,sigfac=3.0,sigsamp=3.0,xlabel=None):


    sz1 = len(dist1)
    med1 = np.median(dist1)
    sig1 = rb.std(dist1)
    min1 = med1 - sigfac*sig1
    max1 = med1 + sigfac*sig1
#    min1 = np.float(np.sort(dist1)[np.round(frac*sz1)])
#    max1 = np.float(np.sort(dist1)[np.round((1.-frac)*sz1)])
    
    sz2 = len(dist2)
    med2 = np.median(dist2)
    sig2 = rb.std(dist2)
    min2 = med2 - sigfac*sig2
    max2 = med2 + sigfac*sig2
#    min2 = np.float(np.sort(dist2)[np.round(frac*sz2)])
#    max2 = np.float(np.sort(dist2)[np.round((1.-frac)*sz2)])
    
    inds1, = np.where((dist1 >= min1) & (dist1 <= max1))
    inds2, = np.where((dist2 >= min2) & (dist2 <= max2))

    aspect = (max1-min1)/(max2-min2)
    X, Y = np.mgrid[min1:max1:100j, min2:max2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([dist1, dist2])

    kernel = KDE(values,var_type='cc',bw=[sig1/sigsamp,sig2/sigsamp])
    Z = np.reshape(kernel.pdf(positions).T, X.shape)

    ax = plt.subplot(position)
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,aspect=aspect,\
              extent=[min1, max1, min2, max2],origin='upper')
#    print "Row plot min1: %.5f" % min1
#    print "Row plot max1: %.5f" % max1
#    print "Row plot min2: %.5f" % min2
#    print "Row plot max2: %.5f" % max2
    clev = np.exp(np.log(np.max(Z))-0.5)
    cset = ax.contour(X,Y,Z,[clev],colors='w',linewidth=5,linestyles='dotted')
#    ax.plot(val1,val2, 'wx', markersize=3)
    ax.set_xlim(min1, max1)
    ax.set_ylim(min2, max2)
    ax.set_yticklabels(())
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
        tick.label.set_rotation('vertical')
    ax.set_xlabel(xlabel,fontsize=12)
    return


def middle_plot(dist1,dist2,position,val1=False,val2=False,sigfac=3.0,sigsamp=3.0):

    sz1 = len(dist1)
    med1 = np.median(dist1)
    sig1 = rb.std(dist1)
    min1 = med1 - sigfac*sig1
    max1 = med1 + sigfac*sig1
#    min1 = np.float(np.sort(dist1)[np.round(frac*sz1)])
#    max1 = np.float(np.sort(dist1)[np.round((1.-frac)*sz1)])

    sz2 = len(dist2)
    med2 = np.median(dist2)
    sig2 = rb.std(dist2)
    min2 = med2 - sigfac*sig2
    max2 = med2 + sigfac*sig2
#    min2 = np.float(np.sort(dist2)[np.round(frac*sz2)])
#    max2 = np.float(np.sort(dist2)[np.round((1.-frac)*sz2)])

    inds1, = np.where((dist1 >= min1) & (dist1 <= max1))
    inds2, = np.where((dist2 >= min2) & (dist2 <= max2))

    aspect = (max1-min1)/(max2-min2)
    X, Y = np.mgrid[min1:max1:100j, min2:max2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([dist1, dist2])

    kernel = KDE(values,var_type='cc',bw=[sig1/sigsamp,sig2/sigsamp])
    Z = np.reshape(kernel.pdf(positions).T, X.shape)

    ax = plt.subplot(position)
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,aspect=aspect,\
              extent=[min1, max1, min2, max2],origin='upper')
#    print "Middle plot min1: %.5f" % min1
#    print "Middle plot max1: %.5f" % max1
#    print "Middle plot min2: %.5f" % min2
#    print "Middle plot max2: %.5f" % max2
    clev = np.exp(np.log(np.max(Z))-0.5)
    cset = ax.contour(X,Y,Z,[clev],colors='w',linewidth=5,linestyles='dotted')
#    ax.plot(val1,val2, 'wx', markersize=3)
    ax.set_xlim(min1, max1)
    ax.set_ylim(min2, max2)
    ax.set_xticklabels(())
    ax.set_yticklabels(())
    return



def corner_plot(dist1,dist2,position,val1=False,val2=False,\
                sigfac=3.0,sigsamp=3.0,xlabel=None,ylabel=None):

    sz1 = len(dist1)
    med1 = np.median(dist1)
    sig1 = rb.std(dist1)
    min1 = med1 - sigfac*sig1
    max1 = med1 + sigfac*sig1
#    min1 = np.float(np.sort(dist1)[np.round(frac*sz1)])
#    max1 = np.float(np.sort(dist1)[np.round((1.-frac)*sz1)])

    sz2 = len(dist2)
    med2 = np.median(dist2)
    sig2 = rb.std(dist2)
    min2 = med2 - sigfac*sig2
    max2 = med2 + sigfac*sig2
#    min2 = np.float(np.sort(dist2)[np.round(frac*sz2)])
#    max2 = np.float(np.sort(dist2)[np.round((1.-frac)*sz2)])

    inds1, = np.where((dist1 >= min1) & (dist1 <= max1))
    inds2, = np.where((dist2 >= min2) & (dist2 <= max2))

    aspect = (max1-min1)/(max2-min2)
    X, Y = np.mgrid[min1:max1:100j, min2:max2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([dist1, dist2])

    kernel = KDE(values,var_type='cc',bw=[sig1/sigsamp,sig2/sigsamp])
    Z = np.reshape(kernel.pdf(positions).T, X.shape)

    ax = plt.subplot(position)
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,aspect=aspect,\
              extent=[min1, max1, min2, max2],origin='upper')
    clev = np.exp(np.log(np.max(Z))-0.5)
    cset = ax.contour(X,Y,Z,[clev],colors='w',linewidth=5,linestyles='dotted')
#    ax.plot(val1,val2, 'wx', markersize=3)
    ax.set_xlim(min1, max1)
    ax.set_ylim(min2, max2)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
        tick.label.set_rotation('vertical')
    ax.set_xlabel(xlabel,fontsize=12)
    ax.set_ylabel(ylabel,fontsize=12)

    return



def distparams(dist):

#    vals = np.linspace(np.min(dist)*0.5,np.max(dist)*1.5,1000)
#    try:
#        kde = gaussian_kde(dist)
#        pdf = kde(vals)
#        dist_c = np.cumsum(pdf)/np.nansum(pdf)
#        func = sp.interpolate.interp1d(dist_c,vals,kind='linear')
#        lo = np.float(func(math.erfc(1./np.sqrt(2))))
#        hi = np.float(func(math.erf(1./np.sqrt(2))))
#        med = np.float(func(0.5))
#        mode = vals[np.argmax(pdf)]
#        disthi = np.linspace(.684,.999,100)
#        distlo = disthi-0.6827
#        disthis = func(disthi)
#        distlos = func(distlo)
#        interval = np.min(disthis-distlos)
#    except:
    print 'Using "normal" stats.'
    interval = 2.0*np.std(dist)
    med = np.median(dist)
    mode = med
    lo = med-interval/2.0
    hi = med+interval/2.0
    
    return med,mode,np.abs(interval),lo,hi

