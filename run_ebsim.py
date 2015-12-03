import numpy as np
from ebsim import *
from ebsim_results import *


def ebsim_core(core,ncores=16,nwalkers=1000,burnsteps=1000,mcmcsteps=1000,clobber=False,
               network=None,thin=10,fullanalysis=False,norun=False,no_analyze=False,
               cadence='short'):

    short = True if cadence == 'short' else False
    long = True if cadence == 'long' else False

    # get array of input values and unique sequence number
    params,seq = param_sequence()
    # Total number of fits to be done
    nproc = len(seq)

    # Number of tasks per core
    ntask = nproc/ncores

    # Assign each core "ntask" tasks
    coretask = np.ones(ncores,dtype='int')*ntask

    # Distribute the remained amongst the cores
    rem = np.remainder(nproc,ncores)
    for i in range(rem):
        coretask[i] += 1

    coreseq = range(coretask[core-1])+np.sum(coretask[0:core-1])

    print 'Sequence for core number '+str(core)+':'
    print str(coreseq)

    for i in range(len(coreseq)):
        n = coreseq[i]
        param = params[n,:]
        print ' '
        print ' '
        print ' '
        print ' '
        print '---------------------------------------'
        print 'Starting sequence number '+str(n)+' ...'
        print '---------------------------------------'
        print 'Cadence           = '+cadence
        print 'Period            = %.2f days' % param[0]
        ppm = param[1]*1e6
        print 'Photometric Noise = %.2f ppm' % ppm
        print 'RV Samples        = %.0f' % param[2]
        print 'Radius Ratio      = %.2f' % param[3]
        print 'Impact Parameter  = %.2f' % param[4]

        if norun:
            analyze_run(n,network=network,thin=thin,full=fullanalysis,cadence=cadence)

        elif no_analyze:
            fit_sequence(n,nwalkers=nwalkers,cadence=cadence,
                         burnsteps=burnsteps,mcmcsteps=mcmcsteps,
                         clobber=False,network=network)

        else:
            fit_sequence(n,nwalkers=nwalkers,cadence=cadence,
                         burnsteps=burnsteps,mcmcsteps=mcmcsteps,
                         clobber=False,network=network)

            analyze_run(n,network=network,thin=thin,full=fullanalysis,cadence=cadence)
        

def param_sequence():
    """
    Routine to create a sequence of input parameters for a simulated EB fitting sequence
    
    The parameters that are varied are photometric noise, integration time, RV samples,
    radius ratio of the stars, and impact parameter

    """

    # Period
    periods = np.array([4.53,9.06,13.59])
    # number of eclipse pairs = 7, 3, 2
    
    # Photometric noise (as a fraction of total flux)
    photnoise = np.array([10,100,1000,10000])/1e6

#    # Integration time in seconds
#    itime = np.array([60,900,1800])

    # Total number of RV samples
    RVsamples = np.array([10,50,100])

    # Radius ratio
    Rratio = np.array([0.2,0.6,1.0])

    # Imapct parameters of the primary eclipse
    impact = np.array([0.05,0.75,-999])

    # Total number of iterations needed in full sequence
    niter = len(periods)*len(photnoise)*len(RVsamples)*len(Rratio)*len(impact)
    
    # Array of parameters for each iteration
    params = np.zeros((niter,5))

    # Create array of parameters
    seq_num = []
    index = 0
    for i in range(len(periods)):
        for j in range(len(photnoise)):
            for k in range(len(RVsamples)):
                for l in range(len(Rratio)):
                    for m in range(len(impact)):
                        impval = 1+0.5*Rratio[l] if m == 2 else impact[m]
                        params[index,:] = np.array([periods[i],photnoise[j],RVsamples[k],Rratio[l],impval])
                        seq_num.append(int(index))
                        index += 1


    # params = [periods,photnoise,RVsamples,Rratio,impact]
    
    return params,seq_num


def params_from_seq(seq):
    """
    Return set of simulation parameters for a given sequence number.
    """
    
    par,s = param_sequence()
    return par[seq,:]



def check(seq_num,network=None,cadence='short'):

    path = get_path(network=network)

    params = params_from_seq(seq_num)

    short = True if cadence == 'short' else False
    long = True if cadence == 'long' else False

    ebpar,data = make_model_data(period=params[0],photnoise=params[1],RVsamples=params[2],
                                 r1=0.5,r2=0.5*params[3],impact= params[4],network=network,
                                 path=path+cadence+'/'+str(seq_num)+'/',short=short,long=long)
    check_model(data)
    return


def fit_sequence(seq_num,nwalkers=1000,burnsteps=1000,mcmcsteps=1000,
                 fit_period=True,fit_limb=True,claret=False,fit_rvs=True,
                 fit_ooe1=False,fit_ooe2=False,fit_L3=False,fit_sp2=False,
                 fit_ellipsoidal=False,write=True,order=3,
                 reduce=10,network=None,thin=1,clobber=False,cadence='short'):

    """ 
    Routine to run the sequence of fits generated by param_sequence

    """

    short = True if cadence == 'short' else False
    long = True if cadence == 'long' else False

    # params = [period,photnoise,RVsamples,Rratio,impact]
    params = params_from_seq(seq_num)

    path = get_path(network=network)

    ebpar,data = make_model_data(period=params[0],photnoise=params[1],RVsamples=params[2],
                                  r1=0.5,r2=0.5*params[3],impact=params[4],
                                  path=path+cadence+'/'+str(seq_num)+'/',network=network,
                                 long=long,short=short)

    f = open(path + cadence+'/'+str(seq_num) + "/"+ "initialparams.txt","w")
    f.writelines([str(p) + '\n' for p in params])
    f.close()

    fitinfo = fit_params(ebpar,nwalkers=nwalkers,burnsteps=burnsteps,mcmcsteps=mcmcsteps,
                         clobber=clobber,fit_period=fit_period,fit_limb=fit_limb,claret=claret,
                         fit_rvs=fit_rvs,fit_ooe1=fit_ooe1,fit_ooe2=fit_ooe2,fit_L3=fit_L3,
                         fit_sp2=fit_sp2,fit_ellipsoidal=fit_ellipsoidal,write=write,order=order,
                         thin=thin)
    
    lnprob,chains = ebsim_fit(data,ebpar,fitinfo)

    return


#----------------------------------------------------------------------
# GET_PATH
#----------------------------------------------------------------------
def get_path(network=None):
    """
    get_path:
    ---------
    set path given network choice

    inputs:
    -------
    "network": name of network choices are:
               None: local path
               astro: Caltech astronomy network
               gps: Caltech geology and planetary science network
               eb: local path but for an individual KIC target
    example:
    --------
    path = get_path(network='bellerophon')

    """

    # Set correct paths here
    if network == None or network=='bellerophon':
        path = '/home/administrator/Simulations/'
    elif network=='swift':
        path = '/Users/jonswift/Astronomy/EBs/Simulations/'
    elif network=='doug':
        path = '/home/douglas/Simulations/'
    elif network=='bellerophon-old':
        path = '/home/administrator/old_Simulations/old_02December2015/'

    return path




def thin_chains(koi,planet,thin=10,short=False,network=None,clip=False,limbmodel='quad',rprior=False):
    
    lc,pdata,sdata = get_koi_info(koi,planet,short=short,network=network,\
                                      clip=clip,limbmodel=limbmodel,rprior=rprior)
    t = time.time()
    print 'Importing MCMC chains'
    rprsdist = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_rchain.txt')
    ddist    = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_dchain.txt')*24.
    bdist    = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_bchain.txt')
    tdist    = (np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_t0chain.txt')) * 24.0 * 3600.0 + pdata[0,3]
    pdist    = (np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_pchain.txt')-pdata[0,4])*24.*3600.
    q1dist   = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_q1chain.txt')
    q2dist   = np.loadtxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_q2chain.txt')
    print done_in(t)

    rprsdist = rprsdist[0::thin]
    ddist    = ddist[0::thin]
    tdist    = tdist[0::thin]
    bdist    = bdist[0::thin]
    pdist    = pdist[0::thin]
    q1dist   = q1dist[0::thin]
    q2dist   = q2dist[0::thin]

    t = time.time()
    print 'Exporting thinned chains'
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_thin_rchain.txt',rdist)
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_thin_dchain.txt',ddist)
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_thin_bchain.txt',bdist)
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_thin_t0chain.txt',tdist)
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_thin_pchain.txt',pdist)
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_thin_q1chain.txt',q1dist)
    np.savetxt(path+'MCMC/'+name+stag+ctag+rtag+lptag+ltag+'fit_thin_q2chain.txt',q2dist)
    print done_in(t)
    
    return



#----------------------------------------------------------------------
# EBLIST
#----------------------------------------------------------------------
def eblist(ebpar):

    """ 
    ----------------------------------------------------------------------
    eblist:
    --------
    Check the list of EBs with preliminary data.

    inputs:
    -------
    "path": choose correct path (see get_path routine)
    
    example:
    --------
    In[1]: ebs = eblist(network=None)
    ----------------------------------------------------------------------
    """

    # Directories correspond to KIC numbers
    files = glob.glob(ebpar['path']+'[0-9]*')

    
    eblist = np.zeros(len(files))
    for i in xrange(0,len(files)):
        eblist[i] = files[i].split('/')[-1]

    # Format output
    eblist = np.array(eblist).astype(int)

    return eblist
