import ebsim as ebs
import numpy as np
import constants as c

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

obsdur = 130.0
int = 1800.0

# Adjust this so that the out of eclipse light variations are 2% peak to peak
spotamp1 = 0.13
# From Swift AC analysis
spotP1 = 3.99*86400/period
# Just a guess, could be anything
spotfrac1 = 0.75
spotbase1 = -0.012
# Estimated from preliminary fit residuals to Kepler data
photnoise=300.0/1e6


ebpar,data = ebs.make_model_data(m1=m1/c.Msun, m2=m2/c.Msun, r1=r1/c.Rsun, r2=r2/c.Rsun,
                                 impact=impact, period=period/86400.0, t0=t0, L3=0.0,
                                 vsys=vsys, photnoise=photnoise, RVnoise=RVnoise,
                                 RVsamples=RVsamples, obsdur=obsdur, int=int, durfac=20.0,
                                 spotamp1=spotamp1, spotP1=spotP1, spotfrac1=spotfrac1,
                                 spotbase1=spotbase1, network='swift', J=J, l1=l1/c.Lsun, l2=l2/c.Lsun)
                                 

ebs.check_model(data)
