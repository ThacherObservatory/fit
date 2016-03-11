import numpy as np
import ebsim as ebs
ebin = ebs.ebinput() 

nphot = 3
band = ['Kp','J','K']
q1a = [None,None,None]
q2a = [None,None,None]
q1b = [None,None,None]
q2b = [None,None,None]
photnoise = [0.0003,0.01,0.01]
obsdur = [30,8./24.,7./24.]
int = [1800.0,60.0,60.0]
durfac = [5.0,3.0,2]
spotamp1 = [None,None,None]
spotP1 = [0.0,0.0,0.0]
P1double = [0.0,0.0,0.0]
spotfrac1 = [None,None,None]
spotbase1 = [None,None,None]
spotamp2 = [None,None,None]
spotP2 = [0.0,0.0,0.0] 
P2double = [0.0,0.0,0.0]
spotfrac2 = [None,None,None]
spotbase2 = [None,None,None]

RVsamples = 20
RVnoise = 3
data_dict = ebs.make_model_data(ebin,nphot=nphot,band=band,photnoise=photnoise,
                                obsdur=obsdur,int=int,durfac=durfac,
                                q1a=q1a,q1b=q1b,q2a=q2a,q2b=q2b,
                                spotamp1=spotamp1,spotP1=spotP1,P1double=P1double,
                                spotfrac1=spotfrac1,spotbase1=spotbase1,
                                spotamp2=spotamp2,spotP2=spotP2,P2double=P2double,
                                spotfrac2=spotfrac2,spotbase2=spotbase2,
                                RVsamples=RVsamples,RVnoise=RVnoise,network='swift')

ebs.check_model(data_dict)
