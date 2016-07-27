# -*- coding: utf-8 -*-
"""
Created on Wed Jul 06 10:23:49 2016

@author: dng5
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as sLA
import time

hbar = 1.0545718e-34 # reduced Planck constant m^2 kg/s
mRb =1.44467e-25 #mass of rubidium in kg
lambdaR = 790e-9 # Raman wavelength in m
lambdaL = 1.064e-6 #lattice wavelength in m
Erecoil = (2.0*np.pi*hbar)**2.0/(2.0*mRb*lambdaL**2.0) #recoil energy


a=-40*np.pi
b=40*np.pi
alpha=1.0e-6
xSteps=1024


latticeRampOnt=0.03 # in seconds
ramanRampOnt=0.02 # in seconds
omegaMax=0.5
delta0=0.0
deltaN=0.0
phi=0.8*2.0*np.pi
epsilon=0.048
Vmax=6.0
x0max=-1964.0
xHopt=10.0e-6 # in seconds
phi=4.0/3.0

tMax=0.090 # in seconds
tstep=0.1 # in recoils


def params(t, Vmax, omegaMax, x0Max, delta0):
    latticeRampOntr=latticeRampOnt*Erecoil/hbar
    ramanRampOntr=ramanRampOnt*Erecoil/hbar
    xHoptr=xHopt*Erecoil/hbar
    if t<latticeRampOntr:
        omega=0.0
        x0=0.0
        V0=Vmax*t/latticeRampOntr
        updating=True
        delta=delta0+deltaN*np.sin(2.0*np.pi*t*hbar/Erecoil+phi)
    elif t<latticeRampOntr+ramanRampOntr:
        omega=omegaMax*(t-latticeRampOntr)/ramanRampOntr
        V0=Vmax
        x0=0
        delta=delta0+deltaN*np.sin(2.0*np.pi*t*hbar/Erecoil+phi)
        updating=True
    elif t< latticeRampOntr+ramanRampOntr+xHoptr:
        V0=Vmax
        omega=omegaMax
        x0=x0Max*(t-latticeRampOntr-ramanRampOntr)/xHoptr
        delta=delta0+deltaN*np.sin(2.0*np.pi*t*hbar/Erecoil+phi)
        updating=True
    else:
        V0=Vmax
        omega=omegaMax
        x0=x0Max
        delta=delta0+deltaN*np.sin(2.0*np.pi*t*hbar/Erecoil+phi)
        if deltaN==0.0:
            updating=False
        else:
            updating=True
    return V0,omega,x0,delta,updating   
    
def Vho(x,alpha=1.0,x0=0.0):
    return alpha*((x-x0)**2.0)
    
def vLatHo(x,V0=6.0,alpha=0.00015,x0=0.0):
    return (V0/1.0)*(np.sin(x))**2.0+Vho(x,alpha=alpha,x0=x0)

def Fx(S):
    F=np.zeros((2*S+1,2*S+1))
    for i in range(2*S+1):
        for j in range(2*S+1):
            if np.abs(i-j)==1:
                F[i,j]=(1.0/2.0)*np.sqrt(S*(S+1)-(i-S)*(j-S))
    return F
    
        
def vRaman(x,omega=1.0,delta=0.0,epsilon=0.048,phi=4.0/3.0):
    x=np.array(x)    
    s=1
    v=v=np.einsum('i,jk->ijk',omega*np.exp(1.0j*2.0*phi*x),Fx(s)*np.sqrt(2.0)/2.0)
    v=np.array([np.triu(v[i])+np.conjugate(np.triu(v[i],1)).transpose() for i in range(x.size)])
    v+=np.array([np.diag([epsilon+delta,0.0,epsilon-delta])]*x.size)
    return v


def getEigenHam2(a,b,xSteps,V,omega=1.0,delta=0.0,epsilon=0.048,phi=4.0/3.0,*args,**kwargs):
    s=1
    xgrid=np.linspace(a,b,xSteps)
  #  print '1. Xgrid:' +str(xgrid.shape)
    eigE=np.zeros((xgrid.size,2*s+1),dtype=complex)
    eigV=np.zeros((xgrid.size,2*s+1,2*s+1),dtype=complex)
    eigVdagger=np.zeros((xgrid.size,2*s+1,2*s+1),dtype=complex)
    for i,x in enumerate(xgrid):
        Vgrid=np.diag(np.array([V(x,*args,**kwargs)]*(2*s+1)))
        Vspin=vRaman(x,omega=omega,delta=delta,epsilon=epsilon,phi=phi)
        Vtot=Vgrid+Vspin
        eigE[i],eigV[i]=sLA.eigh(Vtot)
        eigVdagger[i]=np.conj(eigV[i]).transpose()

    return eigE,eigV,eigVdagger 

def getEigenHam3(a,b,xSteps,V,omega=1.0,delta=0.0,epsilon=0.048,phi=4.0/3.0,*args,**kwargs):
    s=1
    xgrid=np.linspace(a,b,xSteps)
  #  print '1. Xgrid:' +str(xgrid.shape)
    eigE=np.zeros((xgrid.size,2*s+1),dtype=complex)
    eigV=np.zeros((xgrid.size,2*s+1,2*s+1),dtype=complex)
    eigVdagger=np.zeros((xgrid.size,2*s+1,2*s+1),dtype=complex)
    
    Vgrid=np.array([np.diag([V(xgrid,*args,**kwargs)[i]]*3) for i in range(xgrid.size)])
    Vspin=vRaman(xgrid,omega=omega,delta=delta,epsilon=epsilon,phi=phi)
    Vtot=Vgrid+Vspin
    eigE,eigV=np.linalg.eig(Vtot)
    eigVdagger=np.swapaxes(np.conj(eigV),1,2)

    return eigE,eigV,eigVdagger 

def splitStepPropagatorEigB2(psi,dt,a,b,eigE,eigV,eigVdagger):
    xgrid=np.linspace(a,b,psi.shape[0])
    U=np.exp(-1.0j*eigE*dt/2.0)
    psi1=U*psi
    psi1=np.einsum('ijk,ik->ij',eigV,psi1)
    kgrid=np.fft.fftfreq(xgrid.size,d=((b-a)/xgrid.size/(2.0*np.pi)))
    fft1=np.fft.fft(psi1.transpose())
    psi2=np.exp(-1.0j*dt*(kgrid**2.0))*fft1
    fft2=np.fft.ifft(psi2).transpose()
    psi3=np.einsum('ijk,ik->ij',eigVdagger,fft2)
    psi3=U*psi3
    return psi3
    
def splitStepPropagatorUncoupledSpins(psi,V,dt,a,b,*args,**kwargs):
    xgrid=np.linspace(a,b,psi.shape[0])
    Vgrid=V(xgrid,*args,**kwargs)
    U=np.array([np.exp(-1.0j*Vgrid*dt/2.0)]*3).transpose()
    psi1=U*psi
    kgrid=np.fft.fftfreq(xgrid.size,d=((b-a)/xgrid.size/(2.0*np.pi)))
    psi2=np.exp(-1.0j*dt*(kgrid**2.0))*np.fft.fft(psi1.transpose())
    psi3=U*np.fft.ifft(psi2).transpose()
    return psi3

def propagateInTime(psi0,V,a,b,tf,dt,omegaMax=1.0,delta0=0.0,epsilon=0.048,phi=4.0/3.0,x0max=0.0,Vmax=6.0,**kwargs):
    xgrid=np.linspace(a,b,psi0.shape[0])
    dx=(b-a)/(psi0.shape[0])
    tgrid=np.arange(dt,tf,dt)
    fracM=np.zeros(tgrid.size)
    frac0=np.zeros(tgrid.size)
    fracP=np.zeros(tgrid.size)
    com0=np.zeros(tgrid.size)
    
    for ind,t in enumerate(tgrid):
        psiMag=psi0*np.conj(psi0)
        fracM[ind]=dx*np.sum(psiMag[:,2])
        frac0[ind]=dx*np.sum(psiMag[:,1])
        fracP[ind]=dx*np.sum(psiMag[:,0]) 
        com0[ind]=np.sum(xgrid*(psiMag[:,1]+psiMag[:,0]+psiMag[:,2])*dx)
        
        V0,omega,x0,delta,updating=params(t,Vmax,omegaMax,x0max,delta0)
        change=0
        if omega==0.0:
            psi0=splitStepPropagatorUncoupledSpins(psi0,V,dt,a,b,V0=V0,x0=x0,**kwargs)
        elif updating:    
            eigE,eigV,eigVdagger=getEigenHam3(a,b,psi0.shape[0],V,omega=omega,delta=delta,epsilon=epsilon,phi=phi,x0=x0,V0=V0,**kwargs)
            psi0eigB=np.einsum('ijk,ik->ij',eigVdagger,psi0)
            psi0eigB=splitStepPropagatorEigB2(psi0eigB,dt,a,b,eigE,eigV,eigVdagger)
            psi0=np.einsum('ijk,ik->ij',eigV,psi0eigB)
        else:
            if change==0:
                eigE,eigV,eigVdagger=getEigenHam3(a,b,psi0.shape[0],V,omega=omega,delta=delta,epsilon=epsilon,phi=phi,x0=x0,V0=V0,**kwargs)
                change+=1
            psi0eigB=np.einsum('ijk,ik->ij',eigVdagger,psi0)
            psi0eigB=splitStepPropagatorEigB2(psi0eigB,dt,a,b,eigE,eigV,eigVdagger)
            psi0=np.einsum('ijk,ik->ij',eigV,psi0eigB)
            
        
    return fracM,frac0,fracP,com0,tgrid,psi0   
    
    

xgrid=np.linspace(a,b,xSteps)
kgrid=np.fft.fftfreq(xgrid.size,d=((b-a)/xgrid.size/(2.0*np.pi)))
dx=xgrid[1]-xgrid[0]
psiAn=np.exp(-np.sqrt(alpha)*xgrid**2.0/2.0)
psiAn=psiAn/np.sqrt(np.dot(psiAn,np.conj(psiAn)*dx))
Vgrid=vLatHo(xgrid,alpha=alpha)

psi0=np.swapaxes(np.array([np.zeros(psiAn.size),psiAn,np.zeros(psiAn.size)],dtype=complex),0,1)
#fig2=plt.figure()
#pan2=fig2.add_subplot(1,1,1)
#pan2.plot(xgrid,Vgrid)

Vgrid=vLatHo(xgrid, alpha=alpha,x0=19640,V0=0.0)
print np.gradient(Vgrid)[xSteps/2.0]/(xgrid[1]-xgrid[0])

s=1

t1=time.clock()
fracM,frac0,fracP,com0,tgrid,psiOut=propagateInTime(psi0,vLatHo,a,b,tMax*Erecoil/hbar,tstep,omegaMax=omegaMax,delta0=delta0,epsilon=epsilon,phi=phi,alpha=alpha,Vmax=Vmax,x0max=x0max)
t2=time.clock()
print 'Time propagation completed in %f seconds' %(t2-t1)
psiOut=psiOut.reshape(xgrid.size,2*s+1).transpose()
    
#    
#deltaList = np.arange(-0.15,0.15,0.003)
#fracMofD=np.zeros(deltaList.size)
#frac0ofD=np.zeros(deltaList.size)
#fracPofD=np.zeros(deltaList.size)
#
#
#for ind,delta in enumerate(deltaList):
#    
#    t1=time.clock()
#    fracM,frac0,fracP,com0,tgrid,psiOut=propagateInTime(psi0,vLatHo,a,b,tMax*Erecoil/hbar,tstep,omegaMax=omegaMax,delta0=delta0,epsilon=epsilon,phi=4.0/3.0,alpha=alpha,Vmax=Vmax,x0max=x0max)
#    fracMofD[ind]=fracM[-1]
#    frac0ofD[ind]=frac0[-1]
#    fracPofD[ind]=fracP[-1]
#    t2=time.clock()
#    print 'Time propagation completed in %f seconds for index %i' %(t2-t1,ind)
#    psiOut=psiOut.reshape(xgrid.size,2*s+1).transpose()
#
#
#fig1=plt.figure()
#pan1=fig1.add_subplot(1,1,1)
#pan1.plot(deltaList,fracPofD,'b-', label='mF=+1')   
#pan1.plot(deltaList,frac0ofD,'g-', label='mF=0') 
#pan1.plot(deltaList,fracMofD,'r-', label='mF=-1')
#pan1.set_xlabel('Detuning')
#pan1.set_title(r'$\Omega$=%.2f,V=%.2f,$\delta$=%.3f,$\alpha$=%.0f e-6,$x_0$=%.0f,'%(omegaMax,Vmax,delta,alpha*1e6,x0max)+'\n'+ r'$t_{latramp}$=%.3f,$t_{ramanramp}$=%.3f,xSteps=%.0f,tstep=' %(latticeRampOnt,ramanRampOnt,xSteps)+str(np.round(tstep*1e6*hbar/Erecoil,3)))

#fig=plt.figure()
#pan=fig.add_subplot(1,1,1)
#pan.plot(xgrid,psiOut[0]*np.conj(psiOut[0]),'b-', label='mF=+1')   
#pan.plot(xgrid,psiOut[1]*np.conj(psiOut[1]),'g-', label='mF=0') 
#pan.plot(xgrid,psiOut[2]*np.conj(psiOut[2]),'r-', label='mF=-1')
#pan.plot(xgrid, psiAn*np.conj(psiAn),'k-')
#fig.show()
#
#psiFFt=np.fft.fft(psiOut)
#fig=plt.figure()
#pan=fig.add_subplot(1,1,1)
#pan.plot(kgrid,psiFFt[0]*np.conj(psiFFt[0]),'bo', label='mF=+1')   
#pan.plot(kgrid,psiFFt[1]*np.conj(psiFFt[1]),'go', label='mF=0') 
#pan.plot(kgrid,psiFFt[2]*np.conj(psiFFt[2]),'ro', label='mF=-1')
##

#dataFile=np.load('..\\Raman\\29Jun2016_files_37-146.npz')
#imbal=dataFile['imbalArray']
#signalGood=dataFile['signalGood']
#cutoff=0.35
#fieldGoodArray=((imbal<cutoff) & signalGood)
#fractionP=dataFile['fractionP'][fieldGoodArray]
#fraction0=dataFile['fraction0'][fieldGoodArray]
#fractionM=dataFile['fractionM'][fieldGoodArray]
#time=dataFile['tlist'][fieldGoodArray]+(latticeRampOnt+ramanRampOnt)*hbar/Erecoil

fig1=plt.figure()
pan1=fig1.add_subplot(1,1,1)
pan1.plot(tgrid*hbar*1e3/Erecoil,fracP,'b-', label='mF=+1')   
pan1.plot(tgrid*hbar*1e3/Erecoil,frac0,'g-', label='mF=0') 
pan1.plot(tgrid*hbar*1e3/Erecoil,fracM,'r-', label='mF=-1')
pan1.set_title(r'$\Omega$=%.2f,V=%.2f,$\delta_0$=%.3f,$\delta_N$=%.3f,$\phi$=%.3f,$\alpha$=%.0f e-6,$x_0$=%.0f,'%(omegaMax,Vmax,delta0,deltaN,phi,alpha*1e6,x0max)+'\n'+ r'$t_{latramp}$=%.3f,$t_{ramanramp}$=%.3f,xSteps=%.0f,tstep=' %(latticeRampOnt,ramanRampOnt,xSteps)+str(np.round(tstep*1e6*hbar/Erecoil,3)))
pan1.set_xlim(latticeRampOnt*1e3+ramanRampOnt*1e3,tMax*1e3)
#pan1.plot(time*1.0e3,fractionP,'bo', label=r'$m_F$=+1')
#pan1.plot(time*1.0e3,fraction0,'go', label=r'$m_F$=0')
#pan1.plot(time*1.0e3,fractionM,'ro', label=r'$m_F$=-1')
#
#fig2=plt.figure()
#pan2=fig2.add_subplot(1,1,1)
#pan2.plot(tgrid*hbar*1e3/Erecoil,com0,'b-')