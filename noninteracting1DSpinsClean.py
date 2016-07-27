# -*- coding: utf-8 -*-
"""
Created on Tue Jul 05 15:39:40 2016

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



a=-50*np.pi
b=50*np.pi
alpha=1.0e-6
xSteps=4096
x0=-10476
dt=0.1
omega=0.5
delta=0.02
epsilon=0.048
V0=6.0
phi=0.0
tMax=0.01*Erecoil/hbar

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
    v=np.outer(omega*np.exp(1.0j*2.0*phi*x),Fx(s)*np.sqrt(2.0)/2.0).reshape(x.size,2*s+1,2*s+1)
    v=sLA.block_diag(*[np.triu(v[i])+np.conjugate(np.triu(v[i],1)).transpose() for i in range(x.size)])
    v+=sLA.block_diag(*[np.diag([epsilon-delta,0.0,epsilon+delta])]*x.size)
    return v
    
def splitStepPropagator0(psi,V,dt,a,b,*args,**kwargs):
    xgrid=np.linspace(a,b,psi.size)
    Vgrid=V(xgrid,*args,**kwargs)
    psi1=np.exp(-1.0j*Vgrid*dt/2.0)*psi
    kgrid=np.fft.fftfreq(xgrid.size,d=(np.abs(xgrid[1]-xgrid[0]))/(2.0*np.pi))
    psi2=np.exp(-1.0j*dt*(kgrid**2.0)/2.0)*np.fft.fft(psi1)
    psi3=np.exp(-1.0j*Vgrid*dt/2.0)*np.fft.ifft(psi2)
    return psi3
   
def getGroundState0(psi0,V,a,b,dt,*args,**kwargs):
  #  xgrid=np.linspace(a,b,psi0.size)
    dx=(b-a)/psi0.size
    psi0=psi0/np.sqrt(np.dot(psi0,np.conj(psi0)*dx))
    psi1=splitStepPropagator0(psi0,V,np.complex(0.0,-1.0)*dt,a,b,*args,**kwargs)
    psi1=psi1/np.sqrt(np.dot(psi1,np.conj(psi1)*dx))
    diff=np.sum(np.abs(psi0*np.conj(psi0)*dx-psi1*np.conj(psi1)*dx))
    i=0
    while diff>0.2e-5:
        psi0=psi1
        psi1=splitStepPropagator0(psi0,V,-1.0j*dt,a,b,*args,**kwargs)
        psi1=psi1/np.sqrt(np.dot(psi1,np.conj(psi1)*dx))
        diffLast=diff
        diff=np.sum(np.abs(psi0*np.conj(psi0)*dx-psi1*np.conj(psi1)*dx))
        if diffLast<diff:
            print 'Not converging! Difference went up from %f to %f' %(diffLast,diff)
            break
        i+=1
    print i
    print np.sum(np.abs(psi0*np.conj(psi0)*dx-psi1*np.conj(psi1)*dx))
    return psi1


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
        eigE[i],eigV[i]=sLA.eig(Vtot)
        eigVdagger[i]=np.conj(eigV[i]).transpose()

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


def getGroundState(psi0,V,a,b,dt,omega=1.0,delta=0.0,epsilon=0.048,phi=4.0/3.0,*args,**kwargs):
    dx=(b-a)/(psi0.shape[0])
    t0=time.clock()
    eigE,eigV,eigVdagger=getEigenHam2(a,b,psi0.shape[0],V,omega=omega,delta=delta,epsilon=epsilon,phi=phi,*args,**kwargs)
    t1=time.clock()
    print 'Got EigenHam in '+str(t1-t0)+' seconds!'
    psi0=psi0/np.sqrt(np.sum(psi0*np.conj(psi0)*dx))
    psi0eigB=np.einsum('ijk,ik->ij',eigVdagger,psi0)
    t0=time.clock()
    psi1eigB=splitStepPropagatorEigB2(psi0eigB,dt*np.complex(0.0,-1.0),a,b,eigE,eigV,eigVdagger)
    t1=time.clock()
    psi1eigB=psi1eigB/np.sqrt(np.sum(psi1eigB*np.conj(psi1eigB)*dx))
    t2=time.clock()
    print 'Completed one time step in '+str(t1-t0)+' seconds!'
    print 'Then normalized wavefunction in '+str(t2-t1)+' seconds!'
    diff=np.sum(np.abs(psi0eigB*np.conj(psi0eigB)*dx-psi1eigB*np.conj(psi1eigB)*dx))
    i=0
    while diff>0.1e-5:
        psi0eigB=psi1eigB
        psi1eigB=splitStepPropagatorEigB2(psi0eigB,dt*np.complex(0.0,-1.0),a,b,eigE,eigV,eigVdagger)
        psi1eigB=psi1eigB/np.sqrt(np.sum(psi1eigB*np.conj(psi1eigB)*dx))
        diffLast=diff
        diff=np.sum(np.abs(psi0eigB*np.conj(psi0eigB)*dx-psi1eigB*np.conj(psi1eigB)*dx))
        if diffLast<diff:
            print 'Not converging! Difference went up from %f to %f' %(diffLast,diff)
            break
        i+=1
    print i
    print diff
    psi1=np.einsum('ijk,ik->ij',eigV,psi1eigB)
    psi1=psi1.transpose()
    return psi1

def propagateInTime(psi0,V,a,b,tf,dt,omega=1.0,delta=0.0,epsilon=0.048,phi=4.0/3.0,*args,**kwargs):
    xgrid=np.linspace(a,b,psi0.shape[0])
    dx=(b-a)/(psi0.shape[0])
    tgrid=np.arange(dt,tf,dt)
    fracM=np.zeros(tgrid.size)
    frac0=np.zeros(tgrid.size)
    fracP=np.zeros(tgrid.size)
    com0=np.zeros(tgrid.size)
    eigE,eigV,eigVdagger=getEigenHam2(a,b,psi0.shape[0],V,omega=omega,delta=delta,epsilon=epsilon,phi=phi,*args,**kwargs)
    for ind,t in enumerate(tgrid):
        psiMag=psi0*np.conj(psi0)
        fracM[ind]=dx*np.sum(psiMag[:,2])
        frac0[ind]=dx*np.sum(psiMag[:,1])
        fracP[ind]=dx*np.sum(psiMag[:,0]) 
        com0[ind]=np.sum(xgrid*(psiMag[:,1]+psiMag[:,0]+psiMag[:,2])*dx)
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


#start=time.clock()
#psi1spin = getGroundState0(psiAn,vLatHo,a,b,0.01,alpha=alpha)
#end=time.clock()
#print 'Got single spin ground state in %f seconds' %(end-start)
#fig1=plt.figure()
#pan1=fig1.add_subplot(1,1,1)
#pan1.plot(xgrid,psi1spin*np.conj(psi1spin),'b-')
#pan1.plot(xgrid,psiAn,'k-')


psi0spins=np.swapaxes(np.array([np.sqrt(0.2)*psiAn,np.sqrt(0.6)*psiAn,np.sqrt(0.2)*psiAn],dtype=complex),0,1)
fig2=plt.figure()
pan2=fig2.add_subplot(1,1,1)
pan2.plot(xgrid,Vgrid)

Vgrid=vLatHo(xgrid, alpha=alpha,x0=x0,V0=0.0)
print np.gradient(Vgrid)[xSteps/2.0]/(xgrid[1]-xgrid[0])

start=time.clock()
psi = getGroundState(psi0spins,vLatHo,a,b,0.001,omega=omega,delta=delta,epsilon=epsilon,phi=phi,alpha=alpha,V0=V0,x0=0.0)
end = time.clock()
print 'Got multi-spin ground state in %f seconds' %(end-start)

fig=plt.figure()
pan=fig.add_subplot(1,1,1)
pan.plot(xgrid,psi[0]*np.conj(psi[0]),'b-', label='mF=+1')   
pan.plot(xgrid,psi[1]*np.conj(psi[1]),'g-', label='mF=0') 
pan.plot(xgrid,psi[2]*np.conj(psi[2]),'r-', label='mF=-1')
pan.plot(xgrid, psiAn*np.conj(psiAn),'k-')

psiFFt=np.fft.fft(psi)
fig=plt.figure()
pan=fig.add_subplot(1,1,1)
pan.plot(kgrid,psiFFt[0]*np.conj(psiFFt[0]),'bo', label='mF=+1')   
pan.plot(kgrid,psiFFt[1]*np.conj(psiFFt[1]),'go', label='mF=0') 
pan.plot(kgrid,psiFFt[2]*np.conj(psiFFt[2]),'ro', label='mF=-1')

print np.sum(psi[0]*np.conj(psi[0])*dx)
print np.sum(psi[1]*np.conj(psi[1])*dx)
print np.sum(psi[2]*np.conj(psi[2])*dx)

s=1
t1=time.clock()
fracM,frac0,fracP,com0,tgrid,psiOut=propagateInTime(psi.transpose(),vLatHo,a,b,tMax,dt,omega=omega,delta=delta,epsilon=epsilon,phi=phi,alpha=alpha,x0=0.0)
t2=time.clock()
print 'Time propagation completed in %f seconds' %(t2-t1)
psiOut=psiOut.reshape(xgrid.size,2*s+1).transpose()
fig=plt.figure()
pan=fig.add_subplot(1,1,1)
pan.plot(xgrid,psiOut[0]*np.conj(psiOut[0]),'b-', label='mF=+1')   
pan.plot(xgrid,psiOut[1]*np.conj(psiOut[1]),'g-', label='mF=0') 
pan.plot(xgrid,psiOut[2]*np.conj(psiOut[2]),'r-', label='mF=-1')
fig.show()
#
fig1=plt.figure()
pan1=fig1.add_subplot(1,1,1)
pan1.plot(tgrid*1e3*hbar/Erecoil,fracP,'b-', label='mF=+1')   
pan1.plot(tgrid*1e3*hbar/Erecoil,frac0,'g-', label='mF=0') 
pan1.plot(tgrid*1e3*hbar/Erecoil,fracM,'r-', label='mF=-1')
fig2=plt.figure()
pan2=fig2.add_subplot(1,1,1)
pan2.plot(tgrid,com0,'b-')