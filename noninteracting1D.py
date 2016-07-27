# -*- coding: utf-8 -*-
"""
Created on Thu Jun 09 11:48:15 2016

@author: dng5
"""
import numpy as np
import matplotlib.pyplot as plt

def splitStepPropagator(psi,V,dt,a,b,*args,**kwargs):
    xgrid=np.linspace(a,b,psi.size)
    Vgrid=V(xgrid,*args,**kwargs)
    psi1=np.exp(-1.0j*Vgrid*dt/2.0)*psi
    kgrid=np.fft.fftfreq(xgrid.size,d=(np.abs(xgrid[1]-xgrid[0]))/(2.0*np.pi))
    psi2=np.exp(-1.0j*dt*(kgrid**2.0)/2.0)*np.fft.fft(psi1)
    psi3=np.exp(-1.0j*Vgrid*dt/2.0)*np.fft.ifft(psi2)
    return psi3

def Vho(x,alpha=1.0):
    return alpha*(x**2.0)/2.0
    
def vLatHo(x,V0=6.0,alpha=0.00015):
    return V0*(np.sin(x))**2.0+Vho(x,alpha=alpha)
    

    
    
def getGroundState(psi0,V,a,b,dt,*args,**kwargs):
  #  xgrid=np.linspace(a,b,psi0.size)
    dx=(b-a)/psi0.size
    psi0=psi0/np.sqrt(np.dot(psi0,np.conj(psi0)*dx))
    psi1=splitStepPropagator(psi0,V,np.complex(0.0,-1.0)*dt,a,b,*args,**kwargs)
    psi1=psi1/np.sqrt(np.dot(psi1,np.conj(psi1)*dx))
    i=0
    while np.sum(np.abs(psi0*np.conj(psi0)*dx-psi1*np.conj(psi1)*dx))>0.2e-6:
        psi0=psi1
        psi1=splitStepPropagator(psi0,V,-1.0j*dt,a,b,*args,**kwargs)
        psi1=psi1/np.sqrt(np.dot(psi1,np.conj(psi1)*dx))
        i+=1
    print i
    print np.sum(np.abs(psi0*np.conj(psi0)*dx-psi1*np.conj(psi1)*dx))
#    fig=plt.figure()
#    pan=fig.add_subplot(1,1,1)
#    pan.plot(xgrid,psi1*np.conj(psi1))
#    psiAn=np.exp(-xgrid**2.0/2.0)
#    psiAn=psiAn/np.sqrt(np.dot(psiAn,np.conj(psiAn)*dx))
#    #pan.plot(xgrid, psiAn*np.conj(psiAn),'g-')
#    psiAn2=np.exp(-xgrid**2.0/2.0)*xgrid
#    psiAn2=psiAn2/np.sqrt(dx*np.dot(psiAn2,np.conj(psiAn2)))
#   # pan.plot(xgrid, psiAn2*np.conj(psiAn2),'g-')
    return psi1
    
a=-40.0*np.pi
b=40.0*np.pi
alpha=1.0e-6
psi0=np.ones(4096)+0.0j
xgrid=np.linspace(a,b,psi0.size)
dx=xgrid[1]-xgrid[0]
psiAn2=np.exp(-xgrid**2.0/2.0)*xgrid
psiAn=np.exp(-np.sqrt(alpha)*xgrid**2.0/2.0)
psiAn=psiAn/np.sqrt(np.dot(psiAn,np.conj(psiAn)*dx))
Vgrid=vLatHo(xgrid,alpha=alpha)

psi1spin = getGroundState(psi0,vLatHo,a,b,0.1,alpha=alpha,V0=0.0)
fig1=plt.figure()
pan1=fig1.add_subplot(1,1,1)
pan1.plot(xgrid,psi1spin*np.conj(psi1spin),'b-')
pan1.plot(xgrid,psiAn*np.conj(psiAn),'k-')
    