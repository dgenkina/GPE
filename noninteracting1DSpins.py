# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:43:01 2016

@author: dng5
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as sLA
import time

def splitStepPropagator(psi,V,dt,a,b,omega=1.0,delta=0.0,epsilon=0.048,phi=4.0/3.0,*args,**kwargs):
    s=1
    xgrid=np.linspace(a,b,psi.size/(2*s+1))
  #  print '1. Xgrid:' +str(xgrid.shape)
    Vgrid=V(xgrid,*args,**kwargs)
 #   print '2. Vgrid:'+str(Vgrid.shape)
    Vgrid = np.diag(np.array([Vgrid]*(2*s+1)).transpose().flatten())
#    print '3. Vgrid, updated:'+str(Vgrid.shape)
    Vspin=vRaman(xgrid,omega=omega,delta=delta,epsilon=epsilon,phi=phi)
    #print '4. Vspin:' +str(Vspin.shape)
    Vtot=Vgrid+Vspin
   # print '5. Vtot:' +str(Vtot.shape)
    eigE,eigV=sLA.eig(Vtot)
    eigV=eigV+0.0j
  #  print '5a. eigV:' +str(eigV.shape)
    eigVdagger=np.conj(eigV).transpose()
    U=np.diag(np.exp(-1.0j*eigE*dt/2.0))
 #   print '5b. U:' +str(U.shape)
    psi1=np.dot(eigV,np.dot(U,np.dot(eigVdagger,psi)))
#    print '6. psi1:'+str( psi1.shape)
    kgrid=np.fft.fftfreq(xgrid.size,d=(np.abs(xgrid[1]-xgrid[0]))/(2.0*np.pi))
    #print '7. kgrid:' +str(kgrid.shape)
    fft1=np.fft.fft(psi1.reshape(xgrid.size,2*s+1).transpose())
   # print '8. fft1:' +str(fft1.shape)
    psi2=np.exp(-1.0j*dt*(kgrid**2.0)/2.0)*fft1
   # print '9. psi2:' +str(psi2.shape)
    fft2=np.fft.ifft(psi2).transpose().flatten()
   # print '10. fft2:'+str(fft2.shape)
    psi3=np.dot(eigV,np.dot(U,np.dot(eigVdagger,fft2)))
   # psi3=np.dot(np.exp(-1.0j*Vtot*dt/2.0),fft2)
   # print psi3.shape
    return psi3#, Vtot,eigE,eigV,eigVdagger,U,psi1,kgrid,fft1,psi2,fft2,psi3
    
def getEigenHam(a,b,xSteps,V,omega=1.0,delta=0.0,epsilon=0.048,phi=4.0/3.0,*args,**kwargs):
    s=1
    xgrid=np.linspace(a,b,xSteps)
  #  print '1. Xgrid:' +str(xgrid.shape)
    eigEloc=np.zeros((xgrid.size,2*s+1),dtype=complex)
    eigVloc=np.zeros((xgrid.size,2*s+1,2*s+1),dtype=complex)
    for i,x in enumerate(xgrid):
        Vgrid=np.diag(np.array([V(x,*args,**kwargs)]*(2*s+1)))
        if i==0:
            print x, V(x,*args,**kwargs)
        Vspin=vRaman(x,omega=omega,delta=delta,epsilon=epsilon,phi=phi)
        Vtot=Vgrid+Vspin
        eigEloc[i],eigVloc[i]=sLA.eig(Vtot)
        eigVloc[i]=eigVloc[i]+0.0j


    eigE=eigEloc.flatten()
 #   print '1a. eigE:' +str(eigE.shape)
    eigV=sLA.block_diag(*[eigVloc[i] for i in range(xgrid.size)]) 
#    print '1b. eigV:' +str(eigV.shape)
    #print '5a. eigV:' +str(eigV.shape)
    eigVdagger=np.conj(eigV).transpose()
    return eigE,eigV,eigVdagger

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
    
def splitStepPropagator1(psi,dt,a,b,eigE,eigV,eigVdagger):
    s=1
    t0=time.clock()
    xgrid=np.linspace(a,b,psi.size/(2*s+1))
    U=np.diag(np.exp(-1.0j*eigE*dt/2.0))
    t1=time.clock()
#    print 'U:' +str(t1-t0)
    psi1=np.dot(eigV,np.dot(U,np.dot(eigVdagger,psi)))
    t2=time.clock()
 #   print 'psi1:' +str(t2-t1)
    kgrid=np.fft.fftfreq(xgrid.size,d=(np.abs(xgrid[1]-xgrid[0]))/(2.0*np.pi))
    t3=time.clock()
 #   print 'kgrid:' +str(t3-t2)
    fft1=np.fft.fft(psi1.reshape(xgrid.size,2*s+1).transpose())
    t4=time.clock()
 #   print 'fft1' +str(t4-t3)
    psi2=np.exp(-1.0j*dt*(kgrid**2.0)/2.0)*fft1
    t5=time.clock()
 #   print 'psi2:' +str(t5-t4)
    fft2=np.fft.ifft(psi2).transpose().flatten()
    t6=time.clock()
#    print 'fft2:' +str(t6-t5)
    psi3=np.dot(eigV,np.dot(U,np.dot(eigVdagger,fft2)))
    t7=time.clock()
#    print 'psi3:' +str(t7-t6)
    return psi3#, Vtot,eigE,eigV,eigVdagger,U,psi1,kgrid,fft1,psi2,fft2,psi3
    
def splitStepPropagatorEigB(psi,dt,a,b,eigE,eigV,eigVdagger):
    s=1
    t0=time.clock()
    xgrid=np.linspace(a,b,psi.size/(2*s+1))
    U=np.exp(-1.0j*eigE*dt/2.0)
    t1=time.clock()
#    print 'U:' +str(t1-t0)
    psi1=U*psi
    t2=time.clock()
 #   print 'psi1:' +str(t2-t1)
    kgrid=np.fft.fftfreq(xgrid.size,d=(np.abs(xgrid[1]-xgrid[0]))/(2.0*np.pi))
    t3=time.clock()
 #   print 'kgrid:' +str(t3-t2)
    fft1=np.fft.fft(psi1.reshape(xgrid.size,2*s+1).transpose())
    t4=time.clock()
 #   print 'fft1' +str(t4-t3)
    psi2=np.exp(-1.0j*dt*(kgrid**2.0)/2.0)*fft1
    t5=time.clock()
 #   print 'psi2:' +str(t5-t4)
    fft2=np.fft.ifft(psi2).transpose().flatten()
    t6=time.clock()
#    print 'fft2:' +str(t6-t5)
    psi3=U*fft2
    t7=time.clock()
#    print 'psi3:' +str(t7-t6)
    return psi3#, Vtot,eigE,eigV,eigVdagger,U,psi1,kgrid,fft1,psi2,fft2,psi3

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
    
def vRaman1(x,omega=1.0,delta=0.0,epsilon=0.048,phi=4.0/3.0):    
    s=1
    v=omega*np.exp(1.0j*2.0*phi*np.pi*x)*Fx(s)*np.sqrt(2.0)/2.0
    v=np.triu(v)+np.conjugate(np.triu(v,1)).transpose() 
    v+=np.diag([epsilon-delta,0.0,epsilon+delta])
    return v
    
    
def getGroundState(psi0,V,a,b,dt,omega=1.0,delta=0.0,epsilon=0.048,phi=4.0/3.0,*args,**kwargs):
  #  s=1
   # psi0=psi0.transpose().flatten()psi0eigB=np.array(map(np.dot,eigVdagger,psi0))
 #   xgrid=np.linspace(a,b,psi0.shape[0])
    dx=(b-a)/(psi0.shape[0])
    t0=time.clock()
    eigE,eigV,eigVdagger=getEigenHam2(a,b,psi0.shape[0],V,omega=omega,delta=delta,epsilon=epsilon,phi=phi,*args,**kwargs)
    t1=time.clock()
    print 'Got EigenHam in '+str(t1-t0)+' seconds!'
 #   psi0=psi0/np.sqrt(np.dot(psi0,np.conj(psi0)*dx))
    psi0=psi0/np.sqrt(np.sum(psi0*np.conj(psi0)*dx))
  #  psi0eigB=np.dot(eigVdagger,psi0)
    psi0eigB=np.einsum('ijk,ik->ij',eigVdagger,psi0)
    t0=time.clock()
    psi1eigB=splitStepPropagatorEigB2(psi0eigB,dt*np.complex(0.0,-1.0),a,b,eigE,eigV,eigVdagger)
    t1=time.clock()
    psi1eigB=psi1eigB/np.sqrt(np.sum(psi1eigB*np.conj(psi1eigB)*dx))
    t2=time.clock()
    print 'Completed one time step in '+str(t1-t0)+' seconds!'
    print 'Then normalized wavefuntion in '+str(t2-t1)+' seconds!'
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

 #   psi1=np.dot(eigV,psi1eigB)
    psi1=np.einsum('ijk,ik->ij',eigV,psi1eigB)
#    psi1=psi1.reshape(xgrid.size,2*s+1).transpose()
    psi1=psi1.transpose()
    return psi1

def propagateInTime(psi0,V,a,b,tf,dt,omega=1.0,delta=0.0,epsilon=0.048,phi=4.0/3.0,*args,**kwargs):
    #s=1
   # psi0=psi0.transpose().flatten()
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
  #      psiMag=psiMag.reshape(xgrid.size,2*s+1).transpose()
        fracM[ind]=dx*np.sum(psiMag[:,2])
        frac0[ind]=dx*np.sum(psiMag[:,1])
        fracP[ind]=dx*np.sum(psiMag[:,0]) 
        com0[ind]=np.sum(xgrid*psiMag[:,1]*dx)
        psi0eigB=np.einsum('ijk,ik->ij',eigVdagger,psi0)
        psi0eigB=splitStepPropagatorEigB2(psi0eigB,dt,a,b,eigE,eigV,eigVdagger)
        psi0=np.einsum('ijk,ik->ij',eigV,psi0eigB)
        
    return fracM,frac0,fracP,com0,tgrid,psi0

    
a=-48*np.pi
b=48*np.pi
alpha=1.0e-6
xSteps=8192
xgrid=np.linspace(a,b,xSteps)
kgrid=np.fft.fftfreq(xgrid.size,d=((b-a)/xgrid.size/(2.0*np.pi)))
dx=xgrid[1]-xgrid[0]
psiAn=np.exp(-np.sqrt(alpha)*xgrid**2.0/2.0)
psiAn=psiAn/np.sqrt(np.dot(psiAn,np.conj(psiAn)*dx))
psi0=np.array([np.array([0.0+0.0j,1.0+0.0j,0.0+0.0j])]*xSteps).transpose()
psi0ho=np.swapaxes(np.array([np.sqrt(0.2)*psiAn,np.sqrt(6.0)*psiAn,np.sqrt(0.2)*psiAn],dtype=complex),0,1)
psiAn2=np.exp(-xgrid**2.0/2.0)*xgrid
psiAn1=np.exp(-(xgrid-2.0)**2.0/2.0)
Vgrid=vLatHo(xgrid, alpha=alpha,x0=2200,V0=0.0)
print np.gradient(Vgrid)[xSteps/2.0]/(xgrid[1]-xgrid[0])

#fig1=plt.figure()
#pan1=fig1.add_subplot(1,1,1)
#pan1.plot(xgrid,Vgrid)
#start=time.clock()
#print 'legen.... '
#psi = getGroundState(psi0ho,vLatHo,a,b,0.01,omega=1.0,delta=0.0,epsilon=0.048,phi=4.0/3.0,alpha=alpha)
#end = time.clock()
#print 'dary.'
#print end-start
#
#fig=plt.figure()
#pan=fig.add_subplot(1,1,1)
#pan.plot(xgrid,psi[0]*np.conj(psi[0]),'b-', label='mF=+1')   
#pan.plot(xgrid,psi[1]*np.conj(psi[1]),'g-', label='mF=0') 
#pan.plot(xgrid,psi[2]*np.conj(psi[2]),'r-', label='mF=-1')
#pan.plot(xgrid, psiAn*np.conj(psiAn),'k-')
#
#psiFFt=np.fft.fft(psi)
#fig=plt.figure()
#pan=fig.add_subplot(1,1,1)
#pan.plot(kgrid,psiFFt[0]*np.conj(psiFFt[0]),'bo', label='mF=+1')   
#pan.plot(kgrid,psiFFt[1]*np.conj(psiFFt[1]),'go', label='mF=0') 
#pan.plot(kgrid,psiFFt[2]*np.conj(psiFFt[2]),'ro', label='mF=-1')
#
#print np.sum(psi[0]*np.conj(psi[0])*dx)
#print np.sum(psi[1]*np.conj(psi[1])*dx)
#print np.sum(psi[2]*np.conj(psi[2])*dx)

s=1
t1=time.clock()
fracM,frac0,fracP,com0,tgrid,psiOut=propagateInTime(psi.transpose(),vLatHo,a,b,500.0,0.01,omega=1.0,delta=0.0,epsilon=0.048,phi=4.0/3.0,alpha=alpha,x0=2200)
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
pan1.plot(tgrid,fracP,'b-', label='mF=+1')   
pan1.plot(tgrid,frac0,'g-', label='mF=0') 
pan1.plot(tgrid,fracM,'r-', label='mF=-1')
fig2=plt.figure()
pan2=fig2.add_subplot(1,1,1)
pan2.plot(tgrid,com0,'b-')