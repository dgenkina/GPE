# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:10:55 2016

@author: dng5
"""

import numpy as np
import time
from scipy import linalg as sLA

xSteps=512
spins=3

a=np.array(rand(xSteps,spins,spins))
b=np.array(rand(xSteps,spins))

a1=sLA.block_diag(*[a[i] for i in range(xSteps)]) 
b1=b.flatten()
b2=np.swapaxes(np.swapaxes(np.array([b,b,b]),0,1),1,2)

t1=time.clock()
c=np.dot(a1,b1)
t2=time.clock()
d=np.array(map(np.dot,a,b))
t3=time.clock()
f=[np.dot(a[i],b[i]) for i in range(xSteps)]
t4=time.clock()

print t2-t1
print t3-t2
print t4-t3

#
#timeit.timeit('np.)
