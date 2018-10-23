# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:25:03 2018

@author: EduardoArnoldo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Initial Parameters
h =0.05
delta = 0.1*h**2
N = 500
sigma = 0.2
r = 0.1
E = 1
T = 1

#Discretizing axes
time = np.arange(0,T+delta, delta)
M = len(time - 1)

Sf = np.zeros(M)
p = np.zeros([N,M])    

#Typical change of variables
a = (sigma**2 * delta)/(2*h**2) - delta*(r - (sigma**2)/2 )/(2*h)
b =(1 - (sigma**2 * delta)/(h**2) -delta*r )
c =  (sigma**2 * delta)/(2 * h**2) + delta * (r-sigma**2/2)/(2*h) 
alpha = ((h**2/sigma**2)*r+1)*E 
beta = ( (h**2)/2 + 1 + h)

Sf[0] = E 

# Looping to get parameters
for j in range (0,M-1):
    #For i=0 and i=1
    Sf[j+1] = Sf[j]*(alpha - (a+1/(2*h))*p[0,j] - b*p[1,j] +(1/(2*h)-c)*p[2,j])/(beta*Sf[j] +(p[2,j]-p[0,j])/(2*h))
    p[0,j+1] = E - Sf[j+1]
    p[1,j+1] = alpha - beta * Sf[j+1]
    
    #For the rest of the values
    for i in range(2, N-1):
        p[i,j+1] = a*p[i-1,j] + b*p[i,j] + c*p[i+1,j] + (Sf[j+1]-Sf[j])*(p[i+1,j]-p[i-1,j])/(2*h*Sf[j])
        if i == N-1:
            p[i,j+1] = a*p[i-1,j] + b*p[i,j] + (Sf[j+1]-Sf[j])*(-p[i-1,j])/(2*h*Sf[j])


#Plotting SF
fig1 = plt.figure(1)    
plt.title('Sf vs time')
plt.xlabel('t')
plt.ylabel('Sf')
plt.grid()
plt.plot(time[1:],Sf[1:])
plt.show()
 

#Now recovering values
x = np.linspace(0,5,N+1)
S = np.exp(x)*Sf[-1]
plt.plot(S[:70],p[:,-1][:70])
plt.grid()
plt.title(" V(S,0) vs S")
plt.show()
