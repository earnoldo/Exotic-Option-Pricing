# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 18:16:06 2018

@author: EduardoArnoldo
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

T=1
# I can't use his parameters so I will modify the size of h
N=70
M=35
h=12.8/N
dt=0.01
O=int(1/dt)
E=3
l=E/M
sigma=0.2
r=0.1

# This will become a function
Si=[i*h for i in range(1,N+1)] # From 0 ->N
Ij=[j*l for j in range(1,M+1)] # From 0->M
tau=[k*dt for k in range(O)] # From 0 to T

M_matrix=np.zeros([M*N,M*N])

for i in range(N):
    for j in range(M):
        p=N*j+i
        
        ## V(i+1,j,k+1) coefficient
        if(p!=M*N-1):
            if(i%N!=0):
                M_matrix[p,p+1]=-((1/2)*(sigma**2)*(Si[i]**2)/(h**2)+r*Si[i]/(2*h))
        
        ## V(i-1,j,k+1) coefficient
        if(p!=0):
            if(i!=0):#I changed he index inside
                M_matrix[p,p-1]=-((1/2)*(sigma**2)*(Si[i]**2)/(h**2)-r*Si[i]/(2*h))
            
        ## V(i,j,k+1) coefficient
        M_matrix[p,p]=((1/2)*(sigma**2)*(Si[i]**2)*2/(h**2)+r+Si[i]/l)
        
        ## V(i,j+1,k+1) coefficient
        if(j!=(M-1)):
            M_matrix[p,p+N]=-(Si[i]/l)
      
A=dt*M_matrix+np.identity(len(M_matrix))
        

# Function of solutions
Vijk=np.zeros([M*N,O])

#Initial value vector
v=np.zeros(M*N)
for i in range (N):
    for j in range(M):
        p=N*j+i
        v[p]=max(Ij[j]-E,0)
        
Vijk[:,0]=v

#Doing Forward Euler
for k in range(1,O):

    #Building vector b        
    b=np.zeros(M*N)
    for j in range(M):
        ## BC for i=0
        i=0
        p=N*j+i
        px=np.exp(-r*tau[k])*max(Ij[j]-E,0)
        b[p]=px*((1/2)*(sigma**2)*Si[i]**2*dt/(h**2)-r*Si[i]*dt/(2*h))
        
        ## BC for i=N
        i=N
        p=N*j+i-1
        px=(Si[i-1]+h)/(r*T)*(1-np.exp(-r*tau[k]))+(Ij[j]-E)*np.exp(-r*tau[k])
        b[p]=px*((1/2)*(sigma**2)*Si[i-1]**2*dt/(h**2)+r*Si[i-1]*dt/(2*h))
    
    for i in range(N):
        ## BC for j=M-1
        j=M-1
        p=N*j+i
        px=Si[i]/(r*T)*(1-np.exp(-r*tau[k]))
        b[p]=px*(Si[i]*dt/l)
        
    
    #Implicit Euler
    Vijk[:,k]=np.dot(linalg.inv(A),Vijk[:,k-1])+np.dot(linalg.inv(A),b)
    
    
    if k%10==0:
        print(k)
    

tmp=Vijk[:N,99]
Si=[i*h for i in range(1,N+1)] # From 0 ->N


#Now, plotting the european option
import scipy.stats as ss

# Standard Black Scholes
def BSprice(style,S0, K, r, sigma, T,div):
    d1=(np.log(S0/K) + (r -div+ sigma**2 / 2) * T)/(sigma * np.sqrt(T))
    d2=(np.log(S0 / K) + (r-div - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    if style=="C":
        return S0 *np.exp(-div*T)* ss.norm.cdf(d1) - K * np.exp(-r * T) * ss.norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * ss.norm.cdf(-d2) - S0 * np.exp(-div*T)*ss.norm.cdf(-d1)
    

Eur=[BSprice("C",Si[i], 3, 0.1, 0.2, 1,0) for i in range(30)]


# NOW PLOTTING
#I'm plotting just 30 elements, but you can plot the full 70 elements
plt.plot(Si[:30],Eur,label="European Call Option")
plt.plot(Si[:30],tmp[:30],label='Arithmetic Average Call Option')
plt.grid()
plt.legend()
plt.xlabel("Price")
plt.ylabel("Value")
plt.show







