# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 18:24:50 2018

@author: EduardoArnoldo
"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt


# Standard Black Scholes
def BSprice(style,S0, K, r, sigma, T,div):
    d1=(np.log(S0/K) + (r -div+ sigma**2 / 2) * T)/(sigma * np.sqrt(T))
    d2=(np.log(S0 / K) + (r-div - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    if style=="C":
        return S0 *np.exp(-div*T)* ss.norm.cdf(d1) - K * np.exp(-r * T) * ss.norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * ss.norm.cdf(-d2) - S0 * np.exp(-div*T)*ss.norm.cdf(-d1)
    

# MODIFIED Black Scholes
def BSprice2(style,S0, K, r, sigma, T,div,Z):
    d1=(np.log(S0/K) + (r -div+ sigma**2 / 2) * T-np.log(Z/K))/(sigma * np.sqrt(T))
    d2=(np.log(S0 / K) + (r-div - sigma**2 / 2) * T-np.log(Z/K)) / (sigma * np.sqrt(T))
    if style=="C":
        return S0 *np.exp(-div*T)* ss.norm.cdf(d1) - K * np.exp(-r * T) * ss.norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * ss.norm.cdf(-d2) - S0 * np.exp(-div*T)*ss.norm.cdf(-d1)
    
    
h=0.05
delta=0.01
N=25
sigma=0.2
r=0.01
E=1
T=1

M=int(T/delta)

#Vector of prices
Z=0.6
N=50
#Si=[Z+h*i for i in range(1,N+1)]

Si=[Z+h*i for i in range(1,N+1)]
#Vector of times to maturity
tj=[delta*j for j in range(M)]
#Matrix of Solutions
V=np.zeros((N,M))


#Solution only makes sense for Z<E
for i in range(N):
    for j in range(M):
        tmp=BSprice("P",Si[i], E, r, sigma, tj[j],0)
        tmp2=BSprice("P",(Z**2/Si[i]), E, r, sigma, tj[j],0)
        tmp3=BSprice2("P",Si[i], E, r, sigma, tj[j],0,Z)
        tmp4=BSprice2("P",(Z**2/Si[i]), E, r, sigma, tj[j],0,Z)    
        tmp5=tmp-(Si[i]/Z)**(-2*r/(sigma**2)+1)*tmp2
        tmp6=tmp3-(Si[i]/Z)**(-2*r/(sigma**2)+1)*tmp4
        V[i,j]=tmp5-tmp6
        
        
s_x=list(np.linspace(0,0.6,10))
v_x=[0 for i in range(10)]
plt.plot(s_x+Si,v_x+list(V[:,M-1]),'b',label = 'Up & Out Put with Z = 0.6')
plt.legend()
plt.xlabel('S')
plt.ylabel('V(S,0)') 
plt.show
               
