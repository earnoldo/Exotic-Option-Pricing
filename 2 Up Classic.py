# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:26:18 2018

@author: EduardoArnoldo
"""
import numpy as np
import matplotlib.pyplot as plt


h=0.01
delta=0.01
N=1024
sigma=0.2
r=0.01
E=1
T=1
M=int(T/delta)

#################### BARRIER ###################

Z=1.2##CAN BE CHANGED TO 0.6

################################################


#Vector of prices
N=int(Z/h)

Si=[0+h*i for i in range(1,N+1)]

#Vector of times to maturity
tj=[delta*j for j in range(M)]


#Creating Black Scholes Matrix
M_matrix=np.zeros([N,N])
M_matrix[0,0]=-(sigma**2)*(Si[0]**2)/(h**2)-r
M_matrix[0,1]=(sigma**2)*(Si[0]**2)/(2*h**2)+r*Si[0]/(2*h)
M_matrix[N-1,N-1]=-(sigma**2)*(Si[N-1]**2)/(h**2)-r
M_matrix[N-1,N-2]=(sigma**2)*(Si[N-1]**2)/(2*h**2)-r*Si[N-1]/(2*h)
for i in range(1,N-1):
    M_matrix[i,i-1]=(sigma**2)*(Si[i]**2)/(2*h**2)-r*Si[i]/(2*h)
    M_matrix[i,i]=-(sigma**2)*(Si[i]**2)/(h**2)-r
    M_matrix[i,i+1]=(sigma**2)*(Si[i]**2)/(2*h**2)+r*Si[i]/(2*h)

#Matrix A
A=np.identity(N)-delta*M_matrix 

#Final Condition
V0=np.zeros(N)
for i in range(N):
    if Si[i]<Z:
        V0[i]=max(E-Si[i],0)
        
#Matrix of Solutions
V=np.zeros((N,M))
V[:,0]=V0

#Implicit Euler
for j in range(1,M):    
    bj=np.zeros(N)
    bj[0]=((sigma**2)*(Si[0]**2)/(2*h**2)-r*Si[0]/(2*h))*delta*E*np.exp(-r*tj[j])
    V[:,j]=np.dot(np.linalg.inv(A),V[:,j-1]+bj)
    

## Ploting for Z = 1.2
v_x=[0 for i in range(10)]
s_x=list(np.linspace(1.2,3,10))
plt.plot(Si+s_x,list(V[:,M-1])+v_x,'b',label = 'Up & Out Put with Z = 1.2')

# Ploting if Z=0.6
#v_x=[0 for i in range(10)]
#s_x=list(np.linspace(0.6,2,10))
#plt.plot(Si+s_x,list(V[:,M-1])+v_x,'b',label = 'Up & Out Put with Z = 0.6')

plt.legend()
plt.xlabel('S')
plt.ylabel('V(S,0)') 
plt.show
