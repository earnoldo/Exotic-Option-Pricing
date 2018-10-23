# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:53:47 2018

@author: EduardoArnoldo
"""

import numpy as np
import matplotlib.pyplot as plt

def alpha(t,E,r):
    return ( E * np.exp(-r*t) )

h = 0.01
delta = 0.01
N = 500
sigma = 0.2
r = 0.1
E = 1
T = 1

s_list = np.linspace(h, N*h, N)
time = np.arange(0, T+delta, delta)
M = np.zeros((N,N))

for i in range(N):
    if i == 0:
        M[i,0] = -(sigma**2 * s_list[i]**2)/(h**2) - r
        M[i,1] = (sigma**2  * s_list[i]**2)/(2*h**2) + (r * s_list[i])/(2*h)
    elif i == N-1:
        M[i, N-2] = (sigma**2 * s_list[i]**2)/(2*h**2) - (r * s_list[i])/(2*h)
        M[i, N-1] = -(sigma**2 * s_list[i]**2)/(h**2) - r

    else:
        M[i,i-1] = (sigma**2 * s_list[i]**2)/(2*h**2) - (r * s_list[i])/(2*h)
        M[i,i] = -(sigma**2 * s_list[i]**2)/(h**2) - r
        M[i,i+1] = (sigma**2 * s_list[i]**2)/(2*h**2) +  (r * s_list[i])/(2*h)         

I = np.identity(N)
M = np.array(M)
A = I - delta * M
D = np.diag(A)
L = -(np.tril(A) - np.diagflat(D))
U = -(A + L - np.diagflat(D))
D = A + L + U

spectral_radii_list = []
w_list = np.arange(0,2.1,0.1) #finding optimal w for relaxation

for w in w_list:
    T_w = np.matmul(np.linalg.inv((D - w*L)),((1-w)*D + w*U))
    eigvalue = np.linalg.eigvals(T_w)
    spectral_radii = max(np.abs(eigvalue))
    spectral_radii_list.append(spectral_radii)

w_opt = w_list[np.argmin(spectral_radii_list)]

T_w_optimal = np.matmul(np.linalg.inv((D - w_opt*L)),((1-w_opt)*D + w_opt*U))
inverse_matrix = np.linalg.inv(D - w_opt*L) * w_opt
V_put = E - np.array(s_list)
V_put[V_put < 0] = 0
tolerance = 10**-5

V_put_copy = V_put.copy()
Sf = []
for i in time:
    e = 1
    b = [0 for i in range(N)]
    b[0] = ((sigma*sigma*s_list[0]*s_list[0])/(2*h*h) - (r*s_list[0])/(2*h)) * alpha(i+delta,E,r)
    B = V_put + delta * np.array(b)
    c_w = np.matmul(inverse_matrix,B)
    New_V = np.matmul(T_w_optimal,V_put) + c_w

    while e > tolerance:
        V_put = New_V
        New_V = np.matmul(T_w_optimal,V_put) + c_w
        difference = V_put - New_V
        e = np.linalg.norm(difference,np.inf)

    V_put = np.fmax(New_V,V_put_copy) 
    diff = New_V - V_put_copy
    loc = np.argwhere(diff>0)[0][0]
    Sf.append(s_list[loc])


s_list = np.arange(0, N*h+h, h)

Sf_t0 = Sf[::-1]
   
plt.title('Sf vs time')
plt.xlabel('t')
plt.ylabel('Sf')
plt.grid()
plt.plot(time,Sf_t0)
plt.show()
    

fig2 = plt.figure(2)
plt.title('V(S,0) vs S')
plt.xlabel('S')
plt.ylabel('V(S,0)')
plt.plot(s_list[1:300], V_put[:299],label = 'V(S,0) American put')
plt.grid()
plt.legend()
plt.show()