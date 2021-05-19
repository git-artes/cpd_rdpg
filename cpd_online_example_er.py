#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 02:42:50 2020

@author: flarroca
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import cpd_online
from graspologic.utils import augment_diagonal
from scipy.sparse.linalg import eigs

plt.close(fig='all')


############################################
# A online CPD to detect changes on an Erdos-Renyi
############################################

######################
#### historic dataset
######################

# grafos = []
grafos_historicos = []

# you may try changing n and/or the probabilities
n = 100
p1 = 0.7
cp = True
if cp:
    p2 = 0.6
else:
    p2 = p1
cps_reales = [800, 1100]
T = np.max(cps_reales)+1000
m = 100
exp = 1.5

for i in np.arange(m):
    grafos_historicos.append(nx.erdos_renyi_graph(n,p1))   

cusum = cpd_online.cpd_online_CUSUM(hfun='resid',exp=exp)
cusum.init(grafos_historicos)

X0 = np.sqrt(p1)*np.ones((n,1))
Q = np.ones((1,1))
P = X0@Q@X0.T

###################
# the incoming graphs are observed
###################

signal_errors = np.zeros((T))
i=0

for _ in np.arange(cps_reales[0]):
    new_graph = nx.erdos_renyi_graph(n,p1)
    signal_errors[i] = cusum.new_graph(new_graph)
    i = i+1

for _ in np.arange(cps_reales[1]-cps_reales[0]):
    new_graph = nx.erdos_renyi_graph(n,p2)
    signal_errors[i] = cusum.new_graph(new_graph)
    i = i+1

for _ in np.arange(T-cps_reales[1]):
    new_graph = nx.erdos_renyi_graph(n,p2)
    signal_errors[i] = cusum.new_graph(new_graph)
    i = i+1
 
k = np.arange(T)+1

plt.plot(signal_errors,label='Signal error')
plt.grid()

############################
# I now estimate the intervals where the signal error should live
############################

(m_k, sigma) = cusum.estimate_confidence_intervals(weighted=False, graphs=grafos_historicos)

#plt.hlines([np.sqrt(scipy.stats.chi2.ppf(0.999,df=n*(n-1)/2)*p1*(1-p1)), np.sqrt(scipy.stats.chi2.ppf(0.001,df=n*(n-1)/2)*p1*(1-p1))],0,len(signal_errors),color='red')

#############################
# I compute the actual intervals
#############################

df =  n*(n-1)/2

p_est = np.mean(cusum.Xhat0@cusum.Q@cusum.Xhat0.T)
P_est = p_est*np.ones_like(P)
X0_error = P_est - cusum.Xhat0@cusum.Q@cusum.Xhat0.T
epsilon = np.triu(X0_error,1)
error_norm = np.linalg.norm(epsilon)**2



m_k2 = p_est*(1-p_est)*df*k + error_norm*(k**2)
wmk = n*(k**exp)
m_k2 = m_k2/wmk

var_k2 = p_est*(1-p_est)*(2*df*p_est*(1-p_est)*(k**2) + 4*error_norm*(k**3))
var_k2 =var_k2/wmk**2
sigma2 = np.sqrt(var_k2)

# var_k_bound = p_est2*(1-p_est2)*(2*df*p_est2*(1-p_est2)*(k**2) + 4*error_bound*(k**3))
# var_k_bound = var_k_bound/wmk**2
# sigma_bound = np.sqrt(var_k_bound)

plt.plot(k,m_k2,color='r',label='Actual mean')
plt.plot(k,m_k,color='g',label='Estimated mean')
#plt.plot(k,m_k_bound,color='r',label='Media con cota')
plt.plot(k,m_k+2*sigma,color='g',linestyle='--')
plt.plot(k,m_k-2*sigma,color='g',linestyle='--')
plt.plot(k,m_k2+2*sigma2,color='r',linestyle='--')
plt.plot(k,m_k2-2*sigma2,color='r',linestyle='--')
#plt.plot(k,m_k_bound+2*sigma_bound,color='r',linestyle='--')
#plt.plot(k,m_k_bound-2*sigma_bound,color='r',linestyle='--')

# if cp:
#     detected_cp = np.where(signal_errors > m_k+2*sigma)[0][0]
#     plt.axvline(cps_reales[0],color='black',linewidth=2,linestyle='dashed', label="Punto de cambio real")
#     plt.axvline(detected_cp,color='grey',linewidth=2,linestyle='dashed',label='Punto de cambio detectado')
    
plt.title(r'Online CPD for an ER (p=0.7->p=0.8)',fontsize=26)
plt.legend(fontsize=16)
plt.show()
