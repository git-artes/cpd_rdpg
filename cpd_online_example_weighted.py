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
import graspologic as gp
from scipy.sparse.linalg import eigs

import random 

plt.close(fig='all')


########################################################################################
# graphs are weighted ER but a subset suddenly changes its distribution (just the mean)
########################################################################################

# grafos = []
grafos_historicos = []

n = [80, 20]

p1 = [[0.5, 0.5],
      [0.5, 0.5]]
wt1 = [[np.random.normal, np.random.normal],
      [np.random.normal, np.random.normal]]
wtargs1 = [[dict(loc=5, scale=0.1), dict(loc=5, scale=0.1)],
          [dict(loc=5, scale=0.1), dict(loc=5, scale=0.1)]]

# p2 = [[0.8, 0.2],
#       [0.2, 0.5]]
p2 = p1
wt2 = wt1
wtargs2 = [[dict(loc=5, scale=0.1), dict(loc=3, scale=0.1)],
          [dict(loc=3, scale=0.1), dict(loc=5, scale=0.1)]]

cps_reales = [800, 1100]
T = np.max(cps_reales)+1000
m = 100
exp = 1.5

#### The historic data is generated and the cusum algorithm initialized

for i in np.arange(m):
    grafos_historicos.append(nx.from_numpy_array(gp.simulations.sbm(n=n, p=p1, wt=wt1, wtargs=wtargs1, directed=False)))

cusum = cpd_online.cpd_online_CUSUM(hfun='resid',exp=exp)
cusum.init(grafos_historicos)

#### new graphs are processed

signal_errors = np.zeros((T))
i=0

for _ in np.arange(cps_reales[0]):
    new_graph = gp.simulations.sbm(n=n, p=p1, wt=wt1, wtargs=wtargs1, directed=False)
    signal_errors[i] = cusum.new_graph(nx.from_numpy_array(new_graph))
    i = i+1

for _ in np.arange(cps_reales[1]-cps_reales[0]):
    new_graph = gp.simulations.sbm(n=n, p=p2, wt=wt2, wtargs=wtargs2, directed=False)
    signal_errors[i] = cusum.new_graph(nx.from_numpy_array(new_graph))
    i = i+1

for _ in np.arange(T-cps_reales[1]):
    new_graph = gp.simulations.sbm(n=n, p=p2, wt=wt2, wtargs=wtargs2, directed=False)
    signal_errors[i] = cusum.new_graph(nx.from_numpy_array(new_graph))
    i = i+1
 
k = np.arange(T)+1

plt.figure(1)
#plt.plot(signal_errors**2 - k*p1*(1-p1)*n*(n-1)/2)
plt.plot(signal_errors,label='Error signal')
plt.grid()
    

######## the intervals are estimated

sigma_entries = cusum.estimate_adjacency_variance(weighted=True, graphs=grafos_historicos)
(error_norm_sq, error_norm_ij) = cusum.cross_validate_model_error(grafos_historicos)

wmk = np.sum(n)*(k**exp)

m_k = np.sum(sigma_entries)*k + error_norm_sq*(k**2)
m_k = m_k/wmk

# var_k = residual_variance*(2*residual_variance*(k**2) + 4*error_norm_sq*(k**3))
var_k = 2*np.square(np.linalg.norm(sigma_entries,2))*(k**2) + 4*np.dot(sigma_entries,error_norm_ij)*(k**3)
var_k =var_k/wmk**2
sigma = np.sqrt(var_k)

########## plot the results 

plt.figure(1)
plt.plot(k,m_k,color='g',label='Estimated mean')
#plt.plot(k,m_k_bound,color='r',label='Media con cota')
plt.plot(k,m_k+3*sigma,color='g',linestyle='--')
plt.plot(k,m_k-3*sigma,color='g',linestyle='--')

#plt.plot(k,m_k_bound+2*sigma_bound,color='r',linestyle='--')
#plt.plot(k,m_k_bound-2*sigma_bound,color='r',linestyle='--')

# if cp:
#     detected_cp = np.where(signal_errors > m_k+2*sigma)[0][0]
#     plt.axvline(cps_reales[0],color='black',linewidth=2,linestyle='dashed', label="Punto de cambio real")
#     plt.axvline(detected_cp,color='grey',linewidth=2,linestyle='dashed',label='Punto de cambio detectado')
    
plt.title(r'Online CPD for a weighted ER -> weighted SBM (gaussian with mean=5 -> mean=3)' ,fontsize=26)
plt.legend(fontsize=16)
plt.show()
