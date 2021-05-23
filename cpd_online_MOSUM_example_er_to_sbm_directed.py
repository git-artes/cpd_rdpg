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
# graphs are ER but a subset suddenly become a (directive) community (get more followers)
########################################################################################

# grafos = []
grafos_historicos = []

n = [80, 20]

p1 = [[0.5, 0.5],
      [0.5, 0.5]]

p2 = [[0.5, 0.5],
      [0.7, 0.5]]

cps_reales = [800, 1100]
T = np.max(cps_reales)+1000
m = 100
exp = 1.5

#### The historic data is generated and the cusum algorithm initialized

for i in np.arange(m):
    grafos_historicos.append(nx.stochastic_block_model(sizes=n, p=p1, directed=True))

cusum = cpd_online.cpd_online_MOSUM(hfun='resid',exp=exp)
cusum.init(grafos_historicos)

#### new graphs are processed

signal_errors = np.zeros((T))
i=0

for _ in np.arange(cps_reales[0]):
    new_graph = nx.stochastic_block_model(sizes=n, p=p1, directed=True)
    signal_errors[i] = cusum.new_graph(new_graph)
    i = i+1

for _ in np.arange(cps_reales[1]-cps_reales[0]):
    new_graph = nx.stochastic_block_model(sizes=n, p=p2, directed=True)
    signal_errors[i] = cusum.new_graph(new_graph)
    i = i+1

for _ in np.arange(T-cps_reales[1]):
    new_graph = nx.stochastic_block_model(sizes=n, p=p2, directed=True)
    signal_errors[i] = cusum.new_graph(new_graph)
    i = i+1
 
k = np.arange(T)+1

plt.figure(1)
#plt.plot(signal_errors**2 - k*p1*(1-p1)*n*(n-1)/2)
plt.plot(signal_errors,label='Error signal')
plt.grid()
    

######## the intervals are estimated

(m_k, sigma) = cusum.estimate_confidence_intervals(weighted=True, graphs=grafos_historicos)

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
    
plt.title(r'Online CPD for a directed ER -> directed SBM' ,fontsize=26)
plt.legend(fontsize=16)
plt.show()
