#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 02:42:50 2020

@author: flarroca
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cpd

import graspy as gy
import ruptures as rpt

plt.close(fig='all')


# ##############################################################
# # pruebas del algoritmo en weighted SBM: cambio de varianza
# ##############################################################

# grafos = []

# n = [50, 50]
# p1 = 0.7
# p2 = 0.8
# cps_reales = [30, 50]
# T = np.sum(cps_reales)+30

# p = [[0.5, 0.5],
#       [0.5, 0.5]]
# wt1 = [[np.random.normal, np.random.normal],
#       [np.random.normal, np.random.normal]]
# wtargs1 = [[dict(loc=5, scale=0.1), dict(loc=5, scale=0.1)],
#           [dict(loc=5, scale=0.1), dict(loc=5, scale=0.1)]]
# wt2 = [[np.random.normal, np.random.normal],
#       [np.random.normal, np.random.normal]]
# wtargs2 = [[dict(loc=5, scale=0.1), dict(loc=5, scale=1)],
#           [dict(loc=5, scale=1), dict(loc=5, scale=1)]]

# for i in np.arange(cps_reales[0]):
#     weights = gy.simulations.sbm(n=n, p=p, wt=wt1, wtargs=wtargs1)**4
#     grafos.append(nx.from_numpy_array(weights))   
# for i in np.arange(cps_reales[1]-cps_reales[0]):
#     weights = gy.simulations.sbm(n=n, p=p, wt=wt2, wtargs=wtargs2)**4
#     grafos.append(nx.from_numpy_array(weights))   
# for i in np.arange(T-cps_reales[1]):
#     weights = gy.simulations.sbm(n=n, p=p, wt=wt1, wtargs=wtargs1)**4
#     grafos.append(nx.from_numpy_array(weights))   
    
# tau = 2*np.log(len(grafos)*np.sum(n)/2)/3

# algo = cpd.nrdpgwbs(tau, 10, d=1)
# #algo = cpd.nrdpgwbs(2,10)
# algo.fit(grafos)
# plt.plot(algo.Y)
# plt.show()
# algo.find_segments(0,len(grafos))   
# print("puntos de cambio detectados por NRDPG-WBS: " + str(algo.estimated_change_points))
# print("puntos de cambios reales: "+str(cps_reales))

##############################################################
# pruebas del algoritmo en weighted SBM: cambio de a una
##############################################################

grafos = []

n = [50, 50]
cps_reales = [30, 50, 80]
T = cps_reales[-1]+30

#conectividad
p1 = [[0.5, 0.3],
      [0.3, 0.05]]
# p2 = [[0.5, 0.3], 
#       [0.3, 0.3]]
p2=p1
p3 = p2
p4 = p2
p5 = p2

# distros
wt1 = [[np.random.normal, np.random.normal],
      [np.random.normal, np.random.normal]]
wt2 = wt1
wt3 = wt1
wt4 = wt1
wt5 = [[np.random.normal, np.random.normal],
      [np.random.poisson, np.random.normal]]

#parametros
wtargs1 = [[dict(loc=5, scale=0.1), dict(loc=5, scale=0.1)],
          [dict(loc=5, scale=0.1), dict(loc=5, scale=0.1)]]
wtargs2 = wtargs1
wtargs3 = [[dict(loc=5, scale=0.1), dict(loc=3, scale=0.1)],
          [dict(loc=5, scale=0.1), dict(loc=5, scale=0.1)]]
wtargs4 = [[dict(loc=5, scale=0.1), dict(loc=3, scale=0.1)],
          [dict(loc=5, scale=np.sqrt(5)), dict(loc=5, scale=0.1)]]
wtargs5 = [[dict(loc=5, scale=0.1), dict(loc=3, scale=0.1)],
          [dict(lam=5), dict(loc=5, scale=1)]]

perm = np.eye(np.sum(n))
idx = np.random.permutation(np.arange(perm.shape[0]))
perm = perm[idx,:]

for i in np.arange(cps_reales[0]):
    weights = gy.simulations.sbm(n=n, p=p2, wt=wt2, wtargs=wtargs2, directed=True)**4
    weights = perm.T@weights@perm
    grafos.append(nx.from_numpy_array(weights,create_using=nx.DiGraph()))   
for i in np.arange(cps_reales[1]-cps_reales[0]):
    weights = gy.simulations.sbm(n=n, p=p3, wt=wt3, wtargs=wtargs3, directed=True)**4
    weights = perm.T@weights@perm
    grafos.append(nx.from_numpy_array(weights,create_using=nx.DiGraph()))   
for i in np.arange(cps_reales[2]-cps_reales[1]):
    weights = gy.simulations.sbm(n=n, p=p4, wt=wt4, wtargs=wtargs4, directed=True)**4
    weights = perm.T@weights@perm
    grafos.append(nx.from_numpy_array(weights,create_using=nx.DiGraph()))   
for i in np.arange(T-cps_reales[2]):
    weights = gy.simulations.sbm(n=n, p=p5, wt=wt5, wtargs=wtargs5, directed=True)**4
    weights = perm.T@weights@perm
    grafos.append(nx.from_numpy_array(weights,create_using=nx.DiGraph()))       

tau = np.log(len(grafos)*np.sum(n)/2)/3

algo = cpd.nrdpgwbs(tau, 20, d=1)
#algo = cpd.nrdpgwbs(2,10)
algo.fit(grafos,shuffle=False)
plt.plot(algo.Y)
plt.show()
algo.find_segments(0,len(grafos))   
print("puntos de cambio detectados por NRDPG-WBS: " + str(algo.estimated_change_points))
print("puntos de cambios reales: "+str(cps_reales))
cpd.display(algo.Y,cps_reales+[T],algo.estimated_change_points)

# ##############################################################
# # pruebas del algoritmo en weighted SBM con cuatro cambios: conectividad, media, varianza y distro
# ##############################################################


# grafos = []

# n = [50, 50]
# cps_reales = [30, 50, 70, 100]
# T = 130

# #conectividad
# p1 = [[0.5, 0.3],
#      [0.3, 0.05]]
# p2 = [[0.5, 0.3], 
#       [0.3, 0.3]]
# p3 = p2
# p4 = p2
# p5 = p2

# # distros
# wt1 = [[np.random.normal, np.random.normal],
#       [np.random.normal, np.random.normal]]
# wt2 = wt1
# wt3 = wt1
# wt4 = wt1
# wt5 = [[np.random.normal, np.random.normal],
#       [np.random.normal, np.random.poisson]]

# #parametros
# wtargs1 = [[dict(loc=5, scale=0.1), dict(loc=5, scale=0.1)],
#           [dict(loc=5, scale=0.1), dict(loc=5, scale=0.1)]]
# wtargs2 = wtargs1
# wtargs3 = [[dict(loc=8, scale=0.1), dict(loc=5, scale=1)],
#           [dict(loc=5, scale=1), dict(loc=5, scale=0.1)]]
# wtargs4 = [[dict(loc=8, scale=0.1), dict(loc=5, scale=1)],
#           [dict(loc=5, scale=1), dict(loc=5, scale=np.sqrt(5))]]
# wtargs5 = [[dict(loc=8, scale=0.1), dict(loc=5, scale=1)],
#           [dict(loc=5, scale=1), dict(lam=5)]]

# for i in np.arange(cps_reales[0]):
#     weights = gy.simulations.sbm(n=n, p=p1, wt=wt1, wtargs=wtargs1)**4
#     grafos.append(nx.from_numpy_array(weights))   
# for i in np.arange(cps_reales[1]-cps_reales[0]):
#     weights = gy.simulations.sbm(n=n, p=p2, wt=wt2, wtargs=wtargs2)**4
#     grafos.append(nx.from_numpy_array(weights))   
# for i in np.arange(cps_reales[2]-cps_reales[1]):
#     weights = gy.simulations.sbm(n=n, p=p3, wt=wt3, wtargs=wtargs3)**4
#     grafos.append(nx.from_numpy_array(weights)) 
# for i in np.arange(cps_reales[3]-cps_reales[2]):
#     weights = gy.simulations.sbm(n=n, p=p4, wt=wt4, wtargs=wtargs4)**4
#     grafos.append(nx.from_numpy_array(weights)) 
# for i in np.arange(T-cps_reales[3]):
#     weights = gy.simulations.sbm(n=n, p=p5, wt=wt5, wtargs=wtargs5)**4
#     grafos.append(nx.from_numpy_array(weights))   
    
# tau = 2*np.log(len(grafos)*np.sum(n)/2)/3

# algo = cpd.nrdpgwbs(tau, 10, d=3)
# #algo = cpd.nrdpgwbs(2,10)
# algo.fit(grafos)
# plt.plot(algo.Y)
# plt.show()
# algo.find_segments(0,len(grafos))   
# print("puntos de cambio detectados por NRDPG-WBS: " + str(algo.estimated_change_points))
# print("puntos de cambios reales: "+str(cps_reales))