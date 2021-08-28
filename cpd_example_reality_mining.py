#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:13:39 2021

@author: flarroca
"""

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms as transforms
from itertools import cycle
import graspologic as gp

import cpd
import cpd_online

# import funciones_aux

import warnings
warnings.filterwarnings('ignore')

plt.close(fig='all')
plt.rcParams['lines.linewidth'] = 3
# plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 28

COLOR_CYCLE = ["#4286f4", "#f44174"]


# Dataset downloaded from http://realitycommons.media.mit.edu/realitymining.html
# Has to be pre-processed with the following code in order to obtain reality-mining-proximity.txt: https://github.com/j2kun/reality-mining

# let us load the data
df_proximity = pd.read_csv('reality-mining-proximity.txt', sep='\t', parse_dates=['start','end'])

# how long they've been near each other
df_proximity['proximity_duration'] = df_proximity.end-df_proximity.start

# let us group them by `win_days` days (change as needed, now it's 1 day) 
t0 = df_proximity.start.min()
win_days =1
df_proximity['delta_dias'] = (df_proximity.start-t0).dt.days//win_days
# We remove a couple of outliers with contacts with a duration of over a day
df_proximity = df_proximity[df_proximity.proximity_duration.dt.days<1]

grouped_ = df_proximity.groupby('delta_dias')
# grouped_ = df_proximity.groupby('week')
grafos = []
fechas = []

# we may consider either the weighted or unweighted graphs
weighted = True

for dia, grupo_proximity in grouped_:
    proximity = grupo_proximity.groupby(['id1', 'id2']).proximity_duration.sum().reset_index()
    # the edge's weight is going to be the proximity duration in minutes
    proximity['duracion'] = proximity.proximity_duration.dt.seconds/60
    proximity.rename(columns={'duracion':'weight'}, inplace=True)
    if weighted: 
        G = nx.from_pandas_edgelist(proximity, source='id1', target='id2', create_using=nx.Graph, edge_attr='weight')
    else: 
        G = nx.from_pandas_edgelist(proximity, source='id1', target='id2', create_using=nx.Graph)
    
    grafos.append(G)
    fechas.append(grupo_proximity.start.min())

#pd.DataFrame(index=fechas, data=[g.number_of_nodes() for g in grafos]).plot(title='nodos')
#pd.DataFrame(index=fechas, data=[g.number_of_edges() for g in grafos]).plot(title='aristas')

##################
# we construct the training set
##################

# the date below is the first on the dataset. We take a month for training. 
fecha_historico_inicio = np.datetime64('2004-07-19')
fecha_historico_fin = np.datetime64('2004-08-19')

cuales = [(fecha>=fecha_historico_inicio) & (fecha<=fecha_historico_fin) for fecha in fechas]

grafos_historicos = list(np.array(grafos)[cuales])
fechas_historicas = np.array(fechas)[cuales]

# all nodes are considered
# nodes_list = list(set().union(*[grafo.nodes() for grafo in grafos_historicos]))
nodes_list = list(set(df_proximity.id1.unique()).union(df_proximity.id2.unique()))

# I add the nodes to all graphs. It may happen that certain nodes were not active.
for grafo in grafos_historicos: 
    grafo.add_nodes_from(nodes_list) 

#######################
# armado de grafos a monitorear
#######################

# we start monitoring right after training and do it for all graphs available
fecha_observacion_inicio = fecha_historico_fin
# fecha_observacion_fin = np.datetime64('2004-12-01')
fecha_observacion_fin = np.datetime64('2005-05-01')

cuales = [(fecha>fecha_observacion_inicio) & (fecha<=fecha_observacion_fin) for fecha in fechas]

fechas_observacion = np.array(fechas)[cuales]

grafos = list(np.array(grafos)[cuales])

# I add the nodes to all graphs. It may happen that certain nodes were not active.
for grafo in grafos: 
    grafo.add_nodes_from(nodes_list) 
    
# just in case, I remove those nodes that are not in my nodes_list. In this case in unnecesary anyway.
grafos = [grafo.subgraph(nodes_list).copy() for grafo in grafos]

###########################
# let's generate some curves
###########################

T = len(grafos)
m = len(grafos_historicos)
n = len(nodes_list)
exp = 1.5

cusum = cpd_online.cpd_online_MOSUM(hfun='resid',exp=exp)
cusum.init(grafos_historicos)

signal_errors = []

for grafo in grafos: 
    signal_errors.append(cusum.new_graph(grafo))


#plt.figure(1)    
#ax = plt.gca()
#pd.DataFrame({'signal error':signal_errors}, index=np.array(fechas)[cuales]).plot(figsize=(12,6), ax = ax)

############################
# compute the intervals
############################

(m_k, sigma) = cusum.estimate_confidence_intervals(weighted=weighted, graphs=grafos_historicos, nboots=100)

max_curve  =  m_k+3*sigma
detected_cp = np.where(signal_errors > max_curve)[0]
if detected_cp.size != 0:
    detected_cp = detected_cp[0]
    fecha_cp = fechas_observacion[detected_cp]
    print(f"Online cp detected on {fechas_observacion[detected_cp]}")

cpd_offline = True

grafos_offline = grafos_historicos + grafos
fechas_offline = np.concatenate((fechas_historicas,fechas_observacion))

if cpd_offline:
    tau = np.log(len(grafos_offline)*n/2)/4
         
    algo = cpd.nrdpgwbs(tau, 50)
    algo.fit(grafos_offline,shuffle=False)
     
    algo.find_segments(0,len(grafos_offline))
    print(f"Offline cp detected on : " + str([fechas_offline[cp].strftime('%d/%m/%y') for cp in algo.estimated_change_points]))

print(f"Total number of graphs: {len(grafos_offline)}")       
print(f"Number of graphs on the training set: {len(grafos_historicos)}")

fig, ax = plt.subplots()
# plt.title(titulo, fontsize=20)
fig.autofmt_xdate()
ax.plot(pd.DatetimeIndex(fechas_observacion),signal_errors,label=r'$\omega[k]\Gamma[m,k]$')
# ax.plot(fechas_observacion,signal_errors,label=r'$\omega[k]\Gamma[m,k]$')
plt.plot(pd.DatetimeIndex(fechas_observacion),m_k,color='g',label=r'Estimated mean')
plt.plot(pd.DatetimeIndex(fechas_observacion),max_curve,color='g',linestyle='dashed',label=r'Threshold')
plt.ylabel(r'Weighted statistic',fontsize=42)



#plt.gca().axes.get_yaxis().set_ticklabels([])
locator = mdates.AutoDateLocator(minticks=7, maxticks=10)
formatter = mdates.DateFormatter('%b/%Y')
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
#ax.tick_params(axis='x', rotation=30)
plt.legend(fontsize=40,loc='upper right')
plt.grid()


fig2, ax2 = plt.subplots()

if cpd_offline:
    ax2.plot(pd.DatetimeIndex(fechas_offline),algo.Y)
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)
    
    
    cps_fechas_off = [fechas_offline[cp].to_datetime64() for cp in sorted(algo.estimated_change_points)]
    #cps_fechas = [fechas_observacion[0].to_datetime64()] + cps_fechas_off + [fechas_observacion[-1].to_datetime64()]
    #cps_fechas_off = [fechas_offline[0].to_datetime64()] + cps_fechas_off + [fechas_offline[-1].to_datetime64()]
    xmin,xmax = ax.get_xlim()
    xmin2,xmax2 = ax2.get_xlim()
    
    cps_data_coord = mdates.date2num(cps_fechas_off)
    cps_data_coord_obs = [xmin] + list(cps_data_coord) + [xmax]
    cps_data_coord_obs2 = [xmin2] + list(cps_data_coord) + [xmax2]
    
    alpha = 0.2  # transparency of the colored background
    color_cycle = cycle(COLOR_CYCLE)
    trans2 = transforms.blended_transform_factory(ax2.transData, ax2.transAxes)
     
    for (start, end), col in zip(cpd.pairwise(cps_data_coord_obs2), color_cycle):
        ax2.axvspan(max(0, start), end, facecolor=col, alpha=alpha)
        
    for (start, end), col in zip(cpd.pairwise(cps_data_coord_obs), color_cycle):
        ax.axvspan(max(0, start), end, facecolor=col, alpha=alpha)

if detected_cp.size != 0:
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    x_pos = mdates.date2num(fecha_cp.to_datetime64())
    ax.axvline(x_pos,linestyle='--',linewidth=3,color='black')
    
# special dates from https://arxiv.org/pdf/1403.0989.pdf

special_dates = {'start of the semester':'06/09/2004', \
                 'sponsor week':'18/10/2004', \
                 'exam week':'08/11/2004', \
                 'Thanksgiving':'25/11/2004', \
                     'Last week of classes':'06/12/2004', \
                         'Finals week':'13/12/2004', \
                             'Independent activities':'03/01/2005',\
                                 'MLK day':'17/01/2005',\
                                     'Start of semester':'31/01/2005',\
                                         'Exam week':'28/2/2005',\
                                             'Spring break':'21/03/2005'}

plot_special_dates = True
if plot_special_dates:
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for special_date in special_dates:
        date = pd.to_datetime(special_dates[special_date],format='%d/%m/%Y')
        x_pos = mdates.date2num(date.to_datetime64())
        ax.axvline(x_pos,linestyle='--',linewidth=3,color='red')
    
ax.autoscale(enable=True, axis='x', tight=True)
fig.subplots_adjust(left=0.1,right=0.98,bottom=0.08,top=0.97)
ax2.autoscale(enable=True, axis='x', tight=True)
fig2.subplots_adjust(left=0.06,right=0.98,bottom=0.08,top=0.97)
    
