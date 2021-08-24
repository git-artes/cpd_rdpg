# -*- coding: utf-8 -*-

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms as transforms
import locale

import cpd_online
import cpd
import random
from itertools import cycle

import funciones_aux
import graspologic as gp

import warnings
warnings.filterwarnings('ignore')

plt.close(fig='all')
plt.rcParams['lines.linewidth'] = 3
# plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 28

COLOR_CYCLE = ["#4286f4", "#f44174"]

## get AllMatches.csv from https://www.fing.edu.uy/owncloud/index.php/s/V2tk4MxZxAvNidx/download
df_matches = pd.read_csv('AllMatches.csv')

df_matches['anio'] = pd.to_datetime(df_matches.Date,format='%m/%d/%Y').dt.year

all_graphs = []

weighted = True
anios = df_matches.anio.unique()
for anio in anios:
    df_matches_recortado = df_matches[df_matches.anio==anio]
    
    # I count how many matches they played regardless of who was home or away.
    ordered_matches = pd.DataFrame(df_matches_recortado[['HomeTeam','AwayTeam']].apply(lambda x: sorted([x['HomeTeam'],x['AwayTeam']]),axis=1).tolist())
    ordered_matches.columns = ['team1','team2']
    num_matchs = ordered_matches.groupby(['team1','team2']).size().reset_index()
    num_matchs.columns = ['team1','team2','weight']
    
    # two version of the graph: weighted and unweighted
    if weighted:
        G = nx.from_pandas_edgelist(num_matchs,source='team1',target='team2',edge_attr='weight',create_using=nx.Graph())
    else: 
        G = nx.from_pandas_edgelist(num_matchs,source='team1',target='team2',create_using=nx.Graph())
    
    all_graphs.append(G)
    

#######################
# I construct the sets
#######################


################ historical data ####


anio_historico_inicial = 1940
anio_historico_final = 1960

cuales = [((anio>=anio_historico_inicial) & (anio<=anio_historico_final)) for anio in anios]
anio_historicos = np.array(anios)[cuales]


grafos_historicos = list(np.array(all_graphs)[cuales])

# nodes to consider
nodes_list = ['Argentina', 'Uruguay', 'Chile', 'Paraguay', 'Brazil', 'Bolivia', 'Peru', 'Ecuador', 'Colombia', 'Venezuela']


# I add the nodes to the historic data (it may happen that a certain country does not play certain years)
for grafo in grafos_historicos: 
    grafo.add_nodes_from(nodes_list) 

# I keep only those nodes that interest me
grafos_historicos = [grafo.subgraph(nodes_list).copy() for grafo in grafos_historicos]

################# monitoring period ########

anio_observacion_inicio = anio_historico_final
anio_observacion_final = np.max(anios)

cuales = [(anio>anio_observacion_inicio) & (anio<=anio_observacion_final) for anio in anios]

anio_observaciones = np.array(anios)[cuales]

grafos = list(np.array(all_graphs)[cuales])
# grafos = list(np.array(all_graphs_unweighted)[cuales])

# I add the nodes to the historic data 
# (it may happen that a certain country does not play certain years or certain countries that do not exist any longer)
for grafo in grafos: 
    grafo.add_nodes_from(nodes_list) 
    
# I keep only those nodes that interest me
grafos = [grafo.subgraph(nodes_list).copy() for grafo in grafos]

# ploteo cosas para que ver cuando estan medianamente quietos los grafos
# pd.DataFrame(index=anio_observaciones, data=[g.number_of_nodes() for g in grafos]).plot(title='nodos')
# pd.DataFrame(index=(list(anio_historicos)+list(anio_observaciones)), data=[g.number_of_edges() for g in (grafos_historicos+grafos)]).plot(title='aristas')

###########################
# checking for changes via the offline method
###########################

grafos_offline = grafos_historicos + grafos
fechas_offline = np.concatenate((anio_historicos,anio_observaciones))

check_offline = True
if check_offline:
    tau = np.log(len(grafos_offline)*grafos_historicos[0].number_of_nodes()/2)/2
         
    algo = cpd.nrdpgwbs(tau, 50)
    algo.fit(grafos_offline)
     
    algo.find_segments(0,len(grafos_offline))
    print(f"Change-points detected by the offline method: " + str([(anio_historico_inicial+cp) for cp in algo.estimated_change_points]))

###########################
# and now the online method
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



# I compute the intervals and detect the change-points

(m_k, sigma) = cusum.estimate_confidence_intervals(weighted=weighted, graphs=grafos_historicos)
thresh  =  m_k+3*sigma
detected_cp = np.where(signal_errors > thresh)[0][0]

###########################
# now plot stuff
###########################

fig, ax = plt.subplots()
ax.plot(anio_observaciones,signal_errors,label=r'$\omega[k]\Gamma[m,k]$')
ax.plot(anio_observaciones,m_k,color='g',label=r'Estimated mean')
ax.plot(anio_observaciones,thresh,color='g',linestyle='dashed',label=r'Threshold')
# ax.set_title("En 1986 la copa america pasa de un caos a organizarse cada dos años",fontsize=20)

cps_fechas_off = [fechas_offline[cp]for cp in sorted(algo.estimated_change_points)]
xmin,xmax = ax.get_xlim()
cps_fechas = [xmin] + cps_fechas_off + [xmax]

alpha = 0.2
color_cycle = cycle(COLOR_CYCLE)
trans2 = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    
for (start, end), col in zip(cpd.pairwise(cps_fechas), color_cycle):
    ax.axvspan(max(0, start), end, facecolor=col, alpha=alpha)

ax.axvline(anio_observaciones[detected_cp],linestyle='--',linewidth=3,color='black')

ax.set_xlabel(r"Year", fontsize=42)
ax.set_ylabel(r"Weighted statistic", fontsize=42)
#ax.tick_params(axis='x',labelsize=16)
#ax.tick_params(axis='y',labelsize=16)
ax.autoscale(enable=True, axis='x', tight=True)

fig.subplots_adjust(left=0.06,right=0.98,bottom=0.11,top=0.97)

ax.grid(True)
plt.legend(fontsize=40,loc='upper left')

#############################################################
# finally, plot the resulting embeddings of the historic data and the last ones (to see the detected change)
###########################################################

grafos_final = grafos[-len(grafos_historicos)//2:]
cusum_aux = cpd_online.cpd_online_MOSUM(hfun='resid',exp=3/2)
cusum_aux.init(grafos_final)

grafos_promedio = [cusum.avg_graph_np, cusum_aux.avg_graph_np]
oe = gp.embed.OmnibusEmbed(n_elbows = 2)

Zhat = oe.fit_transform(grafos_promedio)
# #            funciones_aux.scatter_multi_grafos_anotado(list(grafosq),Zhat, dims=(1,2,3))

# plt.figure()
funciones_aux.scatter_multi_grafos_anotado(grafos,Zhat, dims=(1,2))

ax = plt.gca()
ax.tick_params(axis='x',labelsize=30)
ax.tick_params(axis='y',labelsize=30)
labels = []
for nodo in np.sort(grafos[0].nodes()):
    labels.append(nodo)
    

for i,txt in enumerate(labels):
    if txt in ['Uruguay','Brazil','Argentina','Chile','Paraguay']:
        ax.annotate(txt,(Zhat[1,i,0],Zhat[1,i,1]), fontsize=35)
    else:
        ax.annotate(txt,(Zhat[0,i,0],Zhat[0,i,1]), fontsize=35)
        
fig = plt.gcf()
fig.subplots_adjust(left=0.06,right=0.98,bottom=0.05,top=0.97)
            
plt.show()


