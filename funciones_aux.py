#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 13:13:27 2020

@author: flarroca
"""
import matplotlib.pyplot as plt
import networkx as nx
import scipy
import numpy as np

import scipy

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D

class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._xyz = (x,y,z)
        self._dxdydz = (dx,dy,dz)

    def draw(self, renderer):
        x1,y1,z1 = self._xyz
        dx,dy,dz = self._dxdydz
        x2,y2,z2 = (x1+dx,y1+dy,z1+dz)

        xs, ys, zs = proj_transform((x1,x2),(y1,y2),(z1,z2), renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        super().draw(renderer)

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D,'arrow3D',_arrow3D)

def scatter_anotado(list_g, list_Xhat, dims=(1,2), pintar_ejes = True, leyenda=True, anotado= True, maxmin=None, polar=True):
        
    numpy_dims = np.array(dims)
    if(numpy_dims[numpy_dims<=0].size>0):
        print('dims must be strictly positive')
        return

    labels = []
    categories = []
    category_i = 0
    for g in list_g:
        if type(g) is np.ndarray:
            labels.extend([str(i) for i in range(0,g.shape[0])])
            categories.extend([category_i for i in range(0, g.shape [0])])
            np_array = g
            category_i += 1
        else:
            # supogno que es un grafo de networkx. son las dos chances que hay...
#            labels = []
#            categories = []
            for nodo in g.nodes():
                labels.append(nodo)
                node_dict = g.nodes[nodo]
                if node_dict.get('category') is None:
                    categories.append(category_i)
                else:
                    categories.append(str(node_dict.get('category'))+str(category_i))
            np_array = nx.to_numpy_array(g)
        category_i += 1
    
    
#    print(categories)
    if pintar_ejes:
        (w,v) = scipy.sparse.linalg.eigs(np_array, k=list_Xhat[0].shape[1],which='LM')
        
        #ordeno los valores propios por magnitud (para que quede coherente con el orden del embedding)
    #    w = w[np.argsort(-abs(w))]
        # ademas, a veces son de igual magnitud pero distinto signo. pongo primero el positivo
        wabs = np.array(list(zip(-np.abs(w), -np.sign(np.real(w)))), dtype=[('abs', 'f4'), ('sign', 'i4')])
        w = w[np.argsort(wabs,order=['abs','sign'])]
    #    print(w)
        dim_negativas = [i for i, x in enumerate(w) if x < 0]
        if len(dim_negativas)>0:
            print('Dimensiones con valor propio negativo: '+str(np.array(dim_negativas)+1))
    
    # fig, ax = plt.subplots()
    if len(dims)==2:
        if polar:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        else:
            fig, ax = plt.subplots()
        
        Xhat_total = np.vstack(list_Xhat)
        
        if polar:
            complejo = Xhat_total[:,dims[0]-1]+1j*Xhat_total[:,dims[1]-1]
            scatter = ax.scatter(np.angle(complejo),np.abs(complejo),c=[int(cat) for cat in categories], cmap=plt.cm.coolwarm)
        else:
            scatter = ax.scatter(Xhat_total[:,dims[0]-1],Xhat_total[:,dims[1]-1],c=[int(cat) for cat in categories], cmap=plt.cm.coolwarm)
        
        if maxmin is not None:
            ax.set_thetamin(maxmin[0])
            ax.set_thetamax(maxmin[1])
#        ax.legend()
        if leyenda: 
            legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="Classes")
            ax.add_artist(legend1)
        # marco con rojo las dimensiones con valor propio negativo
#        plt.xlabel("dim "+str(dims[0]-1)+" (vp="+str(w[dims[0]-1])+")")
#        plt.ylabel("dim "+str(dims[1]-1)+" (vp="+str(w[dims[1]-1])+")")
        if pintar_ejes:
            if(w[dims[0]-1]<0):
                ax.tick_params(axis='x', colors='red')
            if(w[dims[1]-1]<0):
                ax.tick_params(axis='y', colors='red')
            
        # le pongo nombre a los puntos
        if anotado:
            for i,txt in enumerate(labels):
                if polar:
                    complejo = Xhat_total[i,dims[0]-1]+1j*Xhat_total[i,dims[1]-1]
                    ax.annotate(txt,(np.angle(complejo),np.abs(complejo)), fontsize=12)
                else:
                    ax.annotate(txt,(Xhat_total[i,dims[0]-1],Xhat_total[i,dims[1]-1]))
    elif len(dims)==3:
        ax = plt.axes(projection='3d')
        Xhat_total = np.vstack(list_Xhat)
        ax.scatter3D(xs = Xhat_total[:,dims[0]-1], ys = Xhat_total[:,dims[1]-1], zs = Xhat_total[:,dims[2]-1], c=categories, cmap=plt.cm.coolwarm)
#        plt.xlabel("dim "+str(dims[0]-1)+" (vp="+str(w[dims[0]-1])+")")
#        plt.ylabel("dim "+str(dims[1]-1)+" (vp="+str(w[dims[1]-1])+")")
#        ax.set_zlabel("dim "+str(dims[1]-1)+" (vp="+str(w[dims[2]-1])+")")
        if pintar_ejes:
            if(w[dims[0]-1]<0):
                ax.tick_params(axis='x', colors='red')
            if(w[dims[1]-1]<0):
                ax.tick_params(axis='y', colors='red')
            if(w[dims[2]-1]<0):
                ax.tick_params(axis='z', colors='red')
        
        Xhat_total = np.vstack(list_Xhat)
        if anotado:
            for i,txt in enumerate(labels):
                ax.text(Xhat_total[i,dims[0]-1],Xhat_total[i,dims[1]-1],Xhat_total[i,dims[2]-1],txt)
            
    return ax
            
def scatter_multi_grafos_anotado(grafos, Zhat, dims=(1,2)):
        
    numpy_dims = np.array(dims)
    if(numpy_dims[numpy_dims<=0].size>0):
        print('dims must be strictly positive')
        return

    if type(grafos[0]) is np.ndarray:
        labels = [str(i) for i in range(0,grafos[0].shape[0])]        
        lista_arrays = grafos
    else:
        # supogno que es un grafo de networkx. son las dos chances que hay...
        labels = []
        for nodo in np.sort(grafos[0].nodes()):
            labels.append(nodo)
        lista_arrays = []
        for g in grafos:
            lista_arrays.append(nx.to_numpy_array(g, nodelist=np.sort(g.nodes())))
        
    (w,v) = scipy.sparse.linalg.eigs(lista_arrays[0], k=Zhat[0].shape[1],which='LM')
    
    #ordeno los valores propios por magnitud (para que quede coherente con el orden del embedding)
    w = w[np.argsort(-abs(w))]
#    print(w)
    dim_negativas = [i for i, x in enumerate(w) if x < 0]
    if len(dim_negativas)>0:
        print('Dimensiones con valor propio negativo: '+str(np.array(dim_negativas)+1))
    
    fig, ax = plt.subplots()
    if len(dims)==2:
        for Xhat in Zhat:
            ax.scatter(Xhat[:,dims[0]-1],Xhat[:,dims[1]-1])
#            print(str(labels))
        # las labels se las pongo solo al último para que se vea bien
        #for i,txt in enumerate(labels):
            #ax.annotate(txt,(Xhat[i,dims[0]-1],Xhat[i,dims[1]-1]), fontsize=16)
            
        for igrafo in range(len(Zhat)-1):
            current_xhat = Zhat[igrafo]
            next_xhat = Zhat[igrafo+1]
            for inodo in range(len(current_xhat)):
                (x,y) = current_xhat[inodo,dims[0]-1],current_xhat[inodo,dims[1]-1]
                (nextx,nexty) = next_xhat[inodo,dims[0]-1],next_xhat[inodo,dims[1]-1]
                ax.arrow(x,y,nextx-x,nexty-y,alpha=0.5)

        # marco con rojo las dimensiones con valor propio negativo
        # if(w[dims[0]-1]<0):
        #     ax.tick_params(axis='x', colors='red')
        # if(w[dims[1]-1]<0):
        #     ax.tick_params(axis='y', colors='red')
            
        # le pongo nombre a los puntos
        ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
        ax.grid()


    elif len(dims)==3:
        ax = plt.axes(projection='3d')
        for Xhat in Zhat:
            ax.scatter3D(xs = Xhat[:,dims[0]-1], ys = Xhat[:,dims[1]-1], zs = Xhat[:,dims[2]-1])
        
        # las labels se las pongo solo al último para que se vea bien
        for i,txt in enumerate(labels):
            ax.text(Xhat[i,dims[0]-1],Xhat[i,dims[1]-1],Xhat[i,dims[2]-1],txt)            
            
        for igrafo in range(len(Zhat)-1):
            current_xhat = Zhat[igrafo]
            next_xhat = Zhat[igrafo+1]
            for inodo in range(len(current_xhat)):
                (x,y,z) = current_xhat[inodo,dims[0]-1],current_xhat[inodo,dims[1]-1],current_xhat[inodo,dims[2]-1]
                (nextx,nexty,nextz) = next_xhat[inodo,dims[0]-1],next_xhat[inodo,dims[1]-1],next_xhat[inodo,dims[2]-1]
                ax.arrow3D(x,y,z,nextx-x,nexty-y,nextz-z, alpha=0.1)
        
        # if(w[dims[0]-1]<0):
        #     ax.tick_params(axis='x', colors='red')
        # if(w[dims[1]-1]<0):
        #     ax.tick_params(axis='y', colors='red')
        # if(w[dims[2]-1]<0):
        #     ax.tick_params(axis='z', colors='red')
        
def normalizar_rdpg_directivo(Xhatl,Xhatr):
    dims = Xhatl.shape[1]
    for d in np.arange(dims):
        factor = np.sqrt(np.max(Xhatl[:,d])/np.max(Xhatr[:,d]))
        Xhatl[:,d] = Xhatl[:,d]/factor
        Xhatr[:,d] = Xhatr[:,d]*factor
    return (Xhatl, Xhatr)
