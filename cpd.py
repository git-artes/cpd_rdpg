#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 02:13:28 2020

@author: flarroca
"""

import numpy as np
import scipy
import numpy.matlib
import graspologic as gy
import networkx as nx
import matplotlib.pyplot as plt
from itertools import tee
from itertools import cycle
COLOR_CYCLE = ["#4286f4", "#f44174"]

class nbs():
    def __init__(
        self,
        tau
    ):
        self.tau = tau
        self.estimated_change_points = []
        self.flag = 0
        
    def fit(self,Y):
        """
        Load data and initialize changepoint array.

        Parameters
        ----------
        Y : list of length T with arrays of length n(t)
            Data to detect changepoints.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        # I embed Y into an array of lists simply to be able to slice through indexes
        self.Y = np.array(Y, dtype=object)
#        self.flag = 0
        self.estimated_change_points = []
        self.nt = np.array([len(data) for data in self.Y])
        
        self.Y_concatenated = np.concatenate(self.Y)
        
    def compute_statistic(self,s,e):
        """
        Computes the D^t_{s,e} statistic for all t between s+1 and e-1 and outputs the maximum. 
        """
        
#        Y = self.Y[s:e+1,:]
#        nt = self.Y.shape[1]
        
        statistic = -np.inf
        t_aster = 0
        
        ns = 0
        if s>0:
            ns = np.sum(self.nt[0:s])
        ne = np.sum(self.nt[0:e])
        for t in np.arange(s+1,e):
            # aca parece haber un error porque no detecta los cambios cuando t esta desbalanceado
#            Dt = np.sqrt((t-s+1)*nt*(e-t)*nt/((e-s+1)*nt))*np.max(np.abs(self.ecdf(s,t,e)))
            
            nt = np.sum(self.nt[0:t])
            # Dt = np.sqrt( np.sum(self.nt[s:t+1])*np.sum(self.nt[t+1:e+1])/np.sum(self.nt[s:e+1]) )*np.max(np.abs(self.ecdf(s,t,e)))
            (ksdist,pvalue) = scipy.stats.kstest(self.Y_concatenated[ns:nt],self.Y_concatenated[nt:ne])
            Dt = np.sqrt( np.sum(self.nt[s:t+1])*np.sum(self.nt[t+1:e+1])/np.sum(self.nt[s:e+1]) )*ksdist
            
#            print("D["+str(t)+"]="+str(Dt))
            if Dt>statistic:
                statistic = Dt
                t_aster = t
  
        
        return (statistic, t_aster)
                
        
    def find_segments(self,s,e):
#        print("s: "+str(s)+" e: "+str(e))
#        while e-s>2 and self.flag==0:
        if e-s>2:
            (a,b) = self.compute_statistic(s,e)
            if a<=self.tau:
#                self.flag = 1 # revisar esto del flag...
                return
            else:
#                print("encontre un cp en "+str(b)+" con un estadistico de "+str(a))
                self.estimated_change_points.append(b)
                self.find_segments(s,b-1)
                self.find_segments(b,e)
                
#        return self.estimated_change_points
        
    def ecdf(self,s,t,e):
        """
        Deprecated. I used it to calculate the KS distance, but then I found that scipy
        had a function to calculate it (much faster!). I'm leaving it here just in case. 
        """
        
#        datas = self.Y[s:t,:].reshape(-1)
#        datae = self.Y[t:e,:].reshape(-1)
#        data = self.Y[s:e,:].reshape(-1)
        
#        datas = np.concatenate(self.Y[s:t])
#        datae = np.concatenate(self.Y[t:e])
#        data = np.concatenate(self.Y[s:e])
        
        ns = 0
        if s>0:
            ns = np.sum(self.nt[0:s])
        nt = np.sum(self.nt[0:t])
        ne = np.sum(self.nt[0:e])
        datas = self.Y_concatenated[ns:nt]
        datae = self.Y_concatenated[nt:ne]
        data = self.Y_concatenated[ns:ne]
        
        # is there a faster method?
#        Fs = np.array([sum( datas <= x)/float(len(datas)) for x in data])
#        Fe = np.array([sum( datae <= x)/float(len(datae)) for x in data])
    
#        ### esto se debe poder hacer mas rapido con un tensor que ya cargue el t en la tecera dimension
#        Fs = (np.matlib.repmat(datas.reshape(-1,1),1,data.shape[0]) < \
#            np.matlib.repmat(data,datas.shape[0],1)).sum(axis=0)/datas.size
#        Fe = (np.matlib.repmat(datae.reshape(-1,1),1,data.shape[0]) < \
#            np.matlib.repmat(data,datae.shape[0],1)).sum(axis=0)/datae.size

        Fs = (datas.reshape(-1,1) < data).sum(axis=0)/datas.size
        Fe = (datae.reshape(-1,1) < data).sum(axis=0)/datae.size        
#        print("hizo el ecdf")
        return Fs-Fe

class nwbs(nbs):
    def __init__(
        self,
        tau, 
        M
    ):
        nbs.__init__(self,tau)
        self.M = M
        
    def fit(self,Y):
        """
        Load data, initialize changepoint array and choose random intervals.

        Parameters
        ----------
        Y : array_like of shape (T,n)
            Data to detect changepoints.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        super().fit(Y)
#        self.Y = Y
##        self.flag = 0
#        self.estimated_change_points = []
#        self.intervals = np.sort(np.random.randint(0,Y.shape[0],(self.M,2)))
        self.intervals = np.sort(np.random.randint(0,len(Y),(self.M,2)))
                
        
    def find_segments(self,s,e):
        print("s: "+str(s)+" e: "+str(e))
        list_a = []
        list_b = []
        # for interval in self.intervals:
        for interval in np.vstack((self.intervals,[s,e])):
            # print("interval: "+str(interval))
            if (interval[0]>=s) & (interval[1]<=e):
                sm = np.max([s,interval[0]])
                em = np.min([e,interval[1]])
    #            print("alpha_m "+str(interval[0])+" beta_m: "+str(interval[1]))
    #            print("sm: "+str(sm)+" em: "+str(em))
                if em-sm>2:
                    (am,bm) = self.compute_statistic(sm,em)
                    list_a.append(am)
                    list_b.append(bm)
                # if len(list_a)>0:
                    # print("sm: "+str(sm)+" em: "+str(em)+ " max statistic: "+str(np.max(list_a))+ " en "+str(list_b[np.argmax(list_a)]))
                    print("interval: "+str(interval)+ " sm: "+str(sm)+" em: "+str(em)+ " max statistic: "+str(am)+ " en "+str(bm))
        if len(list_a)>0:
            m_aster = np.argmax(list_a)
            a = list_a[m_aster]
            b = list_b[m_aster]
            
            if a>self.tau:
                print("encontre un cp en "+str(b)+" con un estadistico de "+str(a))
                self.estimated_change_points.append(b)
                self.find_segments(s,b-1)
                self.find_segments(b,e)
            else:
                print("rechace el cp en "+str(b)+" porque "+str(a)+" es mas chico que "+str(self.tau))
                
#        return self.estimated_change_points

class nrdpgwbs(nwbs):

            
    def __init__(
        self,
        tau, 
        M, 
        d=10
    ):
        nwbs.__init__(self,tau,M)
        
        # the embedding method
        self.d = d
        # self.ase = gy.embed.AdjacencySpectralEmbed(n_components=1)
        self.ase = gy.embed.AdjacencySpectralEmbed(n_elbows=2, algorithm='full')
        
    def fit(self,graphs,nodes=None,dims=None,outin='both',shuffle=False):
        """
        Load graphs (and embed them), initialize changepoint array and choose random intervals.

        Parameters
        ----------
        graphs : list-like of networkx graphs (or adjacency matrix) of length T. 
            Data to detect changepoints. Graphs should have the same set of nodes or else a list of nodes should be 
            specified in the nodes parameter (nothing checked).
        nodes : list with array-likes with the nodes' index. The nodes to consider on the statistic (all nodes are 
            considered in the embedding). Default: all nodes are used. 
        dims : array-like. The dimensions of the embedding to be used in the statistic. All values should 
            be smaller than self.d (not checked). Default: all dimensions are used. 
        outin : either 'both', 'out' or 'in'. If the graph is directional, it will use either both embeddings, 
            only the out-degree or the in-degree one.
        shuffle : choose a random selection of node pairs or just the diagonal as in the original paper. 

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        
        T = len(graphs)
                
        if dims is None:
            # if no dims are specified, i'll use them all
            dims = np.arange(self.d)
        if nodes is None:
#            n = graphs[0].number_of_nodes()
#            nodes = np.tile(np.arange(n),(T,1))
            n = [grafo.number_of_nodes() for grafo in graphs]
            nodes = [np.arange(num_nodes) for num_nodes in n]
#            nodes = np.ones(T,n)
        else:
#            n = len(nodes[0])
            n = [len(nodos) for nodos in nodes]
#        print("n: "+str(n))
        
        # I embed the graphs
#        Y = np.zeros((T,int(np.floor(n/2))))
        Y = [np.zeros(int(np.floor(num_nodes/2))) for num_nodes in n]
        for t in np.arange(T):
            g = nx.to_numpy_array(graphs[t])
            g = gy.utils.augment_diagonal(g)
            Xhats = self.ase.fit_transform(g)
            if (type(Xhats)!=tuple):
                Xhat = Xhats
                # el grafo no era direccional y tengo un solo Xhat
                (w,v) = scipy.sparse.linalg.eigs(g, k=Xhat.shape[1],which='LM')
                #I sort the eigenvalues in magnitude (to make it coherent with the embedding)
                # TODO if there is a tie??? SOLVED
                # w = w[np.argsort(-abs(w))]
                wabs = np.array(list(zip(-np.abs(w), -np.sign(np.real(w)))), dtype=[('abs', 'f4'), ('sign', 'i4')])
                w = w[np.argsort(wabs,order=['abs','sign'])]
                #    print(w)
                # I use gRDPG
                
    #            statistic = np.matmul(np.sign(w)*Xhat,Xhat.T)
    #            statistic = np.matmul(np.sign(np.real(w))*Xhat,Xhat.T)
                
                # statistic = np.matmul(np.sign(np.real(w[dims]))*Xhat[:,dims][nodes[t]],Xhat[:,dims][nodes[t]].T)
                statistic = np.matmul(np.sign(np.real(w[:]))*Xhat[:,:][nodes[t]],Xhat[:,:][nodes[t]].T)
                # TODO the theory says I should ignore a random node completely if n is odd. 
                # I'm ignoring only the last node's statistic. 
                if shuffle:
                    halfn = int(np.floor(n[t]/2))
                    choice = np.random.choice(statistic.shape[0],statistic.shape[0],replace=False)
                    Y[t] = statistic[choice[0:halfn],choice[halfn:2*halfn]]
                else: 
    #                Y[t,:] = statistic.diagonal(int(np.floor(n/2)))[0:int(np.floor(n/2))]
                    Y[t] = statistic.diagonal(int(np.floor(n[t]/2)))[0:int(np.floor(n[t]/2))]
            else:
                Xhatout, Xhatin = Xhats
                Xhatl = Xhatout
                Xhatr = Xhatin
                
                (Xhatl,Xhatr) = normalize_rdpg_directive(Xhatl,Xhatr)
                if outin=='in':
                    Xhatl = Xhatin
                    Xhatr = Xhatin
                elif outin=='out':
                    Xhatl = Xhatout
                    Xhatr = Xhatout
                    
                # print("directive")
                # statistic = np.matmul(Xhatl[:,dims][nodes[t]],Xhatr[:,dims][nodes[t]].T)
                statistic = np.matmul(Xhatl[:,:][nodes[t]],Xhatr[:,:][nodes[t]].T)
                if shuffle:
                    halfn = int(np.floor(n[t]/2))
                    choice = np.random.choice(statistic.shape[0],statistic.shape[0],replace=False)
                    Y[t] = statistic[choice[0:halfn],choice[halfn:2*halfn]]
                else: 
        #                Y[t,:] = statistic.diagonal(int(np.floor(n/2)))[0:int(np.floor(n/2))]
                    Y[t] = statistic.diagonal(int(np.floor(n[t]/2)))[0:int(np.floor(n[t]/2))]
            
        
        super(nrdpgwbs, self).fit(Y)
        
def normalize_rdpg_directive(Xhatl,Xhatr):
    dims = Xhatl.shape[1]
    for d in np.arange(dims):
        factor = np.sqrt(np.max(Xhatl[:,d])/np.max(Xhatr[:,d]))
        Xhatl[:,d] = Xhatl[:,d]/factor
        Xhatr[:,d] = Xhatr[:,d]*factor
    return (Xhatl, Xhatr)
        
#        self.Y = Y
#        self.flag = 0
#        self.estimated_change_points = []
#        self.intervals = np.sort(np.random.randint(0,Y.shape[0],(self.M,2)))      

""" Taken from https://github.com/deepcharles/ruptures/blob/master/ruptures/show/display.py. I'm copying it
here and making small convenient modifications. 
.. _sec-display:
Display
====================================================================================================
Description
----------------------------------------------------------------------------------------------------
The function :func:`display` displays a signal and the change points provided in alternating colors.
If another set of change point indexes is provided, they are displayed with dashed vertical dashed lines.
Usage
----------------------------------------------------------------------------------------------------
Start with the usual imports and create a signal.
.. code-block:: python
    import numpy as np
    import matplotlib.pylab as plt
    import ruptures as rpt
    # creation of data
    n, dim = 500, 2  # number of samples, dimension
    n_bkps, sigma = 3, 5  # number of change points, noise standart deviation
    signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma)
    rpt.display(signal, bkps)
If we computed another set of change points, for instance ``[110, 150, 320, 500]``, we can easily compare the two segmentations.
.. code-block:: python
    rpt.display(signal, bkps, [110, 150, 320, 500])
.. figure:: /images/example-display.png
    :scale: 50 %
    Example output of the function :func:`display`.
Code explanation
----------------------------------------------------------------------------------------------------
.. autofunction:: ruptures.show.display.display
"""

def display(signal, true_chg_pts, computed_chg_pts=None, **kwargs):
    """
    Display a signal and the change points provided in alternating colors. If another set of change
    point is provided, they are displayed with dashed vertical dashed lines.
    The following matplotlib subplots options is set by default, but can be changed when calling `display`):
    - "figsize": (10, 2 * n_features),  # figure size
    Args:
        signal (array): signal array, shape (n_samples,) or (n_samples, n_features).
        true_chg_pts (list): list of change point indexes.
        computed_chg_pts (list, optional): list of change point indexes.
        **kwargs : all additional keyword arguments are passed to the plt.subplots call.
    Returns:
        tuple: (figure, axarr) with a :class:`matplotlib.figure.Figure` object and an array of Axes objects.
    """

    if type(signal) != np.ndarray:
        # Try to get array from Pandas dataframe
        signal = signal.values

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    n_samples, n_features = signal.shape

    n_features = 1
    # let's set a sensible defaut size for the subplots
    matplotlib_options = {
        "figsize": (10, 4 * n_features),  # figure size
    }
    # add/update the options given by the user
    matplotlib_options.update(kwargs)

    # create plots
    # n_features = 1
    fig, axarr = plt.subplots(n_features, sharex=True, **matplotlib_options)
    # if n_features == 1:
    #     axarr = [axarr]

    # for axe, sig in zip(axarr, signal.T):
    color_cycle = cycle(COLOR_CYCLE)
    # plot s
    axarr.plot(signal)
    axarr.tick_params(axis = 'both', which = 'major', labelsize = 16)
    axarr.set_xlabel("Time",fontsize=20)
    # print("toda la bola")
    # axarr.set_ylabel(ylabel,fontsize=14)

    # color each (true) regime
    bkps = [0] + sorted(true_chg_pts)
    alpha = 0.2  # transparency of the colored background

    for (start, end), col in zip(pairwise(bkps), color_cycle):
        axarr.axvspan(max(0, start - 0.5), end - 0.5, facecolor=col, alpha=alpha)

    color = "k"  # color of the lines indicating the computed_chg_pts
    linewidth = 3  # linewidth of the lines indicating the computed_chg_pts
    linestyle = "--"  # linestyle of the lines indicating the computed_chg_pts
    # vertical lines to mark the computed_chg_pts
    if computed_chg_pts is not None:
        for bkp in computed_chg_pts:
            if bkp != 0 and bkp < n_samples:
                axarr.axvline(
                    x=bkp - 0.5,
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                )

    fig.tight_layout()

    return fig, axarr

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)