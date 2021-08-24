#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:29:14 2020

@author: flarroca
"""

import numpy as np
import networkx as nx
import graspologic as gy
import scipy
import random

class cpd_online_CUSUM():
    """An online CUSUM algorithm for sequences of graphs. It keeps the sum of 
    all previous signal errors (which may be either the gradient or the residual)."""
    
    def __init__(self, exp=3/2, hfun = 'resid'):
        """
        It creates the CUSUM algorithm object. It will use a historic dataset to estimate the RDPG parameters 
        (and the resulting connectivity matrix). As new graphs are input, a monitoring function is computed and 
        accumulated. When this value exceeds a certain threshold, a CPD should be signalled. 

        Parameters
        ----------
        exp : float, optional
            The exponential to be used on the weights (wmk = n*(.k**.exp)). The default is 3/2.
        hfun : str, optional
            The type of monitoring function to use. Although both the residual ('resid') and the gradient ('grad') 
            are valid, the former is preferred (and we've tested mostly with it) . The default is 'resid'.

        Returns
        -------
        None.

        """
        self.exp = exp
        self.hfun = hfun
        
    def init(self, graphs):
        """
        This method provides the historic dataset that is used to compute the RDPG parameters. 

        Parameters
        ----------
        graphs : list of networkx graphs. 
            The historic dataset with m graphs.

        Returns
        -------
        None.

        """
        
        self.n = graphs[0].number_of_nodes()
        self.m = len(graphs)
        self.k = 0
        
        (avg_graph_np, Xhats) = self.compute_avg_graph_and_embed(graphs)
        self.avg_graph_np = avg_graph_np
        
        if type(Xhats) is tuple:
            self.directed = True
            (self.Xlhat0, self.Xrhat0) = Xhats
            if self.hfun=='grad':
                self.S = np.zeros_like(np.concatenate((self.Xlhat0, self.Xrhat0), axis=1))
            elif self.hfun== 'resid':
                self.S = np.zeros_like(avg_graph_np)
            
            self.Phat = self.Xlhat0@self.Xrhat0.T
            
        else: 
            self.directed = False
            self.Xhat0 = Xhats
            
            self.Q = self.compute_Q(avg_graph_np, self.Xhat0.shape[1])
            
            if self.hfun=='grad':
                self.S = np.zeros_like(self.Xhat0)
            elif self.hfun== 'resid':
                self.S = np.zeros_like(avg_graph_np)
            
            self.Phat = self.Xhat0@self.Q@self.Xhat0.T
            
        # self.Phat = np.maximum(np.minimum(self.Phat,1),0)
        self.k = 0
      
        
    def compute_Q(self, matrix, d):
        """
        In order to consider the so-called generalized RDPG, we compute here the Q matrix (Phat = Xhat@Q@Xhat.T). 
        This implies computing the d largest (in absolute value) eigen-values and checking their sign. 

        Parameters
        ----------
        matrix : numpy array
            The matrix for which the ASE is being performed.
        d : integer
            The dimension of the embedding.

        Returns
        -------
        A numpy array of shape (d,d) with the sign of the corresponding eigenvalues. 

        """
        (w,v) = scipy.sparse.linalg.eigs(matrix, k=d,which='LM')
        wabs = np.array(list(zip(-np.abs(w), -np.sign(np.real(w)))), dtype=[('abs', 'f4'), ('sign', 'i4')])
        w = w[np.argsort(wabs,order=['abs','sign'])]
        return np.diag(np.sign(w.real))
        
    def compute_avg_graph_and_embed(self, graphs, d = None):
        """
        Given a list of networkx graphs, this method first computes the average adjacency matrix, and then 
        the corresponding RDPG (the resulting Xhat or (Xlhat,Xrhat) depending on whether the matrix is symmetric or not). 

        Parameters
        ----------
        graphs : list of networkx graphs. 
            The graphs for which the average and the resulting ASE should be computed. 
        d : int, optional
            The dimension of the resulting ASE. If None, it is automatically computed. The default is None.

        Returns
        -------
        avg_graph_np : An (n,n) numpy array (with n the number of nodes)
            The resulting average adjacency matrix.
        Xhats : either a single (n,d) numpy array or a tuple with two arrays (depending on whether the graph is directive or not).
            The ASE of avg_graph_np.

        """
        n = graphs[0].number_of_nodes()
        avg_graph_np = np.zeros((n,n))
        
        for g in graphs:
            # I sort them, so that if nodes came in a different order they are assigned the same position
            avg_graph_np = avg_graph_np + nx.to_numpy_array(g, nodelist=np.sort(g.nodes()))
            # avg_graph_np = avg_graph_np + nx.to_numpy_array(g)
            
        avg_graph_np = avg_graph_np/len(graphs)
        avg_graph_np = gy.utils.augment_diagonal(avg_graph_np)
        
        if d is None:
            self.ase = gy.embed.AdjacencySpectralEmbed(n_elbows=2, diag_aug=False, algorithm='full')
        else: 
            self.ase = gy.embed.AdjacencySpectralEmbed(n_components=d, diag_aug=False, algorithm='full')
        Xhats = self.ase.fit_transform(avg_graph_np)
        
        return (avg_graph_np, Xhats)
    
    def reset(self):
        """
        This method resets the cusum object using the same historic dataset. 

        Returns
        -------
        None.

        """
        self.k = 0
        if self.directed:
            if self.hfun=='grad':
                self.S = np.zeros_like(np.concatenate((self.Xlhat0, self.Xrhat0), axis=1))
            elif self.hfun== 'resid':
                self.S = np.zeros((self.n,self.n))
        else:
            if self.hfun=='grad':
                self.S = np.zeros_like(self.Xhat0)
            elif self.hfun== 'resid':
                self.S = np.zeros((self.n,self.n))

    
    def new_graph(self, g):
        """
        Input a new graph to the cusum object. I.e. it computes the corresponding monitoring function
        and adds the result to the accumulated error matrix. It returns the resulting weighted squared Frobenius norm
        (np.linalg.norm(self.S)**2/wmk). 

        Parameters
        ----------
        g : A networkx graph. 
            The new incoming graph.

        Returns
        -------
        float
            The new resulting statistic.

        """
        current_H = self.compute_current_H(g)
        self.S = self.S + current_H
        
        # gamma = 0
        self.k = self.k+1
        
        wmk = self.n*(self.k**self.exp)
        #wmk = 1/np.sqrt(self.k)
        #wmk = 1/self.k
        # return wmk**2*np.linalg.norm(self.S)
        return np.linalg.norm(self.S)**2/wmk
    
    def compute_current_H(self,g):
        """
        Compute the monitoring function for a new graph. 

        Parameters
        ----------
        g : networkx graph
            A new incoming graph.

        Returns
        -------
        current_H : An (n,n) numpy array.
            The monitoring function evaluated at g (see the constructor to check what the monitoring function may be).

        """
        # I sort them, so that if nodes came in a different order they are assigned the same position
        g_np = nx.to_numpy_array(g, nodelist=np.sort(g.nodes()))
        # g_np = nx.to_numpy_array(g)
        g_np = gy.utils.augment_diagonal(g_np)
        
        if self.hfun == 'grad':
            if self.directed:
                current_Hl = (g_np - self.Xlhat0@self.Xrhat0.T)@self.Xrhat0
                current_Hr = (g_np - self.Xlhat0@self.Xrhat0.T).T@self.Xlhat0
                current_H = np.concatenate((current_Hl,current_Hr), axis=1)
            else: 
                current_H = 4*(self.Xhat0@self.Q@self.Xhat0.T - g_np)@(self.Xhat0@self.Q)
        elif self.hfun == 'resid':
            if self.directed:
                current_H = (g_np - self.Xlhat0@self.Xrhat0.T)
                np.fill_diagonal(current_H,0)
            else: 
                current_H = (self.Xhat0@self.Q@self.Xhat0.T - g_np)
                current_H = np.triu(current_H,1)
        
        return current_H
    
    def estimate_adjacency_variance(self, weighted=False, graphs=[]):
        """
        Estimate the variance of the entries of the adjacency matrix. If the 
        graph is not weighted, this amounts to p_{ij}(1-p_{ij}), and p_{ij} 
        is esimated through the ASE of the historical dataset. If the graph
        is weighted, then we first compute the ASE of the squared (entry-wise)
        adjacency matrix (i.e. for each graph in the histroic dataset, we compute the
        entry-wise square, average it, and compute the ASE). This will provide us
        an estimation of the squared adjacency matrix. We then substract the square 
        of the mean adjacency matrix (already estimated) to obtain the variance. 
                          

        Parameters
        ----------
        weighted : boolean, optional
            If the graphs are weighted or not. The default is False.
        graphs : list of networkx graphs, optional
            If the graphs are weighted, then you have to provide the historical dataset
            since we do not keep it as an attribute. 
            

        Returns
        -------
        sigma_entries : An numpy vector with size depending on whether the 
        graphs were directional (n*n-n) or not (n*(n-1)/2). 
            The estimated variance for all i,j. If the matrices were not directed, 
            we output only the triu entries. Else, we output all but the diagonal.

        """
        if not weighted:
            # sigma_entries = self.Phat*(1-self.Phat)
            # As expected, when the graph is too sparse the embedding does not work. The 
            # result is a too small variance. 
            #TODO Look for a better lower-bound.
            sigma_entries = np.minimum(np.maximum(self.Phat,1/self.n),1)*(1-np.minimum(np.maximum(self.Phat,1/self.n),1))
            # sigma_entries = self.avg_graph_np*(1-self.avg_graph_np)
            if not self.directed:
                sigma_entries = sigma_entries[np.triu_indices(sigma_entries.shape[0], 1)]
            else: 
                sigma_entries = np.concatenate([sigma_entries[np.triu_indices(sigma_entries.shape[0], 1)], sigma_entries[np.tril_indices(sigma_entries.shape[0], 1)]])
        else: 
            if not self.directed:
                (avg_graph_np, Xhat0_quad) = self.compute_avg_graph_and_embed([nx.from_numpy_array(nx.to_numpy_array(g, nodelist=np.sort(g.nodes()))**2) for g in graphs])
                # (avg_graph_np, Xhat0_quad) = self.compute_avg_graph_and_embed([nx.from_numpy_array(nx.to_numpy_array(g)**2) for g in graphs])
                Q = self.compute_Q(avg_graph_np, Xhat0_quad.shape[1])
                
                P_quad = Xhat0_quad@Q@Xhat0_quad.T
                sigma_entries = P_quad - (self.Phat)**2
                sigma_entries = sigma_entries[np.triu_indices(sigma_entries.shape[0], 1)]
            else: 
                (avg_graph_np, Xhats0_quad) = self.compute_avg_graph_and_embed([nx.from_numpy_array(nx.to_numpy_array(g, nodelist=np.sort(g.nodes()))**2, create_using=nx.DiGraph) for g in graphs])
                # (avg_graph_np, Xhats0_quad) = self.compute_avg_graph_and_embed([nx.from_numpy_array(nx.to_numpy_array(g)**2, create_using=nx.DiGraph) for g in graphs])
                
                P_quad = Xhats0_quad[0]@Xhats0_quad[1].T
                sigma_entries = P_quad - (self.Phat)**2
                sigma_entries = np.concatenate([sigma_entries[np.triu_indices(sigma_entries.shape[0], 1)], sigma_entries[np.tril_indices(sigma_entries.shape[0], 1)]])
                
                # This is another possible estimate of the variance (basically E{(A-E{A})^2} instead of E{A^2}-E{A}^2 as before). 
                # We think that the other one works better, but I'll leave the code here just in case. 
                # quad_graphs = [nx.from_numpy_array((nx.to_numpy_array(g, nodelist=np.sort(g.nodes()))-self.Phat)**2, create_using=nx.DiGraph) for g in graphs]
                # (avg_graph_np, Xhats0_quad) = self.compute_avg_graph_and_embed(quad_graphs)
                # P_quad = Xhats0_quad[0]@Xhats0_quad[1].T
                # sigma_entries = P_quad
                # sigma_entries = np.concatenate([sigma_entries[np.triu_indices(sigma_entries.shape[0], 1)], sigma_entries[np.tril_indices(sigma_entries.shape[0], 1)]])
            
        return sigma_entries
    
    def cross_validate_model_error(self, graphs, nboot = 20):
        """
        Estimate the error of the model used on this CUSUM. 
        
        We compare the ASE of one randomly selected graph from `graphs` 
        and the ASE of the (averaged) rest. The difference between these two is 
        assumed similar to the error of the model error of using a single graph. 
        To estimate the error of using all `graphs`, we divide by m**0.5, where 
        m is the length of `graphs`. This is repeated `n_boot` times. 
        
        Note that we suppose `graphs` are actually the historic data, which we 
        do not keep as an attribute and thus need to be passed as an argument. We 
        will then perform all ASEs assuming the same dimension as with the 
        historic data. 

        Parameters
        ----------
        graphs : list of networkx graphs
            The graphs for which the error model is to be calculated. Typically, 
            the historic data. 
        nboot : int, optional
            The number of times to compute the error. The default is 20.

        Returns
        -------
        error_norm_sq : float
            After computing each error matrix, we take the squared norm, divide by `len(graphs)` and return a certain quantile 
            over all random selections.
        error_norm_ij : numpy array of shape (n*(n-1)/2,) (if undirected) or (n*(n-1)) (if directed). 
            The entry-wise quantile of the absolute error matrix (divided by `len(graphs)`).

        """
        errors_cv = []
        errors_cv_sq = []
        
        if nboot < len(graphs):
            indexes = np.random.choice(len(graphs), nboot, replace=False)
        else:
            indexes = np.arange(len(graphs))
            
        indexes = np.random.choice(len(graphs), nboot)
        
        # print(indexes)
        # for idx,g in enumerate(grafos_historicos):
        # for _ in np.arange(nboot):
        for idx in indexes:
            
            # idx = random.randint(0,len(graphs)-1)
            graphs_one_out = graphs[:idx] + graphs[idx+1:]
            
            if not self.directed:
                # I'll use the d (dimension) obtained from the historic data
                d = self.Xhat0.shape[1]
                
                (avg_graph_np, Xhat_one_out) = self.compute_avg_graph_and_embed(graphs_one_out,d)
                Q_one_out = self.compute_Q(avg_graph_np, Xhat_one_out.shape[1])
                P_one_out = Xhat_one_out@Q_one_out@Xhat_one_out.T
            
                (avg_graph_np, Xhat_one) = self.compute_avg_graph_and_embed([graphs[idx]],d)
                Q_one = self.compute_Q(avg_graph_np, Xhat_one.shape[1])

                P_one = Xhat_one@Q_one@Xhat_one.T
            
                E = P_one_out - P_one
                
                # gy.plot.heatmap(E)
                
                E_vec = E[np.triu_indices(E.shape[0], 1)]
                
                errors_cv.append(E_vec)
                errors_cv_sq.append(np.linalg.norm(E_vec)**2/(len(graphs)-1))
                
            else: 
                # I'll use the d (dimension) obtained from the historic data
                d = self.Xlhat0.shape[1]
                
                (avg_graph_np, Xhats_one_out) = self.compute_avg_graph_and_embed(graphs_one_out,d)
                P_one_out = Xhats_one_out[0]@Xhats_one_out[1].T
            
                (avg_graph_np, Xhats_one) = self.compute_avg_graph_and_embed([graphs[idx]],d)
                P_one = Xhats_one[0]@Xhats_one[1].T
            
                E = P_one_out - P_one
                
                # gy.plot.heatmap(E)

                E_vec = np.concatenate([E[np.triu_indices(E.shape[0], 1)], E[np.tril_indices(E.shape[0], 1)]])
                
                errors_cv.append(E_vec)
                errors_cv_sq.append(np.linalg.norm(E_vec)**2/(len(graphs)-1))
            
        # print(errors_cv_sq)
        error_norm_sq = np.quantile(errors_cv_sq, 0.99)
        error_norm_ij = np.quantile(np.array(errors_cv)**2/len(graphs),0.99, axis=0)
        
        return (error_norm_sq, error_norm_ij)
    
    def estimate_confidence_intervals(self, weighted=False, graphs=[], nboots=20, sigma_entries = None, error_norm_sq = None, error_norm_ij = None):
        """
        Given the current `k` state of the algorithm, it computes a whole confidence interval from 
        0 to k (current time). 

        Parameters
        ----------
        weighted : bool, optional
            Whether the graphs are weighted. The default is False.
        graphs : list of networkx graphs, optional
            The historic dataset, which we are not keeping as a class attribute. The default is [].
        nboots : int, optional
            The number of iterations to estimate the errors involved. The default is 20.
        sigma_entries : numpy array of shape (n,n), optional
            The variance of the adjancecy matrix. If None, it is estimated by this object. The default is None.
        error_norm_sq : float, optional
            The squared frobenius norm of the difference between the actual probability matrix and the 
            one inferred from the historic data. If None, it is estimated by this object. The default is None.
        error_norm_ij : numpy array of shape (n*(n-1)/2,) (if undirected) or (n*(n-1)) (if directed). 
            The square of the difference between the actual probability matrix and the one inferred from the 
            historic data. If None, it is estimated by this object. The default is None.            

        Returns
        -------
        m_k : 1-d numpy array. 
            The error signal's estimated mean from k=0 to the current time.
        sigma_k : 1-d numpy array.
            The error signal's estimated standard deviation from k=0 to the current time.

        """

        if sigma_entries is None:
            sigma_entries = self.estimate_adjacency_variance(weighted=weighted, graphs=graphs)
        if (error_norm_sq is None) or (error_norm_ij is None):
            (error_norm_sq, error_norm_ij) = self.cross_validate_model_error(graphs, nboots)
        
        t = np.arange(1,self.k+1)
        
        wmk = self.n*(t**self.exp)
        
        m_k = np.sum(sigma_entries)*t + error_norm_sq*(t**2)
        #print("np.sum(sigma_entries): "+str(np.sum(sigma_entries)))
        #print("error_norm_sq: "+str(error_norm_sq))
        m_k = m_k/wmk
        
        # var_k = residual_variance*(2*residual_variance*(k**2) + 4*error_norm_sq*(k**3))
        var_k = 2*np.square(np.linalg.norm(sigma_entries,2))*(t**2) + 4*np.dot(sigma_entries,error_norm_ij)*(t**3)
        var_k =var_k/wmk**2
        sigma_k = np.sqrt(var_k)
        
        return (m_k, sigma_k)
    
class cpd_online_mMOSUM(cpd_online_CUSUM):
    """An online mMOSUM algorithm for sequences of graphs. It keeps a FIFO queue of variable size of 
    all previous signal errors (which may be either the gradient or the residual)."""
    
    def __init__(self, exp=3/2, hfun = 'grad', bw=0.4):
        """
        It creates the mMOSUM algorithm object. It will use a historic dataset to estimate the RDPG parameters 
        (and the resulting connectivity matrix). As new graphs are input, a monitoring function is computed and 
        kept in a FIFO queue with variable length (of size `n_samples = int(np.ceil(self.bw*self.k))`). 
        When the weighted frobenius norm of its sum exceeds a certain threshold, a CPD should be signalled. 

        Parameters
        ----------
        exp : float, optional
            The exponential to be used on the weights (wmk = n*(.k**.exp)). The default is 3/2.
        hfun : str, optional
            The type of monitoring function to use. Although both the residual ('resid') and the gradient ('grad') 
            are valid, the former is preferred (and we've tested mostly with it) . The default is 'resid'.
        bw : float, optional
            The bandwidth of the mMOSUM algorithm. The size of the FIFO queue is a percentage of the current time, given by `bw`. 

        Returns
        -------
        None.

        """
        cpd_online_CUSUM.__init__(self,exp, hfun)
        self.bw = bw
        
        self.historic_H = None
    
    def reset(self):
        """
        Restarts the algorithm, but keeps the historic data and the resulting RDPG estimation. 

        Returns
        -------
        None.

        """
        cpd_online_CUSUM.reset(self)
        self.historic_H = None
    
    def new_graph(self,g):
        """
        Input a new graph to the mMOSUM object. I.e. it computes the corresponding monitoring function
        and adds the result to the variable-size FIFO queue of residuals (with a size equal to
        `n_samples = int(np.ceil(self.bw*self.k))`). It returns the resulting weighted squared Frobenius norm
        of the sum (np.linalg.norm(self.S)**2/wmk). 

        Parameters
        ----------
        g : A networkx graph. 
            The new incoming graph.

        Returns
        -------
        float
            The resulting statistic.

        """
        current_H = self.compute_current_H(g)
        
        if self.historic_H is None:
            self.historic_H = np.zeros((current_H.shape[0],current_H.shape[1],1))
        
        self.k = self.k+1            
        n_samples = int(np.ceil(self.bw*self.k))
        
        temp_historic_H = np.zeros((current_H.shape[0],current_H.shape[1],n_samples))
        if n_samples>self.historic_H.shape[2]:
            # I have to add a new sample
            temp_historic_H[:,:,1:] = self.historic_H[:,:,:]
            temp_historic_H[:,:,0] = current_H
        else:
            # I apply a FIFO queue
            temp_historic_H[:,:,1:] = self.historic_H[:,:,0:-1]
            temp_historic_H[:,:,0] = current_H
        self.historic_H = temp_historic_H
        
        self.S = np.sum(self.historic_H,axis=2)
        
        wmk = self.n*(n_samples**self.exp)
        
        return np.linalg.norm(self.S)**2/wmk
    
    def estimate_confidence_intervals(self, weighted=False, graphs=[], nboots=20, sigma_entries = None, error_norm_sq = None, error_norm_ij = None):
        """
        Given the current `k` state of the algorithm, it computes a whole confidence interval from 
        0 to k (current time). 

        Parameters
        ----------
        weighted : bool, optional
            Whether the graphs are weighted. The default is False.
        graphs : list of networkx graphs, optional
            The historic dataset, which we are not keeping as a class attribute. The default is [].
        nboots : int, optional
            The number of iterations to estimate the errors involved. The default is 20.
        sigma_entries : numpy array of shape (n,n), optional
            The variance of the adjancecy matrix. If None, it is estimated by this object. The default is None.
        error_norm_sq : float, optional
            The squared frobenius norm of the difference between the actual probability matrix and the 
            one inferred from the historic data. If None, it is estimated by this object. The default is None.
        error_norm_ij : numpy array of shape (n*(n-1)/2,) (if undirected) or (n*(n-1)) (if directed). 
            The square of the difference between the actual probability matrix and the one inferred from the 
            historic data. If None, it is estimated by this object. The default is None.
            
        Returns
        -------
        m_k : 1-d numpy array. 
            The error signal's estimated mean from k=0 to the current time.
        sigma_k : 1-d numpy array.
            The error signal's estimated standard deviation from k=0 to the current time.

        """
        
        if sigma_entries is None:
            sigma_entries = self.estimate_adjacency_variance(weighted=weighted, graphs=graphs)
        if (error_norm_sq is None) or (error_norm_ij is None):
            (error_norm_sq, error_norm_ij) = self.cross_validate_model_error(graphs, nboots)
        
        t = np.arange(1,self.k+1)
        n_samples = (np.ceil(self.bw*t)).astype('int')
        
        wmk = self.n*(n_samples**self.exp)
        
        m_k = np.sum(sigma_entries)*n_samples + error_norm_sq*(n_samples**2)
        m_k = m_k/wmk
        
        # var_k = residual_variance*(2*residual_variance*(k**2) + 4*error_norm_sq*(k**3))
        var_k = 2*np.square(np.linalg.norm(sigma_entries,2))*(n_samples**2) + 4*np.dot(sigma_entries,error_norm_ij)*(n_samples**3)
        var_k =var_k/wmk**2
        sigma_k = np.sqrt(var_k)
        
        return (m_k, sigma_k)
    
class cpd_online_MOSUM(cpd_online_CUSUM):
    """An online MOSUM algorithm for sequences of graphs. It keeps a FIFO queue of fixed size of 
    all previous signal errors (which may be either the gradient or the residual)."""
    
    def __init__(self, exp=3/2, hfun = 'grad', win_relative=1):
        """
        It creates the MOSUM algorithm object. It will use a historic dataset to estimate the RDPG parameters 
        (and the resulting connectivity matrix). As new graphs are input, a monitoring function is computed and 
        kept in a FIFO queue with length `win_relative` times the size of the historic data. 
        When the weighted frobenius norm of its sum exceeds a certain threshold, a CPD should be signalled. 

        Parameters
        ----------
        exp : float, optional
            The exponential to be used on the weights (wmk = n*(.k**.exp)). The default is 3/2.
        hfun : str, optional
            The type of monitoring function to use. Although both the residual ('resid') and the gradient ('grad') 
            are valid, the former is preferred (and we've tested mostly with it) . The default is 'resid'.
        win_relative : float, optional
            The size of the moving window. It is relative to the historic data's size.

        Returns
        -------
        None.

        """        
        cpd_online_CUSUM.__init__(self,exp, hfun)
        self.historic_H = None
        self.win_relative = win_relative
        
    def init(self, graphs):
        """
        This method provides the historic dataset that is used to compute the RDPG parameters. 
        Plus, it computes the window size. 

        Parameters
        ----------
        graphs : list of networkx graphs. 
            The historic dataset with m graphs.

        Returns
        -------
        None.

        """
        super().init(graphs)
        self.win = int(self.m*self.win_relative)
        
    def reset(self):
        """
        Restarts the algorithm, but keeps the historic data and the resulting RDPG estimation. 

        Returns
        -------
        None.

        """
        cpd_online_CUSUM.reset(self)
        self.historic_H = None
        
    def new_graph(self,g):
        """
        Input a new graph to the MOSUM object. I.e. it computes the corresponding monitoring function
        and adds the result to the FIFO queue of residuals (with a size equal to
        `self.win = self.m*self.win_relative`). It returns the resulting weighted squared Frobenius norm
        of the sum (np.linalg.norm(self.S)**2/wmk). 

        Parameters
        ----------
        g : A networkx graph. 
            The new incoming graph.

        Returns
        -------
        float
            The resulting statistic.

        """
        
        current_H = self.compute_current_H(g)
        
        if self.historic_H is None:
            self.historic_H = np.zeros((current_H.shape[0],current_H.shape[1],self.win))
        
        self.k = self.k+1            
        
        # I apply a FIFO queue
        self.historic_H[:,:,1:] = self.historic_H[:,:,0:-1]
        self.historic_H[:,:,0] = current_H

        self.S = np.sum(self.historic_H,axis=2)
        
        win = np.minimum(self.win,self.k)
        wmk = self.n*(win**self.exp)
        
        return np.linalg.norm(self.S)**2/wmk
    
    def estimate_confidence_intervals(self, weighted=False, graphs=[], nboots=20, sigma_entries = None, error_norm_sq = None, error_norm_ij = None):
        """
        Given the current `k` state of the algorithm, it computes a whole confidence interval from 
        0 to k (current time). 

        Parameters
        ----------
        weighted : bool, optional
            Whether the graphs are weighted. The default is False.
        graphs : list of networkx graphs, optional
            The historic dataset, which we are not keeping as a class attribute. The default is [].
        nboots : int, optional
            The number of iterations to estimate the errors involved. The default is 20.
        sigma_entries : numpy array of shape (n,n), optional
            The variance of the adjancecy matrix. If None, it is estimated by this object. The default is None.
        error_norm_sq : float, optional
            The squared frobenius norm of the difference between the actual probability matrix and the 
            one inferred from the historic data. If None, it is estimated by this object. The default is None.
        error_norm_ij : numpy array of shape (n*(n-1)/2,) (if undirected) or (n*(n-1)) (if directed). 
            The square of the difference between the actual probability matrix and the one inferred from the 
            historic data. If None, it is estimated by this object. The default is None.

        Returns
        -------
        m_k : 1-d numpy array. 
            The error signal's estimated mean from k=0 to the current time.
        sigma_k : 1-d numpy array.
            The error signal's estimated standard deviation from k=0 to the current time.

        """
        
        if sigma_entries is None:
            sigma_entries = self.estimate_adjacency_variance(weighted=weighted, graphs=graphs)
        if (error_norm_sq is None) or (error_norm_ij is None):
            (error_norm_sq, error_norm_ij) = self.cross_validate_model_error(graphs, nboots)
        
        t = np.arange(1,self.k+1)
        win = np.minimum(self.win*np.ones_like(t),t)
        
        wmk = self.n*(win**self.exp)
        
        m_k = np.sum(sigma_entries)*win + error_norm_sq*(win**2)
        m_k = m_k/wmk
        
        # var_k = residual_variance*(2*residual_variance*(k**2) + 4*error_norm_sq*(k**3))
        var_k = 2*np.square(np.linalg.norm(sigma_entries,2))*(win**2) + 4*np.dot(sigma_entries,error_norm_ij)*(win**3)
        var_k =var_k/wmk**2
        #print(sigma_entries)
        sigma_k = np.sqrt(var_k)
        
        return (m_k, sigma_k)
    
class cpd_online_expCUSUM(cpd_online_CUSUM):
    """An online CUSUM algorithm for sequences of graphs. It keeps an exponentially weighted average
    all previous signal errors (which may be either the gradient or the residual)."""
    
    def __init__(self, exp=3/2, hfun = 'grad', alpha=0.01):
        """
        It creates the  expCUSUM algorithm object. It will use a historic dataset to estimate the RDPG parameters 
        (and the resulting connectivity matrix). As new graphs are input, a monitoring function is computed and 
        an exponentially weighted sum is kept. 
        When the frobenius norm of the weighted average exceeds a certain threshold, a CPD should be signalled. 

        Parameters
        ----------
        exp : float, optional
            The exponential to be used on the weights (wmk = n*(.k**.exp)). The default is 3/2.
        hfun : str, optional
            The type of monitoring function to use. Although both the residual ('resid') and the gradient ('grad') 
            are valid, the former is preferred (and we've tested mostly with it) . The default is 'resid'.
        alpha : float, optional
            The weighting on the exponential sum (i.e. sum <- new_error + sum*(1-alpha)).

        Returns
        -------
        None.

        """        
        cpd_online_CUSUM.__init__(self,exp, hfun)
        self.historic_H = None
        self.alpha = alpha
        
    def new_graph(self, g):
        """
        Input a new graph to the cusum object. I.e. it computes the corresponding monitoring function
        and adds the result to the exponentially weighted error matrix. It returns the resulting weighted squared Frobenius norm
        (np.linalg.norm(self.S)**2/wmk). Experimental and not tested. 

        Parameters
        ----------
        g : A networkx graph. 
            The new incoming graph.

        Returns
        -------
        float
            The new resulting statistic.

        """
        current_H = self.compute_current_H(g)
        # weighted average
        self.S = (1-self.alpha)*self.S + current_H
        
        # gamma = 0
        self.k = self.k+1
                
        win = 1.0/self.alpha 
        win = np.minimum(win,self.k)
        wmk = self.n*(win**self.exp)
        
        return np.linalg.norm(self.S)**2/wmk
    
    def estimate_confidence_intervals(self, weighted=False, graphs=[], nboots=20, sigma_entries = None, error_norm_sq = None, error_norm_ij = None):
        """
        Given the current `k` state of the algorithm, it computes a whole confidence interval from 
        0 to k (current time). Experimental and not tested. 

        Parameters
        ----------
        weighted : bool, optional
            Whether the graphs are weighted. The default is False.
        graphs : list of networkx graphs, optional
            The historic dataset, which we are not keeping as a class attribute. The default is [].
        nboots : int, optional
            The number of iterations to estimate the errors involved. The default is 20.
        sigma_entries : numpy array of shape (n,n), optional
            The variance of the adjancecy matrix. If None, it is estimated by this object. The default is None.
        error_norm_sq : float, optional
            The squared frobenius norm of the difference between the actual probability matrix and the 
            one inferred from the historic data. If None, it is estimated by this object. The default is None.
        error_norm_ij : numpy array of shape (n*(n-1)/2,) (if undirected) or (n*(n-1)) (if directed). 
            The square of the difference between the actual probability matrix and the one inferred from the 
            historic data. If None, it is estimated by this object. The default is None.

        Returns
        -------
        m_k : 1-d numpy array. 
            The error signal's estimated mean from k=0 to the current time.
        sigma_k : 1-d numpy array.
            The error signal's estimated standard deviation from k=0 to the current time.

        """
        
        if sigma_entries is None:
            sigma_entries = self.estimate_adjacency_variance(weighted=weighted, graphs=graphs)
        if (error_norm_sq is None) or (error_norm_ij is None):
            (error_norm_sq, error_norm_ij) = self.cross_validate_model_error(graphs, nboots)
        
        t = np.arange(1,self.k+1)
        win = 2.0/self.alpha - 1 
        win = np.minimum(win*np.ones_like(t),t)
        
        wmk = self.n*(win**self.exp)
        
        m_k = np.sum(sigma_entries)*win + error_norm_sq*(win**2)
        m_k = m_k/wmk
        
        # var_k = residual_variance*(2*residual_variance*(k**2) + 4*error_norm_sq*(k**3))
        var_k = 2*np.square(np.linalg.norm(sigma_entries,2))*(win**2) + 4*np.dot(sigma_entries,error_norm_ij)*(win**3)
        var_k =var_k/wmk**2
        sigma_k = np.sqrt(var_k)
        
        return (m_k, sigma_k)