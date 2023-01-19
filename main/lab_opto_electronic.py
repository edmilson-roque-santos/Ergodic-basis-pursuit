"""
Script for analyzing Opto electronic experimental data

Created on Wed Oct  6 14:53:58 2021

@author: Edmilson Roque dos Santos
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec    
import networkx as nx
import numpy as np
import os
from scipy import stats

import h5dict

from EBP import tools, net_dyn
from EBP.modules.opto_electronic import opto_electronic as opt_elec

#colors = ['palevioletred', 'orange', 'darkviolet', 'darkred', 'darkseagreen']
colors = ['darkgrey', 'orange', 'darkviolet', 'darkslategrey', 'silver']
#silver - match
#orange - false positive
#darkviolet - false negative

def plot_traj_density(X_time_series, parameters):
    
    number_of_iterations = X_time_series.shape[0]
    nodelist = parameters['nodelist']
    
    lower_bound = np.min(X_time_series)
    upper_bound = np.max(X_time_series)
    
    interval = np.arange(lower_bound, upper_bound, 0.001)
    
    fig, ax_f = plt.subplots(2, 1, sharex=True, figsize = (6, 4), dpi = 100)
    #ax_f[0].plot(X_time_series[:number_of_iterations-1, :], X_time_series[1:number_of_iterations, :], 'o', markersize=5)
    
    N = X_time_series.shape[1]
    
    nodelist = list(range(N))
    col = plt.cm.tab20b(np.arange(N))  
    #np.linspace(0,1,N)
    for index in nodelist:
        Opto_orbit = X_time_series[: number_of_iterations, index]
        kernel = stats.gaussian_kde(Opto_orbit, bw_method = 0.05)
        ax_f[1].plot(interval, 
                  kernel(interval)/kernel.integrate_box_1d(lower_bound, upper_bound), 
                  label="{}".format(index),
                  color = col[index])

        ax_f[0].plot(X_time_series[:number_of_iterations-1, index], 
                     X_time_series[1:number_of_iterations, index], 
                     'o', 
                     color = col[index],
                     markersize=5)


    Opto_orbit = X_time_series.T.flatten()
    kernel = stats.gaussian_kde(Opto_orbit, bw_method = 5e-2)
    
    ax_f[1].plot(interval,
              kernel(interval)/kernel.integrate_box_1d(lower_bound, upper_bound), 
              'k--')
    
    fig.suptitle(r'Optoelectronic $\sigma = {}$'.format(parameters['coupling']), fontsize = 20)
    ax_f[0].set_ylabel(r'$y(t + 1)$', fontsize = 16)
    ax_f[1].set_ylabel(r'$Density$', fontsize = 16)
    ax_f[1].set_xlabel(r'$y$', fontsize = 16)
    l = parameters.get('lower_bound', lower_bound)
    u = parameters.get('upper_bound', upper_bound)
    ax_f[1].set_xlim(l, u)
    plt.legend(loc = 0, ncol=5, fontsize = 8)
    if parameters['filename'] == None:
        plt.show()
    else:
        plt.savefig(parameters['filename']+".pdf", format = 'pdf')

def plot_cluster_traj_density(X_time_series, cluster_list, parameters):
   
    number_of_iterations = X_time_series.shape[0]
    nodelist = parameters['nodelist']
    
    lower_bound = np.min(X_time_series)
    upper_bound = np.max(X_time_series)
    
    interval = np.arange(lower_bound, upper_bound, 0.001)
    
    fig, ax_f = plt.subplots(2, 1, sharex=True, figsize = (6, 4), dpi = 100)
   
    N = X_time_series.shape[1]
    col = plt.cm.tab20b(np.arange(N))  
    
    num_clusters = len(cluster_list)
    col_cluster = ['tab:purple', 'tab:orange']
    
    nodelist = np.arange(0, N, dtype = int)
    
    for id_cluster in range(num_clusters):
        id_vec_cluster =  np.asarray(cluster_list[id_cluster], dtype = int)
        mask_cluster = np.isin(nodelist, id_vec_cluster)
        
        X_t_cluster = X_time_series[:, mask_cluster]
        data_cluster = X_t_cluster.T.flatten()
        kernel_cluster = stats.gaussian_kde(data_cluster, bw_method = 0.05)
        
        
        for index in id_vec_cluster:
            Opto_orbit = X_time_series[: number_of_iterations, index]
            kernel = stats.gaussian_kde(Opto_orbit, bw_method = 0.05)
            ax_f[1].plot(interval, 
                      kernel(interval)/kernel.integrate_box_1d(lower_bound, upper_bound), 
                      label="{}".format(index),
                      color = col[index],
                      alpha = 0.35)
    
            ax_f[0].plot(X_time_series[:number_of_iterations-1, index], 
                         X_time_series[1:number_of_iterations, index], 
                         'o', 
                         color = col[index],
                         markersize=5)
            
        ax_f[1].plot(interval,
              kernel_cluster(interval)/kernel_cluster.integrate_box_1d(lower_bound, upper_bound), 
              '--',
              color = col_cluster[id_cluster],
              alpha = 1.0)
   
    fig.suptitle(r'Optoelectronic $\sigma = {}$'.format(parameters['coupling']), fontsize = 20)
    ax_f[0].set_ylabel(r'$y(t + 1)$', fontsize = 16)
    ax_f[1].set_ylabel(r'$Density$', fontsize = 16)
    ax_f[1].set_xlabel(r'$y$', fontsize = 16)
    plt.legend(loc = 0, ncol=5, fontsize = 8)
    if parameters['filename'] == None:
        plt.show()
    else:
        plt.savefig(parameters['filename']+".pdf", format = 'pdf')

def plot_return_map(ax, X_time_series, args):
    '''
    Plot return map for each node from multivariate time series.

    Parameters
    ----------
    ax : Matplotlib Axes object
    
    X_time_series : numpy array - size (length_of_time_series, number_of_vertices)
       Multivariate time series.
    args : dict
        arguments of function to detail some aspects.
        'plot_subset': Boolean
            If shaded area in the subset is highlighted
        'subset': list 
            Define upper and lower bound of subset in phase space
        'filename': str
            Filename for saving pdf 

    Returns
    -------
    None.
    '''
    number_of_iterations = X_time_series.shape[0]
    N = X_time_series.shape[1]
    nodelist = np.arange(N, dtype = int)
    
    lower_bound = np.min(X_time_series)
    upper_bound = np.max(X_time_series)
    
    col = plt.cm.tab20b(nodelist)  
    
    for index in nodelist:
        ax.plot(X_time_series[:number_of_iterations-1, index], 
                X_time_series[1:number_of_iterations, index], 
                'o', 
                color = col[index],
                markersize=5)
    
    if args['plot_subset']:
        y = np.arange(lower_bound, upper_bound + 0.5)
        ax.fill_betweenx(y,
                        args['subset'][0]*np.ones(y.shape[0]),
                        args['subset'][1]*np.ones(y.shape[0]),
                        alpha = 0.7, 
                        color = 'silver') 
        
    ax.set_ylabel(r'$y(t + 1)$')

def plot_kernel_density(ax, X_time_series, args):
    '''
    Plot density function for each node from multivariate time series.

    Parameters
    ----------
    ax : Matplotlib Axes object
    
    X_time_series : numpy array - size (length_of_time_series, number_of_vertices)
       Multivariate time series.
    args : dict
        DESCRIPTION.

    Returns
    -------
    None.
    '''
    number_of_iterations = X_time_series.shape[0]
    N = X_time_series.shape[1]
    nodelist = np.arange(N, dtype = int)
    
    lower_bound = np.min(X_time_series)
    upper_bound = np.max(X_time_series)
    interval = np.arange(lower_bound, upper_bound, 0.001)
       
    col = plt.cm.tab20b(nodelist)  
    
    for index in nodelist:
        Opto_orbit = X_time_series[: number_of_iterations, index]
        kernel = stats.gaussian_kde(Opto_orbit, bw_method = 0.05)
        ax.plot(interval, 
                  kernel(interval)/kernel.integrate_box_1d(lower_bound, upper_bound), 
                  label="{}".format(index),
                  color = col[index])

    ax.set_ylabel(r'$\hat{\nu}(y)$')
        
    ax.set_xlabel(r'$y$')
    ax.set_xlim(lower_bound, upper_bound)

    return ax

def plot_clusterize_density(ax, X_time_series, cluster_list):
    '''
    Clusterize the kernel density functions to visualize the desired number of 
    clusters in cluster_list

    Parameters
    ----------
    ax : Matplotlib Axes object
        
    X_time_series : numpy array - size (length_of_time_series, number_of_vertices)
        Multivariate time series.
    cluster_list : list
        List corresponding to partition of the nodelist caracterizing the clusters.

    Returns
    -------
    None.
    '''  

    number_of_iterations = X_time_series.shape[0]
    N = X_time_series.shape[1]
    nodelist = np.arange(N, dtype = int)
    
    lower_bound = np.min(X_time_series)
    upper_bound = np.max(X_time_series)
    interval = np.arange(lower_bound, upper_bound, 0.001)
    
    
    col = plt.cm.tab20b(nodelist)  
    
    num_clusters = len(cluster_list)
    col_cluster = ['darkblue', 'brown']
    legend_ = [r' $\rho^1$ : Nodes 1 to 5', r'$\rho^2$ : Nodes 6 to 17']
    nodelist = np.arange(0, N, dtype = int)
    
    for id_cluster in range(num_clusters):
        id_vec_cluster =  np.asarray(cluster_list[id_cluster], dtype = int)
        mask_cluster = np.isin(nodelist, id_vec_cluster)
        
        X_t_cluster = X_time_series[:, mask_cluster]
        data_cluster = X_t_cluster.T.flatten()
        kernel_cluster = stats.gaussian_kde(data_cluster, bw_method = 0.05)
        
        for index in id_vec_cluster:
            Opto_orbit = X_time_series[: number_of_iterations, index]
            kernel = stats.gaussian_kde(Opto_orbit, bw_method = 0.05)
            ax.plot(interval, 
                      kernel(interval)/kernel.integrate_box_1d(lower_bound, upper_bound),
                      color = col[index],
                      alpha = 0.25)
       
        ax.plot(interval,
              kernel_cluster(interval)/kernel_cluster.integrate_box_1d(lower_bound, upper_bound), 
              '--',
              label = legend_[id_cluster],
              color = col_cluster[id_cluster],
              alpha = 1.0)
    
    

def plot_panel_pipeline(X_time_series, args):
    '''
    
    Parameters
    ----------
    X_time_series : numpy array - size (length_of_time_series, number_of_vertices)
        Multivariate time series.
    args : dict
        arguments of function to detail some aspects.
        'plot_subset': Boolean
            If shaded area in the subset is highlighted
        'subset': list 
            Define upper and lower bound of subset in phase space
        'filename': str
            Filename for saving pdf 
    Returns
    -------
    None.

    '''
    fig = plt.figure(figsize = (16, 5), dpi = 100)
    gs = GridSpec(nrows=2, ncols=3)
    
    args['plot_subset'] = True
    args['subset'] = [3.4, 4.5]
    
    ax_0 = fig.add_subplot(gs[0, 0])
    ax_1 = fig.add_subplot(gs[1, 0])
    ax_0.set_title(args['title'][0])
    y = X_time_series[args['start_measurement']: args['start_measurement']\
                      + args['number_of_iterations'], :]
    
    plot_return_map(ax_0, y, args)
    plot_kernel_density(ax_1, y, args)
    ax_1.legend(loc = 0, ncol=5, fontsize = 8)

    args['plot_subset'] = not True
    data = opt_elec.select_subset_phase_space(X_time_series, args['subset'][0],
                                              args['subset'][1])
    print(data.shape)
    ax_2 = fig.add_subplot(gs[0, 1])
    ax_2.set_title(args['title'][1])
    ax_3 = fig.add_subplot(gs[1, 1])
    
    plot_return_map(ax_2, data, args)
    plot_kernel_density(ax_3, data, args)
    
    ax_4 = fig.add_subplot(gs[:, 2])
    plot_clusterize_density(ax_4, data, args['clusters_list'])
    ax_4.set_title(args['title'][2])
    ax_4.set_ylabel(r'$\hat{\nu}(y)$', fontsize = 15)
    ax_4.set_xlabel(r'$y$', fontsize = 15)
    ax_4.legend(loc = 0, ncol=5, fontsize = 12)
    
    if args['filename'] == None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(args['filename']+".pdf", format = 'pdf')
    
    return
  
def plot_subset_clustering(X_time_series, args):
    '''
    
    Parameters
    ----------
    X_time_series : numpy array - size (length_of_time_series, number_of_vertices)
        Multivariate time series.
    args : dict
        arguments of function to detail some aspects.
        'plot_subset': Boolean
            If shaded area in the subset is highlighted
        'subset': list 
            Define upper and lower bound of subset in phase space
        'filename': str
            Filename for saving pdf 
       
    Returns
    -------
    None.

    '''
   
    fig = plt.figure(figsize=(11, 4), dpi = 300)
    gs = GridSpec(1, 3, figure=fig, wspace = 0.25)
    
    ax = fig.add_subplot(gs[0])
    ax_plot_true_net(ax, probed_node=6)
    ax.set_title(r'a) Original network')
    
    args['plot_subset'] = False
    args['subset'] = [3.4, 4.5]
    data = X_time_series
    
    ax_1 = fig.add_subplot(gs[1])
    plot_return_map(ax_1, data, args)  
    ax_1.set_xlabel(r'$y(t)$')
    ax_1.set_title(r'b) Return map')
    
    
    ax_2 = fig.add_subplot(gs[2])
    plot_clusterize_density(ax_2, data, args['clusters_list'])
    ax_2.set_ylabel(r'$\rho(y)$')
    ax_2.set_xlabel(r'$y$')
   
    ax_2.legend(loc = 0,fontsize = 12)
    ax_2.set_title(r'c) Clustering density functions')

    if args['filename'] == None:
        plt.show()
    else:
        plt.savefig(args['filename']+".pdf", format='pdf', bbox_inches='tight')

    
    return    
  
def plot_subset_selection(outfolder = 'Figures', filename = 'default',
                          plot_pipeline = False):  
    '''
    Generate plot involving data selection subset and clustering of density functions.    

    Parameters
    ----------
    outfolder : str, optional
        Folder of output figure. The default is 'Figures'.
    filename : str, optional
        Filename name for saving pdf figure. The default is 'default'.
    plot_pipeline : boolean, optional
        Boolean for which plot is done. The default is False.
        True: entire pipeline is plotted
        False: only part of pipeline is plotted.
    Returns
    -------
    None.

    '''      
    folder = "data"+"/"+"opto_electronic_data"+"/"+"symmetric_data"+"/"
    coupling_vec = np.arange(0.0156250, 1.093750 + 0.015625, 0.0156250)
    float_formatter = "{:.6f}".format
    
    cluster_list = [np.arange(0, 5, 1, dtype = int), np.arange(5, 17, 1, dtype = int)]
    
    args = dict()
    args['start_measurement'] = 1000
    args['number_of_iterations'] = 5000
    args['clusters_list'] = cluster_list
    args['title'] = ['Subset selection', 'Parabolic shape', 'Clusterizing']
    
    for id_sig in range(10, 11):
        print(coupling_vec[id_sig])
        X_time_series_data = np.loadtxt(folder+"N_symm_data_sigma = {}.txt".format(float_formatter(coupling_vec[id_sig]))) 
        if filename == None:
            args['filename'] = filename
        if filename == 'default':
            args['filename'] = outfolder+"/"+'Data_selection_{}'.format(float_formatter(coupling_vec[id_sig])) #None
        
        if plot_pipeline:
            plot_panel_pipeline(X_time_series_data, args)
        else:
            plot_subset_clustering(X_time_series_data, args)

def weights_from_edges(G, edgeslist):
    '''
    Extract weights from edges.

    Parameters
    ----------
    G : graph
        A networkx graph.
    edgeslist : collection of edge tuples (default=G.edges(data=True))
        
    Returns
    -------
    numpy array
    weights of edgelist
    
    '''
    num_edges = len(edgeslist)
    if num_edges > 0:
        edges_weights = np.zeros(num_edges)
        for edge_id in range(num_edges):
            u, v = edgeslist[edge_id][0], edgeslist[edge_id][1] 
            edges_weights[edge_id] = G.get_edge_data(u, v)['weight']
        
        return np.array(edges_weights)

    else:
        return []

def pos_optoelectronic(cluster_draw = [np.arange(0, 11, 1, dtype=int), 
                                       np.arange(11, 17, 1,dtype = int)]):
    '''
    Position nodes using clustering information in cluster_draw. Each cluster
    is plotted in circular template using nx.circular_layout

    Parameters
    ----------
    cluster_draw : list, optional
        DESCRIPTION. The default is [np.arange(0, 11, 1, dtype=int), 
                                     np.arange(11, 17, 1,dtype = int)].

    Returns
    -------
    pos_true : dict
        A dictionary of positions keyed by node.

    '''
    pos_true = dict()
    for id_cluster in range(len(cluster_draw)):
        cluster_size = cluster_draw[id_cluster].shape[0]
        pos = nx.circular_layout(nx.path_graph(cluster_size), 
                            scale = cluster_size/2,     
                            center = np.array([10*id_cluster, 10*id_cluster]))
        
        for id_node in cluster_draw[id_cluster]:
            pos_true[id_node] = pos[id_node - id_cluster*(11)]
    
    return pos_true

def links_types(G, G_true):
    '''
    Compare a graph G and edges set.

    Parameters
    ----------
    G : graph
        A networkx graph.
    edges_G_true : set
        Edges set from graph that is considered true.

    Returns
    -------
    links : dict
        A dictionary keyed by type of links in the comparison, false positive,
        false negatives, or intersection.

    '''
    
    edges_G_true = G_true.edges()
    edges_G_true = set(edges_G_true)
    
    links = dict()    
    
    links['set_edges_G_estimated'] = set(G.edges())
    links['intersection'] = list(links['set_edges_G_estimated'] & edges_G_true)
    links['intersec_weights'] = weights_from_edges(G, links['intersection'])
    
    links['false_positives'] = list(links['set_edges_G_estimated'] - edges_G_true)
    links['false_pos_weights'] = weights_from_edges(G, links['false_positives'])
    
    links['false_negatives'] = list(edges_G_true - links['set_edges_G_estimated'])
    
    return links

def plot_recons_G(ax, G_estimated, links, pos = pos_optoelectronic()):
    '''
    Plot reconstructed G and compare with true graph using links dictionary.

    Parameters
    ----------
    ax : Matplotlib Axes object
        
    G_estimated : graph
        A networkx graph.
    links : dict
        A dictionary keyed by type of links in the comparison, false positive,
        false negatives, or intersection..
    pos : dict, optional
        A dictionary of positions keyed by node.
        The default is generated by function pos_optoelectronic().

    Returns
    -------
    None.

    '''
    intersection = links['intersection']
    intersec_weights = links['intersec_weights']    
    false_positives = links['false_positives']
    false_pos_weights = links['false_pos_weights']
    false_negatives = links['false_negatives']
    print(false_negatives)
    nx.draw_networkx_nodes(G_estimated, pos = pos, 
                           ax = ax,
                           node_color = colors[3], linewidths= 1.0,
                           node_size = 450,
                           alpha = 1.0)
    
    nx.draw_networkx_nodes(G_estimated, pos = pos, 
                           ax = ax,
                           node_color = colors[0], 
                           node_size = 400,
                           alpha = 1.0)
    
    nx.draw_networkx_edges(G_estimated, pos = pos, edgelist = intersection,
                           ax = ax,
                           width = intersec_weights*2.5,
                           edge_color = colors[4], alpha = 1.0)
    
    nx.draw_networkx_edges(G_estimated, pos = pos, edgelist = false_positives, 
                           edge_color = colors[1],
                           ax = ax,
                           width = false_pos_weights,
                           alpha = 1.0,
                           connectionstyle='arc3,rad=0.2')
    
    nx.draw_networkx_edges(G_estimated, pos = pos, edgelist = false_negatives, 
                           edge_color = colors[2], 
                           ax = ax,
                           width = 1.0,
                           connectionstyle='arc3,rad=0.2')
    
    ax.margins(0.10)
    ax.axis("off")

def plot_reconst_G_optoelectronic(ax, reconst_adj_matrix):
    '''
    Plot reconstructed optoelectronic network and compare with true graph 
    using links dictionary.

    Parameters
    ----------
    ax : Matplotlib Axes object
        Draw the graph in the specified Matplotlib axes.
    
    reconst_adj_matrix : numpy array - size: (number_of_vertices, number_of_vertices)
        Adjacency matrix resulting from greedy network reconstruction algorithm.

    Returns
    -------
    None.

    '''
    A_true, G_true = opt_elec.get_Adj_matrix()
    
    A_estimated = reconst_adj_matrix
    G_estimated = nx.from_numpy_array(A_estimated.T, create_using = nx.Graph)
    
    links = links_types(G_estimated, G_true)    
    
    plot_recons_G(ax, G_estimated, links)
    
    
def plot_weighted_fp_fn_links(reconst_adj_matrix, ax = None, filename = 'opto_electr_crit3'):    
    '''
    Status plot show reconstruct opto electronic graph and histogram of weighted
    edgelist

    Parameters
    ----------
    reconst_adj_matrix : numpy array - size (number_of_vertices, number_of_vertices)
        Adjacency matrix resulting from greedy network reconstruction algorithm
        
    filename : str, optional
        Filename to save pdf. The default is 'opto_electr_crit3'.

    Returns
    -------
    None.

    '''
    separate_fig = False
    if ax == None:
        fig, ax = plt.subplots(1, 2, figsize = (8, 4), dpi = 300)
        separate_fig = True

    ax1 = ax[0]
    plot_reconst_G_optoelectronic(ax1, reconst_adj_matrix)
    if separate_fig:
        ax1.set_title("Reconstructed Optoelectronic network")

    ax2 = ax[1]
    A_true, G_true = opt_elec.get_Adj_matrix()
    A_estimated = reconst_adj_matrix
    G_estimated = nx.from_numpy_array(A_estimated.T, create_using = nx.Graph)
    links = links_types(G_estimated, G_true)    
    intersec_weights = links['intersec_weights']    
    false_pos_weights = links['false_pos_weights']
    
    bins_auto = np.histogram_bin_edges(intersec_weights, bins='auto')
    n, bins, patches = ax2.hist(intersec_weights, bins_auto, 
                                density=False, facecolor=colors[4], alpha=0.75)
    bins_auto = np.histogram_bin_edges(false_pos_weights, bins='auto')
    n, bins, patches = ax2.hist(false_pos_weights, bins_auto, 
                                density=False, facecolor=colors[1], alpha=0.75)
    
    max_fp = np.max(false_pos_weights)
    print(max_fp)
    ax2.vlines(max_fp, 0, np.max(n), colors='k', linestyles='dashed')
    
    ax2.set_xlabel("edge weights")
    ax2.set_ylabel("Histogram")    
    if separate_fig:
        if filename == None:
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(filename+".pdf", format='pdf')
            
    return 

def ax_plot_true_net(ax, probed_node = 0, print_probed = True, plot_net_alone = False):
    '''
    Plot original optoelectronic network   

    Parameters
    ----------
    ax : Matplotlib Axes object
        Draw the graph in the specified Matplotlib axes.
    probed_node : int, optional
        Node being probed is highlighted in 'darkred'. The default is 0.
    print_probed : boolean, optional
        boolean if the probed node is highlighted. The default is True.
    plot_net_alone : boolean, optional
        Title is plotted embedded in the Matplotlib Axes object. The default is False.    

    Returns
    -------
    None.

    '''
    A_true, G_true = opt_elec.get_Adj_matrix()
    pos_true = pos_optoelectronic()

    nx.draw_networkx_nodes(G_true, pos = pos_true, 
                           ax = ax, node_color = colors[3], 
                           linewidths= 1.0,
                           node_size = 250,
                           alpha = 1.0)
    nx.draw_networkx_nodes(G_true, pos = pos_true, 
                           node_color = colors[0], 
                           node_size = 200,
                           ax = ax,
                           alpha = 1.0)
    
    if print_probed:
        nx.draw_networkx_nodes(G_true, pos = pos_true, 
                               ax = ax,
                               nodelist=[probed_node],
                               node_color = colors[3], 
                               node_size = 200,
                               alpha = 1.0)
        
    nx.draw_networkx_edges(G_true, pos = pos_true, 
                           ax = ax,
                           edgelist = list(G_true.edges()), 
                           edge_color = colors[4],
                           arrowsize = 7,
                           width = 0.65,
                           alpha = 1.0)
    ax.margins(0.3)
    ax.axis("off")
    if plot_net_alone:
        ax.set_title('{}'.format('Original Network'))
        

def ax_relaxing_path(ax, info_x_eps, node, threshold):
    '''
    Plot the relaxing path for a given node 

    Parameters
    ----------
    ax : Matplotlib Axes object
        Draw the graph in the specified Matplotlib axes.
    info_x_eps : TYPE
        DESCRIPTION.
    node : TYPE
        DESCRIPTION.
    threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    x_eps_path = info_x_eps[node]['x_eps_path']
    noise_vector = info_x_eps['noise_vector']
    noise_min = info_x_eps[node]['noise_min']
    noise_flag = info_x_eps[node]['eps_flag']['eps']
    
    
    indices_list_x_eps = np.arange(0, x_eps_path.shape[0], 1, dtype = int)
    for id_eps in range(noise_vector.shape[0]):
        supp_t = np.where(np.absolute(x_eps_path[:, id_eps]) >= threshold)[0]
        if supp_t.shape[0] > 0:
            ax.scatter(np.ones(supp_t.shape[0])*noise_vector[id_eps], indices_list_x_eps[supp_t], 
                       c = 'tab:blue', marker = 'o', alpha = 1.0)
        
        else:
            ax.scatter(np.ones(1)*noise_vector[id_eps], indices_list_x_eps[1], 
                       c = 'w', marker = 'o')
     
    ax.vlines(noise_vector[noise_min_loc(info_x_eps, node)], -0.5, 
              indices_list_x_eps[-1]+0.5, 
              colors = 'black', 
              linestyles = 'dashed', linewidths=1.8)
    
    ax.vlines(noise_flag, -0.5, indices_list_x_eps[-1]+0.5, colors = 'dimgray',
              linestyles = 'dashed', linewidths=1.8)
    
    ax.set_ylabel(r'$x(\epsilon)$ indices')
    ax.set_ylim(-0.5,  indices_list_x_eps[-1]+0.5)
    ax.set_xlabel(r'$\epsilon$')
    ax.set_xlim(noise_vector[0], noise_vector[-1]+1e-2)
    ax.invert_yaxis()
    ax.set_title(r'Relaxing path')
    ax.set_title(r'b)', loc = 'left')

def position_true_diff(node, nodelist_, nodelist, subgraph_nodelist):
    '''
    Generates the position dictionary for plotting relaxing path.    

    Parameters
    ----------
    node : int
        Probed node in the relaxing path algorithm.
    nodelist_ : list
        complete nodelist in the reconstructed Graph union True graph.
    nodelist : list
        Nodelist of the reconstructed Graph.
    subgraph_nodelist : list
        Nodelist of the true Graph.

    Returns
    -------
    pos : dict
        positions of the nodes for plotting relaxing path.
    diff_G : list
        Nodelist of symmetric difference.

    '''
    mask_intrue = np.isin(nodelist_, subgraph_nodelist)
    mask_inrecons = np.isin(nodelist_, nodelist)
    
    intersec_G = list(nodelist_[(mask_intrue)&(mask_inrecons)])
    intersec_G.remove(node)
    theta = np.linspace(0.25, 0.75, len(intersec_G) + 1)[:-1] * 2 * np.pi
    theta = theta.astype(np.float32)
    pos = np.column_stack(
        [np.cos(theta), np.sin(theta)]
    )

    pos_intersec = dict(zip(intersec_G, pos))
    
    diff_G = list(nodelist_[(~mask_intrue)|(~mask_inrecons)])
    theta = np.linspace(0.85, 1.15, len(diff_G) + 1)[:-1] * 2 * np.pi
    theta = theta.astype(np.float32)
    pos = np.column_stack(
        [np.cos(theta), np.sin(theta)]
    )
    
    pos_diff = dict(zip(diff_G, pos))
    pos = dict(pos_intersec)
    pos.update(pos_diff)
    
    center = [0, 0]
    pos[node] = center
    
    pos = nx.rescale_layout_dict(pos, scale=1)
    
    return pos, diff_G 

def ax_false_proportion(ax, info_x_eps):
    '''
    False link proportion with varying noise level \epsilon

    Parameters
    ----------
    ax : Matplotlib Axes object
        Draw the graph in the specified Matplotlib axes.
    info_x_eps : dict
        Dictionary that contains information from the reconstruction algorithm.

    Returns
    -------
    None.

    '''
    params_node = info_x_eps['params']
    noise_vec = info_x_eps['noise_vector']
    A_true, G_true = opt_elec.get_Adj_matrix()
    size = (noise_vec.shape[0],A_true.shape[0])
    FN, FP = np.zeros(size), np.zeros(size)           
    
    for node in range(A_true.shape[0]):
        x_eps_path = info_x_eps[node]['x_eps_path'] 
        
        for eps_counter in range(noise_vec.shape[0]):
            threshold = noise_vec[eps_counter]/np.sqrt(params_node['L'])
            adj_row = net_dyn.get_adj_row_from_coeff_vec(node, x_eps_path[:, eps_counter], 
                                                         params_node, threshold, True)
            
            #Calculate the fraction link proportion imported from tools
            rel_error = tools.FP_FN_rel_error(node, A_true[:, node], adj_row, True)
            FN[eps_counter, node] = rel_error['FN']
            FP[eps_counter, node] = rel_error['FP']
    
    mean_FP = np.mean(FP, axis = 1)
    ax.plot(noise_vec, mean_FP, '-o', label='FP', color=colors[1])

    mean_FN = np.mean(FN, axis = 1)    
    ax.plot(noise_vec, mean_FN, '-o', label='FN', color=colors[2])
    
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r'$\epsilon$',fontsize = 25)
    ax.legend(loc=0)
    

    return

def ax_probed_node_links(ax, info_x_eps, node, x_eps, eps, threshold):
    '''
    B_eps result: the connections reconstructed for node in the step eps_counter
    of the relaxing path.

    Parameters
    ----------
    ax : Matplotlib Axes object
        Draw the graph in the specified Matplotlib axes.
    info_x_eps : dict
        Dictionary that contains information from the reconstruction algorithm.
    node : int
        Probed node.
    eps_counter : int
        Index of the corresponding parameter epsilon entry to be 
        highlighted using a dashed vertical line.
    threshold : float
        Value that defines the edges in the network. Any entry that has absolute
        value smaller than threshold is equal to zero. 
        
    Returns
    -------
    None.

    '''
    params_node = info_x_eps['params'] 
    #x_eps_path = info_x_eps[node]['x_eps_path'] 
    adj_row = net_dyn.get_adj_row_from_coeff_vec(node, x_eps, 
                                                 params_node, threshold, True)
    #if eps_counter == -1:
    #    adj_row = net_dyn.get_adj_row_from_coeff_vec(node, info_x_eps[node]['min_l2_sol'][:, node], 
    #                                                 params_node, threshold, True)
           
    
    A_true, G_true = opt_elec.get_Adj_matrix()
    #Select neighbors of node and append node itself
    neigh_true = np.append([node], [n for n in G_true[node]])
    #Create a subgraph from neigh_true
    subgraph_G_true = G_true.subgraph(np.unique(neigh_true)).copy()
    edges_G_true = subgraph_G_true.edges()
    #Filter true edges which only involve node
    edges_G_true = filter(lambda c: c[0] == node or c[1] == node, edges_G_true)
        
    subgraph_nodelist = list(subgraph_G_true.nodes())
    #Create a subgraph to be compared to reconstruct subgraph involving node
    H_true = nx.Graph()
    H_true.add_nodes_from(subgraph_nodelist) 
    H_true.add_edges_from(edges_G_true)
    
    mask =  adj_row > 0
    nodelist = np.unique(np.append([node], params_node['cluster_list'][mask]))
    nodelist_ = np.union1d(nodelist, subgraph_nodelist)
    G = nx.Graph()
    G.add_nodes_from(nodelist_)
    for v in params_node['cluster_list'][mask]:
        G.add_edge(node, v, weight = adj_row[v])

    pos, diff_G = position_true_diff(node, nodelist_, nodelist, subgraph_nodelist)
    links = links_types(G, H_true)    
    intersection = links['intersection']
    false_positives = links['false_positives']
    false_negatives = links['false_negatives']
    
    nx.draw_networkx_nodes(G, pos = pos, ax = ax,
                           node_color = colors[3], linewidths= 1.0,
                           node_size = 150,
                           alpha = 1.0)
    nx.draw_networkx_nodes(G, pos = pos, ax = ax,
                           node_color = colors[0], 
                           node_size = 100,
                           alpha = 1.0)
    
    nx.draw_networkx_nodes(G, pos = pos, 
                               ax = ax,
                               nodelist=[node],
                               node_color = colors[3], 
                               node_size = 200,
                               alpha = 1.0)
    
    nx.draw_networkx_edges(G, pos = pos, edgelist = intersection,
                           ax = ax,
                           width = links['intersec_weights'],
                           edge_color = colors[4], alpha = 1.0)
    
    nx.draw_networkx_nodes(G, pos = pos, 
                               ax = ax,
                               nodelist=diff_G,
                               node_color = colors[1], 
                               node_size = 150,
                               alpha = 1.0)
    
    nx.draw_networkx_edges(G, pos = pos, edgelist = false_positives, 
                           edge_color = colors[1],
                           ax = ax,
                           width = links['false_pos_weights'],
                           alpha = 1.0,
                           connectionstyle='arc3,rad=0.2')
    
    nx.draw_networkx_edges(G, pos = pos, edgelist = false_negatives, 
                           edge_color = colors[2], 
                           ax = ax,
                           width = 1.0,
                           connectionstyle='arc3,rad=0.2')
    
    #nx.draw_networkx_labels(G, pos, ax = ax,
    #                        font_family="Computer Modern Serif")
    ax.set_title(r'$\epsilon = {:.4f}$'.format(eps))
    ax.margins(0.10)
    ax.axis("off")

def noise_min_loc(info_x_eps, node):
    '''
    Locate the minimum noise corrsponding to L2 minimum solution

    Parameters
    ----------
    info_x_eps : dict
        Dictionary that contains information from the reconstruction algorithm.
    node : int
        probed node.

    Returns
    -------
    TYPE
        index of the minimum node in the noise vector.

    '''
    mask = np.where(info_x_eps['noise_vector'] >= info_x_eps[node]['noise_min'])[0]
    id_vec_noise = np.arange(0, info_x_eps['noise_vector'].shape[0], 1, dtype = int)

    return id_vec_noise[mask][0]

def plot_B_eps(gr_alg, node, threshold, id_cluster = 0, filename = None):    
    '''
    Plot B_eps relaxing path algorithm for trivial graph partition.

    Parameters
    ----------
    gr_alg : dict
        Greedy Network Recontruction dictionary.
    node : int
        Probed node to be highlighted.
    threshold : TYPE
        DESCRIPTION.
    id_cluster : int, optional
        Cluster id for identification. The default is 0 because is the trivial
        graph partition.
    filename : str, optional
        Filename to save figure. In case of None, the figure is only shown. 
        The default is None.

    Returns
    -------
    None.

    '''
    fig_ = plt.figure(figsize=(15, 5), dpi = 300)
    subfigs = fig_.subfigures(1, 3, width_ratios = [0.6, 1.2, 0.8], 
                              wspace = 0.75)
    
    fig = subfigs[0]
    gs = GridSpec(1, 1, figure=fig,left=0.01, right=1.0, top = 0.990, 
                  bottom = 0.01)
    
    ax1 = fig.add_subplot(gs[0])
    ax_plot_true_net(ax1, probed_node=node)
    fig.suptitle(r'a) Original Network') 
    
    #======================================================#
    #======================================================#
    fig1 = subfigs[1]
    
    gs1 = GridSpec(1, 3, figure=fig1, 
                   left=0.01, right=1.0, top = 0.80, bottom = 0.05)
    
    info_dict = gr_alg['info_x_eps'][id_cluster]

    ax3 = fig1.add_subplot(gs1[0])
    x_eps = info_dict[node]['min_l2_sol']#[:, node]
    eps_min = info_dict[node]['noise_min']
    threshold_noise = eps_min/np.sqrt(gr_alg['PHI'].shape[1])
    ax_probed_node_links(ax3, info_dict, node, x_eps, eps_min, threshold_noise)
    
    
    ax3 = fig1.add_subplot(gs1[1])
    eps_counter = noise_min_loc(info_dict, node)
    x_eps = info_dict[node]['x_eps_path'][:, eps_counter]
    eps = info_dict['noise_vector'][eps_counter]
    threshold_noise = eps/np.sqrt(gr_alg['PHI'].shape[1])
    ax_probed_node_links(ax3, info_dict, node,  x_eps, eps, threshold_noise)
    
    ax4 = fig1.add_subplot(gs1[2])
    eps_counter = info_dict[node]['eps_flag']['eps_counter']
    x_eps = info_dict[node]['x_eps_path'][:, eps_counter]
    eps = info_dict[node]['eps_flag']['eps']
    threshold_noise = eps/np.sqrt(gr_alg['PHI'].shape[1])
    ax_probed_node_links(ax4, info_dict, node,  x_eps, eps, threshold_noise)
    
    fig1.suptitle(r'b) $\mathcal{B}_{\varepsilon}$ relaxing path')

    #======================================================#
    #======================================================#
    fig2 = subfigs[2]
    gs2 = GridSpec(1, 1, figure=fig2,left=0.2, right=1.0, top = 0.850, 
                  bottom = 0.05)
    ax5 = fig2.add_subplot(gs2[0])
    
    fig2.suptitle(r'c) False links proportion', x=0.6)
    
    ax_false_proportion(ax5, gr_alg['info_x_eps'][id_cluster])
    
    if filename == None:
        plt.show()
    else:
     
        plt.savefig(filename+".pdf", format='pdf', bbox_inches='tight')
        
    return

def plot_B_eps_complete(gr_alg, node, threshold, id_cluster = 0, filename = None):    
    '''
    Plot B_eps relaxing path algorithm for trivial graph partition.

    Parameters
    ----------
    gr_alg : dict
        Greedy Network Recontruction dictionary.
    node : int
        Probed node to be highlighted.
    threshold : TYPE
        DESCRIPTION.
    id_cluster : int, optional
        Cluster id for identification. The default is 0 because is the trivial
        graph partition.
    filename : str, optional
        Filename to save figure. In case of None, the figure is only shown. 
        The default is None.

    Returns
    -------
    None.

    '''
    fig_ = plt.figure(figsize=(15, 5.0), dpi = 300)
    subfigs = fig_.subfigures(1, 3, width_ratios = [0.8, 1.5, 1.2], 
                              wspace = 0.75)
    
    fig = subfigs[0]
    gs = GridSpec(1, 1, figure=fig,left=0.01, right=1.0, top = 0.990, 
                  bottom = 0.01)
    
    ax1 = fig.add_subplot(gs[0])
    ax_plot_true_net(ax1, probed_node=node)
    fig.suptitle(r'a) Original Network') 
    
    #======================================================#
    #======================================================#
    fig1 = subfigs[1]
    
    gs1 = GridSpec(1, 3, figure=fig1, 
                   left=0.01, right=1.0, top = 0.80, bottom = 0.05)
    
    ax3 = fig1.add_subplot(gs1[0])
    eps_counter = noise_min_loc(gr_alg['info_x_eps'][id_cluster], node)    
    ax_probed_node_links(ax3, gr_alg['info_x_eps'][id_cluster], node, eps_counter, threshold)
    
    ax3 = fig1.add_subplot(gs1[1])
    eps_counter = noise_min_loc(gr_alg['info_x_eps'][id_cluster], node) + 1
    ax_probed_node_links(ax3, gr_alg['info_x_eps'][id_cluster], node, eps_counter, threshold)
       
    eps_counter = gr_alg['info_x_eps'][id_cluster][node]['eps_flag']['eps_counter']
    ax4 = fig1.add_subplot(gs1[2])
    ax_probed_node_links(ax4, gr_alg['info_x_eps'][id_cluster], node, eps_counter, threshold)
    
    fig1.suptitle(r'b) $\mathcal{B}_{\varepsilon}$ relaxing path')

    #======================================================#
    #======================================================#
    fig2 = subfigs[2]
    gs2 = GridSpec(1, 2, figure=fig2, 
                   left=0.01, right=1.0, top = 0.850, bottom = 0.15,
                   wspace = 0.25)
    ax5 = fig2.add_subplot(gs2[0, 0])
    ax6 = fig2.add_subplot(gs2[0, 1])
    
    fig2.suptitle(r'c) Reconstructed optoelectronic network')
    plot_weighted_fp_fn_links(gr_alg['A'], ax =  [ax5, ax6], filename = None)
     
    if filename == None:
        plt.show()
    else:
     
        plt.savefig(filename+".png", format='png', bbox_inches='tight')
        
    return     
    
def interval_intersec(I_1, I_2):
 
    try:
        max_ = np.max(I_1)
    except:
        max_ = 0
    min_ = np.min(I_2)
 
    d = min_ - max_

    return d    
    
def compare_basis(exp_dictionary):
    
    A_true, G_true = opt_elec.get_Adj_matrix()
    edges_G_true = list(G_true.edges())
    exp_vec = exp_dictionary['exp_params'].keys()
    lgth_endpoints = exp_dictionary['lgth_endpoints']
    
    lgth_vector = np.arange(lgth_endpoints['0'], lgth_endpoints['1'],
                                      lgth_endpoints['2'], dtype = int)
    
    FN_comparison = np.zeros((len(exp_vec), lgth_vector.shape[0]))
    d_matrix = np.zeros((len(exp_vec), lgth_vector.shape[0]))

    for exp_ in exp_vec:
        for id_key in range(len(exp_dictionary[exp_].keys())):
            key = list(exp_dictionary[exp_].keys())[id_key]
            A_est = exp_dictionary[exp_][key]['A']
            G_est = nx.from_numpy_array(A_est.T, create_using = nx.Graph)
            links = links_types(G_est, G_true)        
    
            intersec_weights, false_pos_weights = links['intersec_weights'],\
                links['false_pos_weights']
            d_matrix[exp_, id_key] = interval_intersec(false_pos_weights,
                                                       intersec_weights)
            
            FN = len(links['false_negatives'])/len(edges_G_true)
            FN_comparison[exp_, id_key] = FN
            
    return lgth_vector, FN_comparison, d_matrix


def plot_false_proportion(ax, lgth_vector, f_comp_vec, std, plot_std = True, 
                          plot_legend = False):
    '''
    Plot sharing x axis the false proportion for each length of time series.

    Parameters
    ----------
    ax : Matplotlib Axes object
        Draw the graph in the specified Matplotlib axes.
    lgth_vector : numpy array
        Array of length of time series to be tested.
    f_comp_vec : numpy array
        Array of false proportion (False positive or false negative) to be plotted.
    plot_legend : Boolean, optional
        To plot legend. The default is False.

    Returns
    -------
    None.

    '''
    
    #labels = ['BP', 'normcols', 'EBP']
    #markers = ['s', 'D', 'o']
    #colors = ['tab:purple', 'tab:orange', '#24AD7A']
    
    #For ring plot, I selected only norm cols and EBP
    labels = ['BP', 'EBP'] #'normcols'
    markers = ['s', 'o']
    colors = ['tab:purple', '#24AD7A']
    
    for id_exp in range(f_comp_vec.shape[0]):    
        ax.plot(lgth_vector, f_comp_vec[id_exp, :], '-', 
                label = labels[id_exp], marker = markers[id_exp], 
                color = colors[id_exp])
        if plot_std:
            ax.fill_between(lgth_vector, f_comp_vec[id_exp, :] - std[id_exp, :],
                            f_comp_vec[id_exp, :] + std[id_exp, :], 
                    color = colors[id_exp], alpha = 0.4)
        
    if plot_legend:
        #ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
        #          mode="expand", borderaxespad=0, ncol=3)
        ax.legend(loc=0, fontsize = 16)
        
        
def plot_interval_intersec(ax, lgth_vector, d_matrix, plot_legend = False): 
    
    labels = ['BP', 'BP + cols normalized', 'Ergodic BP']
    for id_exp in range(d_matrix.shape[0]):    
        ax.plot(lgth_vector, d_matrix[id_exp, :], 'o-', label = labels[id_exp])
    
    ax.hlines(0.0, lgth_vector[0], lgth_vector[-1]+1)
    ax.set_ylabel(r'd')
    ax.set_xlabel(r'$n$')
    if plot_legend:
        ax.legend(loc=0)


def plot_comparison_analysis(exp_dictionary):    
    
    lgth_vector, FN_comparison, d_matrix = compare_basis(exp_dictionary)
    
    fig = plt.figure(dpi = 300)
    gs = GridSpec(nrows=2, ncols=2, figure=fig)
    
    ax_0 = fig.add_subplot(gs[0, 0:])
    plot_false_proportion(ax_0, lgth_vector, FN_comparison)
    
    ax_1 = fig.add_subplot(gs[1, 0:])
    plot_interval_intersec(ax_1, lgth_vector, d_matrix)
    
def fig_3_script(filename = None):

    ##### Identification for output
    folder = "data"+"/"+"opto_electronic_data"+"/"+"symmetric_data"+"/"
    outfilename = os.path.join(folder, 'subset_data')
    outfilename = os.path.join(outfilename, "")
    hdf5 = h5dict.File(outfilename+"subset_3_4_4_5"+".hdf5", 'r')    
    
    id_sig = 10
    coupling_vec = np.arange(0.0156250, 1.093750 + 0.015625, 0.0156250)
    X_time_series = hdf5[coupling_vec[id_sig]]
    X_t = X_time_series[:None,:]
    
    hdf5.close()

    args= dict()
    args['plot_subset'] = False
    args['subset'] = [3.4, 4.5]
    
    args['clusters_list'] = [np.arange(5), np.arange(5, 16)]
    args['filename'] = filename
    
    plot_subset_clustering(X_t, args)
    
    
    
    
    
    
    
    
    

