"""
To measure the dependence of number of cluster versus number of nodes sharing
inter cluster connections.

Created on Thu Jul 15 20:12:29 2021

@author: edinh
"""

import networkx as nx
import numpy as np

import matplotlib.pyplot as plt

# Set plotting parameters
params_plot = {'axes.labelsize': 22,
              'axes.titlesize': 18,
              'axes.linewidth': 1.0,
              'axes.xmargin':0, 
              'axes.ymargin': 0,
              'legend.fontsize': 20,
              'xtick.labelsize': 18,
              'ytick.labelsize': 18,
              'figure.figsize': (8, 6),
              'figure.titlesize': 18,
              'font.serif': 'Computer Modern Serif',
              'mathtext.fontset': 'cm'
             }

plt.rcParams.update(params_plot)
plt.rc('text', usetex=True)

def R_net_arbitrary(q, N, cluster_dict):
    
    t_tot = N**(q + 1)
    
    t_GNR = 0
    tau_j = 0
    for id_cluster in cluster_dict.keys():
        if not type(id_cluster) == str:
            p_j = cluster_dict[id_cluster]['cluster_size']
            tau_j = tau_j + cluster_dict[id_cluster]['tau']
            t_GNR = t_GNR + p_j**(q + 1) 
    t_GNR = t_GNR  + tau_j**(q + 1)            
    return t_GNR/t_tot

def tau_calculation(G, partition_list):

    cluster_dict = dict() 
    cluster_dict['kappa'] = len(partition_list)
    
    id_cluster = 0
    for cluster in partition_list:
        cluster_dict[id_cluster] = dict()
        cluster_dict[id_cluster]['tau'] = 0
        cluster_dict[id_cluster]['cluster_size'] = cluster.shape[0]
        for node in cluster:
            neigh_node = np.array(list(G[node]))
            mask_intra_cluster = np.isin(cluster, neigh_node)
            intra_connections = cluster[mask_intra_cluster].shape[0]
            inter_connections = neigh_node.shape[0] - intra_connections
            if inter_connections > 0:
                cluster_dict[id_cluster][node] =  [intra_connections, inter_connections] #cluster[mask_intra_cluster]#np.array([intra_connections, inter_connections])
                cluster_dict[id_cluster]['tau'] = cluster_dict[id_cluster]['tau'] + 1
            #else:
            #    print('Nodes sharing only intra-cluster connections', node, [intra_connections, inter_connections])
        id_cluster = id_cluster + 1    
            
    return cluster_dict            
            
def g_lambda(x, q):
    
    y = (1 - 1/x**q)
    y = y**(1/(1 + q))
    return y



def plot_diagram(q):
    xmin, xhalf, xmax = 1, 10, 20
    kappa = np.arange(xmin, xmax, 0.01)
    y1 = g_lambda(kappa, q)
    fig, ax = plt.subplots(dpi = 400)
    ax.plot(kappa, y1, 'k', lw=2.2)
    ax.hlines(1.0, xmin, xmax, colors = 'k')
    
    ax.fill_between(kappa, y1, y2 = 1.0, color='#7C3CB4', alpha = 1.0)
    ax.text(1.5, 0.92, r'loses', fontsize = 27)

    ax.fill_between(kappa, y1, color='#05752B', alpha = 0.4)
    ax.text(14, 0.2, r'wins', fontsize = 27)
    ax.set_xlabel(r"Number of clusters")
    ax.set_ylabel(r" Fraction of bridge nodes")
    x_ticks = [xmin, xhalf, xmax]
    ax.set_xticks(x_ticks)
    ax.set_ylim(0.0, 1.001)
    #plt.savefig('num_cluster_vs_bridge_nodes.png', format='png', bbox_inches='tight')

def plot_q_values(q):
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    xmin, xhalf, xmax = 1, 2.5, 5
    kappa = np.arange(xmin, xmax, 0.01)
    fig, ax = plt.subplots(figsize = (8, 4), dpi = 200)
    
    counter = 0
    for q_i in q:
        y1 = g_lambda(kappa, q_i)
        ax.plot(kappa, y1, 'k', label = r'$q = {}$'.format(q_i), lw = 1.5, linestyle = linestyles[counter])
        counter = counter + 1
    ax.legend(loc=0)
    ax.hlines(1.0, xmin, xmax, colors = 'k')
    
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$g_{q}(t)$")
    ax.set_ylim(0.0, 1.001)
    fig.tight_layout()
    #plt.savefig('g_q_function.pdf', format='pdf', bbox_inches='tight')

   
def kappa_vs_tau():    
    net_name = 'cat_cortex_network_52'
    G = nx.read_edgelist("network_structure/{}.txt".format(net_name), nodetype = int)
                
    N = len(nx.nodes(G))
    print(N)
    partition_list = [np.arange(0, 16, 1, dtype = int) , np.arange(16, 23, 1, dtype = int),
                      np.arange(23, 39, 1, dtype = int), np.arange(39, N, 1, dtype = int)]
    cluster_dict = tau_calculation(G, partition_list)          
    return cluster_dict


plot_diagram_ = False
plot_g_function = True
            
if plot_diagram_:           
    plot_diagram(1)    

if plot_g_function:
    plot_q_values(np.arange(0.9, 1.3, 0.1))                      
    
