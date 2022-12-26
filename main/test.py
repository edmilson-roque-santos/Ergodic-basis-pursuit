"""
Created on Tue Dec 20 12:46:02 2022

@author: Edmilson Roque dos Santos
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.special

from EBP import net_dyn, tools
from EBP.base_polynomial import pre_settings as pre_set 
from EBP.base_polynomial import triage as trg
import lab_opto_electronic as lab_opto

def num_basis_compare():
    N_vec = np.arange(3, 5, 1, dtype = int)
    r_vec = np.arange(1, 4)
    
    
    r = r_vec[2]
    
    L_new = np.empty(N_vec.size)
    L_old = np.empty(N_vec.size)
    for id_ in range(N_vec.size):
        L_new[id_] = scipy.special.comb(N_vec[id_], 2, exact = True)*scipy.special.comb(r, 2, exact = True) + N_vec[id_]*r + 1
        L_old[id_] = np.sum(np.fromiter((scipy.special.comb(l + N_vec[id_] - 1, l, exact = True) for l in range(0, r + 1)), int))
        
    plt.plot(N_vec, L_new)
    plt.plot(N_vec, L_old)

def star_test():
    G = nx.star_graph(15)
    params = dict()
    params['number_of_vertices'] = len(nx.nodes(G))
    N = params['number_of_vertices']
    A = nx.to_numpy_array(G, nodelist = list(range(params['number_of_vertices'])))
    A = np.asarray(A)
    
    params['adj_matrix'] = A
    params['exp_name'] = 'test'
    params['network_name'] = 'test'
    params['length_of_time_series'] = 50
    params['coupling'] = 1e-2
    
    params['max_deg_monomials'] = 3
    
    net_dynamics_dict = dict()
    net_dynamics_dict['adj_matrix'] = params['adj_matrix']
    
    r = 3.990
    net_dynamics_dict['f'] = lambda x: r*x*(1 - x)
    net_dynamics_dict['h'] = lambda x: (x**1)*(A.T @ x**1)
    net_dynamics_dict['max_degree'] = np.max(np.sum(A, axis=0))
    net_dynamics_dict['coupling'] = params['coupling']#*net_dynamics_dict['max_degree']
    net_dynamics_dict['random_seed'] = 1
    X_time_series = net_dyn.gen_net_dynamics(params['length_of_time_series'] , net_dynamics_dict)
    
    params['expansion_crossed_terms'] = True
    params['build_from_reduced_basis'] = True
    params = trg.triage_params(params)
    params['nodelist'] = np.arange(0, N, 1, dtype = int)
    params['filename'] = None
    lab_opto.plot_traj_density(X_time_series, params)








