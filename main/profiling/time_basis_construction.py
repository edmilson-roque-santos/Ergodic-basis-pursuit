'''
To quantify the time of execution of creating the adapted basis when compared
to the non-adapted one.
'''

import networkx as nx 
import numpy as np
import os
import sympy as spy 
import time


import base_polynomial.poly_library as polb
import base_polynomial.triage as trg
import base_polynomial.pre_settings as pre_set 
import net_dyn

import tools

############# Construct the parameters dictionary ##############
parameters = dict()

exp_name = "time_execution"
parameters['exp_name'] = exp_name
parameters['Nseeds'] = 1

num_of_vertices = np.arange(3, 4, 1, dtype = int)
max_degree_poly = np.arange(5, 6, 1, dtype = int)
num_N, num_deg = num_of_vertices.shape[0], max_degree_poly.shape[0]
time_exec = np.zeros((num_N, num_deg, 1))

for id_N in range(num_N):
    for id_deg in range(num_deg):
        N = num_of_vertices[id_N]   
        parameters['number_of_vertices'] = N
        
        G = nx.cycle_graph(parameters['number_of_vertices'])
        parameters['network_name'] = "cycle_{}".format(N)    
        A = nx.to_numpy_array(G, nodelist = list(range(N)))
        A = np.asarray(A)
        parameters['adj_matrix'] = A
        
        parameters['length_of_time_series'] = 100
        #==========================================================#
        net_dynamics_dict = dict()
        net_dynamics_dict['adj_matrix'] = parameters['adj_matrix']
        
        r = 3.990
        net_dynamics_dict['f'] = lambda x: r*x*(1 - x)
        net_dynamics_dict['h'] = lambda x: (x**3)*(A @ x**1)
        net_dynamics_dict['coupling'] = 1e-3
        X_t = net_dyn.gen_net_dynamics(parameters['length_of_time_series'],
                                       net_dynamics_dict)    
        #==========================================================#
        parameters['X_time_series_data'] = X_t
        
        parameters['max_deg_monomials'] = max_degree_poly[id_deg]
        parameters['max_deg_generating'] = 20
        parameters['expansion_crossed_terms'] = True
        parameters['use_canonical'] = False
        parameters['normalize_cols'] = False
        parameters['use_orthonormal'] = True
        parameters['use_kernel'] = True
        parameters['normalize_coupling_function'] = False
        parameters['noisy_measurement'] = False
        
        start_time = time.time()
        orthnorm_func = pre_set.create_orthnormfunc_kde(parameters, save_orthnormfunc = False)
        time_exec[id_N, id_deg, 0] = time.time() - start_time
        
        print('N: {} and D: {}'.format(num_of_vertices[id_N], max_degree_poly[id_deg]))

filename = "time_exec_N_{}_D_{}.txt".format(num_of_vertices[-1], max_degree_poly[-1])
np.savetxt(filename, time_exec[0, :, 0])









