"""
Script for multiprocessing experiments

Created on Mon Dec 26 11:26:51 2022

@author: Edmilson Roque dos Santos
"""


import h5dict
from multiprocessing import Pool 
import os
import time

from EBP import net_dyn, tools
from EBP.base_polynomial import pre_settings as pre_set 

import lab_opto_electronic as lab_opto
import net_reconstr 
import compare_basis_choice as cbc

def realization(ps_pool):
    
    params = ps_pool.copy()
    X_time_series = generate_map_iteration_Logistic_map(A, coupling, number_of_iterations, ps_pool['random_seed'])

    # Extract the time series for the state and map
    X_t = X_time_series[:-1, :]
    X_t_dot = X_time_series[1:, :]
    params['length_of_time_series'] = X_t.shape[0]
    
    error_estimative = np.zeros(len(ps_pool['node_trial_vector']))
    num_nonzeros_vec = np.zeros(len(ps_pool['node_trial_vector']))
    
    X_t_test = X_t[:, ps_pool['id_trial']]
            
    PHI, dictionary_matrix = Dictionary_matrix_dif_basis(X_t_test, ps_pool)
    L = dictionary_matrix['L']
 
    b = X_t_dot[:, ps_pool['node_test']]
   
    for test_index in range(len(ps_pool['node_trial_vector'])):
        number_of_terms = (ps_pool['node_trial_vector'][test_index] + 1)*order + 1
       
        coefficient_vector, num_nonzeros_vec[test_index] = l_1_optimization(b, PHI[:, : number_of_terms], ps_pool['noisy_measurement'], params, dictionary_matrix)
    
        A_est = estimate_adjacency_matrix_node(node_test, coefficient_vector, dictionary_matrix, params, ps_pool['threshold_connect'])  
        
        error_estimative[test_index] = np.sum(np.absolute(A[node_test, :] - A_est))/(N - 1)
        
    trials_results = dict()    
    trials_results['error_seed_{}'.format(ps_pool['random_seed'])] = error_estimative
    trials_results['num_nonzeros_vec_seed_{}'.format(ps_pool['random_seed'])] = num_nonzeros_vec
    
    save_hdf5(ps_pool['hdf5_filename'], ps_pool, to_store=trials_results, try_append=True)
    print('\t\t Executed seed {} with {}s'.format(ps_pool['random_seed'], time.time()))

##### Randomness
Nseeds = 50
parameters['Nseeds'] = Nseeds
MonteCarlo_seeds = np.arange(1, Nseeds + 1)     # Seed for random number generator
nproc = 10
ti = time.time()
ps = []
for seed in MonteCarlo_seeds:
    ps.append(parameters.copy())
    ps[-1]['random_seed'] = seed
    ps[-1]['hdf5_filename'] = scenario_output+'_'+exp_name+".hdf5"
    
pool = Pool(processes=nproc)
pool.map(realization, ps)
pool.close()

tf = time.time()
execution_time = tf - ti
print('>> total execution time was {:.2f} min, using {:} realizations of the adaptative analysis'.format(execution_time/60, Nseeds))



