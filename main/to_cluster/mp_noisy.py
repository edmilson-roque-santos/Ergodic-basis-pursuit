"""
Script for multiprocessing experiments: noisy measurements

Created on Sun Feb  2 11:13:18 2025

@author: Edmilson Roque dos Santos
"""

import time

import compare_basis_choice as cbc

def realization(ps_pool):
    cbc.ring_graph_noisy_lgth_script(ps_pool)
    print('\t\t Executed seed {} with {}s'.format(ps_pool, time.time()))

def worker_function(inputs):
    """Wrapper to apply my_function in parallel."""
    #with Pool(processes=num_processes) as pool:
    
    #pool = Pool(processes=num_processes)
    #results = pool.map(realization, inputs)
    #pool.close()  
    results = []
    for id_ps in inputs:
        results.append(realization(id_ps))
        
    return results

##### Randomness
Nseeds = 10
ps = range(1, Nseeds + 1) # Seed for random number generator

ti = time.time()    
results = worker_function(ps)
tf = time.time()
execution_time = tf - ti

print('>> total execution time was {:.2f} min, using {:} realizations of the adaptative analysis'.format(execution_time/60, Nseeds))

