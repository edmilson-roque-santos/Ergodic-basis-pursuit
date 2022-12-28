"""
Script for multiprocessing experiments

Created on Mon Dec 26 11:26:51 2022

@author: Edmilson Roque dos Santos
"""


from multiprocessing import Pool 
import numpy as np
import time

import compare_basis_choice as cbc

def realization(ps_pool):
    cbc.ring_graph_lgth_script(ps_pool)
    print('\t\t Executed seed {} with {}s'.format(ps_pool, time.time()))

##### Randomness
Nseeds = 10
MonteCarlo_seeds = np.arange(1, Nseeds + 1)     # Seed for random number generator
nproc = 10
ti = time.time()
ps = range(1, Nseeds + 1)
    
pool = Pool(processes=nproc)
pool.map(realization, ps)
pool.close()

tf = time.time()
execution_time = tf - ti
print('>> total execution time was {:.2f} min, using {:} realizations of the adaptative analysis'.format(execution_time/60, Nseeds))



