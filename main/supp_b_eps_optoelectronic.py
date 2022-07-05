"""
Reconstruction network from multivariate time series in the subset of the phase space
for the optoelectronic experimental data.

Created on Mon Oct 25 14:39:42 2021

@author: Edmilson Roque dos Santos
"""

import cvxpy as cp
import numpy as np
import os
import h5dict

from EBP import tools
from EBP.base_polynomial import greedy_algorithms as gnr_alg

import lab_opto_electronic as lab_opto

exp_name = "gnr_opto_crossed"

############# Construct the parameters dictionary ##############
parameters = dict()

parameters['exp_name'] = exp_name
parameters['Nseeds'] = 1

parameters['network_name'] = "opto_electronic"
parameters['max_deg_monomials'] = 2
use_crossed_terms = True
parameters['expansion_crossed_terms'] = use_crossed_terms

parameters['use_kernel'] = True
parameters['noisy_measurement'] = True
opt = True#False# 
parameters['use_canonical'] = not opt
parameters['normalize_cols'] = False
parameters['use_orthonormal'] = opt

coupling_vec = np.arange(0.0156250, 1.093750 + 0.015625, 0.0156250)


lgth_time_series = None

'''
From numerical observation we noted the density function clustering only 
occurs from length of time series 125. So, we implemented this difference in 
the next lines.
'''
if parameters['use_orthonormal']:
    if lgth_time_series == None:
        orthnorm_folder = 'gen_orthf_cluster'
        use_single = False
    else:    
        if (lgth_time_series > 125):
            use_single = False
            
            if use_crossed_terms:
                orthnorm_folder = 'gen_orthf_cluster'
                orthnorm_folder = os.path.join(orthnorm_folder, exp_name)

            else:
                orthnorm_folder = 'gen_orthf_cluster'
            
        else:
            orthnorm_folder = 'gen_orthf_single'
            use_single = True


##### Identification for output
folder = "data"+"/"+"opto_electronic_data"+"/"+"symmetric_data"+"/"
outfilename = os.path.join(folder, 'subset_data')
if parameters['use_orthonormal']:
    outfile_functions = os.path.join(outfilename, orthnorm_folder)
    if os.path.isdir(outfile_functions) == False:
        os.makedirs(outfile_functions)
    outfile_functions = os.path.join(outfile_functions, "")
    
outfilename = os.path.join(outfilename, "")

hdf5 = h5dict.File(outfilename+"subset_3_4_4_5"+".hdf5", 'r')    

id_sig = 10
parameters['coupling'] = coupling_vec[id_sig]
X_time_series = hdf5[coupling_vec[id_sig]]

hdf5.close()

X_t = X_time_series[:lgth_time_series,:]

parameters['lower_bound'] = np.min(X_t)
parameters['upper_bound'] = np.max(X_t)

parameters['number_of_vertices'] = X_t.shape[1]

parameters['X_time_series_data'] = X_t

params = parameters.copy()

if params['use_orthonormal']:
    if lgth_time_series == None:
        params['orthnorm_func_filename'] = outfile_functions+\
            "orthnorm_sig_{:.6f}_deg_{}".format(coupling_vec[id_sig], 
                                                parameters['max_deg_monomials'] )
    else:   
        output_orthnormfunc_filename = outfile_functions+\
            "orthnorm_sig_{:.6f}_deg_{}_lgth_{}".format(coupling_vec[id_sig], 
                                                        parameters['max_deg_monomials'],
                                                        lgth_time_series)
        
        if not os.path.isfile(output_orthnormfunc_filename):
            print("Feature missing on the code.")
            '''
            prepro.generate_orthonorm_funct(exp_name = orthnorm_folder,
                                            max_deg_monomials = params['max_deg_monomials'],
                                            lgth_time_series = lgth_time_series,
                                            use_single = use_single,
                                            use_crossed_terms = use_crossed_terms)
            '''
            params['orthnorm_func_filename'] = output_orthnormfunc_filename
        if os.path.isfile(output_orthnormfunc_filename):
            params['orthnorm_func_filename'] = output_orthnormfunc_filename
            
        
    params['orthnormfunc'] = tools.SympyDict.load(params['orthnorm_func_filename'])
    
    params['build_from_reduced_basis'] = False
    params['save_orthnormfunc'] = False

params['cluster_list'] = [np.arange(0, 17, 1, dtype = int),]

params_ = params.copy()

threshold_connect = 1e-8
tolerance = 1e-8
fixed_search_set = True
relaxing_path = np.linspace(0.15, 0.40, 25) 
select_criterion = 'crit_3'
solver_optimization = cp.ECOS

gr_alg = gnr_alg.GR_algorithm(X_t, params['cluster_list'], params_, 
                 tolerance, threshold_connect, fixed_search_set, 
                 relaxing_path, select_criterion, solver_optimization)

outfolder = 'Figures_supp'
filename = None#outfolder+'/'+'Fig_1_v3_0_01'

node = 6
lab_opto.plot_B_eps(gr_alg, node, threshold_connect, filename = filename)






