"""
Script for comparison of nonadapted and adapted basis to a given network dynamics.

Created on Thu Jan  6 12:47:14 2022

@author: Edmilson Roque dos Santos
"""

import cvxpy as cp
import h5dict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec    
import networkx as nx 
import numpy as np
import os
from scipy.linalg import norm
import seaborn as sns
import sympy as spy


from EBP import net_dyn, tools
from EBP.base_polynomial import pre_settings as pre_set 

import lab_opto_electronic as lab_opto

import net_reconstr 

colors = ['darkgrey', 'orange', 'darkviolet', 'darkslategrey', 'silver']
folder_name = 'results'
ortho_folder_name = 'ortho_func_folder'
#=============================================================================#
#Simulation fix a network and increase length of time series 
#=============================================================================#

def out_dir_ortho(net_name, exp_name, params):
    '''
    Create the folder name for save orthonormal functions 
    locally inside results folder.

    Parameters
    ----------
    net_name : str
        Network structure filename.
    exp_name : str
        Filename.
    params : dict
        

    Returns
    -------
    out_results_direc : str
        Out results directory.

    '''
        
    out_results_direc = os.path.join(folder_name, ortho_folder_name)
    out_results_direc = os.path.join(out_results_direc, net_name)
    out_results_direc = os.path.join(out_results_direc, exp_name)
    out_results_direc = os.path.join(out_results_direc, '')
    if os.path.isdir(out_results_direc ) == False:
        
        try:
            os.makedirs(out_results_direc)
        except:
            'Folder has already been created'
    #For coupling analysis it is necessary to save each orthonormal function 
    #with respect to this coupling.
    filename = 'onf_deg_{}_lgth_ts_{}_coupling_{}_crossed_{}_seed_{}'.format(params['max_deg_monomials'],
                                                              params['length_of_time_series'], 
                                                              params['coupling'],
                                                              params['expansion_crossed_terms'],
                                                              params['random_seed'])
    out_results_direc = os.path.join(out_results_direc, filename)
    
    return out_results_direc

def compare_script(script_dict):
    '''
    Script for basis choice comparison. 

    Parameters
    ----------
    script_dict : dict
    Dictionary with specifier of the comparison script
    Keys:
        opt_list : list of boolean
            Each entry determines which basis is selected. 
            Order: #canonical, normalize_cols, orthonormal
        lgth_time_series : float
            Length of time series.
        exp_name : str
            Filename.
        net_name: str
            Network structure filename.
        id_trial: numpy array 
            Set of nodes to be reconstructed
            
    Returns
    -------
    dictionary result from net reconstruction algorithm.

    '''
    ############# Construct the parameters dictionary ##############
    parameters = dict()
    
    parameters['exp_name'] = script_dict['exp_name']
    parameters['Nseeds'] = 1
    parameters['random_seed'] = script_dict.get('random_seed', 1)
    parameters['network_name'] = script_dict['net_name']
    parameters['max_deg_monomials'] = 3
    parameters['expansion_crossed_terms'] = False#True#
    
    parameters['use_kernel'] = True
    parameters['noisy_measurement'] = False
    parameters['use_canonical'] = script_dict['opt_list'][0]
    parameters['normalize_cols'] = script_dict['opt_list'][1]
    parameters['use_orthonormal'] = script_dict['opt_list'][2]
    parameters['length_of_time_series'] = script_dict['lgth_time_series']
    
    try:
        G = script_dict['G']
    except:
        G = nx.read_edgelist("network_structure/{}.txt".format(parameters['network_name']),
                            nodetype = int, create_using = nx.Graph)
        
    parameters['number_of_vertices'] = len(nx.nodes(G))
    A = nx.to_numpy_array(G, nodelist = list(range(parameters['number_of_vertices'])))
    A = np.asarray(A)
    parameters['adj_matrix'] = A
    parameters['coupling'] = 1e-3
    #==========================================================#
    net_dynamics_dict = dict()
    net_dynamics_dict['adj_matrix'] = parameters['adj_matrix']
    
    r = 3.990
    net_dynamics_dict['f'] = lambda x: r*x*(1 - x)
    net_dynamics_dict['h'] = lambda x: (A.T @ x**2)#(x**1)*(A.T @ x**1)
    net_dynamics_dict['max_degree'] = np.max(np.sum(A, axis=0))
    net_dynamics_dict['coupling'] = parameters['coupling']#*net_dynamics_dict['max_degree']
    net_dynamics_dict['random_seed'] = parameters['random_seed']
    X_time_series = net_dyn.gen_net_dynamics(script_dict['lgth_time_series'], net_dynamics_dict)    
    #==========================================================#    
    
    net_dict = dict()

    mask_bounds = (X_time_series < 0) | (X_time_series > 1) | (np.any(np.isnan(X_time_series)))
    if np.any(mask_bounds):
        raise ValueError("Network dynamics does not live in a compact set ")
        
    if not np.any(mask_bounds):

        X_t = X_time_series[:script_dict['lgth_time_series'],:]
        
        parameters['lower_bound'] = np.min(X_t)
        parameters['upper_bound'] = np.max(X_t)
        
        parameters['number_of_vertices'] = X_t.shape[1]
        
        parameters['X_time_series_data'] = X_t
        
        params = parameters.copy()
        
        if params['use_orthonormal']:
            out_dir_ortho_folder = out_dir_ortho(script_dict['net_name'], 
                                                 script_dict['exp_name'], params)
            
            output_orthnormfunc_filename = out_dir_ortho_folder
        
            if not os.path.isfile(output_orthnormfunc_filename):
                params['orthnorm_func_filename'] = output_orthnormfunc_filename
                params['orthnormfunc'] = pre_set.create_orthnormfunc_kde(params)    
    
            if os.path.isfile(output_orthnormfunc_filename):
                params['orthnorm_func_filename'] = output_orthnormfunc_filename
                      
            params['build_from_reduced_basis'] = True
        
        params['cluster_list'] = [np.arange(0, params['number_of_vertices'], 1, dtype = int)]
        params['threshold_connect'] = 1e-8
        
        if script_dict['id_trial'] != None:
            params['id_trial'] = script_dict['id_trial']
        
        solver_optimization = cp.ECOS
        
        net_dict = net_reconstr.reconstr(X_t, params, solver_optimization)
    
    return net_dict


    
def save_dict(dictionary, out_dict):
    '''
    Save dictionary in the output dictionary and avoids some keys that are not
    allowed in hdf5.

    Parameters
    ----------
    dictionary : dict
    out_dict : dict

    Returns
    -------
    None.

    '''
    keys = dictionary.keys()
    for key in keys:
        try:
            out_dict[key] = dictionary[key]
        except:
            print("Error: not possible to save", key)

def out_dir(net_name, exp_name): 
    '''
    Create the folder name for save comparison  
    locally inside results folder.

    Parameters
    ----------
    net_name : str
        Network structure filename.
    exp_name : str
        Filename.
    
    Returns
    -------
    out_results_direc : str
        Out results directory.

    '''       
    out_results_direc = os.path.join(folder_name, net_name)
    out_results_direc = os.path.join(out_results_direc, exp_name)
    out_results_direc = os.path.join(out_results_direc, '')
    
    if os.path.isdir(out_results_direc) == False:
        try:
            os.makedirs(out_results_direc)
        except:
            'Folder has already been created'
    return out_results_direc

def compare_setup(exp_name, net_name, lgth_endpoints, random_seed = 1, 
                  save_full_info = False):
    '''
    
    Parameters
    ----------
    exp_name : str
        filename.
    net_name : str
        Network structure filename.
    lgth_endpoints : list
        Start, end and space for length time vector.
    random_seed : int
        Seed for the random pseudo-generator.
    save_full_info : dict, optional
        To save the library matrix. The default is False.

    Returns
    -------
    exp_dictionary : TYPE
        DESCRIPTION.

    '''
    exp_params = dict()
    #canonical
    #exp_params[0] = [True, False, False]
    #normalize_cols
    exp_params[0] = [True, True, False]
    #orthonormal
    exp_params[1] = [False, False, True]
    
    length_time_series_vector = np.arange(lgth_endpoints[0], lgth_endpoints[1],
                                          lgth_endpoints[2], dtype = int)
    
    #Filename for output results
    out_results_direc = out_dir(net_name, exp_name)
    filename = "lgth_endpoints_{}_{}_{}_seed_{}".format(lgth_endpoints[0], lgth_endpoints[1],
                                                lgth_endpoints[2], random_seed) 
    
    if os.path.isfile(out_results_direc+filename+".hdf5"):
        out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'r')
        exp_dictionary = out_results_hdf5.to_dict()  
        out_results_hdf5.close()      
        return exp_dictionary
    
    else:
        out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'a')    
        out_results_hdf5['lgth_endpoints'] = lgth_endpoints
        out_results_hdf5['exp_params'] = dict() 
        out_results_hdf5['exp_params'] = exp_params
        
        for key in exp_params.keys():    
            out_results_hdf5[key] = dict()
            for lgth_time_series in length_time_series_vector:
                print('exp:', key, 'n = ', lgth_time_series)
                
                script_dict = dict()
                script_dict['opt_list'] = exp_params[key]
                script_dict['lgth_time_series'] = lgth_time_series
                script_dict['exp_name'] = exp_name
                script_dict['net_name'] = net_name
                script_dict['id_trial'] = None
                script_dict['random_seed'] = random_seed
                
                net_dict = compare_script(script_dict)
                out_results_hdf5[key][lgth_time_series] = dict()
                out_results_hdf5[key][lgth_time_series]['A'] = net_dict['A']
                if save_full_info:
                    out_results_hdf5[key][lgth_time_series]['PHI.T PHI'] = net_dict['PHI.T PHI']
                    out_results_hdf5[key][lgth_time_series]['params'] = dict()
                    save_dict(net_dict['params'], out_results_hdf5[key][lgth_time_series]['params'])            
                
                
        exp_dictionary = out_results_hdf5.to_dict()        
        out_results_hdf5.close()
        return exp_dictionary

#=============================================================================#
#Simulation increasing the network and finding length of time series such that
#the reconstruction is successed. 
#=============================================================================#
def quick_comparison(net_dict, net_name):
    '''
    False positives and false negatives proportions between two graphs.

    Parameters
    ----------
    net_dict : dict
        Output results dictionary.
    net_name : str
        Network structure filename.

    Returns
    -------
    FP : float
        False positive proportion.
    FN : float
        False negative proportion.

    '''
    G_true, edges_G_true, N = get_G_true(net_name) 

    A_est = net_dict['A']
    G_est = nx.from_numpy_array(A_est, create_using = nx.Graph)
    links = lab_opto.links_types(G_est, G_true)        

    total_connections = 0.5*N*(N-1)
    total_fp_connections = (total_connections-len(edges_G_true))
    
    if total_fp_connections == 0:
        FP = 0
    else:
        FP = len(links['false_positives'])/total_fp_connections
    FN = len(links['false_negatives'])/len(edges_G_true)
    
    return FP, FN        
    
def determine_critical_n(exp_param, size, exp_name, net_info, id_trial = None, 
                         random_seed = 1):
    '''
    Determine the minimum length of time series for a successfull reconstruction.

    Parameters
    ----------
    exp_param : list
        Set the optlist for compare_script.
    size : float
        Network size.
    exp_name : str
        Filename.
    net_class : str
        Common network structure filename.
    id_trial : numpy array
        Set of nodes to be reconstructed.
    random_seed : int
        Seed for the random pseudo-generator.

    Returns
    -------
    n_critical : float
        minimum length of time series.

    '''
    net_name = net_info['net_class']+"_{}".format(size)
    
    if not os.path.isfile('network_structure/'+net_name):
        try:
            true_graph = net_info['gen'](size, 'network_structure/'+net_name)
        except:
            print("There is already a net!")
            
    size_step = int(np.round(size/10))
    lgth_time_series_vector = np.arange(5, 3*size**2, size_step, dtype = int)
    id_, max_iterations = 0, 100
    
    find_critical = True
    while (find_critical) and (id_ < max_iterations):
        lgth_time_series = lgth_time_series_vector[id_]
        print('lgth:', lgth_time_series)
        
        script_dict = dict()
        script_dict['opt_list'] = exp_param
        script_dict['lgth_time_series'] = lgth_time_series
        script_dict['exp_name'] = exp_name
        script_dict['net_name'] = net_name
        script_dict['id_trial'] = id_trial
        script_dict['random_seed'] = random_seed
        script_dict['G'] = true_graph
        
        net_dict = compare_script(script_dict)
        FP, FN = quick_comparison(net_dict, net_name)
        print('FP, FN:', FP, FN)
        if (FP == 0) and (FN == 0):
            find_critical = False
            print('Net Recovered!')
        id_ = id_ + 1
    
    n_critical = lgth_time_series
    return n_critical

def compare_setup_critical_n(exp_name, net_info, size_endpoints, id_trial,
                             random_seed = 1, save_full_info = False):
    '''
    Comparison script to growing the net size and evaluate the critical length of 
    time series for a successful reconstruction.
    
    Parameters
    ----------
    exp_name : str
        Filename.
    net_name : str
        Network structure filename.
    size_endpoints : list
        Start, end and space for size vector.
    random_seed : int
        Seed for the random pseudo-generator.
    save_full_info : dict, optional
        To save the library matrix. The default is False.

    Returns
    -------
    exp_dictionary : dict
        Output results dictionary.

    '''
    exp_params = dict()
    #canonical
    #exp_params[0] = [True, False, False]
    #normalize_cols
    exp_params[0] = [True, True, False]
    #orthonormal
    exp_params[1] = [False, False, True]
    
    size_vector = np.arange(size_endpoints[0], size_endpoints[1],
                                          size_endpoints[2], dtype = int)
    
    #Filename for output results
    out_results_direc = out_dir(net_info['net_class'], exp_name)
        
    filename = "size_endpoints_{}_{}_{}_seed_{}".format(size_endpoints[0], 
                                                        size_endpoints[1],
                                                        size_endpoints[2], 
                                                        random_seed) 
    
    if os.path.isfile(out_results_direc+filename+".hdf5"):
        out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'r')
        exp_dictionary = out_results_hdf5.to_dict()  
        out_results_hdf5.close()      
        return exp_dictionary
    
    else:
        out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'a')    
        out_results_hdf5['size_endpoints'] = size_endpoints
        out_results_hdf5['exp_params'] = dict() 
        out_results_hdf5['exp_params'] = exp_params
        
        for key in exp_params.keys():    
            out_results_hdf5[key] = dict()
            for size in size_vector:
                print('exp:', key, 'N = ', size)
                
                n_critical = determine_critical_n(exp_params[key], size, exp_name, 
                                                  net_info, id_trial, random_seed)
                
                out_results_hdf5[key][size] = dict()
                out_results_hdf5[key][size]['n_critical'] = n_critical
                
        exp_dictionary = out_results_hdf5.to_dict()        
        out_results_hdf5.close()
        return exp_dictionary
    
#=============================================================================#
#For a fixed noise magnitude value, simulate 
#increasing the length of time series such that
#the reconstruction is successed. 
#=============================================================================#
def compare_script_noisy(script_dict):
    '''
    Script for the reconstruction of coupled Logistic maps under noisy measurements. 

    Parameters
    ----------
    script_dict : dict
    Dictionary with specifier of the comparison script
    Keys:
        opt_list : list of boolean
            Each entry determines which basis is selected. 
            Order: #canonical, normalize_cols, orthonormal
        lgth_time_series : float
            Length of time series.
        exp_name : str
            Filename.
        net_name: str
            Network structure filename.
        id_trial: numpy array 
            Set of nodes to be reconstructed
            
    Returns
    -------
    dictionary result from net reconstruction algorithm.

    '''
    ############# Construct the parameters dictionary ##############
    parameters = dict()
    
    parameters['exp_name'] = script_dict['exp_name']
    parameters['Nseeds'] = 1
    parameters['random_seed'] = script_dict.get('random_seed', 1)
    parameters['network_name'] = script_dict['net_name']
    parameters['max_deg_monomials'] = 3
    parameters['expansion_crossed_terms'] = True#
    
    parameters['use_kernel'] = True
    parameters['noisy_measurement'] = True
    parameters['noise_magnitude'] = script_dict['noise_magnitude']
    parameters['use_canonical'] = script_dict['opt_list'][0]
    parameters['normalize_cols'] = script_dict['opt_list'][1]
    parameters['use_orthonormal'] = script_dict['opt_list'][2]
    parameters['length_of_time_series'] = script_dict['lgth_time_series']
    
    try:
        G = script_dict['G']
    except:
        G = nx.read_edgelist("network_structure/{}.txt".format(parameters['network_name']),
                            nodetype = int, create_using = nx.Graph)
        
    parameters['number_of_vertices'] = len(nx.nodes(G))
    A = nx.to_numpy_array(G, nodelist = list(range(parameters['number_of_vertices'])))
    A = np.asarray(A)
    parameters['adj_matrix'] = A
    parameters['coupling'] = 1e-3
    #==========================================================#
    net_dynamics_dict = dict()
    net_dynamics_dict['adj_matrix'] = parameters['adj_matrix']
    
    r = 3.990
    net_dynamics_dict['f'] = lambda x: r*x*(1 - x)
    net_dynamics_dict['h'] = lambda x: (A.T @ x**2)#(x**1)*(A.T @ x**1)
    net_dynamics_dict['h_dict'] = lambda x: spy.Matrix(A.T) @ x.applyfunc(lambda e: e**2)
    net_dynamics_dict['max_degree'] = np.max(np.sum(A, axis=0))
    net_dynamics_dict['coupling'] = parameters['coupling']#*net_dynamics_dict['max_degree']
    net_dynamics_dict['random_seed'] = parameters['random_seed']
    X_time_series = net_dyn.gen_net_dynamics(script_dict['lgth_time_series'], net_dynamics_dict) 
    
    
    mask_bounds = (X_time_series < 0) | (X_time_series > 1) | (np.any(np.isnan(X_time_series)))
    if np.any(mask_bounds):
        raise ValueError("Network dynamics does not live in a compact set ")
    
    parameters['perturbation_noise'] = 1e-4
    Z = net_dyn.noise_generation(parameters, 
                                 generate_overall = script_dict['generate_overall'])
    X_time_series = X_time_series + Z
    #==========================================================#    
    
    net_dict = dict()
    
    if not np.any(mask_bounds):

        X_t = X_time_series[:script_dict['lgth_time_series'],:]
        
        parameters['lower_bound'] = np.min(X_t)
        parameters['upper_bound'] = np.max(X_t)
        
        parameters['number_of_vertices'] = X_t.shape[1]
        
        parameters['X_time_series_data'] = X_t
        
        params = parameters.copy()
        
        if params['use_orthonormal']:
            out_dir_ortho_folder = out_dir_ortho(script_dict['net_name'], 
                                                 script_dict['exp_name'], params)
            
            output_orthnormfunc_filename = out_dir_ortho_folder
        
            if not os.path.isfile(output_orthnormfunc_filename):
                params['orthnorm_func_filename'] = output_orthnormfunc_filename
                params['orthnormfunc'] = pre_set.create_orthnormfunc_kde(params)    
    
            if os.path.isfile(output_orthnormfunc_filename):
                params['orthnorm_func_filename'] = output_orthnormfunc_filename
                      
            params['build_from_reduced_basis'] = True
        
        params['cluster_list'] = [np.arange(0, params['number_of_vertices'], 1, dtype = int)]
        params['threshold_connect'] = 1e-8
        
        if script_dict['id_trial'] != None:
            params['id_trial'] = script_dict['id_trial']
        
        solver_optimization = cp.ECOS
        
        net_dict = net_reconstr.reconstr(X_t, params, solver_optimization)
        
        # Compute the true coefficient matrix of the network dynamics
        net_dict['c_matrix_true'] = net_dyn.get_true_coeff_net_dyn(net_dynamics_dict, net_dict['info_x_eps']['params'])
        
        # Small code to compute approximately the correct value of the
        # magnitude of the vector perturbation. 
        net_dict['B'] = X_t[1:, :]
        
        if params['use_orthonormal']:
            R = net_dict['info_x_eps']['params']['R']
            net_dict['c_matrix_true'] = (np.linalg.inv(R) @ net_dict['c_matrix_true'])
       
        cnu = np.sqrt(net_dict['B'].shape[0])*net_dict['c_matrix_true']

        if parameters['normalize_cols']:
            norm_column = net_dict['info_x_eps']['params']['norm_column']
            for id_node in range(cnu.shape[1]):
                cnu[:, id_node] = norm_column*cnu[:, id_node]
        
        Z_bar = net_dict['B'] - (net_dict['PHI'] @ cnu)
        net_dict['eps_magnitude'] = norm(Z_bar, axis = 0)

    return net_dict


def compare_setup_noisy(exp_name, net_name, 
                        noise_mag_endpoints,
                        lgth_endpoints, 
                        random_seed = 1, 
                        save_full_info = False):
    '''
    
    Parameters
    ----------
    exp_name : str
        filename.
    net_name : str
        Network structure filename.
    noise_magnitude : float
        Noise magnitude of the perturbation.
    lgth_endpoints : list
        Start, end and space for length time vector.
    random_seed : int
        Seed for the random pseudo-generator.
    save_full_info : dict, optional
        To save the library matrix. The default is False.

    Returns
    -------
    exp_dictionary : TYPE
        DESCRIPTION.

    '''
    exp_params = dict()
    #canonical
    #exp_params[0] = [True, False, False]
    #normalize_cols
    exp_params[0] = [True, True, False] 
    #orthonormal
    exp_params[1] = [False, False, True]
    
    noise_mag_vector = np.geomspace(noise_mag_endpoints[0], noise_mag_endpoints[1],
                                   noise_mag_endpoints[2])
    
    
    length_time_series_vector = np.arange(lgth_endpoints[0], lgth_endpoints[1],
                                          lgth_endpoints[2], dtype = int)
    
    #Filename for output results
    out_results_direc = out_dir(net_name, exp_name)
    filename = "noise_{}_{}_{}_lgth_p_{}_{}_{}_seed_{}".format(noise_mag_endpoints[0],
                                                               noise_mag_endpoints[1],
                                                               noise_mag_endpoints[2],
                                                               lgth_endpoints[0], 
                                                               lgth_endpoints[1],
                                                               lgth_endpoints[2], 
                                                               random_seed) 
    if os.path.isfile(out_results_direc+filename+".hdf5"):
        out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'r')
        exp_dictionary = out_results_hdf5.to_dict()  
        out_results_hdf5.close()      
        return exp_dictionary
    
    else:
        out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'a')    
        out_results_hdf5['lgth_endpoints'] = lgth_endpoints
        out_results_hdf5['noise_mag_endpoints'] = noise_mag_endpoints
        out_results_hdf5['exp_params'] = dict() 
        out_results_hdf5['exp_params'] = exp_params
        
        for key in exp_params.keys():    
            out_results_hdf5[key] = dict()
            for lgth_time_series in length_time_series_vector:
                out_results_hdf5[key][lgth_time_series] = dict()
                for noise_magnitude in noise_mag_vector:
                    print('exp:', key, 'n = ', lgth_time_series, 'eps = ', noise_magnitude)
                    
                    script_dict = dict()
                    script_dict['opt_list'] = exp_params[key]
                    script_dict['lgth_time_series'] = lgth_time_series
                    script_dict['noise_magnitude'] = noise_magnitude
                    script_dict['generate_overall'] = False
                    script_dict['exp_name'] = exp_name
                    script_dict['net_name'] = net_name
                    script_dict['id_trial'] = None
                    script_dict['random_seed'] = random_seed
                    
                    
                    net_dict = compare_script_noisy(script_dict)
                    out_results_hdf5[key][lgth_time_series][noise_magnitude] = dict()
                    out_results_hdf5[key][lgth_time_series][noise_magnitude]['c_matrix_true'] = net_dict['c_matrix_true']
                    out_results_hdf5[key][lgth_time_series][noise_magnitude]['x_eps_matrix'] = net_dict['x_eps_matrix']
                    out_results_hdf5[key][lgth_time_series][noise_magnitude]['eps_magnitude'] = net_dict['eps_magnitude']                    
                    if save_full_info:
                        out_results_hdf5[key][lgth_time_series][noise_magnitude]['PHI.T PHI'] = net_dict['PHI.T PHI']
                        out_results_hdf5[key][lgth_time_series][noise_magnitude]['params'] = dict()
                        save_dict(net_dict['params'], out_results_hdf5[key][lgth_time_series][noise_magnitude]['params'])            
                    
        exp_dictionary = out_results_hdf5.to_dict()        
        out_results_hdf5.close()
        return exp_dictionary

#=============================================================================#
#Lab Analysis
#=============================================================================#
def get_G_true(net_name):
    '''
    Obtain the G true from the filename

    Parameters
    ----------
    net_name : str
        Network structure filename.

    Returns
    -------
    G_true : networkx graph
        True graph structure.
    edges_G_true : list
        All edges from the true graph.
    N: int
        Net size
        
    '''
    G_true = nx.read_edgelist("network_structure/{}.txt".format(net_name),
                        nodetype = int, create_using = nx.Graph)
    
    N = len(nx.nodes(G_true))
    A = nx.to_numpy_array(G_true, nodelist = list(range(N)))
    A = np.asarray(A)
    G_true = nx.from_numpy_array(A, create_using = nx.Graph)
    edges_G_true = list(G_true.edges())
    
    return G_true, edges_G_true, N

def compare_basis(exp_dictionary, net_name):
    '''
    Given a experiment dict, it calculates the performance of the reconstruction.

    Parameters
    ----------
    exp_dictionary : dict
        Output results dictionary.
    net_name : str
        Filename.

    Returns
    -------
    lgth_vector : numpy array 
        Array with length of time series vector.
    FP_comparison : numpy array
        False positive proportion for each length of time series.
    FN_comparison : numpy array
        False negative proportion for each length of time series.
    d_matrix : TYPE
        DESCRIPTION.

    '''
    G_true, edges_G_true, N = get_G_true(net_name) 
    
    exp_vec = list(exp_dictionary['exp_params'].keys())
    lgth_endpoints = exp_dictionary['lgth_endpoints']
    
    lgth_vector = np.arange(lgth_endpoints[0], lgth_endpoints[1],
                                      lgth_endpoints[2], dtype = int)
    
    FP_comparison = np.zeros((len(exp_vec), lgth_vector.shape[0]))
    FN_comparison = np.zeros((len(exp_vec), lgth_vector.shape[0]))

    for id_exp in range(len(exp_vec)):
        for id_key in range(len(lgth_vector)):
            key = lgth_vector[id_key]
            A_est = exp_dictionary[exp_vec[id_exp]][key]['A']
            
            G_est = nx.from_numpy_array(A_est, create_using = nx.Graph)
            links = lab_opto.links_types(G_est, G_true)        
    
            total_connections = 0.5*N*(N-1)
            FP = len(links['false_positives'])/(total_connections-len(edges_G_true))
            FP_comparison[id_exp, id_key] = FP
            
            FN = len(links['false_negatives'])/len(edges_G_true)
            FN_comparison[id_exp, id_key] = FN
            
    return lgth_vector, FP_comparison, FN_comparison

def compare_basis_net_size(exp_dictionary):
    '''
    Given a experiment dict, it calculates the performance of the reconstruction.

    Parameters
    ----------
    exp_dictionary : dict
        Output results dictionary.
    
    Returns
    -------
    size_vector : numpy array 
        Array with length of time series vector.
    n_critical_comparison : numpy array
        n_critical for each length of time series.
    
    '''
    exp_vec = list(exp_dictionary['exp_params'].keys())
    size_endpoints = exp_dictionary['size_endpoints']
    
    size_vector = np.arange(size_endpoints[0], size_endpoints[1],
                                      size_endpoints[2], dtype = int)
    
    n_critical_comparison = np.zeros((len(exp_vec), size_vector.shape[0]))
    
    for id_exp in range(len(exp_vec)):
        for id_key in range(len(size_vector)):
            key = size_vector[id_key]
            n_critical = exp_dictionary[exp_vec[id_exp]][key]['n_critical']
            n_critical_comparison[id_exp, id_key] = n_critical
            
            
    return size_vector, n_critical_comparison

def compare_l2_error(exp_dictionary):
    
    seed_vec = list(exp_dictionary.keys())
    exp_vec = [0, 1]
    noise_mag_endpoints = exp_dictionary[seed_vec[0]]['noise_mag_endpoints']
    noise_mag_vector = np.geomspace(noise_mag_endpoints[0], noise_mag_endpoints[1],
                                   noise_mag_endpoints[2])
    
    lgth_endpoints = exp_dictionary[seed_vec[0]]['lgth_endpoints']
    lgth_vector = np.arange(lgth_endpoints[0], lgth_endpoints[1],
                                      lgth_endpoints[2], dtype = int)
    
    
    exp_results = dict()
    for id_exp in exp_vec:
        exp_results[id_exp] = dict()
        
        for id_seed, seed in enumerate(seed_vec):
        
            ave_l2_error_comparison = np.zeros((len(seed_vec), noise_mag_vector.shape[0]))
            l2_error_comparison = np.zeros(noise_mag_vector.shape[0])
            
            for id_key in range(len(lgth_vector)):    
                key = lgth_vector[id_key]
                
                for id_noise, noise in enumerate(noise_mag_vector):
                    c_matrix_true = exp_dictionary[seed][id_exp][key][noise]['c_matrix_true']
                    x_eps_matrix = exp_dictionary[seed][id_exp][key][noise]['x_eps_matrix']
                    noise_magnitude = exp_dictionary[seed][id_exp][key][noise]['eps_magnitude']
                    
                    l2_error_comparison[id_noise] = norm(c_matrix_true - x_eps_matrix, 
                                                         axis = 0).mean()#/noise
                    
            ave_l2_error_comparison[id_seed, :] = l2_error_comparison
        
        exp_results[id_exp]['ave_l2'] = ave_l2_error_comparison
        exp_results[id_exp]['noise_mag_mean'] = noise_magnitude.mean()
    return key, noise_mag_vector, exp_results



def plot_l2_error_vs_noise_mag(exp_dictionary, 
                               plot_dict = {'ax': None, 
                                            'color': '#24AD7A',
                                            'marker':'o',
                                            'legend':r''}):
    
    # Plot the relative error with respect to the eps parameter
    
    if plot_dict['ax'] is None:
        fig, ax = plt.subplots(figsize = (6, 3), dpi = 300)
    else:
        ax = plot_dict['ax']
    
    exp_vec = [0, 1]
    labels = ['BP', 'EBP']
    markers = ['s', 'o']
    colors = ['tab:purple', '#24AD7A']
    
    lgth, noise_mag_vector, exp_results = compare_l2_error(exp_dictionary)
    
    for id_exp in exp_vec:
        
        ave_l2_error_comparison = exp_results[id_exp]['ave_l2']
        noise_magnitude = exp_results[id_exp]['noise_mag_mean']
        
        mean, std = ave_l2_error_comparison.mean(axis = 0), ave_l2_error_comparison.std(axis = 0)
        
        
        ax.plot(noise_mag_vector,
                mean,
                '-',
                label = labels[id_exp],
                color = colors[id_exp],
                marker = markers[id_exp],
                alpha = 1.0)
    
        ax.fill_between(noise_mag_vector,
                        mean - std,
                        mean + std,
                        color = colors[id_exp],
                        alpha = 0.4)    
        
    ax.legend(loc=1)
    
    
    ax.set_xscale('log')
    ax.set_xlabel(r'Parameter $\epsilon$')
    #ax.set_ylabel(r'$\langle \|c_{}$'.format('{\nu}')+r'$^{}$'.format('{\star}')+r'$(\epsilon) - c_{}\|_2\rangle$'.format('{\nu}'))
    ax.set_ylabel(r'$\langle E(\epsilon)\rangle$')

    # Plot an inset with the results with the EBP in log-log scale    
    id_exp = 1
    x1, x2, y1, y2 = 1e-3, 1e-2, 1e-6, 1e-3  # subregion of the original image
    axins = ax.inset_axes(
    [0.15, 0.6, 0.45, 0.35],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    
                
    axins.plot(noise_mag_vector, 
               mean, 
               '-',
               label = labels[id_exp],
               color = colors[id_exp],
               marker = markers[id_exp],
               alpha = 1.0)
    
    lower = mean - std
    axins.fill_between(noise_mag_vector,
                    lower,
                    mean + std,
                    color = colors[id_exp],
                    alpha = 0.4)
    
    axins.vlines(noise_magnitude, 1e-6, 
              1e-4,
              linestyle = 'dashed', color = 'black')
    axins.text(noise_magnitude + 5e-4, 
               2e-6, 
               r'$\epsilon_{}$'.format('{\star}'),
               fontsize = 15)
    axins.set_xscale("log")
    axins.set_yscale("log")
    
def get_min_l2_error(exp_dictionary):
    
    seed_vec = list(exp_dictionary.keys())
    exp_vec = [0, 1]
    noise_mag_endpoints = exp_dictionary[seed_vec[0]]['noise_mag_endpoints']
    noise_mag_vector = np.geomspace(noise_mag_endpoints[0], noise_mag_endpoints[1],
                                   noise_mag_endpoints[2])
    
    lgth_endpoints = exp_dictionary[seed_vec[0]]['lgth_endpoints']
    lgth_vector = np.arange(lgth_endpoints[0], lgth_endpoints[1],
                                      lgth_endpoints[2], dtype = int)
    
    
    exp_results = dict()
    for id_exp in exp_vec:
        exp_results[id_exp] = dict()
        
        for id_seed, seed in enumerate(seed_vec):
        
            min_l2_error_comparison = np.zeros((len(seed_vec), len(lgth_vector)))
            noise_mag_error = np.zeros((len(seed_vec), len(lgth_vector)))
            l2_error_comparison = np.zeros(noise_mag_vector.shape[0])
            
            for id_key in range(len(lgth_vector)):    
                key = lgth_vector[id_key]
                
                for id_noise, noise in enumerate(noise_mag_vector):
                    c_matrix_true = exp_dictionary[seed][id_exp][key][noise]['c_matrix_true']
                    x_eps_matrix = exp_dictionary[seed][id_exp][key][noise]['x_eps_matrix']
                    noise_magnitude = exp_dictionary[seed][id_exp][key][noise]['eps_magnitude']
                    
                    l2_error_comparison[id_noise] = norm(c_matrix_true - x_eps_matrix, 
                                                         axis = 0).mean()#/noise
                    
                min_l2_error_comparison[id_seed, id_key] = noise_mag_vector[l2_error_comparison.argmin()]
                noise_mag_error[id_seed, id_key] = noise_magnitude.mean()
                
        exp_results[id_exp]['min_ave_l2'] = min_l2_error_comparison
        exp_results[id_exp]['noise_mag_mean'] = noise_mag_error
        
    return lgth_vector, exp_results
    
def plot_l2_error_vs_lgth_vector(exp_dictionary, 
                                 plot_dict = {'ax': None, 
                                            'color': '#24AD7A',
                                            'marker':'o',
                                            'legend':r''}):
    
    # Plot the relative error with respect to the eps parameter
    
    if plot_dict['ax'] is None:
        fig, ax = plt.subplots(figsize = (6, 3), dpi = 300)
    else:
        ax = plot_dict['ax']
    
    exp_vec = [1]
    labels = ['BP', 'EBP']
    markers = ['s', 'o']
    colors = ['tab:purple', '#24AD7A']
    
    lgth_vector, exp_results = get_min_l2_error(exp_dictionary)
    
    for id_exp in exp_vec:
        
        min_l2_error_comparison = exp_results[id_exp]['min_ave_l2']
        noise_magnitude = exp_results[id_exp]['noise_mag_mean']
        
        mean, std = min_l2_error_comparison.mean(axis = 0), min_l2_error_comparison.std(axis = 0)
        
        
        ax.plot(lgth_vector,
                mean,
                '-',
                label = r'',
                color = colors[id_exp],
                marker = markers[id_exp],
                alpha = 1.0)
    
        ax.fill_between(lgth_vector,
                        mean - std,
                        mean + std,
                        color = colors[id_exp],
                        alpha = 0.4)    
        
        ax.plot(lgth_vector,
                noise_magnitude.mean(axis = 0),
                '--',
                label = r'$\|\bar{u}\|_2$',
                color = 'k',
                alpha = 0.6)
        
    ax.legend(loc=0)
    
    #ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'length of time series $n$')
    ax.set_ylabel(r'$\epsilon_{}$'.format('{\star}'))
    
    
def ax_plot_true_net(ax, G_true, pos_true, probed_node = 0, 
                     print_probed = True, plot_net_alone = False):
    '''
    Plot original network   

    Parameters
    ----------
    ax : Matplotlib Axes object
        Draw the graph in the specified Matplotlib axes.
    G_true: networkx graph
        Graph structure 
    pos_true: dict 
        Dictionary
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
    nx.draw_networkx_nodes(G_true, pos = pos_true, 
                           ax = ax, node_color = colors[3], 
                           linewidths= 1.0,
                           node_size = 150,
                           alpha = 1.0)
    nx.draw_networkx_nodes(G_true, pos = pos_true, 
                           node_color = colors[0], 
                           node_size = 100,
                           ax = ax,
                           alpha = 1.0)
    
    if print_probed:
        nx.draw_networkx_nodes(G_true, pos = pos_true, 
                               ax = ax,
                               nodelist=[probed_node],
                               node_color = colors[3], 
                               node_size = 100,
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

def ax_plot_ring_graph(ax, plot_net_alone=False):
    '''
    Plot the ring graph

    Parameters
    ----------
    ax : Matplotlib Axes object
        Draw the graph in the specified Matplotlib axes.
    plot_net_alone : boolean, optional
        To plot the network itself outside an environment. The default is False.

    Returns
    -------
    None.

    '''
    N = 10
    G_true = nx.cycle_graph(N, create_using=nx.Graph())
    pos_true = nx.circular_layout(G_true)
    nx.draw_networkx_nodes(G_true, pos = pos_true,
                           ax = ax, node_color = colors[3], 
                           linewidths= 1.0,
                           node_size = 150,
                           alpha = 1.0)
    nx.draw_networkx_nodes(G_true, pos = pos_true,
                           node_color = colors[0], 
                           node_size = 100,
                           ax = ax,
                           alpha = 1.0)
    
    nx.draw_networkx_edges(G_true,pos = pos_true, 
                           ax = ax,
                           edgelist = list(G_true.edges()), 
                           edge_color = colors[4],
                           arrows = True,
                           arrowsize = 7,
                           width = 0.65,
                           alpha = 1.0)
    ax.margins(0.3)
    ax.axis("off")
    if plot_net_alone:
        ax.set_title('{}'.format('Original Network'))

def plot_comparison_analysis(ax, exp_dictionary, net_name, plot_legend):    
    '''
    To plot a comparison between EBP and BP for increasing the length of time series.

    Parameters
    ----------
    ax : Matplotlib Axes object
        Draw the graph in the specified Matplotlib axes.
    exp_dictionary : dict
        Dictionary carrying the information about the experiments to be plotted.
    net_name : str
        Network filename.
    plot_legend : boolean
        To plot the legend inside the ax panel.

    Returns
    -------
    None.

    '''
    seeds = list(exp_dictionary.keys())
    Nseeds = int(len(seeds))
    
    lgth_endpoints = exp_dictionary[seeds[0]]['lgth_endpoints']
    lgth_vector = np.arange(lgth_endpoints[0], lgth_endpoints[1],
                                      lgth_endpoints[2], dtype = int)
    
    FP_comparison, FN_comparison = np.zeros((Nseeds, 2, lgth_vector.shape[0])), np.zeros((Nseeds, 2, lgth_vector.shape[0]))
    
    for id_seed in range(Nseeds):
        lgth_vector, FP_comparison[id_seed, :, :], FN_comparison[id_seed, :, :]  = compare_basis(exp_dictionary[seeds[id_seed]], 
                                                                            net_name)
    
    avge_FP_comparison = FP_comparison.mean(axis = 0)    
    std_FP_comparison = FP_comparison.std(axis = 0)    
    avge_FN_comparison = FN_comparison.mean(axis = 0)    
    std_FN_comparison = FN_comparison.std(axis = 0)    
    
    ind_start = np.where(avge_FN_comparison[0, :] == 0)[0][0]
        
    lab_opto.plot_false_proportion(ax, lgth_vector[ind_start:], 
                                   avge_FP_comparison[:, ind_start:], 
                                   std_FP_comparison[:, ind_start:], 
                                   True, 
                                   plot_legend)
    
    ax.set_ylabel(r'FP')
    plt.setp(ax.get_xticklabels(), visible=True)
    
    #lab_opto.plot_false_proportion(ax, lgth_vector, avge_FN_comparison, std_FN_comparison, True, True)
    #ax.set_ylabel(r'FN')
    
    ax.set_xlabel(r'length of time series $n$')
    
def plot_comparison_n_critical(ax, exp_dictionary, plot_legend):    
    '''
    To plot the comparison between EBP and BP in the experiment: n_c vs N

    Parameters
    ----------
    ax : Matplotlib Axes object
        Draw the graph in the specified Matplotlib axes.
    exp_dictionary : dict
        Dictionary carrying the information about the experiments to be plotted.
    plot_legend : boolean
        To plot the legend inside the ax panel.

    Returns
    -------
    None.

    '''
    seeds = list(exp_dictionary.keys())
    Nseeds = int(len(seeds))
    
    size_endpoints = exp_dictionary[seeds[0]]['size_endpoints']
    size_vector = np.arange(size_endpoints[0], size_endpoints[1],
                                      size_endpoints[2], dtype = int)
    
    n_c_comparison = np.zeros((Nseeds, 2, size_vector.shape[0]))
    
    
    for id_seed in range(Nseeds):
        size_vector, n_c_comparison[id_seed, :, :] = compare_basis_net_size(exp_dictionary[seeds[id_seed]])
    
    avge_nc_comparison = n_c_comparison.mean(axis = 0)    
    std_nc_comparison = n_c_comparison.std(axis = 0)     
        
    lab_opto.plot_false_proportion(ax, size_vector, avge_nc_comparison, 
                                   std_nc_comparison, True, plot_legend)
    
    
    ax.set_ylabel(r'$n_0$')
    plt.setp(ax.get_xticklabels(), visible=True)
    
    #lab_opto.plot_false_proportion(ax[1], lgth_vector, FN_comparison, True)
    #ax[1].set_ylabel(r'FN')
    ax.set_xlabel(r'$N$')
    
def plot_lgth_dependence(net_name, exps_dictionary, title, filename = None):    
    '''
    Plot the reconstruction performance vs length of time series.


    Parameters
    ----------
    net_name : str
        Network filename.
    exps_dictionary : dict
        Dictionary carrying the information about the experiments to be plotted.
    title : str
        Title to be plotted.
    filename : str, optional
        Saving pdf filename. The default is None.

    Returns
    -------
    None.

    '''
    keys = list(exps_dictionary.keys())
    n_cols = int(len(keys))
    
    fig_ = plt.figure(figsize = (6, 3), dpi = 300)
    subfigs = fig_.subfigures(1, 2, width_ratios = [0.9, 1.1])
    
    fig = subfigs[0]
    
    gs = GridSpec(nrows=1, ncols=1, figure=fig)
    
    ax_0 = fig.add_subplot(gs[0])
    
    ax_plot_ring_graph(ax_0)

    fig.suptitle(r'a) Original Network') 
    plot_legend = True
    for id_col in range(n_cols):
        fig1 = subfigs[id_col+1]
        
        gs1 = GridSpec(nrows=1, ncols=1, figure=fig1)
        
        exp_dictionary = exps_dictionary[keys[id_col]]
        
        ax1 = fig1.add_subplot(gs1[0])
        #ax2 = fig1.add_subplot(gs1[1])
        
        plot_comparison_analysis(ax1, exp_dictionary, net_name, plot_legend)
        if plot_legend:
            plot_legend = False
        fig1.suptitle(title[id_col])
    
    fig_.suptitle('fig')
    if filename == None:
        plt.show()
    else:
     
        plt.savefig(filename+".pdf", format='pdf', bbox_inches='tight')
        
    return     

def plot_n_c_size(exps_dictionary, title, net_info, fig_ = None, filename = None,
                  plot_legend_global = True):    
    '''
    Plot the n_c vs N.

    Parameters
    ----------
    exp_dictionary : dict
        Dictionary carrying the information about the experiments to be plotted.
    title : str
        Title to be plotted.
    filename : str, optional
        Saving pdf filename. The default is None.

    Returns
    -------
    None.

    '''
    
    keys = list(exps_dictionary.keys())
    n_cols = int(len(keys))
    
    if fig_ == None:
        fig_ = plt.figure(figsize = (6, 3), dpi = 300)
    subfigs = fig_.subfigures(1, 2, width_ratios = [0.8, 1.0])
    
    fig = subfigs[0]
    
    gs = GridSpec(nrows=1, ncols=1, figure=fig)
    
    ax_0 = fig.add_subplot(gs[0])
    
    G_true, pos_true = net_info['G'], net_info['pos']
        
    ax_plot_true_net(ax_0, G_true, pos_true, probed_node = 0, 
                     print_probed = False, plot_net_alone = False)
    
    fig.suptitle(title[0], x = 0.45) 
    
    if plot_legend_global:
        plot_legend = True
    else:
        plot_legend = False
    for id_col in range(n_cols):
        fig1 = subfigs[id_col+1]
        
        gs1 = GridSpec(nrows=1, ncols=1, figure=fig1)
        exp_dictionary = exps_dictionary[keys[id_col]]
        ax1 = fig1.add_subplot(gs1[0])
        #ax2 = fig1.add_subplot(gs1[1])
        
        plot_comparison_n_critical(ax1, exp_dictionary, plot_legend)
        if plot_legend:
            plot_legend = False
        fig1.suptitle(title[1], x = 0.05)
    
    fig_.suptitle('fig')

    if fig_ == None:
        if filename == None:
            plt.show()
        else:
            plt.savefig(filename+".pdf", format='pdf', bbox_inches='tight')
    else:            
        return fig_

def fig_1_plot(exps, net_info, titles, filename = None):
    '''
    EBP vs BP comparison figure.

    Parameters
    ----------
    exps : dict
        Dictionary carrying the information about the experiments to be plotted.
    net_info : dict
        Network information.
            'net_name': Filename of the network to be evaluated.
            'net_class': Class of graph to be increased in order to perform the experiment.
    titles : dict
        Titles to be plotted.
    filename : str, optional
        Saving pdf filename. The default is None.

    Returns
    -------
    None.

    '''
    
    fig = plt.figure(figsize = (8, 3), dpi = 300)
        
    gs = GridSpec(nrows=1, ncols=3, figure=fig, width_ratios=(0.8, 1.2, 0.8))
    
    #===========================================#
    ax_0 = fig.add_subplot(gs[0])
    ax_plot_ring_graph(ax_0)
    ax_0.set_title(r'a) Original Network') 
    #===========================================#
    
    ax_1 = fig.add_subplot(gs[1])
    plot_comparison_analysis(ax_1, exps['lgth'][0], net_info['net_name'], True)
    ax_1.set_title(titles['lgth'][0])
    
    #===========================================#
    
    keys = list(exps['n_c'].keys())
    n_cols = int(len(keys))

    plot_legend = False
    delta = [10, 80]
    slope = [20, 20]
    for id_col in range(n_cols):
        exp_dictionary = exps['n_c'][keys[id_col]]
        size_endpoints = exp_dictionary[1]['size_endpoints']

        ax1 = fig.add_subplot(gs[2 + id_col])
        
        plot_comparison_n_critical(ax1, exp_dictionary, plot_legend)
        
        
        N_vector = np.arange(size_endpoints[0] + delta[id_col], size_endpoints[1] - delta[id_col], 0.1)
        ax1.plot(N_vector, slope[id_col]*np.log(N_vector), 'k--', lw = 1)
        
        ax1.set_title(titles['n_c'][id_col])
    
    gs.tight_layout(fig)
    if filename == None:
        plt.show()
    else:
     
        plt.savefig(filename+".pdf", format='pdf', bbox_inches='tight')
        
    return     
#=============================================================================#
#Scripts
#=============================================================================#
def ring_N_16(net_name = 'ring_graph_N=16', Nseeds = 10):
    '''
    Setting the experiment of increasing the length of time series.

    Parameters
    ----------
    net_name : str, optional
        Name of the network to be evaluated. The default is 'ring_graph_N=16'.
    Nseeds : int, optional
        Total number of seeds. The default is 10.

    Returns
    -------
    exps_dictionary : dict
        Experiment dictionary with information gathered from the hdf5 file.
    title : TYPE
        DESCRIPTION.

    '''
    lgths_endpoints = [[10, 510, 12]]
    #exps_name = ["gnr_logistc_compar_deg_2", "gnr_logistc_compar_deg_3"]
    #title = ['b) deg 2', 'c) deg 3']
    exps_name = ["logc_lgth_3_99_0_001"]
    title = [r'b) Dependence on $n$']
    exps_dictionary = dict()
    
    for id_exp in range(len(exps_name)):
        exps_dictionary[id_exp] = dict()
        lgth_endpoints = lgths_endpoints[id_exp]
        exp_name = exps_name[id_exp]
        out_results_direc = os.path.join(folder_name, net_name)
        out_results_direc = os.path.join(out_results_direc, exp_name)
        out_results_direc = os.path.join(out_results_direc, '')
        
        if os.path.isdir(out_results_direc) == False:
            print("Failed to find the desired result folder !")
        
        for seed in range(1, Nseeds + 1):
            exps_dictionary[id_exp][seed] = dict()
         
            filename = "lgth_endpoints_{}_{}_{}_seed_{}".format(lgth_endpoints[0], lgth_endpoints[1],
                                                        lgth_endpoints[2], seed) 
            
            if os.path.isfile(out_results_direc+filename+".hdf5"):
                out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'r')
                exp_dictionary = out_results_hdf5.to_dict()  
                out_results_hdf5.close()
            
            exps_dictionary[id_exp][seed] = exp_dictionary
    
    return exps_dictionary, title

def ring_net_plot_script(Nseeds = 10):
    exps_dictionary, title = ring_N_16(net_name = 'ring_graph_N=16', Nseeds = Nseeds)
    plot_lgth_dependence('ring_graph_N=16', exps_dictionary, title, filename = None)

def exp_setting_n_c(exps_name, sizes_endpoints, net_class = 'ring_graph', Nseeds = 10):
    '''
    Setting the experiment of determing the critical length of time series for
    a successfull reconstruction.

    Parameters
    ----------
    exps_name : list
        Name of experiments to be read from the file.
    sizes_endpoints : list
        Endpoints of the arrays to determine the numpy arrays.
    net_class : str, optional
        Class of graph to be increased in order to perform the experiment. The default is 'ring_graph'.
    Nseeds : int, optional
        Total number of seeds. The default is 10.

    Returns
    -------
    exps_dictionary : dict
        Experiment dictionary with information gathered from the hdf5 file.

    '''
    exps_dictionary = dict()
    
    for id_exp in range(len(exps_name)):
        exps_dictionary[id_exp] = dict()
        size_endpoints = sizes_endpoints[id_exp]

        exp_name = exps_name[id_exp]
        out_results_direc = os.path.join(folder_name, net_class)
        out_results_direc = os.path.join(out_results_direc, exp_name)
        out_results_direc = os.path.join(out_results_direc, '')
        
        if os.path.isdir(out_results_direc ) == False:
            print("Failed to find the desired result folder !")
        
        for seed in range(1, Nseeds + 1):    
            exps_dictionary[id_exp][seed] = dict()
            
            filename = "size_endpoints_{}_{}_{}_seed_{}".format(size_endpoints[0], size_endpoints[1],
                                                        size_endpoints[2], seed) 
            
            if os.path.isfile(out_results_direc+filename+".hdf5"):
                try:
                    out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'r')
                    exp_dictionary = out_results_hdf5.to_dict()  
                    out_results_hdf5.close()
                    exps_dictionary[id_exp][seed] = exp_dictionary
                except:
                    print('Failed to open the desired file!')    
                    #exp_dictionary = dict()
                    del exps_dictionary[id_exp][seed]
            else:
                print('Failed to find the desired file!')
                
                print(out_results_direc+filename+".hdf5")

    return exps_dictionary

def exp_setting_noisy(exp_name, 
                      noise_mag_endpoints,
                      lgth_endpoints,
                      net_name = 'ring_graph_N=16', Nseeds = 10):
    '''
    Setting the experiment of increasing the noise magnitude and length of time series for
    a successfull reconstruction.

    Parameters
    ----------
    exp_name : list
        Name of experiments to be read from the file.
    noise_mag_endpoints : list
        Endpoints of the arrays to determine the numpy arrays.
    lgth_endpoints : list
        Endpoints of the arrays to determine the numpy arrays.
    net_name : str, optional
        Class of graph to be increased in order to perform the experiment. The default is 'ring_graph'.
    Nseeds : int, optional
        Total number of seeds. The default is 10.

    Returns
    -------
    exps_dictionary : dict
        Experiment dictionary with information gathered from the hdf5 file.

    '''
    exps_dictionary = dict()
    
    noise_mag_vector = np.geomspace(noise_mag_endpoints[0], noise_mag_endpoints[1],
                                   noise_mag_endpoints[2])
    
    
    length_time_series_vector = np.arange(lgth_endpoints[0], lgth_endpoints[1],
                                          lgth_endpoints[2], dtype = int)
    
    
    out_results_direc = out_dir(net_name, exp_name)
       
    for lgth_time_series in length_time_series_vector:
        
        for noise_magnitude in noise_mag_vector:
            
            for seed in range(1, Nseeds + 1):  
                
                filename = "noise_{}_{}_{}_lgth_p_{}_{}_{}_seed_{}".format(noise_mag_endpoints[0],
                                                                       noise_mag_endpoints[1],
                                                                       noise_mag_endpoints[2],
                                                                       lgth_endpoints[0], 
                                                                       lgth_endpoints[1],
                                                                       lgth_endpoints[2], 
                                                                       seed) 
                
                if os.path.isfile(out_results_direc+filename+".hdf5"):
                    try:
                        out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'r')
                        exp_dictionary = out_results_hdf5.to_dict()  
                        out_results_hdf5.close()
                        exps_dictionary[seed] = exp_dictionary
                    except:
                        print('Failed to open the desired file!')    
                        #exp_dictionary = dict()
                        del exps_dictionary[lgth_time_series][noise_magnitude][seed]
        
                else:
                    print('Failed to find the desired file!')
                    
                    print(out_results_direc+filename+".hdf5")
        
    return exps_dictionary


def exp_setting_noisy_one_exp(exp_name, 
                              noise_mag_endpoints,
                              lgth_endpoints,
                              net_name = 'ring_graph_N=16', Nseeds = 10):
    '''
    Setting the experiment of increasing the noise magnitude and length of time series for
    a successfull reconstruction.

    Parameters
    ----------
    exp_name : list
        Name of experiments to be read from the file.
    noise_mag_endpoints : list
        Endpoints of the arrays to determine the numpy arrays.
    lgth_endpoints : list
        Endpoints of the arrays to determine the numpy arrays.
    net_name : str, optional
        Class of graph to be increased in order to perform the experiment. The default is 'ring_graph'.
    Nseeds : int, optional
        Total number of seeds. The default is 10.

    Returns
    -------
    exps_dictionary : dict
        Experiment dictionary with information gathered from the hdf5 file.

    '''
    exps_dictionary = dict()
    
    noise_mag_vector = np.geomspace(noise_mag_endpoints[0], noise_mag_endpoints[1],
                                   noise_mag_endpoints[2])
    
    
    length_time_series_vector = np.arange(lgth_endpoints[0], lgth_endpoints[1],
                                          lgth_endpoints[2], dtype = int)
    
    
    out_results_direc = out_dir(net_name, exp_name)
       
    for seed in range(1, Nseeds + 1):  
                
        filename = "noise_{}_{}_{}_lgth_p_{}_{}_{}_seed_{}".format(noise_mag_endpoints[0],
                                                               noise_mag_endpoints[1],
                                                               noise_mag_endpoints[2],
                                                               lgth_endpoints[0], 
                                                               lgth_endpoints[1],
                                                               lgth_endpoints[2], 
                                                               seed) 
            
        if os.path.isfile(out_results_direc+filename+".hdf5"):
            try:
                out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'r')
                exp_dictionary = out_results_hdf5.to_dict()  
                out_results_hdf5.close()
                exps_dictionary[seed] = exp_dictionary
            except:
                print('Failed to open the desired file!')    
                #exp_dictionary = dict()
                del exps_dictionary[seed]

        else:
            print('Failed to find the desired file!')
            
            print(out_results_direc+filename+".hdf5")

    return exps_dictionary


def ring_graph_lgth_script(rs):
    '''
    Script to generate an experiment of varying length of time series and
    obtaining the network reconstruction.

    Parameters
    ----------
    rs : int
        Int for the seed of the random pseudo-generator.

    Returns
    -------
    None.

    '''
    exp_name = 'logc_lgth_3_99_0_001'
    net_name = 'ring_graph_N=40'
    lgth_endpoints = [10, 510, 12]
    compare_setup(exp_name, net_name, lgth_endpoints, random_seed = rs, 
                      save_full_info = False)


def ring_graph_noisy_lgth_script(rs):
    '''
    Script to generate an experiment of varying length of time series for
    a given noise magnitude and
    obtaining the network reconstruction.

    Parameters
    ----------
    rs : int
        Int for the seed of the random pseudo-generator.

    Returns
    -------
    None.

    '''
    
    
    '''
    Description of a few experiments. 
    
    #The first experiment (02/02/25) [100, 101, 10]
    #The second experiment (02/02/25) [50, 51, 10] 
    #The third experiment (02/02/25) [200, 201, 10] 
    
    # This experiment normalizes the perturbation vector in the dynamics.
    # Let us analyze the effect if one does not normalize it.
    exp_name = 'logc_lgth_3_99_0_001_nsy_eps_1e-5'
    
    # This experiment does not normalize the perturbation vector in the dynamics.
    However, the comparison with respect to the true vector is done on the canonical 
    basis. 
    'logc_lgth_3_99_0_001_unnorm_nsy_eps_1e-5'
    
    #This experiment compares the reconstructed vector and true vector
    #with respect to the orthonormal basis.
    exp_name = 'logc_lgth_3_99_0_001_unnorm_eps_1e-5_orth_basis'
    
    # This experiment compares all basis functions. norm can and orth
    # for a fixed amount of length of time series, for different noise magnitudes.
    exp_name = 'log_unnorm_eps_1e-4_comparison'
    
    # This experiment compares all basis functions. norm can and orth
    # for varying amount of length of time series, for different noise magnitudes.
    'log_unnorm_eps_1e-4_comparison_lgth_vary'
    
    '''
    
    exp_name = 'log_unnorm_eps_1e-4_comparison'
    net_name = 'ring_graph_N=16'
    noise_mag_endpoints = [1e-3, 1e-2, 30]
    
    lgth_endpoints = [200, 201, 10]
    exp_dictionary = compare_setup_noisy(exp_name, net_name, 
                                        noise_mag_endpoints,
                                        lgth_endpoints, 
                                        random_seed = rs, 
                                        save_full_info = False)

    return exp_dictionary

def ring_graph_noisy_exp(Nseeds, lgth_endpoints, one_experiment = False):
    # The description of each experiment is described in the method above: 
        #ring_graph_noisy_lgth_script
    #exp_name = 'logc_lgth_3_99_0_001_nsy_eps_1e-5'
    #exp_name = 'logc_lgth_3_99_0_001_unnorm_nsy_eps_1e-5'
    #exp_name = 'logc_lgth_3_99_0_001_unnorm_eps_1e-5_orth_basis'
    net_name = 'ring_graph_N=16'
    noise_mag_endpoints = [1e-3, 1e-2, 30]
    
    if not one_experiment:
        exp_name = 'log_unnorm_eps_1e-4_comparison'    
        exps = exp_setting_noisy(exp_name, 
                                 noise_mag_endpoints,
                                 lgth_endpoints,
                                 net_name = 'ring_graph_N=16', Nseeds = Nseeds)
    
    else:
        exp_name = 'log_unnorm_eps_1e-4_comparison_lgth_vary'
        exps = exp_setting_noisy_one_exp(exp_name, 
                                         noise_mag_endpoints, 
                                         lgth_endpoints, 
                                         'ring_graph_N=16', 
                                         Nseeds)
        
    return exps

def fig_ave_l2_over_nodes_noise_mag(filename = None):
    
    plot_dict = dict()
    
    fig, ax = plt.subplots(1, 2, figsize = (9, 3.5),
                           gridspec_kw={'width_ratios': [2, 1]},
                           dpi = 300)
    
    #=========================================================================#    
    lgth_endpoints = [200, 201, 10]
    exp1 = ring_graph_noisy_exp(10, lgth_endpoints)
    plot_dict['ax'] = ax[0]
    plot_l2_error_vs_noise_mag(exp1, plot_dict)
    #=========================================================================#
    
    lgth_endpoints = [10, 301, 10]
    exp2 = ring_graph_noisy_exp(1, lgth_endpoints, True)
    plot_dict['ax'] = ax[1]
    plot_l2_error_vs_lgth_vector(exp2, plot_dict)
    #=========================================================================#
    #=========================================================================#
    
    if filename == None:
        plt.tight_layout()
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(filename+".pdf", format = 'pdf')


def n_c_plot_script(Nseeds = 10):
    '''
    Script to plot an experiment of determining the critical length 
    of time series as the size of the network is increased.

    Parameters
    ----------
    Nseeds : int, optional
        Total number of seeds of the random pseudo-generator. The default is 10.

    Returns
    -------
    exps_dictionary : dict
        Experiment dictionary with information gathered from the hdf5 file.

    '''
    title = [r'b) deg 3']
    exps_name = ['lattice_neighs_3_deg_3_x_2']#['gnet_deg_3_3_99']#['growing_net_deg_3_3_99_0_001_N']
    size_endpoints = [[10, 555, 55]]#[[3, 51, 5]]
    exps_dictionary = exp_setting_n_c(exps_name, size_endpoints, 
                                             net_class = 'lattice_neighs_3',
                                             Nseeds = Nseeds)
    
    plot_n_c_size(exps_dictionary, title, fig_ = None, filename = None)
    return exps_dictionary
    
def ring_graph_script(rs):
    '''
    Script to generate an experiment of determining the critical length 
    of time series as the size of the network is increased.

    Parameters
    ----------
    rs : int
        Int for the seed of the random pseudo-generator.

    Returns
    -------
    None.

    '''
    exp_name = 'lattice_neighs_3_deg_3_x_2'
    
    net_info = dict()
    net_info['net_class'] = 'lattice_neighs_3'
    net_info['gen'] = tools.make_ring_lattice
    size_endpoints = [10, 555, 55]
    id_trial = None #np.array([0])
    compare_setup_critical_n(exp_name, net_info, size_endpoints, id_trial, 
                             random_seed = rs, save_full_info = False)


  

def fig_1_setup(Nseeds = 10, filename = None):
    '''
    To plot the comparison of EBP and BP in a ring net. This stands for the
    Figure 1 of the article.

    Parameters
    ----------
    Nseeds : int, optional
        Number of seeds in the experiment. The default is 10.
    filename : str, optional
        Saving pdf filename. The default is None.

    Returns
    -------
    exps : dict
        Dictionary carrying the information about the experiments to be plotted.

    '''
    net_info = dict()
    net_info['net_name'] = 'ring_graph_N=40'
    net_info['net_class'] = 'ring_graph'
    
    exps = dict()
    titles = dict()
    
    exps['lgth'], titles['lgth'] = ring_N_16(net_info['net_name'], Nseeds = Nseeds)
    
    exps_name = ['growing_net_deg_3_3_99_0_001_N']#,'gnet_deg_3_3_99_deg_1']
    size_endpoints = [[3, 51, 5]]#, [10, 555, 55]]
    
    exps['n_c'] = exp_setting_n_c(exps_name, size_endpoints, 
                                             net_class = net_info['net_class'],
                                             Nseeds = Nseeds)
    
    titles['n_c'] = [ r'c) $h_{ij}(x_i, x_j) = x_i x_j$']#, r'd) $h(x_i, x_j) = x_j^2$']

    fig_1_plot(exps, net_info, titles, filename = filename)
   
    return exps

def diff_nets_n_c_plot(Nseeds = 10, filename = None):
    '''
    Script to plot for different network structures 
    the experiment of determining the critical length 
    of time series as the size of the network is increased.

    Parameters
    ----------
    Nseeds : int, optional
        Total number of seeds of the random pseudo-generator. The default is 10.

    Returns
    -------
    exps_dictionary : dict
        Experiment dictionary with information gathered from the hdf5 file.

    '''
    
    fig_ = plt.figure(figsize = (5, 7), dpi = 300)
    subfigs = fig_.subfigures(3, 1)
    #=========================================================================#
    title = [r'a) $ \Delta = 2$', r'b)']
    exps_name = ['gnet_deg_3_3_99_deg_1']
    size_endpoints = [[10, 555, 55]]
    exps_dictionary = exp_setting_n_c(exps_name, size_endpoints, 
                                             net_class = 'ring_graph',
                                             Nseeds = Nseeds)
    
    net_info = dict()
    net_info['G'] = tools.ring_graph(10)
    net_info['pos'] = nx.circular_layout(net_info['G'])

    plot_n_c_size(exps_dictionary, title, net_info, fig_ = subfigs[0], 
                  filename = None, plot_legend_global = True)
    
    #=========================================================================#
    title = [r'c) $  \Delta = 6$', r'd)']
    exps_name = ['lattice_neighs_3_deg_3_x_2']
    size_endpoints = [[10, 555, 55]]
    exps_dictionary = exp_setting_n_c(exps_name, size_endpoints, 
                                             net_class = 'lattice_neighs_3',
                                             Nseeds = Nseeds)
    
    net_info = dict()
    net_info['G'] = tools.make_ring_lattice(10)
    net_info['pos'] = nx.circular_layout(net_info['G'])
    
    plot_n_c_size(exps_dictionary, title, net_info, fig_ = subfigs[1], 
                  filename = None, plot_legend_global = False)
    #=========================================================================#
    title = [r'e) $ \Delta = N$', r'f)']
    exps_name = ['gnet_deg_3_3_99']
    size_endpoints = [[10, 555, 55]]
    exps_dictionary = exp_setting_n_c(exps_name, size_endpoints, 
                                             net_class = 'star_graph',
                                             Nseeds = Nseeds)
    
    net_info = dict()
    net_info['G'] = tools.star_graph(10)
    net_info['pos'] = nx.spring_layout(net_info['G'])
    
    plot_n_c_size(exps_dictionary, title, net_info, fig_ = subfigs[2], 
                  filename = None, plot_legend_global = False)
    #=========================================================================#
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename+".pdf", format='pdf', bbox_inches='tight')
        
    return

def test_rgraph(rs):
    exp_name = 'test_rgraph'
    net_name = 'ring_graph_N=16'
    lgth_endpoints = [90, 91, 5]
    exp_dictionary = compare_setup(exp_name, net_name, lgth_endpoints, random_seed = rs, 
                      save_full_info = True)

    return exp_dictionary

def test_script(rs):
    exp_name = 'test_lattice'
    net_info = dict()
    net_info['net_class'] = 'lattice_neighs_3'
    net_info['gen'] = tools.make_ring_lattice
    size_endpoints = [10, 555, 55]
    id_trial = None #np.array([0])
    exp_dictionary = compare_setup_critical_n(exp_name, net_info, size_endpoints, 
                             id_trial, random_seed = rs, 
                             save_full_info = False)    
    return exp_dictionary


def test_noisy_script(rs):
    exp_name = 'test_normalized_cols_magnitude_eps_1e-4'
    net_name = 'ring_graph_N=16'
    noise_mag_endpoints = [5e-5, 1e-3, 30]
    #The first experiment (02/02/25) [100, 101, 10]
    #The second experiment (02/02/25) [50, 51, 10] 
    #The third experiment (02/02/25) [200, 201, 10] 
    lgth_endpoints = [100, 101, 10]
    script_dict = dict()
    script_dict['opt_list'] = [True, True, False]
    script_dict['lgth_time_series'] = 200
    script_dict['noise_magnitude'] = 1e-3
    script_dict['generate_overall'] = False
    script_dict['exp_name'] = exp_name
    script_dict['net_name'] = net_name
    script_dict['id_trial'] = None
    script_dict['random_seed'] = rs
    
    net_dict = compare_script_noisy(script_dict)

    return net_dict
    

