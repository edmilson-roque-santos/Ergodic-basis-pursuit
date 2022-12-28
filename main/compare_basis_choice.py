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
        os.makedirs(out_results_direc)
        
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
    parameters['expansion_crossed_terms'] = True
    
    parameters['use_kernel'] = True
    parameters['noisy_measurement'] = False
    parameters['use_canonical'] = script_dict['opt_list'][0]
    parameters['normalize_cols'] = script_dict['opt_list'][1]
    parameters['use_orthonormal'] = script_dict['opt_list'][2]
    parameters['length_of_time_series'] = script_dict['lgth_time_series']
    
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
    net_dynamics_dict['h'] = lambda x: (x**1)*(A.T @ x**1)
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
    
    if os.path.isdir(out_results_direc ) == False:
        os.makedirs(out_results_direc)

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
    
def determine_critical_n(exp_param, size, exp_name, net_class, id_trial = None, 
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

    Returns
    -------
    n_critical : float
        minimum length of time series.

    '''
    net_name = net_class+"_{}".format(size)
    #tools.star_graph(size, 'network_structure/'+net_name)
    tools.ring_graph(size, 'network_structure/'+net_name)
    
    lgth_time_series_vector = np.arange(5, 3*size**2, 5, dtype = int)
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
        
        net_dict = compare_script(script_dict)
        FP, FN = quick_comparison(net_dict, net_name)
        print('FP, FN:', FP, FN)
        print('A', net_dict['A'][:, 0])
        if (FP == 0) and (FN == 0):
            find_critical = False
            print('Net Recovered!')
        id_ = id_ + 1
    
    n_critical = lgth_time_series
    return n_critical

def compare_setup_critical_n(exp_name, net_class, size_endpoints, id_trial,
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
    out_results_direc = out_dir(net_class, exp_name)
        
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
                                                  net_class, id_trial, random_seed)
                
                out_results_hdf5[key][size] = dict()
                out_results_hdf5[key][size]['n_critical'] = n_critical
                
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
    
    exp_vec = exp_dictionary['exp_params'].keys()
    lgth_endpoints = exp_dictionary['lgth_endpoints']
    
    lgth_vector = np.arange(lgth_endpoints['0'], lgth_endpoints['1'],
                                      lgth_endpoints['2'], dtype = int)
    
    FP_comparison = np.zeros((len(exp_vec), lgth_vector.shape[0]))
    FN_comparison = np.zeros((len(exp_vec), lgth_vector.shape[0]))
    d_matrix = np.zeros((len(exp_vec), lgth_vector.shape[0]))

    for exp_ in exp_vec:
        for id_key in range(len(lgth_vector)):
            key = lgth_vector[id_key]
            A_est = exp_dictionary[exp_][key]['A']
            
            G_est = nx.from_numpy_array(A_est, create_using = nx.Graph)
            links = lab_opto.links_types(G_est, G_true)        
    
            intersec_weights, false_pos_weights = links['intersec_weights'],\
                links['false_pos_weights']
            if len(intersec_weights) > 0 or len(false_pos_weights) > 0:
                d_matrix[exp_, id_key] = lab_opto.interval_intersec(false_pos_weights,
                                                                    intersec_weights)
            else:
                d_matrix[exp_, id_key] = np.nan
                print("Failed:", exp_, id_key)
            
            total_connections = 0.5*N*(N-1)
            FP = len(links['false_positives'])/(total_connections-len(edges_G_true))
            FP_comparison[exp_, id_key] = FP
            
            FN = len(links['false_negatives'])/len(edges_G_true)
            FN_comparison[exp_, id_key] = FN
            
    return lgth_vector, FP_comparison, FN_comparison, d_matrix

def compare_basis_net_size(exp_dictionary, net_class):
    '''
    Given a experiment dict, it calculates the performance of the reconstruction.

    Parameters
    ----------
    exp_dictionary : dict
        Output results dictionary.
    net_class : str
        Filename.

    Returns
    -------
    size_vector : numpy array 
        Array with length of time series vector.
    n_critical_comparison : numpy array
        n_critical for each length of time series.
    
    '''
    exp_vec = exp_dictionary['exp_params'].keys()
    size_endpoints = exp_dictionary['size_endpoints']
    
    size_vector = np.arange(size_endpoints['0'], size_endpoints['1'],
                                      size_endpoints['2'], dtype = int)
    
    n_critical_comparison = np.zeros((len(exp_vec), size_vector.shape[0]))
    
    for exp_ in exp_vec:
        for id_key in range(len(size_vector)):
            key = size_vector[id_key]
            n_critical = exp_dictionary[exp_][key]['n_critical']
            n_critical_comparison[exp_, id_key] = n_critical
            
            
    return size_vector, n_critical_comparison

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

def ax_plot_star_graph(ax, plot_net_alone=False):
    
    N = 10
    G_true = nx.star_graph(N, create_using=nx.Graph())
    pos_true = nx.spring_layout(G_true)
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
    
    nx.draw_networkx_nodes(G_true, pos = pos_true,
                            ax = ax,
                            nodelist=[0],
                            node_color = colors[3], 
                            node_size = 200,
                            alpha = 1.0)
        
    nx.draw_networkx_edges(G_true,pos = pos_true, 
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

def plot_comparison_analysis(ax, exp_dictionary, net_name, plot_legend):    
    
    lgth_vector, FP_comparison, FN_comparison, d_matrix = compare_basis(exp_dictionary, 
                                                                        net_name)
    
    lab_opto.plot_false_proportion(ax, lgth_vector, FP_comparison, plot_legend)
    ax.set_ylabel(r'FP')
    plt.setp(ax.get_xticklabels(), visible=True)
    
    #lab_opto.plot_false_proportion(ax[1], lgth_vector, FN_comparison, True)
    #ax[1].set_ylabel(r'FN')
    ax.set_xlabel(r'$n$')
    
def plot_comparison_n_critical(ax, exp_dictionary, net_class, plot_legend):    
    
    size_vector, n_critical_comparison = compare_basis_net_size(exp_dictionary, 
                                                                        net_class)
    
    lab_opto.plot_false_proportion(ax, size_vector, n_critical_comparison, plot_legend)
    ax.set_ylabel(r'$n_c$')
    plt.setp(ax.get_xticklabels(), visible=True)
    
    #lab_opto.plot_false_proportion(ax[1], lgth_vector, FN_comparison, True)
    #ax[1].set_ylabel(r'FN')
    ax.set_xlabel(r'$N$')
    
def plot_lgth_dependence(net_name, exps_dictionary, title, filename = None):    
    
    
    
    keys = list(exps_dictionary.keys())
    n_cols = int(len(keys))
    
    fig_ = plt.figure(figsize = (6, 3), dpi = 300)
    subfigs = fig_.subfigures(1, 2, width_ratios = [0.9, 1.1])
    
    fig = subfigs[0]
    
    gs = GridSpec(nrows=1, ncols=1, figure=fig)
    
    ax_0 = fig.add_subplot(gs[0])
    
    G_true = nx.read_edgelist("network_structure/{}.txt".format(net_name),
                        nodetype = int, create_using = nx.Graph)
    N = len(nx.nodes(G_true))
    A = nx.to_numpy_array(G_true, nodelist = list(range(N)))
    A = np.asarray(A)
    G_true = nx.from_numpy_array(A, create_using = nx.Graph)
    pos_true = nx.circular_layout(G_true)
    
    ax_plot_true_net(ax_0, G_true, pos_true, probed_node = 0, 
                     print_probed = False, plot_net_alone = False)
    
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

def plot_n_c_size(exps_dictionary, title, filename = None):    
    
    net_class = 'star_graph'
    
    keys = list(exps_dictionary.keys())
    n_cols = int(len(keys))
    
    fig_ = plt.figure(figsize = (6, 3), dpi = 300)
    subfigs = fig_.subfigures(1, 2, width_ratios = [0.9, 1.1])
    
    fig = subfigs[0]
    
    gs = GridSpec(nrows=1, ncols=1, figure=fig)
    
    ax_0 = fig.add_subplot(gs[0])
    
    ax_plot_star_graph(ax_0)
    
    fig.suptitle(r'a) Original Network') 
    plot_legend = True
    for id_col in range(n_cols):
        fig1 = subfigs[id_col+1]
        
        gs1 = GridSpec(nrows=1, ncols=1, figure=fig1)
        exp_dictionary = exps_dictionary[keys[id_col]]
        ax1 = fig1.add_subplot(gs1[0])
        #ax2 = fig1.add_subplot(gs1[1])
        
        plot_comparison_n_critical(ax1, exp_dictionary, net_class, plot_legend)
        if plot_legend:
            plot_legend = False
        fig1.suptitle(title[id_col])
    
    fig_.suptitle('fig')
    if filename == None:
        plt.show()
    else:
     
        plt.savefig(filename+".pdf", format='pdf', bbox_inches='tight')
        
    return     
  
#=============================================================================#
#Scripts
#=============================================================================#
def ring_N_16(net_name = 'ring_graph_N=16'):
    
    lgths_endpoints = [[10, 201, 5]]
    #exps_name = ["gnr_logistc_compar_deg_2", "gnr_logistc_compar_deg_3"]
    #title = ['b) deg 2', 'c) deg 3']
    exps_name = ["gnr_logistic_comp_pair_deg_3"]
    title = ['b) deg 3']
    exps_dictionary = dict()
    
    for id_exp in range(len(exps_name)):
        lgth_endpoints = lgths_endpoints[id_exp]

        exp_name = exps_name[id_exp]
        out_results_direc = os.path.join(folder_name, net_name)
        out_results_direc = os.path.join(out_results_direc, exp_name)
        out_results_direc = os.path.join(out_results_direc, '')
        
        if os.path.isdir(out_results_direc ) == False:
            print("Failed to find the desired result folder !")
            
        filename = "lgth_endpoints_{}_{}_{}".format(lgth_endpoints[0], lgth_endpoints[1],
                                                    lgth_endpoints[2]) 
        
        if os.path.isfile(out_results_direc+filename+".hdf5"):
            out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'r')
            exp_dictionary = out_results_hdf5.to_dict()  
            out_results_hdf5.close()
        exps_dictionary[id_exp] = exp_dictionary

    return exps_dictionary, title
def ring_net_plot_script():
    exps_dictionary, title = ring_N_16(net_name = 'ring_graph_N=16')
    plot_lgth_dependence('ring_graph_N=16', exps_dictionary, title, filename = None)

def star_graph(exps_name, size_endpoints, net_class = 'star_graph'):
    title = ['b) deg 3']
    exps_dictionary = dict()
    
    for id_exp in range(len(exps_name)):
        size_endpoints = size_endpoints[id_exp]

        exp_name = exps_name[id_exp]
        out_results_direc = os.path.join(folder_name, net_class)
        out_results_direc = os.path.join(out_results_direc, exp_name)
        out_results_direc = os.path.join(out_results_direc, '')
        
        if os.path.isdir(out_results_direc ) == False:
            print("Failed to find the desired result folder !")
            
        filename = "size_endpoints_{}_{}_{}".format(size_endpoints[0], size_endpoints[1],
                                                    size_endpoints[2]) 
        
        if os.path.isfile(out_results_direc+filename+".hdf5"):
            out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'r')
            exp_dictionary = out_results_hdf5.to_dict()  
            out_results_hdf5.close()
        exps_dictionary[id_exp] = exp_dictionary

    return exps_dictionary, title  

def ring_graph_lgth_script(rs):
    exp_name = 'logistic_lgth_3_99_0_001_N'
    net_name = 'ring_graph_N=16'
    lgth_endpoints = [10, 201, 5]
    compare_setup(exp_name, net_name, lgth_endpoints, random_seed = rs, 
                      save_full_info = False)

def star_graph_plot_script():
    exps_name = ['growing_net_deg_3_3_99_0_001_N']
    size_endpoints = [[3, 51, 5]]
    exps_dictionary, title = star_graph(exps_name, size_endpoints, net_class = 'star_graph')
    plot_n_c_size(exps_dictionary, title, filename = None)

    
def ring_graph_script(rs):
    exp_name = 'growing_net_deg_3_3_99_0_001_N'
    net_class = 'ring_graph'
    size_endpoints = [3, 51, 5]
    id_trial = None #np.array([0])
    compare_setup_critical_n(exp_name, net_class, size_endpoints, id_trial, 
                             random_seed = rs, save_full_info = False)


def test_plot_script():
    exps_name = ['test']
    size_endpoints = [[3, 51, 5]]
    exps_dictionary, title = star_graph(exps_name, size_endpoints, net_class = 'star_graph')
    plot_n_c_size(exps_dictionary, title, filename = None)

    
def test_script():
    exp_name = 'test_ring_2'
    net_class = 'ring_graph'
    size_endpoints = [3, 51, 5]
    id_trial = None #np.array([0])
    compare_setup_critical_n(exp_name, net_class, size_endpoints, id_trial,save_full_info = False)    