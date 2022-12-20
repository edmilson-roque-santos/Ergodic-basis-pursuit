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

from EBP import net_dyn
from EBP.base_polynomial import pre_settings as pre_set 

import lab_opto_electronic as lab_opto
import net_reconstr 

colors = ['darkgrey', 'orange', 'darkviolet', 'darkslategrey', 'silver']
folder_name = 'results'

#=============================================================================#
#Simulation fix a network and increase length of time series 
#=============================================================================#

def compare_script(opt_list, lgth_time_series, exp_name, net_name, id_trial):
    '''
    Script for basis choice comparison. 

    Parameters
    ----------
    opt_list : list of boolean
        Each entry determines which basis is selected. 
        Order: #canonical, normalize_cols, orthonormal
    lgth_time_series : float
        Length of time series.
    exp_name : str
        Filename.
    net_name: str
        Network structure filename.
        
    Returns
    -------
    dictionary result from greedy net algorithm.

    '''
    ############# Construct the parameters dictionary ##############
    parameters = dict()
    
    parameters['exp_name'] = exp_name
    parameters['Nseeds'] = 1
    
    parameters['network_name'] = net_name
    parameters['max_deg_monomials'] = 3
    parameters['expansion_crossed_terms'] = True
    
    parameters['use_kernel'] = True
    parameters['noisy_measurement'] = False
    parameters['use_canonical'] = opt_list[0]
    parameters['normalize_cols'] = opt_list[1]
    parameters['use_orthonormal'] = opt_list[2]
    parameters['length_of_time_series'] = lgth_time_series
    
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
    net_dynamics_dict['coupling'] = parameters['coupling']*net_dynamics_dict['max_degree']
    X_time_series = net_dyn.gen_net_dynamics(lgth_time_series, net_dynamics_dict)    
    #==========================================================#    
    
    X_t = X_time_series[:lgth_time_series,:]
    
    parameters['lower_bound'] = np.min(X_t)
    parameters['upper_bound'] = np.max(X_t)
    
    parameters['number_of_vertices'] = X_t.shape[1]
    
    parameters['X_time_series_data'] = X_t
    
    params = parameters.copy()
    
    if params['use_orthonormal']:
        output_orthnormfunc_filename = pre_set.create_orthnormfunc_filename(params)
    
        if not os.path.isfile(output_orthnormfunc_filename):
            params['orthnorm_func_filename'] = output_orthnormfunc_filename
            params['orthnormfunc'] = pre_set.create_orthnormfunc_kde(params)    

        if os.path.isfile(output_orthnormfunc_filename):
            params['orthnorm_func_filename'] = output_orthnormfunc_filename
                  
        params['build_from_reduced_basis'] = True
    
    params['cluster_list'] = [np.arange(0, params['number_of_vertices'], 1, dtype = int)]
    params['threshold_connect'] = 1e-8
    
    if id_trial != None:
        params['id_trial'] = id_trial
    
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
    out_results_direc = os.path.join(folder_name, net_name)
    out_results_direc = os.path.join(out_results_direc, exp_name)
    out_results_direc = os.path.join(out_results_direc, '')
    
    if os.path.isdir(out_results_direc ) == False:
        os.makedirs(out_results_direc)

    return out_results_direc

def compare_setup(exp_name, net_name, lgth_endpoints, save_full_info = False):
    '''
    
    Parameters
    ----------
    exp_name : TYPE
        DESCRIPTION.
    net_name : TYPE
        DESCRIPTION.
    lgth_endpoints : TYPE
        Start, end and space for length time vector.
    save_full_info : TYPE, optional
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
    filename = "lgth_endpoints_{}_{}_{}".format(lgth_endpoints[0], lgth_endpoints[1],
                                                lgth_endpoints[2]) 
    
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
                net_dict = compare_script(exp_params[key], lgth_time_series, exp_name, 
                                        net_name, None)
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

def compare_setup_critical_n(exp_name, net_name, size_endpoints, save_full_info = False):
    '''
    
    Parameters
    ----------
    exp_name : TYPE
        DESCRIPTION.
    net_name : TYPE
        DESCRIPTION.
    size_endpoints : TYPE
        Start, end and space for size vector.
    save_full_info : TYPE, optional
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
    
    size_vector = np.arange(size_endpoints[0], size_endpoints[1],
                                          size_endpoints[2], dtype = int)
    
    #Filename for output results
    out_results_direc = out_dir(net_name, exp_name)
    filename = "size_endpoints_{}_{}_{}".format(size_endpoints[0], size_endpoints[1],
                                                size_endpoints[2]) 
    
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
                
                fun
                
                net_dict = compare_script(exp_params[key], lgth_time_series, exp_name, 
                                        net_name, None)
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
#Lab Analysis
#=============================================================================#

def compare_basis(exp_dictionary, net_name):
    
    G_true = nx.read_edgelist("network_structure/{}.txt".format(net_name),
                        nodetype = int, create_using = nx.Graph)
    
    N = len(nx.nodes(G_true))
    A = nx.to_numpy_array(G_true, nodelist = list(range(N)))
    A = np.asarray(A)
    G_true = nx.from_numpy_array(A, create_using = nx.Graph)
    edges_G_true = list(G_true.edges())
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

def plot_comparison_analysis(ax, exp_dictionary, net_name, plot_legend):    
    
    lgth_vector, FP_comparison, FN_comparison, d_matrix = compare_basis(exp_dictionary, 
                                                                        net_name)
    
    lab_opto.plot_false_proportion(ax, lgth_vector, FP_comparison, plot_legend)
    ax.set_ylabel(r'FP')
    plt.setp(ax.get_xticklabels(), visible=True)
    
    #lab_opto.plot_false_proportion(ax[1], lgth_vector, FN_comparison, True)
    #ax[1].set_ylabel(r'FN')
    ax.set_xlabel(r'$n$')
    
    
def plot_lgth_dependence(net_name, exps_dictionary, title, filename = None):    
    
    
    
    keys = list(exps_dictionary.keys())
    n_cols = int(len(keys))
    
    fig_ = plt.figure(figsize = (11, 3), dpi = 300)
    subfigs = fig_.subfigures(1, 3, width_ratios = [0.9, 1.1, 1.1])
    
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
        
        plot_comparison_analysis(ax1, exp_dictionary, net_name,plot_legend)
        if plot_legend:
            plot_legend = False
        fig1.suptitle(title[id_col])
    
    fig_.suptitle('fig')
    if filename == None:
        plt.show()
    else:
     
        plt.savefig(filename+".pdf", format='pdf', bbox_inches='tight')
        
    return     

    
def ring_N_16(net_name = 'ring_graph_N=16'):
    
    lgths_endpoints = [[10, 101, 5],[10, 201, 5]]
    exps_name = ["gnr_logistc_compar_deg_2", "gnr_logistc_compar_deg_3"]
    title = ['b) deg 2', 'c) deg 3']
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
    





    