"""
Collection of routines to treat and analyze 
the Opto electronic experimental data

Created on Wed Oct  6 14:41:16 2021

@author: Edmilson Roque dos Santos
"""

import networkx as nx
import numpy as np
import pandas as pd 
import sympy as spy
from scipy import stats

def get_Adj_matrix(net_file = 'symmetric_adj.txt'):
    folder = "data"+"/"+"opto_electronic_data"+"/"

    net_df = pd.read_csv(folder + net_file, sep = ',', header = None)
    A = net_df.to_numpy()
    G = nx.from_numpy_array(A)
    
    return A, G

def pre_treatment_data(filename, sigma_0 = "0.093750"):
    
    df = pd.read_fwf(filename, sep = "\t")
    
    col_name = dict()
    col_name['sigma = '+sigma_0] = "0"
    for id_ in range(1, 17):
        col_name["Unnamed: {}".format(id_)] = '{}'.format(id_ + 1) 
    
    df = df.rename(columns = col_name)
    df['0'], df['1'] = df['0'].str.split("\t", 1).str
    
    location_sigma = df[df["0"].str.contains("sigma")]

    return df, location_sigma

def get_exp_coupling(filename, outfilename,  sigma_0 = "0.093750"):
    '''
    Create a file with the data set for a specific coupling strength.

    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.
    outfilename : TYPE
        DESCRIPTION.
    sigma_0 : TYPE, optional
        DESCRIPTION. The default is "0.093750".

    Returns
    -------
    None.

    '''
    df, location_sigma = pre_treatment_data(filename, sigma_0)
    
    row_list = list(location_sigma.index)
    row_list.insert(0, -1)
    for id_row in range(len(row_list)):
        if(id_row < len(row_list) - 1):
            df_sigma = df.iloc[row_list[id_row] + 1: row_list[id_row + 1] - 1]
        else:
            df_sigma = df.iloc[row_list[id_row] + 1: - 1]
        df_sigma['0'] = df_sigma['0'].astype(float)
        df_sigma['1'] = df_sigma['1'].astype(float)
        
        col_name_= "1"
        last_col = df_sigma.pop(col_name_)
        df_sigma.insert(1, col_name_, last_col)
        
        if id_row == 0:
            np.savetxt(outfilename+'_N_symm_data_sigma = {}.txt'.format(sigma_0), df_sigma.values)
        else:
            np.savetxt(outfilename+'_N_symm_data_{}.txt'.format(df['0'][row_list[id_row]]), df_sigma.values)

def select_subset_phase_space(input_data, lower_bound, upper_bound, 
                              transient = 1000):
    '''
    Select multivariate time series that lies inside the subset 
    [lower_bound, upper_bound].

    Parameters
    ----------
    input_data : numpy array
        Optoelectronic multivariate time series.
    lower_bound : numpy float
        Interval subset Lower bound.
    upper_bound : numpy float
        Interval subset upper bound.
    transient : numpy float, optional
        Number of iterations to be discarded as transient. The default is 1000.

    Returns
    -------
    data : numpy array
        Multivariate time series lying inside the subset determined by
        [lower bound, upper_bound].

    '''
    X_time_series_data = input_data[transient:, :]
    mask_upper = X_time_series_data <= upper_bound
    mask_lower = X_time_series_data >= lower_bound
    
    mask = mask_lower & mask_upper
    block = np.all(mask, axis = 1)
    
    if np.any(block == False):
    
        end = np.where(block == False)[0]
        size = np.diff(end)
        max_loc = np.argmax(size)
        location = [end[max_loc], end[max_loc+1]]
    
        index = np.arange(0, X_time_series_data.shape[0], 1, dtype = int)
        data = X_time_series_data[index[location[0]+1:location[1]], :]

        return data

    else:
        return X_time_series_data

def cluster_moment_est(cluster_list, params):
    '''
    The assumption to estimate the density function for each cluster is that
    all nodes in a given cluster behaves similarly, hence we use all trajectories
    of those nodes to estimate the same density function.
    
    Parameters
    ----------
    cluster_list : list
        List of graph partition containing node list for each cluster.
    params : dict

    Returns
    -------
    parameters : dict
        Updated parameters dictionary to be used throughout the simulation.
        The updated arguments are: the calculation of density functions per cluster
        given by the cluster_list variable.

    '''
    
    parameters = dict()
    
    #If kernel density estimation is used, a data point must be given before hand
    if(params['use_kernel'] and params['use_integral_1d']):
        
        parameters['type_density'] = params.get('type_density', '1d_Kernel')
        parameters['density'] = params.get('density', None)
        if parameters['density'] == None:
            #Gather data points to be used on the kernel density estimator
            parameters['X_time_series_data'] = params.get('X_time_series_data', np.array([]))
            if len(parameters['X_time_series_data'] > 0):
                
                x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, params['number_of_vertices'])]
                num_clusters = len(cluster_list)
                id_vec = np.arange(0, params['number_of_vertices'], dtype = int)
                for id_cluster in range(num_clusters):
                    id_vec_cluster =  np.asarray(cluster_list[id_cluster], dtype = int)
                    mask_cluster = np.isin(id_vec, id_vec_cluster)
                    
                    X_t_cluster = params['X_time_series_data'][:, mask_cluster]
                    data_cluster = X_t_cluster.T.flatten()
                    kernel_cluster = stats.gaussian_kde(data_cluster, bw_method = 5e-2)
                    
                    for id_node in id_vec_cluster:
                        parameters[x_t[id_node]] = dict()
                        parameters[x_t[id_node]]['type_density'] = params.get('type_density', '1d_Kernel')
                        parameters[x_t[id_node]]['density'] = params.get('density', None)
        
                        #Lower and upper bound of the phase space of the isolated dynamics
                        parameters[x_t[id_node]]['lower_bound'] = params.get('lower_bound', np.min(data_cluster))
                        parameters[x_t[id_node]]['upper_bound'] = params.get('upper_bound', np.max(data_cluster))
                        parameters[x_t[id_node]]['density'] =  kernel_cluster#/kernel.integrate_box_1d(parameters[x_t[id_node]]['lower_bound'], parameters[x_t[id_node]]['upper_bound'])
                        parameters[x_t[id_node]]['density_normalization'] = kernel_cluster.integrate_box_1d(parameters[x_t[id_node]]['lower_bound'], parameters[x_t[id_node]]['upper_bound'])

    return parameters

def params_cluster(cluster_list, params):
    '''
    Update symbolic representation of the moments calculation accordingly 
    to cluster_list.

    Parameters
    ----------
    cluster_list : list
        List of graph partition containing node list for each cluster.
    params : dict

    Returns
    -------
    parameters : Updated
        Updated parameters dictionary to be used throughout the simulation.
        
    '''
    parameters = params.copy()
    params_cluster = cluster_moment_est(cluster_list, params)
    x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, params['number_of_vertices'])]
    for id_node in range(params['number_of_vertices']):
        parameters[x_t[id_node]] = params_cluster[x_t[id_node]]
    
    return parameters
