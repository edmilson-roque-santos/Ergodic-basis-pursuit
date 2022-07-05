"""
Script for profiling the PNR algorithm in comparison to using all nodes 
in the reconstruction.

Created on Wed Oct - 2021/22

@author: Edmilson Roque dos Santos
"""
import os
import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt

from nanoprofiler import Profiler

from EBP import tools, net_dyn
from EBP.base_fourier import pre_settings as pre_set
from EBP.base_fourier import greedy_algorithms as gnr_alg


# Set plotting parameters
params_plot = {'axes.labelsize': 16,
              'axes.titlesize': 18,
              'axes.linewidth': 1.0,
              'axes.xmargin':0, 
              'axes.ymargin': 0,
              'legend.fontsize': 16,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'figure.figsize': (10, 5),
              'figure.titlesize': 18,
              'font.serif': 'Computer Modern Serif',
              'mathtext.fontset': 'cm',
              'axes.linewidth': 1.0
             }

plt.rcParams.update(params_plot)
plt.rc('text', usetex=True)

def I_function(x, delta):
    
    return np.sin(x + delta)**2

def opto_dynamics(beta, alpha, delta, A, number_of_iterations):
    #==========================================================#
    net_dynamics_dict = dict()
    net_dynamics_dict['adj_matrix'] = A
    degree = A.sum(axis = 0)
    
    net_dynamics_dict['f'] = lambda x: (beta - degree*alpha)*I_function(x, delta)
    net_dynamics_dict['h'] = lambda x: (A.T @ I_function(x, delta))
    net_dynamics_dict['max_degree'] = 1
    net_dynamics_dict['eps'] = 0.1
    net_dynamics_dict['coupling'] = alpha*net_dynamics_dict['max_degree']
    X_time_series = net_dyn.gen_net_dynamics(number_of_iterations,
                                             net_dynamics_dict,
                                             use_noise = False)    

    return X_time_series

#==========================================================#
#==========================================================#
def main(prop_clusters, lg_ts, exp_name = "PNR_profiling_motif_optoelec"):    
    '''
    Method to be computed the performance.

    Parameters
    ----------
    prop_clusters : array
        Properties of the Random toy modular net
        kappa = prop_cluster[0] 
        nclusters = prop_cluster[1].
    lg_ts : float
        DESCRIPTION.
    exp_name : str, optional
        DESCRIPTION. The default is "PNR_profiling_motif_optoelec".

    Returns
    -------
    None.

    '''
    ############# Construct the parameters dictionary ##############
    parameters = dict()
    parameters['random_seed'] = 1
    parameters['Nseeds'] = 1
    
    lgth_ts = lg_ts
    net_name = 'kappa={}_ncluster={}_lgth_ts={}'.format(prop_clusters[0], 
                                                        prop_clusters[1],
                                                        lgth_ts)
    
    parameters['network_name'] = net_name
    A = tools.random_toy_net_model(random_seed = parameters['random_seed'],
                                   num_of_cluster=prop_clusters[0], 
                                   N_nodes_cluster = prop_clusters[1],
                                   mean_degree_in_cluster = 4, 
                                   mean_degree_intercluster = 4,
                                   filename = None, save_net = False, 
                                   plot_net = False)
    A = np.asarray(A)
    N = A.shape[0]
    
    #Generate artificially the optoelectronic data
    args = dict()
    args['beta'], args['alpha'], args['delta'] = 4.5, 0.05, np.pi/4
    X_time_series = opto_dynamics(args['beta'], args['alpha'], args['delta'], A,
                                  lgth_ts)
    
    X_t = X_time_series
        
    parameters['exp_name'] = exp_name
    parameters['number_of_vertices'] = X_t.shape[1]
    parameters['length_of_time_series'] = X_t.shape[0]
    parameters['X_time_series_data'] = X_t
    parameters['coupling'] = args['alpha']
    
    parameters['normalize_cols'] = False
    parameters['noisy_measurement'] = False
    parameters['noise_magnitude'] = 0.01
    parameters['single_density'] = True
    parameters['use_kernel'] = True
    parameters['use_orthonormal'] = True
    parameters['use_canonical'] = False
    parameters['threshold_connect'] = 1e-8
    
    #Sequence of degrees to be present in the Fourier expansion
    # 1, sin x, cos x, sin 2x, cos 2x, sin 3x, cos 3x
    parameters['deg_array'] = np.array([1, 2]) #np.arange(1, 3, 1, dtype = int)     
    parameters['max_deg_harmonics'] = np.max(parameters['deg_array'])
    
    if parameters['use_orthonormal']:
        parameters['save_orthnormfunc'] = True
        parameters['orthnorm_func_filename'] = pre_set.create_orthnormfunc_filename(parameters)
    
    parameters['cluster_list'] = np.arange(0, N, 1, dtype = int).reshape(prop_clusters[0], 
                                                                         prop_clusters[1])
    
    
    params_ = parameters.copy()
    
    threshold_connect = 1e-8
    tolerance = 1e-8
    fixed_search_set = True
    min_noise = 1e-13
    relaxing_path = np.array([min_noise,1.1*min_noise ])#np.linspace(1e-10, 0.3, 40)
    select_criterion = 'crit_3'
    solver_optimization = cp.ECOS
    
    #start_time = time.time()
    
    gr_alg = gnr_alg.GR_algorithm(X_t, parameters['cluster_list'], params_, 
                     tolerance, threshold_connect, fixed_search_set, 
                     relaxing_path, select_criterion, solver_optimization)
    A_ = gr_alg['A']
    A_[np.absolute(A_) > 0] = 1
    mask = A == A_
    if np.all(mask):
        print('Successful reconstruction!')
        
    #end_time = time.time()
    #print((end_time - start_time)/60, '\n')

def profiling_pnr(prop_clusters, lg_ts, pre_fix_name, opt_name):
    pr = Profiler()
    local_run = []
    for comb in range(prop_clusters.shape[0]):
        exp_name = "exec{}-N{}".format(comb, opt_name)
        pr.start(name = exp_name)
        main(prop_clusters[comb], lg_ts)
        pr.stop()
        local_run.append(exp_name)
    #filename = pre_fix_name
    #pr.save_results("PNR_profiling_motif_optoelec"+"/"+"profiling", filename)

    df_results = pr.results()
    run_time = np.zeros(prop_clusters.shape[0])
    counter = 0
    for key_ in local_run:
        mask_loc = df_results[key_]['function'] == 'GR_algorithm'
        
        mask = (mask_loc)
        df_loc = (df_results[key_][mask])
        run_time[counter] = np.max(df_loc['cumtime'].values)

        counter = counter + 1
    
    del pr 
    
    return run_time, df_results

def running_time_call(kappa_start, kappa_end, N_nodes, lg_ts, 
                      exp_name = "PNR_profiling_motif_optoelec",
                      dir_name = 'profiling'):
    
    pre_fix_name = 'cvx_N_{}_ks{}_kf{}'.format(N_nodes, kappa_start, kappa_end)
        
    kappa_ = np.arange(kappa_start, kappa_end, 1, dtype = int)
    kappa_values = []
    for k in kappa_:
        if np.mod(N_nodes, k) == 0:
            kappa_values.append(k)
    kappa_values = np.array(kappa_values)
        
    N_clusters = np.array(N_nodes/kappa_values, dtype = int)
    prop_clusters = np.dstack((kappa_values, N_clusters))[0]
    run_opt = dict()
    keys = '{}'.format(N_nodes)
    run_opt[keys], df = profiling_pnr(prop_clusters,
                                      lg_ts,
                                      pre_fix_name, 
                                      keys)
    
       
    df_run = pd.DataFrame.from_dict(run_opt)
    df_run['kappa_values'] = kappa_values
    df_run = df_run.set_index('kappa_values')
    
    out_direc = os.path.join(exp_name, dir_name)
    if os.path.isdir(out_direc) == False:
        os.makedirs(out_direc)
    scenario_output = os.path.join(out_direc,'')
    
    df_run.to_csv(scenario_output+pre_fix_name+'.csv')

def plot_opt(ax, pre_fix_name, exp_dictionary,
             exp_name = "PNR_profiling_motif_optoelec", 
             dir_name = 'profiling'):
    
    out_direc = os.path.join(exp_name, dir_name)
    scenario_output = os.path.join(out_direc,'')
    df_plot = pd.read_csv(scenario_output+pre_fix_name+'.csv')
    
    kappas = df_plot['kappa_values'].values
    pos_1 = np.where(kappas == 1)[0]
    time = df_plot['{}'.format(exp_dictionary['N_nodes'])].values
    time = time/time[pos_1]
        
    ax.plot(kappas, time, 'o-', color=exp_dictionary['col'], 
            label=r"$N = {}$".format(exp_dictionary['N_nodes']))
    
    if exp_dictionary['theory']:
        #Theory
        kappas_ = np.arange(exp_dictionary['kappa_start'], np.max(kappas), 0.01)
        
        m_N = lambda x: 4*x + 1
        
        #numerator = (2*exp_dictionary['N_nodes']/kappas_ + 1)*exp_dictionary['N_nodes'] \
        #    +(2*kappas_ + 1)*kappas_
        #denominator = ((2*exp_dictionary['N_nodes'] + 1)*exp_dictionary['N_nodes'])
        
        numerator = m_N(exp_dictionary['N_nodes']/kappas_)*exp_dictionary['N_nodes']\
            + m_N(kappas_)*kappas_
        denominator = (m_N(exp_dictionary['N_nodes'])*exp_dictionary['N_nodes'])
        R_kappa = numerator/denominator
        
        
        ax.plot(kappas_, R_kappa, '--', label=r'estimate $N = {}$'.format(exp_dictionary['N_nodes']), 
                color = 'darkgray')
        
    ax.set_xlabel(r'$\kappa$')
    ax.set_ylabel(r'$\varrho_N(\kappa)$')
    ax.set_ylim(0.0, 1.0) 
        
    ax.legend(loc=0)
    
    return ax

def plot_comparison(exp_name = "PNR_profiling_motif_optoelec", dir_name='profiling',
                    filename = None):
    
    kappa_list = np.array([100, 200, 500])
    boolean_vec = np.array([False]*kappa_list.shape[0])
    boolean_vec[-1] = True
    
    fig, ax = plt.subplots(1, 1, figsize = (5, 5), dpi = 300)
    
    ax1 = ax#[1]
    
    colors = plt.cm.Dark2(np.arange(kappa_list.shape[0]))  
    exp_dictionary = dict()
    for id_kappa in range(kappa_list.shape[0]):
        exp_ = dict()
        exp_['kappa_start'], exp_['kappa_end'], exp_['N_nodes'] = 1, \
                                    kappa_list[id_kappa], kappa_list[id_kappa]
        
        exp_['col'] = colors[id_kappa]
        exp_['theory'] = boolean_vec[id_kappa]
        pre_fix_name = 'cvx_N_{}_ks{}_kf{}'.format(exp_['N_nodes'], exp_['kappa_start'],
                                                   exp_['kappa_end'])
        plot_opt(ax1, pre_fix_name, exp_, exp_name, dir_name)
        
        exp_dictionary[id_kappa] = dict()
        exp_dictionary[id_kappa] = exp_.copy()
    ax1.set_xlim(0, 50)
    #ax1.set_title(r'b)', loc= 'left')
    '''
    ax2 = inset_axes(ax, width="100%", height="100%",
                    bbox_to_anchor=(.5, .57, .45, .5),
                    bbox_transform=ax.transAxes)

    #ax2 = ax[0]
    A = tools.random_toy_net_model(random_seed = 1,
                                   num_of_cluster=5, 
                                   N_nodes_cluster = 20,
                                   mean_degree_in_cluster = 4, 
                                   mean_degree_intercluster = 4,
                                   filename = None, save_net = False, 
                                   plot_net = False)
    A = np.asarray(A)
    ax2.matshow(A, cmap=plt.cm.Greys)
    ax2.set_ylabel(r'node index')
    ax2.set_xlabel(r'node index')
    #ax2.set_title(r'a)', loc = 'left')
    '''
    outfolder = 'Figures'
    if filename == None:
        filename = outfolder+'/'+'rel_perf'
    
    #plt.tight_layout()
    plt.savefig(filename+".pdf", format='pdf', bbox_inches='tight')

run = False#True
plot = True#False

if run:    
    running_time_call(1, 100, 100, 300, dir_name='profiling_deg_2')
if plot:
    plot_comparison(dir_name='profiling_deg_2')    
