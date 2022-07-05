import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec    
import networkx as nx 
from scipy import stats
import matplotlib
import pandas as pd

def get_A_sym(net):
    net_df = pd.read_csv("Joe_data/{}.txt".format(net), sep = ',', header = None)
    A_sym = net_df.to_numpy()
    G = nx.from_numpy_array(A_sym)
    
    return A_sym, G


def plot_Artificial_Exp_comparison(X_time_series, X_time_series_data, parameters):
    
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['axes.linewidth'] = 0.4 #set the value globally
    #matplotlib.rcParams['axes.autolimit_mode'] = 'round_numbers'
    matplotlib.rcParams['axes.xmargin'] = 0
    matplotlib.rcParams['axes.ymargin'] = 0
    plt.rc('text', usetex = True)
    
    start_measurement = parameters['start_measurement']
    number_of_iterations = parameters['number_of_iterations']
    
    N = X_time_series.shape[1]
    nodelist = list(range(N))
    col = plt.cm.tab20b(np.linspace(0, 1, N))  
    
    fig, ax_f = plt.subplots(1, 2, sharey = True, figsize = (5,  4), dpi = 100)
    
    '''
    ax_f[0].plot(X_time_series[:number_of_iterations-1, :], X_time_series[1:number_of_iterations, :], 'o', markersize=5)
    ax_f[0].plot(X_time_series_data[start_measurement: start_measurement + number_of_iterations-1, :], X_time_series_data[1 + start_measurement:start_measurement+number_of_iterations, :], 'x', markersize=5)
    ax_f[0].set_title(r'Length Time series n = {} $\sigma = {}$'.format(parameters['number_of_iterations'], parameters['coupling']), fontsize = 15)
    ax_f[0].set_ylabel(r'$x(t + 1)$', fontsize = 15)
    ax_f[0].set_xlabel(r'$x$', fontsize = 15)
    '''
    real_data = X_time_series_data[start_measurement: start_measurement + number_of_iterations + 1, :]
    
    delta = np.absolute(real_data - X_time_series)
   
    time_length = np.arange(10, number_of_iterations, 5, dtype = int)
    error_over_time = np.zeros((time_length.shape[0], N))
    counter = 0
    for length in time_length:
        #error_over_time[counter, :] = np.sqrt(np.sum(delta[:length, :]**2, axis = 0))#/np.sqrt(length))
        error_over_time[counter, :] = delta[length, :]
        
        counter = counter + 1
    
    cluster_list = parameters['clusters_list']
    num_clusters = len(cluster_list)
    for id_cluster in range(num_clusters):
        id_vec_cluster =  np.asarray(cluster_list[id_cluster], dtype = int)
        
        for index in id_vec_cluster:    
    
        #for index in nodelist:
        #ax_f[0].plot(time_length, error_over_time[:, index], 
        #          label="{}".format(index),
        #          color = col[index])
            lower_bound = np.min(error_over_time[:, index])
            upper_bound = np.max(error_over_time[:, index])
            interval = np.arange(lower_bound, upper_bound, 0.001)
            kernel = stats.gaussian_kde(error_over_time[:, index], bw_method = 0.1)
            ax_f[id_cluster].plot(interval, 
                      kernel(interval)/kernel.integrate_box_1d(lower_bound, upper_bound),
                      label="{}".format(index),
                      color = col[index])              
            ax_f[id_cluster].legend(loc = 0, ncol= 1, fontsize = 8)        
            ax_f[id_cluster].set_xlabel(r"$\zeta$", fontsize = 15)
            print(error_over_time[:, index].mean())
    ax_f[0].set_ylabel(r"Density", fontsize = 15)
    
    #ax_f[1].hist(error_over_time[:, 0], bins = 30, density = True)
    #mean = np.mean(error_over_time[:, node])
    #ax_f[1].vlines(mean, 0, 2)
    #print(mean, np.mean(error_over_time))
    plt.legend(loc = 0)
    plt.tight_layout()
    
    if parameters['filename'] == None:
        plt.show()
    else:
        plt.savefig(parameters['filename']+".pdf", format = 'pdf')
    
def plot_Artificial_Density(X_time_series, parameters):
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['axes.linewidth'] = 0.4 #set the value globally
    #matplotlib.rcParams['axes.autolimit_mode'] = 'round_numbers'
    matplotlib.rcParams['axes.xmargin'] = 0
    matplotlib.rcParams['axes.ymargin'] = 0
    plt.rc('text', usetex = True)
    from matplotlib import cm

    number_of_iterations = X_time_series.shape[0]
    nodelist = parameters['nodelist']
    
    lower_bound = np.min(X_time_series)
    upper_bound = np.max(X_time_series)
    
    interval = np.arange(lower_bound, upper_bound, 0.001)
    
    fig, ax_f = plt.subplots(figsize = (6, 4), dpi = 100)
    ax_f.plot(X_time_series[:number_of_iterations-1, :], X_time_series[1:number_of_iterations, :], 'o', markersize=5)
    ax_c = ax_f.twinx()
    N = X_time_series.shape[1]
    
    nodelist = list(range(N))
    col = plt.cm.gist_yarg(np.linspace(0,1,N))  
    
    for index in nodelist:
        Opto_orbit = X_time_series[: number_of_iterations, index]
        kernel = stats.gaussian_kde(Opto_orbit, bw_method = 0.05)
        ax_c.plot(interval, 
                  kernel(interval)/kernel.integrate_box_1d(lower_bound, upper_bound), 
                  label="{}".format(index),
                  color = col[index])
    
    Opto_orbit = X_time_series.T.flatten()
    kernel = stats.gaussian_kde(Opto_orbit, bw_method = 5e-2)
    
    ax_c.plot(interval,
              kernel(interval)/kernel.integrate_box_1d(lower_bound, upper_bound), 
              'k--')
    
    ax_f.set_title(r'Optoelectronic $\sigma = {}$'.format(parameters['coupling']), fontsize = 20)
    ax_f.set_ylabel(r'$x(t + 1)$', fontsize = 16)
    ax_c.set_ylabel(r'$Density$', fontsize = 16)
    ax_f.set_xlabel(r'$x$', fontsize = 16)
    plt.legend(loc = 0)
    if parameters['filename'] == None:
        plt.show()
    else:
        plt.savefig(parameters['filename'])

def plot_traj_density(X_time_series, parameters):
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['axes.linewidth'] = 0.4 #set the value globally
    #matplotlib.rcParams['axes.autolimit_mode'] = 'round_numbers'
    matplotlib.rcParams['axes.xmargin'] = 0
    matplotlib.rcParams['axes.ymargin'] = 0
    plt.rc('text', usetex = True)
    from matplotlib import cm

    number_of_iterations = X_time_series.shape[0]
    nodelist = parameters['nodelist']
    
    lower_bound = np.min(X_time_series)
    upper_bound = np.max(X_time_series)
    
    interval = np.arange(lower_bound, upper_bound, 0.001)
    
    fig, ax_f = plt.subplots(2, 1, sharex=True, figsize = (6, 4), dpi = 100)
    #ax_f[0].plot(X_time_series[:number_of_iterations-1, :], X_time_series[1:number_of_iterations, :], 'o', markersize=5)
    
    N = X_time_series.shape[1]
    
    nodelist = list(range(N))
    col = plt.cm.tab20b(np.arange(N))  
    #np.linspace(0,1,N)
    for index in nodelist:
        Opto_orbit = X_time_series[: number_of_iterations, index]
        kernel = stats.gaussian_kde(Opto_orbit, bw_method = 0.05)
        ax_f[1].plot(interval, 
                  kernel(interval)/kernel.integrate_box_1d(lower_bound, upper_bound), 
                  label="{}".format(index),
                  color = col[index])

        ax_f[0].plot(X_time_series[:number_of_iterations-1, index], 
                     X_time_series[1:number_of_iterations, index], 
                     'o', 
                     color = col[index],
                     markersize=5)


    Opto_orbit = X_time_series.T.flatten()
    kernel = stats.gaussian_kde(Opto_orbit, bw_method = 5e-2)
    
    ax_f[1].plot(interval,
              kernel(interval)/kernel.integrate_box_1d(lower_bound, upper_bound), 
              'k--')
    
    fig.suptitle(r'Optoelectronic $\sigma = {}$'.format(parameters['coupling']), fontsize = 20)
    ax_f[0].set_ylabel(r'$x(t + 1)$', fontsize = 16)
    ax_f[1].set_ylabel(r'$Density$', fontsize = 16)
    ax_f[1].set_xlabel(r'$x$', fontsize = 16)
    l = parameters.get('lower_bound', lower_bound)
    u = parameters.get('upper_bound', upper_bound)
    ax_f[1].set_xlim(l, u)
    plt.legend(loc = 0, ncol=5, fontsize = 8)
    if parameters['filename'] == None:
        plt.show()
    else:
        plt.savefig(parameters['filename']+".pdf", format = 'pdf')


def plot_data_selection(X_time_series, ax, args):
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['axes.linewidth'] = 0.4 #set the value globally
    #matplotlib.rcParams['axes.autolimit_mode'] = 'round_numbers'
    matplotlib.rcParams['axes.xmargin'] = 0
    matplotlib.rcParams['axes.ymargin'] = 0
    plt.rc('text', usetex = True)

    number_of_iterations = X_time_series.shape[0]
    N = X_time_series.shape[1]
    nodelist = np.arange(N, dtype = int)
    
    lower_bound = np.min(X_time_series)
    upper_bound = np.max(X_time_series)
    interval = np.arange(lower_bound, upper_bound, 0.001)
       
    #========================================================================#
    #Fist column
    #========================================================================#
    col = plt.cm.tab20b(nodelist)  
    
    for index in nodelist:
        Opto_orbit = X_time_series[: number_of_iterations, index]
        kernel = stats.gaussian_kde(Opto_orbit, bw_method = 0.05)
        ax[1].plot(interval, 
                  kernel(interval)/kernel.integrate_box_1d(lower_bound, upper_bound), 
                  label="{}".format(index),
                  color = col[index])

        ax[0].plot(X_time_series[:number_of_iterations-1, index], 
                     X_time_series[1:number_of_iterations, index], 
                     'o', 
                     color = col[index],
                     markersize=5)
    '''
    Opto_orbit = X_time_series.T.flatten()
    kernel = stats.gaussian_kde(Opto_orbit, bw_method = 5e-2)
    
    ax[1].plot(interval,
              kernel(interval)/kernel.integrate_box_1d(lower_bound, upper_bound), 
              'k--')
    '''
    if args['plot_subset']:
        y = np.arange(lower_bound, upper_bound + 0.5)
        ax[0].fill_betweenx(y,
                        args['subset'][0]*np.ones(y.shape[0]),
                        args['subset'][1]*np.ones(y.shape[0]),
                        alpha = 0.7, 
                        color = 'silver') 
        ax[1].legend(loc = 0, ncol=5, fontsize = 8)
                    
    ax[0].set_ylabel(r'$x(t + 1)$', fontsize = 15)
    ax[1].set_ylabel(r'$\hat{\nu}(x)$', fontsize = 15)
        
    ax[1].set_xlabel(r'$x$', fontsize = 15)
    ax[1].set_xlim(lower_bound, upper_bound)

    #========================================================================#
    
    #if args['filename'] == None:
    #   plt.show()
    #else:
    #    plt.savefig(args['filename']+".pdf", format = 'pdf')

    return ax

def plot_panel(X_time_series, args):
    '''                
    fig, ax = plt.subplots(2, 2, figsize = (10, 4), dpi = 100)
    
    args['plot_subset'] = True
    args['subset'] = [3.4, 4.5]
    
    ax[:, 0] = plot_data_selection(X_time_series[args['start_measurement']:, :], ax[:, 0], args)    
    
    args['plot_subset'] = not True
    data = select_subset_phase_space(X_time_series,  args['subset'][0],  args['subset'][1])
    ax[:, 1] = plot_data_selection(data, ax[:, 1], args)    
    plt.savefig(args['filename']+".pdf", format = 'pdf')
    '''
    
    fig = plt.figure(figsize = (14, 4), dpi = 100)
    gs = GridSpec(nrows=2, ncols=3) #, height_ratios=[1, 1, 2])
    
    args['plot_subset'] = True
    args['subset'] = [3.4, 4.5]
    
    ax_0 = fig.add_subplot(gs[0, 0])
    ax_1 = fig.add_subplot(gs[1, 0])
    ax_0.set_title(args['title'][0], fontsize = 13)
    ax1 = plot_data_selection(X_time_series[args['start_measurement']: args['start_measurement'] + args['number_of_iterations'], :], [ax_0, ax_1], args)    
    
    
    args['plot_subset'] = not True
    data = select_subset_phase_space(X_time_series,  args['subset'][0],  args['subset'][1])
    
    ax_2 = fig.add_subplot(gs[0, 1])
    ax_2.set_title(args['title'][1], fontsize = 13)
    ax_3 = fig.add_subplot(gs[1, 1])
    ax2 = plot_data_selection(data, [ax_2, ax_3], args)  
    
    ax_4 = fig.add_subplot(gs[:, 2])
    ax3 = plot_clusterize_density(data, args['clusters_list'], ax_4)
    ax_4.set_title(args['title'][2], fontsize = 13)
    plt.tight_layout()
    plt.savefig(args['filename']+".pdf", format = 'pdf')
    return
    
def plot_cluster_traj_density(X_time_series, cluster_list, parameters):
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['axes.linewidth'] = 0.4 #set the value globally
    #matplotlib.rcParams['axes.autolimit_mode'] = 'round_numbers'
    matplotlib.rcParams['axes.xmargin'] = 0
    matplotlib.rcParams['axes.ymargin'] = 0
    plt.rc('text', usetex = True)
    from matplotlib import cm

    number_of_iterations = X_time_series.shape[0]
    nodelist = parameters['nodelist']
    
    lower_bound = np.min(X_time_series)
    upper_bound = np.max(X_time_series)
    
    interval = np.arange(lower_bound, upper_bound, 0.001)
    
    fig, ax_f = plt.subplots(2, 1, sharex=True, figsize = (6, 4), dpi = 100)
    #ax_f[0].plot(X_time_series[:number_of_iterations-1, :], X_time_series[1:number_of_iterations, :], 'o', markersize=5)
    
    N = X_time_series.shape[1]
    col = plt.cm.tab20b(np.arange(N))  
    
    
    num_clusters = len(cluster_list)
    col_cluster = ['tab:purple', 'tab:orange']
    
    nodelist = np.arange(0, N, dtype = int)
    
    for id_cluster in range(num_clusters):
        id_vec_cluster =  np.asarray(cluster_list[id_cluster], dtype = int)
        mask_cluster = np.isin(nodelist, id_vec_cluster)
        
        X_t_cluster = X_time_series[:, mask_cluster]
        data_cluster = X_t_cluster.T.flatten()
        kernel_cluster = stats.gaussian_kde(data_cluster, bw_method = 0.05)
        
        
        for index in id_vec_cluster:
            Opto_orbit = X_time_series[: number_of_iterations, index]
            kernel = stats.gaussian_kde(Opto_orbit, bw_method = 0.05)
            ax_f[1].plot(interval, 
                      kernel(interval)/kernel.integrate_box_1d(lower_bound, upper_bound), 
                      label="{}".format(index),
                      color = col[index],
                      alpha = 0.35)
    
            ax_f[0].plot(X_time_series[:number_of_iterations-1, index], 
                         X_time_series[1:number_of_iterations, index], 
                         'o', 
                         color = col[index],
                         markersize=5)
            
        ax_f[1].plot(interval,
              kernel_cluster(interval)/kernel_cluster.integrate_box_1d(lower_bound, upper_bound), 
              '--',
              color = col_cluster[id_cluster],
              alpha = 1.0)

    
    fig.suptitle(r'Optoelectronic $\sigma = {}$'.format(parameters['coupling']), fontsize = 20)
    ax_f[0].set_ylabel(r'$x(t + 1)$', fontsize = 16)
    ax_f[1].set_ylabel(r'$Density$', fontsize = 16)
    ax_f[1].set_xlabel(r'$x$', fontsize = 16)
    plt.legend(loc = 0, ncol=5, fontsize = 8)
    if parameters['filename'] == None:
        plt.show()
    else:
        plt.savefig(parameters['filename']+".pdf", format = 'pdf')

def plot_clusterize_density(X_time_series, cluster_list, ax):
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['axes.linewidth'] = 0.4 #set the value globally
    matplotlib.rcParams['axes.xmargin'] = 0
    matplotlib.rcParams['axes.ymargin'] = 0
    plt.rc('text', usetex = True)
    

    number_of_iterations = X_time_series.shape[0]
    N = X_time_series.shape[1]
    nodelist = np.arange(N, dtype = int)
    
    lower_bound = np.min(X_time_series)
    upper_bound = np.max(X_time_series)
    interval = np.arange(lower_bound, upper_bound, 0.001)
    
    
    col = plt.cm.tab20b(nodelist)  
    
    num_clusters = len(cluster_list)
    col_cluster = ['darkblue', 'brown']
    legend_ = ['Nodes 0 to 4', 'Nodes 5 to 16']
    nodelist = np.arange(0, N, dtype = int)
    
    for id_cluster in range(num_clusters):
        id_vec_cluster =  np.asarray(cluster_list[id_cluster], dtype = int)
        mask_cluster = np.isin(nodelist, id_vec_cluster)
        
        X_t_cluster = X_time_series[:, mask_cluster]
        data_cluster = X_t_cluster.T.flatten()
        kernel_cluster = stats.gaussian_kde(data_cluster, bw_method = 0.05)
        
        for index in id_vec_cluster:
            Opto_orbit = X_time_series[: number_of_iterations, index]
            kernel = stats.gaussian_kde(Opto_orbit, bw_method = 0.05)
            ax.plot(interval, 
                      kernel(interval)/kernel.integrate_box_1d(lower_bound, upper_bound),
                      color = col[index],
                      alpha = 0.25)
       
        ax.plot(interval,
              kernel_cluster(interval)/kernel_cluster.integrate_box_1d(lower_bound, upper_bound), 
              '--',
              label = legend_[id_cluster],
              color = col_cluster[id_cluster],
              alpha = 1.0)
    
    ax.set_ylabel(r'$\hat{\nu}(x)$', fontsize = 15)
    ax.set_xlabel(r'$x$', fontsize = 15)
    ax.legend(loc = 0, ncol=5, fontsize = 12)
    
    return ax    
        
        
def plot_Exp_Density(X_time_series_data, parameters):
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['axes.linewidth'] = 0.4 #set the value globally
    #matplotlib.rcParams['axes.autolimit_mode'] = 'round_numbers'
    matplotlib.rcParams['axes.xmargin'] = 0
    matplotlib.rcParams['axes.ymargin'] = 0
    plt.rc('text', usetex = True)
    
    start_measurement = parameters['start_measurement']
    number_of_iterations = parameters['number_of_iterations']
    nodelist = parameters['nodelist']
    
    lower_bound = np.max(X_time_series_data[start_measurement: start_measurement + number_of_iterations])
    upper_bound = np.min(X_time_series_data[start_measurement: start_measurement + number_of_iterations])
    
    interval = np.arange(0.0, upper_bound, 0.001)
    fig, ax_f = plt.subplots()
    ax_f.plot(X_time_series_data[start_measurement: start_measurement + number_of_iterations-1, :], X_time_series_data[1 + start_measurement:start_measurement+number_of_iterations, :], 's', markersize=5)

    ax_c = ax_f.twinx()
    for index in nodelist:
        Opto_orbit = X_time_series_data[start_measurement: start_measurement + number_of_iterations, index]
        kernel = stats.gaussian_kde(Opto_orbit)#, bw_method = 0.05)
        ax_c.plot(interval, kernel(interval), label="{}".format(index))
    
    ax_f.set_title(r'Optoelectronic $\sigma = {}$'.format(parameters['coupling']), fontsize = 20)
    ax_f.set_ylabel(r'$x(t + 1)$', fontsize = 20)
    ax_c.set_ylabel(r'$Density$', fontsize = 20)
    ax_f.set_xlabel(r'$x$', fontsize = 20)
    plt.legend()
    if parameters['filename'] == None:
        plt.show()
    else:
        plt.savefig(parameters['filename'])


