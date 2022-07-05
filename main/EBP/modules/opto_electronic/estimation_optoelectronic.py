import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx 
import pandas as pd
from scipy import stats

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
    
    fig, ax_f = plt.subplots(dpi = 100)
    #ax_f.plot(X_time_series[:number_of_iterations-1, :], X_time_series[1:number_of_iterations, :], 'o', markersize=5)
    ax_f.plot(X_time_series_data[start_measurement: start_measurement + number_of_iterations-1, :], X_time_series_data[1 + start_measurement:start_measurement+number_of_iterations, :], 'x', markersize=5)
    ax_f.set_title(r'Length Time series n = {} $\sigma = {}$'.format(parameters['number_of_iterations'], parameters['coupling']), fontsize = 20)
    ax_f.set_ylabel(r'$x(t + 1)$', fontsize = 20)
    ax_f.set_xlabel(r'$x$', fontsize = 20)
    
    if parameters['filename'] == None:
        plt.show()
    else:
        plt.savefig(parameters['filename'])
    
def plot_Artificial_Density(X_time_series, parameters):
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['axes.linewidth'] = 0.4 #set the value globally
    #matplotlib.rcParams['axes.autolimit_mode'] = 'round_numbers'
    matplotlib.rcParams['axes.xmargin'] = 0
    matplotlib.rcParams['axes.ymargin'] = 0
    plt.rc('text', usetex = True)
    
    number_of_iterations = parameters['number_of_iterations']
    nodelist = parameters['nodelist']
    
    lower_bound = np.max(X_time_series)
    upper_bound = np.min(X_time_series)
    
    interval = np.arange(0.0, upper_bound, 0.001)
    
    fig, ax_f = plt.subplots()
    ax_f.plot(X_time_series[:number_of_iterations-1, :], X_time_series[1:number_of_iterations, :], 'o', markersize=5)
    ax_c = ax_f.twinx()
    nodelist = list(range(X_time_series.shape[1]))
    for index in nodelist:
        Opto_orbit = X_time_series[: number_of_iterations, index]
        kernel = stats.gaussian_kde(Opto_orbit, bw_method = 0.05)
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

def plot_traj_dens(X_time_series, parameters):
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
    plt.legend(loc = 0, ncol=5, fontsize = 8)
    if parameters['filename'] == None:
        plt.show()
    else:
        plt.savefig(parameters['filename']+".pdf", format = 'pdf')

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
