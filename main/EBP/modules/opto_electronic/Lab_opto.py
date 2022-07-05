import numpy as np 
import core_experimental as cexp
import EstimationKDE_Optoelectronic  as est_opto


def data_treatment():

    folder = "Joe_data"
    filename = folder+"/"+"IntertwinedNetwork_optimized_data.txt"    
    outfilename = folder+"/"+"optimized_data"+"/"
    est_opto.get_exp_coupling(filename, outfilename)
    #df,  location_sigma = est_opto.pre_treatment_data(filename)
    
    #df = pd.read_fwf(filename, sep = "\t")



parameters = dict()
parameters['beta'] = 4.5

A_sym, G = est_opto.get_A_sym("symmetric_adj")
parameters['Adj_matrix'] = A_sym
parameters['nodelist'] = np.arange(A_sym.shape[0], dtype = int)
parameters['number_of_iterations'] = 5000

parameters['use_IC_exp'] = True
parameters['use_IC_random'] = False

parameters['start_measurement'] = 1000

parameters['filename'] = None#"noise_est_traj"#None #"OptoE_optmz_traj_dens" #


folder = "Joe_data"
#filename = folder+"/"+"IntertwinedNetwork_optimized_data.txt"    
#outfilename = folder+"/"+"optimized_data"+"/"

filename = folder+"/"+"IntertwinedNetwork_optimized_data.txt"    
outfilename = folder+"/"+"symmetric_data"+"/"

#coupling_vec = np.arange(0.093750, 1.093750 + 0.015625, 0.0156250)
coupling_vec = np.arange(0.0156250, 1.093750 + 0.015625, 0.0156250)
float_formatter = "{:.6f}".format
for id_sig in range(11, 12):#
    parameters['coupling'] = coupling_vec[id_sig]

    if parameters['filename'] != None:
        parameters['filename'] = parameters['filename']+"_sig_{}".format(float_formatter(parameters['coupling']))


    X_time_series_data = np.loadtxt(outfilename+"N_symm_data_sigma = {}.txt".format(float_formatter(parameters['coupling']))) 
    parameters['IC_data'] = X_time_series_data[parameters['start_measurement'],:]
    
    parameters['clusters_list'] = [np.arange(0, 5, 1, dtype = int), np.arange(5, 17, 1, dtype = int)]
    #parameters['cluster_list'] = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16]]
    
    
    '''
    Generate the true network dynamics data to evaluate what type of clustering we might find.
    '''
    X_time_series = cexp.generate_map_iteration_Mach_Zehnder_map(parameters)
    data = X_time_series#est_opto.select_subset_phase_space(X_time_series, 3.4, 4.5)
    '''
    # Noise estimation
    X_time_series = cexp.generate_map_iteration_Mach_Zehnder_map(parameters)
    est_opto.plot_Artificial_Exp_comparison(X_time_series, X_time_series_data, parameters)
    #data = est_opto.select_subset_phase_space(X_time_series_data, 3.4, 4.5)
    #print(coupling_vec[id_sig], data.shape[0])
    '''
    
    
    #Subset selection
    #data = est_opto.select_subset_phase_space(X_time_series_data, 3.4, 4.5)
    #parameters['lower_bound'] = 3.4
    #parameters['upper_bound'] = 4.5
    #est_opto.plot_traj_density(X_time_series_data[parameters['start_measurement']:parameters['number_of_iterations'], :6], parameters)
    est_opto.plot_traj_density(data, parameters)
    '''
    cluster_list = [np.arange(0, 5, 1, dtype = int), np.arange(5, 17, 1, dtype = int)]
    #est_opto.plot_cluster_traj_density(data, cluster_list, parameters)
    
    print(coupling_vec[id_sig], data.shape[0])
    
    
    args = dict()
    args['start_measurement'] = 1000
    args['number_of_iterations'] = 3000
    args['filename'] = 'data_selection_{}'.format(float_formatter(parameters['coupling']))
    args['clusters_list'] = [np.arange(0, 5, 1, dtype = int), np.arange(5, 17, 1, dtype = int)]
    
    args['title'] = ['Subset selection', 'Parabolic shape', 'Clusterizing']
    
    est_opto.plot_panel(X_time_series_data, args)
    '''
    
    
    
    
    
    
    
    
    
    
    
    
    



