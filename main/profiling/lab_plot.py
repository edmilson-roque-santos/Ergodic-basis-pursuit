'''
To plot Execution time as function of degree.
'''
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend',fontsize=14)
import numpy as np 
from scipy import optimize



time_execution = np.loadtxt("time_exec_N_3_D_4.txt")
time_execution_1 = np.loadtxt("time_exec_N_3_D_5.txt")

max_degree_poly = np.arange(1, 6, 1, dtype = int)

time_exec_data = np.zeros((max_degree_poly.shape[0]))
time_exec_data[:4] = time_execution
time_exec_data[4] = time_execution_1

yerr = 1.0
xdata = max_degree_poly
ydata = time_exec_data/60
logy = np.log(ydata)
logyerr = yerr / ydata

# define our (line) fitting function
fitfunc = lambda p, x: p[0] + p[1] * x
errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err

pinit = [1, -1.0]
out = optimize.leastsq(errfunc, pinit,
                       args=(xdata, logy, 1.0), full_output=1)

pfinal = out[0]
covar = out[1]
print(pfinal)

index = pfinal[1]
amp = np.exp(pfinal[0])
print(amp, index)

indexErr = np.sqrt( covar[1][1] )
ampErr = np.sqrt( covar[0][0] ) * amp

x = np.arange(1, 5, 1e-3)
fitting_result = amp*np.exp(index*x)

fit = 1e-4*np.exp(3*max_degree_poly)

fig, ax = plt.subplots(dpi = 100)

ax.semilogy(max_degree_poly, ydata, 'kx')
ax.semilogy(x, fitting_result, 'r--', label = r'$T \propto e^{3 d}$')

ax.set_xlabel(r'Degree $d$')
ax.set_xticks(max_degree_poly) 
ax.set_xticklabels(max_degree_poly) 
ax.set_ylabel(r'Execution time (min) ')
ax.set_title('Polynomial basis')
ax.legend(loc = 0)
plt.tight_layout()
plt.savefig("exec_time_poly.pdf", format='pdf')
