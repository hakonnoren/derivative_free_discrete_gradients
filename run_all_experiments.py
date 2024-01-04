import numpy as np

############  Finite difference convergence #################


from experiments.finite_diff_convergence import compute_finite_diff_errors,plot_finite_diff_errors
from experiments.order import H_dp as H
from experiments.order import f_dp as f

T = 1
Ns = 2**np.arange(1,10,1)
x0 = np.array([0.1,0.2,0.25,-0.3])

metrics = compute_finite_diff_errors(x0,Ns,T,f,H)
plot_finite_diff_errors(metrics,T,Ns,save=True)



################ Testing order, energy preservation and computational time ###############


from get_integrators import get_integrator_dict
from experiments.order import run_experiment,plot_results

tol = 1e-11
tau = 1e-5


####### Double pendulum ######


from experiments.order import Df_dp as Df
from experiments.order import H_dp as H
from experiments.order import f_dp as f

integrator_dict = get_integrator_dict(H,f,Df,tol,tau)


x0 = np.array([0.1,0.2,0.25,-0.3])
T = 10
Ns = T*(2**np.arange(2,7))
metrics = run_experiment(integrator_dict,H,Ns,x0,T=T)
plot_results(metrics,integrator_dict,f,x0,T,Ns,save="double_pendulum")


####### Lennard Jones Oscillator ######


from experiments.order import Df_ljo as Df
from experiments.order import H_ljo as H
from experiments.order import f_ljo as f


integrator_dict = get_integrator_dict(H,f,Df,tol,tau)

x0 = np.array([1.21,0.34])
T = 20
Ns = T*(2**np.arange(2,7))
metrics = run_experiment(integrator_dict,H,Ns,x0,T=T)
plot_results(metrics,integrator_dict,f,x0,T,Ns,save="ljo")


################ Topological Hamiltonian ###############

from experiments.topo_hamiltonian import get_topo_potential,integrate_topo,plot_topo_trajectory

N = 50_000
y0 = np.array([-0.1,0.2,0.,0.])
dt = 0.02


lat,long = 61.22186498817666, 10.524947049598786
H_topo,n = get_topo_potential(lat,long,plot = False)
ys = integrate_topo(H_topo,y0,dt,N,method = "sia_df")
plot_topo_trajectory(ys,H_topo,n,dt,N,save=True)