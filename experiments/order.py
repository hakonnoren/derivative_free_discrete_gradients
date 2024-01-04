import autograd.numpy as np
import autograd
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer modern",
    "font.size": 15,
    "ytick.labelsize": 12,
    "xtick.labelsize": 12,
})

global c
c = 0

############ Defining Lennard Jones Oscillator (ljo)  #############

def H_ljo(x):
    global c
    c +=1 
    return x[1]**2 + .25*(1/(x[0]**12) - 2/(x[0]**6))

DH_ljo = autograd.grad(H_ljo)
DDH_ljo = autograd.hessian(H_ljo)


def f_ljo(x):
    n = x.shape[0]
    S = np.zeros((n,n))
    S[0:n//2,n//2:] = np.eye(n//2)
    S[n//2:,0:n//2] = -np.eye(n//2)

    return np.matmul(S,DH_ljo(x))

Df_ljo = autograd.jacobian(f_ljo)


############ Defining double pendulum problem #############
def H_dp(x):
    global c
    c +=1 
    return (1/2*x[2]**2 + x[3]**2 - x[2]*x[3]*np.cos(x[0]-x[1]))/(1 + np.sin(x[0]-x[1])**2) - 2*np.cos(x[0]) - np.cos(x[1])


DH_dp = autograd.grad(H_dp)
DDH_dp = autograd.hessian(H_dp)


def f_dp(x):
    n = x.shape[0]
    S = np.zeros((n,n))
    S[0:n//2,n//2:] = np.eye(n//2)
    S[n//2:,0:n//2] = -np.eye(n//2)

    return np.matmul(S,DH_dp(x))

Df_dp = autograd.jacobian(f_dp)




def integrate(stepper,H,x,dt,N,i,int_idx,metrics):
    x_list_i = [x]
    H_list = []

    x0 = x
    incr = None
    global c
    c = 0
    t = time.perf_counter()
    for j in range(N):
        x1 = stepper(x0,dt,incr)
        H_list.append(H(x1))
        x_list_i.append(x1)
        incr = x1 - x0
        x0 = x1
    t = time.perf_counter() - t
    plt.show()
    metrics["x_end"][:,i,int_idx] = x0
    metrics["t_comp"][i,int_idx] = t
    metrics["c_eval"][i,int_idx] = c - N
    metrics["H_err"][i][int_idx] = H_list
    metrics["x_list"][i][int_idx] = x_list_i


def run_experiment(stepper_dict,H,Ns,x0,T):
    n_methods = len(stepper_dict)

    metrics = {
            "x_end": np.zeros((x0.shape[0],Ns.shape[0],n_methods)),
            "t_comp": np.zeros((Ns.shape[0],n_methods)),
            "c_eval": np.zeros((Ns.shape[0],n_methods)),
            "H_err": [[[H(x0)]]*n_methods]*Ns.shape[0],
            "x_list": [[[]]*n_methods]*Ns.shape[0]
    }

    for i,N in enumerate(Ns):
        dt = T/N
        for j,stepper in enumerate(stepper_dict.values()):
            integrate(stepper,H,x0,dt,N,i,j,metrics)

    metrics['x_list'] = np.array([metrics['x_list'][-1][i] for i in range(n_methods)])

    return metrics


def plot_results(metrics,stepper_dict,f,x0,T,Ns,save=False):


    marker = "v"
    marker_DF = "x"
    line_style_DF = "--"
    linestyle = "-"
    

    x_end,t_comp,H_err,x_list,c_eval = metrics["x_end"],metrics["t_comp"],metrics["H_err"],metrics["x_list"],metrics["c_eval"]
    n_methods = len(stepper_dict)
    ncol = n_methods//3 + n_methods%3
    labels = list(stepper_dict.keys())

    f0 = lambda t,x : f(x)
    ts_arr = np.linspace(0,T,Ns[-1]+1)

    x_ref = solve_ivp(f0,[0,T],x0,method='DOP853',t_eval=ts_arr,rtol=1e-13,atol=1e-13).y.T

    err = np.zeros_like(x_end[0,:,:])

    
    for i in range(err.shape[0]):
        for j in range(err.shape[1]):
            err[i,j] = np.linalg.norm(x_end[:,i,j] - x_ref[-1])

    Hs_last = metrics['H_err'][-1]

    energies = np.zeros((len(Hs_last[0]),n_methods))
    for k in range(len(Hs_last)):
        energies[:,k] = np.abs(np.array(Hs_last[k]) - Hs_last[k][0])

    

    energy_err = np.zeros((len(metrics['H_err']),n_methods))
    for i in range(len(metrics['H_err'])):
        for k in range(len(metrics['H_err'][i])):
            energy_err[i,k] = np.mean(np.abs(np.array(metrics['H_err'][i][k]) - metrics['H_err'][i][k][0]))


    
    plt.title("Convergence of error")
    for i,e in enumerate(err.T):
        if labels[i][-2:] == "DF":
            plt.loglog(T/Ns,e,line_style_DF,label=labels[i],marker=marker_DF)
        else:
            plt.loglog(T/Ns,e,label=labels[i],marker=marker)
    plt.loglog(T/Ns,8e-1*(T/Ns)**(1),"--",c= "gray",alpha=0.5)
    plt.loglog(T/Ns,8e-1*(T/Ns)**(2),"--",c= "gray",alpha=0.5)
    #plt.loglog(T/Ns,1e-1*(T/Ns)**(3),"--",c= "gray",alpha=0.5)
    plt.loglog(T/Ns,6e-1*(T/Ns)**(4),"--",c= "gray",alpha=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncol=ncol)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncol=ncol)
    #plt.tight_layout()
    #plt.show()
    plt.xlabel(r"Step size $h$")
    plt.ylabel(r"$\|x_N - x(T)\|$")

    if save:
        filename = "figures/convergence_" + save + ".pdf"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    plt.title(r"$L_2$ Error")
    for i in range(n_methods):
        x_int_i = np.array(x_list)[i,:,:]
        if labels[i][-2:] == "DF":
           plt.semilogy(ts_arr,np.linalg.norm(x_int_i - x_ref,axis=1),line_style_DF,label=labels[i])
        else:
            plt.semilogy(ts_arr,np.linalg.norm(x_int_i - x_ref,axis=1),label=labels[i])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncol=ncol)
    plt.xlabel(r"$t_n$")
    plt.ylabel(r"$\|x_n - x(t_n)\|$")

    if save:
        filename = "figures/l2error_" + save + ".pdf"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    plt.title("Computational time")
    for i,e in enumerate(t_comp.T):
        if labels[i][-2:] == "DF":
            plt.loglog(T/Ns,e,line_style_DF,label=labels[i],marker=marker_DF)
        else:
            plt.loglog(T/Ns,e,label=labels[i],marker=marker)
    #plt.loglog(T/Ns,t_comp,label=labels)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncol=ncol)
    plt.xlabel(r"Step size $h$")

    if save:
        filename = "figures/computational_time_" + save + ".pdf"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


    def plot_work_precision(complexity_var,error_var,label_x,label_y,title,filename):

        plt.title(title)
        for i,(t,e) in enumerate(zip(complexity_var.T,error_var.T)):

            if labels[i][-2:] == "DF":
                plt.loglog(t,e,line_style_DF,label=labels[i],marker=marker_DF)
            else:
                plt.loglog(t,e,label=labels[i],marker=marker)

        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncol=ncol)

        plt.xlabel(label_x)
        plt.ylabel(label_y)

        if save:
            dir = "figures/"
            plt.savefig(dir + filename + save + ".pdf", bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    plot_work_precision(t_comp,err,r"Time",r"$\|x_N - x(T)\|$",r"Work-precision ($L_2$, time)","work_precision_l2_t")
    plot_work_precision(c_eval,err,r"Num. evals.",r"$\|x_N - x(T)\|$",r"Work-precision ($L_2$, num. evals.)","work_precision_l2_evals_")
    plot_work_precision(t_comp,energy_err,r"Time",r"$|H(x_n) - H(x_0)|$",r"Work-precision (energy, time)","work_precision_energy_t")
    plot_work_precision(c_eval,energy_err,r"Num. evals.",r"$|H(x_n) - H(x_0)|$",r"Work-precision (energy, num. evals.)","work_precision_energy_evals_")



    plt.title(r"$H(x)$ evaluation count")
    for i,e in enumerate(c_eval.T):
        if labels[i][-2:] == "DF":
            plt.loglog(T/Ns,e,line_style_DF,label=labels[i],marker=marker_DF)
        else:
            plt.loglog(T/Ns,e,label=labels[i],marker=marker)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncol=ncol)
    plt.xlabel(r"Num. evals.")
    plt.xlabel(r"Step size $h$")

    if save:
        filename = "figures/eval_count_" + save + ".pdf"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()



    plt.title("Preservation of energy ")


    #plt.semilogy(ts_arr[:-1],energies,label=labels)
    for i,e in enumerate(energies.T):
        if labels[i][-2:] == "DF":
            plt.semilogy(ts_arr[:-1],e,line_style_DF,label=labels[i])
        else:
            plt.semilogy(ts_arr[:-1],e,label=labels[i])
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncol=ncol)
    plt.xlabel(r"$t_n$")
    plt.ylabel(r"$|H(x_n) - H(x_0)|$")
    if save:
        filename = "figures/energy_" + save + ".pdf"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()