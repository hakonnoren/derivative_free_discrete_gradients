

import numpy as np

from methods.approx_derivative import *
from methods.dg_itoh_abe import *
from scipy.integrate import solve_ivp
import autograd

import matplotlib.pyplot as plt


 
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer modern",
    "font.size": 18,
    "ytick.labelsize": 12,
    "xtick.labelsize": 12,
})



def compute_finite_diff_errors(x0,Ns,T,f,H):

    metrics = {
            "S4": np.zeros((Ns.shape)),
            "leading_order_S":np.zeros((Ns.shape)),
            "F": np.zeros((Ns.shape)),
            "leading_order_F": np.zeros((Ns.shape)),
            "J": np.zeros((Ns.shape))
    }



    f0 = lambda t,x : f(x)


    n = x0.shape[0]
    S = np.zeros((n,n))
    I = np.eye(n)
    S[0:n//2,n//2:] = np.eye(n//2)
    S[n//2:,0:n//2] = -np.eye(n//2)
    Sm = S



    tau = 1e-5
    tau_hess = 1e-4
    num_tau = 1e-15
    dt = 0.1


    for i,N in enumerate(Ns):
        n = x0.shape[0]
        dt = T/N

        leading_order_F = (dt*(tau**2 + num_tau/tau) + dt**3*(tau_hess**2 + num_tau/tau_hess**2))
        leading_order_S = (tau**2 + num_tau/(tau))  + dt**2*(tau_hess**2 + num_tau/tau_hess**2)

        #computing x1
        t = [0,dt]
        x1 = solve_ivp(f0,t,x0,method='DOP853',rtol=1e-13,atol=1e-13).y.T[-1]



        D2 =  autograd.jacobian(lambda x : dg_sia(x0,x,H))(x1)
        D2_tau = D_sia(x0,x1,H,tau=tau,diag = True)

        S4 =  S_sia_exact(x0,x1,H,dt)
        S4_tau =  S_sia(x0,x1,H,dt,tau_hess=tau_hess,tau=tau)

        F = lambda x : x - x0 - dt*S_sia_exact(x0,x,H,dt)@dg_sia(x0,x,H)
        F_tau = lambda x : x - x0 - dt*S_sia(x0,x,H,dt,tau_hess=tau_hess,tau=tau)@dg_sia(x0,x,H)


        J_exact = autograd.jacobian(F)(x1)
        J_tau = I - dt*S4_tau@D2_tau


        metrics["S4"][i] = np.linalg.norm(S4_tau - S4)
        metrics["leading_order_S"][i] = leading_order_S
        metrics["F"][i] = np.linalg.norm(F_tau(x1)-F(x1))
        metrics["leading_order_F"][i] = leading_order_F
        metrics["J"][i] = np.linalg.norm(J_tau-J_exact)

    return metrics






def plot_finite_diff_errors(metrics,T,Ns,save=False):

    label_S = r"$\overline \epsilon^{\frac{2}{3}} + h^2\overline \epsilon^{\frac{1}{2}}$"
    label_F = r"$h\overline \epsilon^{\frac{2}{3}} + h^3\overline \epsilon^{\frac{1}{2}}$"

    labels = [r"$\Vert S_4^{\tau} - S_4 \Vert$",label_S,r"$\Vert F_{\tau}(x) - F(x) \Vert $",label_F,r"$\Vert F_{\tau}'(x) - F'(x) \Vert$"]

    lines = ["-","--","-","--","-"]

    keys = ["S4","leading_order_S"]
    plt.title(r"Convergence for $S_4^{\tau}$")
    for k,li,la in zip(keys,lines[0:2],labels[0:2]):
        plt.loglog(T/Ns,metrics[k],li,label=la)
    #plt.loglog(T/Ns,1e-6*(T/Ns)**(2),"--",label=r"$(h^2)$",c="gray",alpha=0.7)
    plt.legend()
    plt.xlabel(r"$h$")
    plt.ylim(1e-11,1e-7)

    if save:
        filename = "figures/convergence_S" + ".pdf"
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

    keys = ["F","leading_order_F","J"]
    plt.title(r"Convergence for $F_{\tau}$ and $F'_{\tau}$")
    for k,li,la in zip(keys,lines[2:],labels[2:]):
        plt.loglog(T/Ns,metrics[k],li,label=la)
    plt.loglog(T/Ns,5e0*(T/Ns)**(2),"--",label=r"$h^2$")
    #plt.loglog(T/Ns,1e-10*(T/Ns)**(1),"--",label="O1",c= "gray",alpha=0.5)
    plt.xlabel(r"$h$")
    plt.legend()#loc="center left")
    if save:
        filename = "figures/convergence_F"  + ".pdf"
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

