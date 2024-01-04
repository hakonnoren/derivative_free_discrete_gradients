from autograd import numpy as np 
import autograd
from .approx_derivative import D_ia,D_sia,make_DDH
from matplotlib import pyplot as plt



########## Helper function for Itoh-Abe #############

h = lambda x0,x1,i : x1[i] - x0[i]
Xm = lambda x0,x1,m : np.concatenate([x1[:m],x0[m:]])
DGm = lambda x0,x1,m,H : (H(Xm(x0,x1,m+1)) - H(Xm(x0,x1,m)))/(x1[m]-x0[m])

############ Itoh-Abe and its symmetrized version #########


def dg_ia(x0,x1,H):
    """ 
    Itoh-Abe (IA) first order gradient free method
    """
    n = x0.shape[0]
    dg = np.array([DGm(x0,x1,i,H) for i in range(n)])
    return dg

def dg_sia(x0,x1,H):
    """ 
    Symmetrized Itoh-Abe (SIA) second order gradient free method
    """
    n = x0.shape[0]
    dg = np.array([.5*(DGm(x0,x1,i,H) + DGm(x1,x0,i,H)) for i in range(n)])
    return dg


###### Timestepping using (quasi-)Newton iterations #########


def e(i,n):
    e = np.zeros(n)
    e[i] = 1.
    return e

def newton(x0,dt,H,integrator,incr = None,tol=1e-10,tau=1e-5,plot_energy_error = False,dist_measure=False):
    n = x0.shape[0]
    max_iter = 20

    #Canonical structur matrix S
    S = np.zeros((n,n))
    I = np.eye(n)
    S[0:n//2,n//2:] = np.eye(n//2)
    S[n//2:,0:n//2] = -np.eye(n//2)
    Sc = S

    #First step in a trajectory initialized using random perturbation
    #Next sttau initialize with the previous increment, incr = x_i - x_{i-1}
    if type(incr) == np.ndarray:
        x1 = x0 + incr
    else:
        x1 = x0 + dt*np.random.randn(x0.shape[0])

    dg = None
    jac_dg = None


    if integrator[0:2] == "ia":
        S = lambda x1 : Sc
        dg = dg_ia
        jac_dg = jac_ia

        if integrator == "ia_df":
            jac_dg = lambda x0,x1,H : D_ia(x0,x1,H,tau=tau)

    elif integrator[0:3] == "sia":
        dg = dg_sia
        S = lambda x1 : Sc
        jac_dg = jac_sia

        if integrator == "sia_df":
            jac_dg = lambda x0,x1,H : D_sia(x0,x1,H,tau=tau,diag=True)
        
        elif integrator == "sia_4":
            S = lambda x1 : S_sia(x0,x1,H,dt,tau=tau)
            jac_dg = lambda x0,x1,H : D_sia(x0,x1,H,tau=tau,diag=True)

        elif integrator == "sia_4_exact":
            jac_dg = jac_sia
            S = lambda x1 : S_sia_exact(x0,x1,H,dt)


    F = lambda x1,S: x1 - x0 - dt*S@dg(x0,x1,H)
    #if integrator == "sia_4":
    #    jac = lambda x1 : np.array([F(x1 + tau*e(i,n)).T - F(x1 - tau*e(i,n)).T for i in range(n)]).T/(2*tau)
    #else:
    jac = lambda x1,S: I - dt*S@jac_dg(x0,x1,H)



    F_exact = lambda x1 : x1 - x0 - dt*S_sia_exact(x0,x1,H,dt)@dg(x0,x1,H) 
    S_exact = lambda x1 : S_sia_exact(x0,x1,H,dt)

    #Starting iterations
    err = np.inf
    it = 0
    H_err = []
    F_errs = []
    x_true_errs = []
    ferr = []
    S_tau = []

    if dist_measure:
        x_true = newton(x0,dt,H,"sia_4_exact",incr,tol,tau)

    S_x1 = S(x1)
    F_x1 = F(x1,S_x1)
    while err > tol:

        
        x1 = x1 - np.linalg.solve(jac(x1,S_x1),F_x1)
        S_x1 = S(x1)
        F_x1 = F(x1,S_x1)
        H_err.append(np.linalg.norm(H(x1)-H(x0)))
        err = np.linalg.norm(F_x1)
        F_errs.append(err)

        if dist_measure:
            x_true_errs.append(np.linalg.norm(x1-x_true))
            ferr.append(np.linalg.norm(F(x1) - F_exact(x1)))
            S_tau.append(np.linalg.norm(S(x1) - S_exact(x1)))

        it += 1
        if it > max_iter:
            break

    #print(integrator)
    #print(it)

    if plot_energy_error:
        plt.title(integrator)
        plt.semilogy(H_err)
        plt.semilogy(F_errs,'--')
        if dist_measure:
            plt.semilogy(ferr,'o')
            plt.semilogy(S_tau,'x')
            plt.semilogy(x_true_errs)

    return x1


####### Exact derivatives using autograd #########

def jac_ia(x0,x1,H):
    """ 
    Autograd jacobian of the IA DG method
    """
    dg = lambda x : dg_ia(x0,x,H)
    return autograd.jacobian(dg)(x1)

def jac_sia(x0,x1,H):
    """ 
    Autograd jacobian of the SIA DG method
    """
    dg = lambda x : dg_sia(x0,x,H)
    return autograd.jacobian(dg)(x1)

def hessian(H):
    return autograd.hessian(H)


###### Construction of structure matrices S bar #########

def S_sia(x0,x1,H,dt,tau=1e-5,tau_hess=1e-4):
    """ 
    Fourth order approximation to the structure matrix S bar
    for the SIA DG method
    """
 
    n = x0.shape[0]
    S = np.zeros((n,n))
    S[0:n//2,n//2:] = np.eye(n//2)
    S[n//2:,0:n//2] = -np.eye(n//2)

    hess = make_DDH((x0+x1)/2,H,tau=tau_hess)
    Dsia_1 = D_sia(x0,(x0+2*x1)/3,H,tau=tau)
    Dsia_2 = D_sia(x1,(2*x0+x1)/3,H,tau=tau)
    
    Q = .5*(Dsia_1.T - Dsia_1 - Dsia_2.T + Dsia_2)

    S_bar = S + .5*dt*S@Q@S - 1/12*dt**2*S@hess@S@hess@S

    for i in range(n):
        S_bar[i,i] = 0

    return S_bar


def S_sia_exact(x0,x1,H,dt):
    """ 
    Structure matrix S bar using autograd to obtain derivatives
    for the SIA DG method
    """

    DDH = hessian(H)

    n = x0.shape[0]
    S = np.zeros((n,n))
    S[0:n//2,n//2:] = np.eye(n//2)
    S[n//2:,0:n//2] = -np.eye(n//2)

    dg = lambda x : dg_sia(x0,x,H)
    jac = autograd.jacobian(dg)

    hess_1 = jac((x0+2*x1)/3)
    dg = lambda x : dg_sia(x1,x,H)
    jac = autograd.jacobian(dg)
    hess_2 = jac((2*x0+x1)/3)
    
    Q = .5*(hess_1.T - hess_1 - hess_2.T + hess_2)

    S_bar = S + .5*dt*S@Q@S - 1/12*dt**2*S@DDH((x0+x1)/2)@S@DDH((x0+x1)/2)@S
    return S_bar


