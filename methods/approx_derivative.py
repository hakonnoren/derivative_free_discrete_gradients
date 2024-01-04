
import numpy as np

import numpy as np




########## Helper function for Itoh-Abe #############

h = lambda x0,x1,i : x1[i] - x0[i]
Xm = lambda x0,x1,m : np.concatenate([x1[:m],x0[m:]])
DGm = lambda x0,x1,m,H : (H(Xm(x0,x1,m+1)) - H(Xm(x0,x1,m)))/(x1[m]-x0[m])

def e(i,n):
    e = np.zeros(n)
    e[i] = 1.
    return e

def df_mid(x,H,i,tau):
    n = x.shape[0]
    return 1/(2*tau)*(H(x + tau*e(i,n)) - H(x - tau*e(i,n)))

########## Approximated derivatives with finite differences """"""

def Xe(x0,x1,i):
    """ 
    Returns a canonical basis vector e_i multiplied by x1[i] - x0[i].
    This enables exchanging element x0[i] with x1[i] when doing finite differences.
    """
    h = x1[i] - x0[i]
    e = np.zeros(x0.shape[0])
    e[i] = h
    return e


def make_DDH(x,H,tau):

    """ 
    Second order approximation of the Hessian of H, evaluated
    in the midpoint xb = (x0+x1)/2
    """

    n = x.shape[0]
    S = np.array([[H(x + tau*e(j,n)) + H(x - tau*e(j,n)) for j in range(n)]])

    FD = -(S + S.T)
    for i in range(n):
        for j in range(i,n):
            Hij = H(x + tau*e(i,n) + tau*e(j,n)) + H(x - tau*e(i,n) - tau*e(j,n))
            FD[i,j] += Hij 
            if i != j:
                FD[j,i] += Hij
    DDH = (FD + 2*H(x))/(2*tau**2)
    return DDH


def make_DDH_e(x,H,tau):

    """ 
    Second order approximation of the Hessian of H, evaluated
    in the midpoint xb = (x0+x1)/2
    """


    def ddf_ii(x,H,i):
        return 1/(4*tau**2)*(H(x + 2*tau*e(i,n)) - 2*H(x) + H(x - 2*tau*e(i,n)))

    def ddf_ij(x,H,i,j):
        SW = H(x - tau*e(i,n) - tau*e(j,n))
        NW = H(x + tau*e(i,n) - tau*e(j,n))
        SE = H(x - tau*e(i,n) + tau*e(j,n))
        NE = H(x + tau*e(i,n) + tau*e(j,n))
        return 1/(4*tau**2)*(NE - SE - NW + SW)

    
    n = x.shape[0]
    DDH = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i == j:
                DDH[i,j] = ddf_ii(x,H,i)
            elif i < j:
                ddh = ddf_ij(x,H,i,j)
                DDH[i,j] = ddh
                DDH[j,i] = ddh
    return DDH


def D_ia(x0,x1,H,tau=1e-5):

    """
    Second order approximation of the derivative of the SIA DG evaluted in either 

    x1 (if diff_eval = "x1")
    (x0+2*x1)/3 (if diff_eval = "xt")
    """

    n = x0.shape[0]
    d_ia = np.zeros((n,n))

    for i in range(n):
        Xm1 = Xm(x0,x1,i)
        Xm2 = Xm(x0,x1,i+1)
        for j in range(n):
            if i > j:
                d_ia[i,j] = df_mid(Xm2,H,j,tau) - df_mid(Xm1,H,j,tau)
            elif i == j:
                d_ia[i,j] = df_mid(Xm2,H,j,tau) - DGm(x0,x1,i,H)
                
    dx = np.array([x1 - x0])
    return d_ia/dx.T

def D_sia(x0,x1,H,tau=1e-5,diag=False):

    """
    Second order approximation of the derivative of the SIA DG evaluted in either 

    x1 (if diff_eval = "x1")
    (x0+2*x1)/3 (if diff_eval = "xt")
    """

    def df_mid(x,H,i,tau):
        n = x.shape[0]
        return 1/(2*tau)*(H(x + tau*e(i,n)) - H(x - tau*e(i,n)))
    
    
    n = x0.shape[0]
    d_sia = np.zeros((n,n))

    for i in range(n):
        Xm1 = Xm(x0,x1,i)
        Xm2 = Xm(x0,x1,i+1)
        Xm3 = Xm(x1,x0,i)
        Xm4 = Xm(x1,x0,i+1)

        for j in range(n):
            if i > j:
                d_sia[i,j] = df_mid(Xm2,H,j,tau) - df_mid(Xm1,H,j,tau)
            elif j > i:
                d_sia[i,j] = df_mid(Xm3,H,j,tau) - df_mid(Xm4,H,j,tau)
            else:
                if diag:
                    d_sia[i,j] = df_mid(Xm2,H,j,tau) + df_mid(Xm3,H,j,tau) - (DGm(x0,x1,i,H)+ DGm(x1,x0,i,H))
    dx = np.array([x1 - x0])

    return d_sia/(2*dx.T)


def D_sia_fwd(x0,x1,H,tau=1e-8,diag=False):

    """
    Second order approximation of the derivative of the SIA DG evaluted in either 

    x1 (if diff_eval = "x1")
    (x0+2*x1)/3 (if diff_eval = "xt")
    """

    def df_mid(x,H,i,tau=1e-3):
        n = x.shape[0]
        return 1/(2*tau)*(H(x + tau*e(i,n)) - H(x - tau*e(i,n)))
    
    def df_fwd(x,H,i,tau=1e-8):
        n = x.shape[0]
        return H(x + tau*e(i,n))/tau
    
    n = x0.shape[0]
    d_sia = np.zeros((n,n))

    for i in range(n):
        Xm1 = Xm(x0,x1,i)
        Xm2 = Xm(x0,x1,i+1)
        Xm3 = Xm(x1,x0,i)
        Xm4 = Xm(x1,x0,i+1)
        HXm1,HXm2,HXm3,HXm4 = H(Xm1),H(Xm2),H(Xm3),H(Xm4)

        for j in range(n):
            if i > j:
                d_sia[i,j] = df_fwd(Xm2,H,j,tau) - df_fwd(Xm1,H,j,tau) - (HXm2 - HXm1)/tau
            elif j > i:
                d_sia[i,j] = df_fwd(Xm3,H,j,tau) - df_fwd(Xm4,H,j,tau) - (HXm3 - HXm4)/tau
            else:
                if diag:
                    d_sia[i,j] = df_fwd(Xm2,H,j,tau) + df_fwd(Xm3,H,j,tau) - (HXm2 + HXm3)/tau - (DGm(x0,x1,i,H)+DGm(x1,x0,i,H))
    dx = np.array([x1 - x0])

    return d_sia/(2*dx.T)