import autograd
import autograd.numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt

############ Differentials for DGM and EDRK ############



def get_S(n):
    S = np.zeros((n,n))
    I = np.eye(n)
    S[0:n//2,n//2:] = np.eye(n//2)
    S[n//2:,0:n//2] = -np.eye(n//2)
    return S


def DGH(x1, x2,H,DH):
    xh = (x1+x2)/2
    xd = x2-x1
    if (x2==x1).all():
        return DH(xh)
    else:
        return DH(xh) + (H(x2)-H(x1) - np.matmul(DH(xh),xd))/np.dot(xd,xd)*xd
    


########## RK4, MIRK4, MIMP4, DGM4 ##############

def rk4(f, x1, dt):
    k1 = f(x1)
    k2 = f(x1+.5*dt*k1)
    k3 = f(x1+.5*dt*k2)
    k4 = f(x1+dt*k3)
    return 1/6*(k1+2*k2+2*k3+k4)

def mirk4(f, x1, x2, dt, df=None, Df=None):
    if df == None:
        df = f
    xh = (x1+x2)/2
    z = xh - dt/8*(f(x2)-f(x1))
    return 1/6*(df(x1)+df(x2))+2/3*df(z)

def mirk2(f, x1, x2, dt, df=None, Df=None):
    if df == None:
        df = f
    xh = (x1+x2)/2
    return df(xh)


def mimp4(f, x1, x2, dt, df=None, Df=None): # Modified implicit midpoint method

    def Dff(x):
        return np.matmul(Df(x),f(x))
    def DfDf(x):
        return np.matmul(Df(x),Df(x))
    def DDff(x): # D^2 f (f, )
        return autograd.jacobian(Dff)(x) - DfDf(x)

    if df == None:
        df = f
    if Df == None:
        print('Error: Missing argument Df')
    xh = (x1+x2)/2
    return df(xh) + dt**2/12*(-np.matmul(Df(xh), np.matmul(Df(xh),df(xh))) + 1/2*np.matmul(DDff(xh),df(xh)))

def quasi_newton_dgm4(x, xn, dt,H, tol, max_iter,plot_energy_error=False):

    DH = autograd.grad(H)
    DDH = autograd.hessian(H)
    
    def DDGH(x1, x2): # The derivative of the discrete gradient w.r.t. the second argument
        g2 = lambda xn: DGH(x1, xn,H,DH)
        return autograd.jacobian(g2)(x2)
    
    S = get_S(x.shape[0])
    I = np.eye(x.shape[0])
    def St(x, xn, dt):
        xh = (x+xn)/2
        z1 = (xh+xn)/2
        z2 = (xh+x)/2
        SH = np.matmul(S,DDH(xh))
        SHS = np.matmul(SH,S)
        SHSHS = np.matmul(SH,SHS)
        DDGH1 = DDGH(x,1/3*x+2/3*xn) # Jacobian of discrete gradient DGH(x,1/3*x+2/3*xn)  w.r.t. (1/3*x+2/3*xn)
        Q1 = .5*(DDGH1.T-DDGH1)
        DDGH2 = DDGH(xn,1/3*xn+2/3*x)
        Q2 = .5*(DDGH2.T-DDGH2)
        Q = .5*(Q1-Q2)
        SQS = np.dot(S,np.matmul(Q,S))
        return (S + dt*SQS - 1/12*dt**2*SHSHS)

    F = lambda x_hat,S: 1/dt*(x_hat-x) - np.matmul(S, DGH(x,x_hat,H,DH))
    J = lambda x_hat,S: 1/dt*I - np.matmul(S, DDGH(x,x_hat))
   
    H_err = []
    it = 0


    S_xn = St(x,xn,dt)
    F_xn = F(xn,S_xn)
    err = la.norm(F_xn)
    while err > tol:
        xn = xn - la.solve(J(xn,S_xn),F_xn)
        S_xn = St(x,xn,dt)
        F_xn = F(xn,S_xn)
        err = la.norm(F_xn)
        H_err.append(np.linalg.norm(H(xn)-H(x)))
        it += 1
        if it > max_iter:
            break

    if plot_energy_error:
        plt.title("DGM4")
        plt.semilogy(H_err)

    return xn

def quasi_newton(integrator, x, xn,H, f, Df, dt, tol, max_iter,plot_energy_error=False):
    '''
    Integrating one step of the ODE x_t = f, from x to xn,
    with an integrator that we assume to be failry similar to
    the implicit midpoint rule
    Using a quasi-Newton method (i.e. with an approximated
    Jacobian that is exact for the implicit midpoint rule) to
    find xn
    '''

    I = np.eye(x.shape[0])
    F = lambda xn: 1/dt*(xn-x) - integrator(f, x, xn, dt, df=None, Df=Df)
    J = lambda xn: 1/dt*I - 1/2*integrator(f, x, xn, dt, df=Df, Df=Df)
    err = la.norm(F(xn))
    H_err = []
    it = 0

    F_xn = F(xn)
    while err > tol:
        xn = xn - la.solve(J(xn),F_xn)
        F_xn = F(xn)
        err = la.norm(F_xn)
        H_err.append(np.linalg.norm(H(xn)-H(x)))
        it += 1
        if it > max_iter:
            break

    if plot_energy_error:
        plt.semilogy(H_err)
        

    return xn
