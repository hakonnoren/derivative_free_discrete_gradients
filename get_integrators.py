from methods.dg_itoh_abe import newton
from methods.numerical_integrators import rk4,quasi_newton_dgm4,quasi_newton,mirk4


def get_integrator_dict(H,f,Df,tol,tau):

    step_ia = lambda x,dt,incr : newton(x,dt,H,integrator="ia",incr=incr,tol=tol,tau=tau)
    step_ia_df = lambda x,dt,incr : newton(x,dt,H,integrator="ia_df",incr=incr,tol=tol,tau=tau)
    step_sia = lambda x,dt,incr : newton(x,dt,H,integrator="sia",incr=incr,tol=tol,tau=tau)
    step_sia_4 = lambda x,dt,incr : newton(x,dt,H,integrator="sia_4",incr=incr,tol=tol,tau=tau)
    step_sia_df = lambda x,dt,incr : newton(x,dt,H,integrator="sia_df",incr=incr,tol=tol,tau=tau)
    step_sia_4_exact = lambda x,dt,incr : newton(x,dt,H,integrator="sia_4_exact",incr=incr,tol=tol)
    step_rk4 = lambda x,dt,incr : x + dt*rk4(f, x, dt)
    step_mirk4 = lambda x,dt,incr : quasi_newton(mirk4, x, x,H, f, Df, dt, tol, 20)
    step_dgm4 = lambda x,dt,incr :quasi_newton_dgm4(x, x, dt,H, tol = tol, max_iter = 20)

    integrator_dict = {
            "IA" : step_ia,
            "IA DF" : step_ia_df,
            "SIA" : step_sia,
            "SIA DF" : step_sia_df,
            "SIA 4" : step_sia_4_exact,
            "SIA 4 DF" : step_sia_4,
            "RK4" : step_rk4,
            "MIRK4" : step_mirk4,
            "DGM4" : step_dgm4
    }

    return integrator_dict