import pygmt
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from methods.dg_itoh_abe import newton

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer modern",
    "font.size": 15,
    "ytick.labelsize": 12,
    "xtick.labelsize": 12,
})



def get_topo_potential(N,E,plot = False):

    p = 1
    r = [E-p, E+p, N-p, N+p]
    topography_array = pygmt.datasets.load_earth_relief(resolution="01m", region=r).to_numpy()
    n = topography_array.shape[0]
    topography_array = np.flip(topography_array/topography_array.max(),axis=0)
    if plot:
        plt.imshow(topography_array)

    x = np.linspace(-1,1,n)
    y = np.linspace(-1,1,n)

    f = RegularGridInterpolator((x, y),topography_array, method='cubic',bounds_error=False,fill_value=0.)
    
    if plot:
        x = np.linspace(-1,1,4*n)
        y = np.linspace(-1,1,4*n)
        xx, yy = np.meshgrid(x, y,indexing='ij')
        int = f((xx,yy))
        plt.imshow(int)

    
    def H_topo(y,sum = True):

        K = (y[0]**2 + y[1]**2)*0.5
        U_top = f((y[2], y[3]))
        U_cube = (y[2]**2 + y[3]**2)*0.5
        if sum:
            return K+U_top+U_cube
        else:
            return K, U_top, U_cube
        
    return H_topo,n


def integrate_topo(H_topo,y0,dt,N,method = "sia_df"):
    ys = [y0]
    incr = None
    for i in range(N):
        if np.abs(y0[1::2]).max() > 1.:
            break
        
        y1 = newton(y0,dt,H_topo,integrator=method,incr=incr,tol=1e-7,tau=1e-5)
        incr = y1 - y0
        ys.append(y1)
        y0 = y1
    return np.array(ys)





def plot_topo_trajectory(ys,H_topo,n,dt,N,save=False):
    x = np.linspace(-1,1,2*n)
    y = np.linspace(-1,1,2*n)
    xx, yy = np.meshgrid(x, y,indexing='ij')

    K, U_top, U_cube = H_topo([0,0,xx,yy],sum=False)
    U_tot = U_top + U_cube


    ts = np.linspace(0,dt*N,N)
    plt.figure(figsize=(10,4))
    plt.title("Energy drift")
    plt.plot(ts,H_topo(ys[1:].T) - H_topo(ys[0]))
    plt.ylabel(r"$H(q_0,p_0) - H(q_n,p_n)$")
    plt.xlabel(r"$t_n$")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    if save:
        filename = "figures/topo_energy_drift_50k" + ".pdf"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    def plot_trajectory(n_max,save):

        plt.title(r"Coordinate trajectory of $q_n$")
        plt.imshow(U_tot,extent=[-1, 1, 1, -1],cmap="cividis")
        plt.colorbar()

        c1 = plt.contour(yy,xx,U_tot,[H_topo(ys[0])],extent=[-1, 1, 1, -1],colors=contour_color)

        plt.plot([0,0],[0,0],color=contour_color,label=r"$\partial D(H_0)$")
        q = plt.plot(ys[:n_max,3],ys[:n_max,2],c="orange",alpha=1,label=r"$q_n$")

        plt.legend()
        plt.xticks(np.linspace(-1,1,5))
        plt.yticks(np.linspace(1,-1,5),np.linspace(1,-1,5)[::-1])
        plt.xlabel(r"$q_1$")
        plt.ylabel(r"$q_2$")

        if save:
            filename = "figures/topo_trajectory_" + str(n_max) + ".pdf"
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    contour_color = "white"

    plot_trajectory(n_max=1_000,save=save)
    plot_trajectory(n_max=10_000,save=save)
    plot_trajectory(n_max=50_000,save=save)


    plt.title(r"Total potential $U(q)$")
    contour_color = "white"
    plt.imshow(U_tot,extent=[-1, 1, 1, -1],cmap="cividis")
    plt.colorbar()
    c1 = plt.contour(yy,xx,U_tot,[H_topo(ys[-1])],extent=[-1, 1, 1, -1],colors=contour_color)

    plt.xticks(np.linspace(-1,1,5))
    plt.yticks(np.linspace(1,-1,5),np.linspace(1,-1,5)[::-1])
    plt.xlabel(r"$q_1$")
    plt.ylabel(r"$q_2$")


    if save:
        filename = "figures/topo_potential" + ".pdf"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
