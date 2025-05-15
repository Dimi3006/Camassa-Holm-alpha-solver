import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solver_core import *

if __name__ == "__main__":

    """ENTER SIMULATION DATA HERE"""
    "----------------------------------------------------------------------------------------------------------------------"
    '''Define simulation paramaters'''
    dx = 0.02
    alpha = 0.01
    nu = 0.01
    a = -6
    b = 6
    T = 0.35
    J = int((b-a)/dx)
    save = False

    '''Initialize initial values for filtered velocity u'''
    x_list = np.linspace(a, b, J)
    u_start = u_compact(x_list) # compact support case
    # u_start = u_peakon(x_list) # peakon case
    # u_start = u_peakonantipeakon(x_list) # peakon-antipeakon example
    "----------------------------------------------------------------------------------------------------------------------"

    linestyles = ['-', '--', '-.', ':'] 
    dt_list = [0.03, 0.003, 0.0003]
    for i, dt in enumerate(dt_list):

        N = int(T/dt)
        t_list = np.linspace(0, T, N)

        q_list, u_list, P_list, energy_list, mass_list = solve_CamassaHolm(u_start, dx, dt, a, b, alpha, nu, T)

        """Evaluate simulation results"""

        '''Plot the discrete approximation at different times'''
        # Make sure to choose times that are less than T
        # Convert in terms of time steps
        plot_time = int(0.3/dt)
        linestyle = linestyles[i % len(linestyles)]  # Wiederholt Muster bei Bedarf
        plt.plot(x_list, u_list[plot_time], label=r'$dt$' + f'={dt}', linestyle=linestyle)

    plt.xlabel(r'$x$')
    plt.ylabel(r'Discrete approximations to $u$')
    plt.xlim((a, b))
    plt.ylim((0, 2.7))
    plt.legend()
    plt.title(r'Discrete approximations for $\nu$' + f'={nu}' + r' and $\alpha$' + f'={alpha}')
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    if save:
        plt.savefig(file_path, format="pgf", bbox_inches="tight")
    plt.show()




