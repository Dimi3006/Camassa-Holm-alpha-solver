import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solver_core import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Define the folder path
folder_path = r'C:\Users\dimit\vscode\Camassa_Holm_Alpha_solver'

# Ensure the folder exists
os.makedirs(folder_path, exist_ok=True)

# Define the full file path
file_path = os.path.join(folder_path, 'counterexample_energies.pgf')


if __name__ == "__main__":

    """ENTER SIMULATION DATA HERE"""
    "----------------------------------------------------------------------------------------------------------------------"
    '''Define simulation paramaters'''
    dx = 0.02
    dt = 0.003
    nu = 0.01
    a = -6
    b = 6
    T = 0.35
    J = int((b-a)/dx)
    N = int(T/dt)
    save = False

    '''Initialize initial values for filtered velocity u'''
    x_list = np.linspace(a, b, J)
    t_list = np.linspace(0, T, N)
    u_start = u_compact(x_list) # compact support case
    # u_start = u_peakon(x_list) # peakon case
    # u_start = u_peakonantipeakon(x_list) # peakon-antipeakon example
    "----------------------------------------------------------------------------------------------------------------------"

    # Prepare figure
    fig, ax = plt.subplots()
    # Store all energy curves
    energy_curves = []

    linestyles = ['-', '--', '-.', ':'] 
    alpha_list = [0.1, 0.01, 0.001, 0.0001]
    for i, alpha in enumerate(alpha_list):

        q_list, u_list, P_list, energy_list, mass_list = solve_CamassaHolm(u_start, dx, dt, a, b, alpha, nu, T)

        """Evaluate simulation results"""

        '''Plot the discrete approximation at different times'''
        # Make sure to choose times that are less than T
        # Convert in terms of time steps
        plot_time = int(0.3/dt)
        linestyle = linestyles[i % len(linestyles)]  # Wiederholt Muster bei Bedarf
        ax.plot(t_list, energy_list, label = r'$\alpha$' + f'={alpha}', linestyle=linestyle)
        energy_curves.append((alpha, energy_list, linestyle))

        # Formatting main plot
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'Discrete $\alpha$-energy')
    ax.set_title(r'Approximations of $E_\alpha[u^n]$ over time')
    ax.set_xlim((0, T))
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Create zoom-in inset
    axins = inset_axes(ax, width="20%", height="20%", loc='lower center')

    # Zoom region
    x1, x2 = 0.18, 0.2
    axins.set_xlim(x1, x2)

    # Dynamically calculate y-range
    all_zoom_ys = []
    for _, energy, _ in energy_curves:
        idx_zoom = np.where((t_list >= x1) & (t_list <= x2))
        all_zoom_ys.extend(np.array(energy)[idx_zoom])
    y1, y2 = 3.9, 4
    axins.set_ylim(y1, y2)

    # Plot all curves in inset
    for alpha, energy, linestyle in energy_curves:
        axins.plot(t_list, energy, linestyle=linestyle)

    # Format inset
    axins.tick_params(left=False, bottom=False)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    axins.set_xticklabels([])
    axins.set_yticklabels([])

    # Save and show
    plt.savefig(file_path, format="pgf", bbox_inches="tight")
    plt.show()




