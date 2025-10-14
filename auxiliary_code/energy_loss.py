import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solver_core import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Define the folder path
folder_path = r'C:\Users\dimit\vscode\Camassa_Holm_Alpha_solver\auxiliary_code'

# Ensure the folder exists
os.makedirs(folder_path, exist_ok=True)

# Define the full file path
file_path = os.path.join(folder_path, 'energy_loss.pgf')

def viscous_loss_energy(q_list, dx, nu):
    return nu * np.linalg.norm(q_list[:-1])**2 * dx

def compute_burgers_energy(u_list, dx):
    """
    Compute the discrete energy of burgers term at time t^n
    :param u_list: list containing approximation of u^n
    :param dx: spatial discretization parameter
    :return: discrete energy at time t^n
    """
    return (0.5 * np.linalg.norm(u_list[:-1])**2) * dx

def compute_gradient_energy(q_list, alpha, dx):
    """
    Compute the discrete energy of gradient term at time t^n
    :param u_list: list containing approximation of u^n
    :param q_list: list containing approximation of q^n
    :param dx: spatial discretization parameter
    :return: discrete energy at time t^n
    """
    return (0.5 * alpha**2 * np.linalg.norm(q_list[:-1])**2) * dx

def solve_CamassaHolm(u_start, dx, dt, a, b, alpha, nu, T):
    """
    Solve the Camassa-Holm equation with given parameters
    :param u_start: initial condition for u
    :param dx: spatial discretization parameter
    :param dt: time step
    :param alpha: filtering parameter
    :param nu: viscosity parameter
    :param T: final time
    :return: lists of discrete approximations to q, u, P, E, U at each time step
    """
    J = int((b-a)/dx)
    N = int(T/dt)
    x_list = np.linspace(a, b, J)
    t_list = np.linspace(0, T, N)

    # Initialize lists to store results
    q_start = backward_diff(u_start, dx)

    '''Build initial discrete energy and discrete integral'''
    initial_burgers_energy = compute_burgers_energy(u_start, dx)
    initial_alpha_energy = compute_gradient_energy(q_start, alpha, dx)
    initial_mass = compute_mass(u_start, dx)
    initial_energy_loss = viscous_loss_energy(np.array([q_start]), dx, nu)
    '''Solve P-equation for the initial time'''
    A_sparse = P_matrix(dx, alpha, J)
    P_0 = linear_solver(A_sparse, get_rhs(u_start, dx, alpha))


    '''Perform simulation'''
    u_list = [u_start]
    P_list = [P_0]
    burgers_energies = [initial_burgers_energy]
    alpha_energies = [initial_alpha_energy]
    mass = [initial_mass]
    q_list = [q_start]
    energy_loss =[initial_energy_loss]
    print(f'Solving Camassa-Holm equation with alpha={alpha}, nu={nu}, dx={dx}, dt={dt}')
    print('Starting simulation')
    for _ in tqdm.tqdm(range(len(t_list)-1)):
        q_list.append(backward_diff(u_list[-1], dx))
        u_list.append(u_new(u_list[-1], dx, dt, P_list[-1], nu))
        P_list.append(linear_solver(A_sparse, get_rhs(u_list[-1], dx, alpha)))
        burgers_energies.append(compute_burgers_energy(u_list[-1], dx))
        alpha_energies.append(compute_gradient_energy(q_list[-1], alpha, dx))
        mass.append(compute_mass(u_list[-1], dx))
        energy_loss.append(viscous_loss_energy(q_list[-1], dx, nu))

    return q_list, u_list, P_list, burgers_energies, alpha_energies, mass, energy_loss


if __name__ == "__main__":

    """ENTER SIMULATION DATA HERE"""
    "----------------------------------------------------------------------------------------------------------------------"
    '''Define simulation paramaters'''
    dx = 0.002
    nu = 0.01
    a = -6
    b = 6
    T = 0.5
    J = int((b-a)/dx)
    save = False

    '''Initialize initial values for filtered velocity u'''
    x_list = np.linspace(a, b, J)
    u_start = u_compact(x_list) # compact support case
    # u_start = u_peakon(x_list) # peakon case
    # u_start = u_peakonantipeakon(x_list) # peakon-antipeakon example
    "----------------------------------------------------------------------------------------------------------------------"

    # Prepare figure
    fig, ax = plt.subplots()
    # Store all energy curves
    energy_curves = []

    linestyles = ['-', '--', '-.', ':'] 
    alpha_list = [0.001, 0.0001]
    for i, alpha in enumerate(alpha_list):

        dt = compute_timestep(dx, nu, alpha, u_start)
        N = int(T/dt)
        t_list = np.linspace(0, T, N)

        q_list, u_list, P_list, burgers_energies, alpha_energies, mass, energy_loss = solve_CamassaHolm(u_start, dx, dt, a, b, alpha, nu, T)
        total_energy = np.array(burgers_energies) + np.array(alpha_energies)
        total_energy_loss = np.gradient(np.array(total_energy), dt)
        viscous_energy_loss = np.array(energy_loss)
        burgers_energy_loss = np.gradient(np.array(burgers_energies), dt)
        alpha_energy_loss = np.gradient(np.array(alpha_energies), dt)

        """Evaluate simulation results"""

        '''Plot the discrete approximation at different times'''
        # Make sure to choose times that are less than T
        # Convert in terms of time steps
        plot_time = int(0.3/dt)
        linestyle = linestyles[i % len(linestyles)]  # Wiederholt Muster bei Bedarf
        # ax.plot(t_list, total_energy, label=r'Total energy, $\alpha$' + f'={alpha}', linestyle=linestyle)
        # ax.plot(t_list, burgers_energies, label=r'Burgers energy, $\alpha$' + f'={alpha}', linestyle='--')
        # ax.plot(t_list, alpha_energies, label=r'Alpha energy, $\alpha$' + f'={alpha}', linestyle='-.')
        ax.plot(t_list, total_energy_loss, label=r'Total energy loss, $\alpha$' + f'={alpha}', linestyle='-')
        ax.plot(t_list, viscous_energy_loss, label=r'Viscous energy loss, $\alpha$' + f'={alpha}', linestyle='--')
        ax.plot(t_list, burgers_energy_loss, label=r'Burgers energy loss, $\alpha$' + f'={alpha}', linestyle='-.')
        ax.plot(t_list, alpha_energy_loss, label=r'Alpha energy loss, $\alpha$' + f'={alpha}', linestyle=':')
        energy_curves.append((alpha, energy_loss, linestyle))

        # Formatting main plot
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'Discrete $\alpha$-energy-difference for 'r'$\nu$' + f'={nu}')
    ax.set_title(r'$D_+^t E_\alpha[u^n] + \nu \Delta x \sum_j (q_j^n)^2$ over time')
    ax.set_xlim((0, T))
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # # Create zoom-in inset
    # axins = inset_axes(ax, width="20%", height="20%", loc='lower center')

    # # Zoom region
    # x1, x2 = 0.18, 0.2
    # axins.set_xlim(x1, x2)

    # # Dynamically calculate y-range
    # all_zoom_ys = []
    # for _, energy, _ in energy_curves:
    #     idx_zoom = np.where((t_list >= x1) & (t_list <= x2))
    #     all_zoom_ys.extend(np.array(energy)[idx_zoom])
    # y1, y2 = 3.9, 4
    # axins.set_ylim(y1, y2)

    # # Plot all curves in inset
    # for alpha, energy, linestyle in energy_curves:
    #     axins.plot(t_list, energy, linestyle=linestyle)

    # # Format inset
    # axins.tick_params(left=False, bottom=False)
    # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    # axins.set_xticklabels([])
    # axins.set_yticklabels([])

    # Save and show
    plt.savefig(file_path, format="pgf", bbox_inches="tight")
    plt.show()




