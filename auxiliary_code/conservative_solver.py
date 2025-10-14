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
file_path = os.path.join(folder_path, 'counterexample_energies.pgf')

def u_new_split(u_old, dx, dt, P_old, nu):
    """
    Vectorized version for periodic grid where u_old[-1] == u_old[0].
    """
    u_old = np.asarray(u_old)
    P_old = np.asarray(P_old)

    # Only compute on the first J-1 points (excluding duplicate)
    u_inner = u_old[:-1]
    u_plus = np.roll(u_inner, -1)
    u_minus = np.roll(u_inner, 1)

    P_inner = P_old[:-1]
    P_plus = np.roll(P_inner, -1)

    advective_flux = (
        (u_inner + np.abs(u_inner)) / 2 * u_inner - (u_minus + np.abs(u_minus)) / 2 * u_minus+
        (u_plus - np.abs(u_plus)) / 2 * u_plus - (u_inner - np.abs(u_inner)) / 2 * u_inner
    )
    pressure_term = P_plus - P_inner
    diffusion_term = nu * (u_plus - 2 * u_inner + u_minus) / dx

    u_next_inner = u_inner - dt / dx * (0.5 * advective_flux + pressure_term - diffusion_term)

    # Add back the duplicated endpoint to preserve PBC
    u_next = np.append(u_next_inner, u_next_inner[0])

    return u_next

def get_rhs_split(u_list, dx, alpha):
    """
    Construct the right-hand side of the linear system to solve elliptic P-equation AP = rhs
    :param u_list: u approximation of the current time step
    :param dx: spatial discretization parameter
    :param alpha: filtering parameter
    :return: right-hand side vector of the linear system
    """
    u = u_list
    abs_u = np.abs(u)
    rhs = np.zeros_like(u)
    # Vectorized computation for indices 1 to len(u)-1
    rhs[1:] = dx**2 * ((u[:-1] + abs_u[:-1])**2 / 4 + (u[1:] - abs_u[1:])**2 / 4) \
          + (alpha**2 / 2) * (u[1:] - u[:-1])**2
    rhs[0] = rhs[-1]
    return rhs


def solve_CamassaHolm_conservative(u_start, dx, dt, a, b, alpha, nu, T):
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
    initial_energy = compute_energy(u_start, q_start, alpha, dx)
    initial_mass = compute_mass(u_start, dx)

    '''Solve P-equation for the initial time'''
    A_sparse = P_matrix(dx, alpha, J)
    P_0 = linear_solver(A_sparse, get_rhs_split(u_start, dx, alpha))


    '''Perform simulation'''
    u_list = [u_start]
    P_list = [P_0]
    energies = [initial_energy]
    mass = [initial_mass]
    q_list = [q_start]
    print(f'Solving Camassa-Holm equation with alpha={alpha}, nu={nu}, dx={dx}, dt={dt}')
    print('Starting simulation')
    for _ in tqdm.tqdm(range(len(t_list)-1)):
        q_list.append(backward_diff(u_list[-1], dx))
        u_list.append(u_new_split(u_list[-1], dx, dt, P_list[-1], nu))
        P_list.append(linear_solver(A_sparse, get_rhs_split(u_list[-1], dx, alpha)))
        energies.append(compute_energy(u_list[-1], q_list[-1], alpha, dx))
        mass.append(compute_mass(u_list[-1], dx))

    return q_list, u_list, P_list, energies, mass

if __name__ == "__main__":

    """ENTER SIMULATION DATA HERE"""
    "----------------------------------------------------------------------------------------------------------------------"
    '''Define simulation paramaters'''
    # dx = 0.002
    # dt = 0.00001
    # alpha = 0.00001
    # nu = 0.01 # Make sure to set alpha = O(nu^2) to approximate Burgers equation
    dx = 0.02
    alpha = 0.1
    nu = 0.1 # Make sure to set alpha = O(nu^2) to approximate Burgers equation
    a = -6
    b = 6
    T = 1
    J = int((b-a)/dx)

    '''Initialize initial values for filtered velocity u'''
    x_list = np.linspace(a, b, J)
    u_start = u_compact(x_list) # compact support case
    # u_start = u_peakon(x_list) # peakon case
    # u_start = u_peakonantipeakon(x_list) # peakon-antipeakon example

    dt = compute_timestep(dx, nu, alpha, u_start)
    N = int(T/dt)
    t_list = np.linspace(0, T, N)
    "----------------------------------------------------------------------------------------------------------------------"

    # measure time to solve
    start_time = time.time()
    q_list, u_list, P_list, energy_list, mass_list = solve_CamassaHolm_conservative(u_start, dx, dt, a, b, alpha, nu, T)
    stop_time = time.time()
    print(f'Time to solve: {stop_time - start_time} seconds. Simulation finished.')

    """Evaluate simulation results"""

    '''Plot the discrete approximation at different times'''
    # Make sure to choose times that are less than T
    plot_times = [0.1, 0.2, 0.3, 0.4, 0.9]
    plot_discrete_approximation(plot_times, T, dt, u_list, x_list, a, b, alpha, nu, False)


    '''Plot the discrete energy over time'''
    plot_energy(energy_list, t_list, T, alpha, nu, False)

    '''Plot the discrete mass over time'''
    plot_mass(mass_list, t_list, T, alpha, nu, False)