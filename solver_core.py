import numpy as np
import matplotlib.pyplot as plt
import scipy 
from scipy.sparse import lil_matrix, diags
import tqdm
import time
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
sns.set_palette("colorblind")       # Ensures good contrast

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",  # Use LaTeX for rendering text
    "font.family": "serif",        # Use serif fonts (matches LaTeX)
    "text.usetex": True,           # Use LaTeX for all text rendering
    "pgf.rcfonts": False,          # Use Matplotlib settings for font sizes
})

# Define the folder path
folder_path = r'C:\Users\dimit\vscode\Camassa_Holm_Alpha_solver'

# Ensure the folder exists
os.makedirs(folder_path, exist_ok=True)

# Define the full file path
file_path = os.path.join(folder_path, 'my_experiment.pgf')

'''
Implementation of a finite-difference scheme to solve the inviscid and viscous Camassa-Holm equation with PBC 
Dimitrios Geladaris, 30.07.2024
'''

'''Construct initial data'''

def u_peakon(x):
    """
    Construct peakon initial data.
    :param x: List consisting of space points
    :return: y-data list of the initial data
    """
    return 2*np.exp(-np.abs(x))

def u_peakonantipeakon(x):
    """
    Construct antipeakon-peakon initial data.
    :param x: List consisting of space points
    :return: y-data list of the initial data
    """
    return -2*np.exp(-2*np.abs(x-2)) + 2*np.exp(-2*np.abs(x+2))

def u_compact(x):
    """
    Construct initial data with compact support.
    :param x: List consisting of space points
    :return: y-data list of the initial data
    """
    u_compact = np.zeros_like(x)
    for i in range(len(x)):
        if np.abs(x[i]) < 2:
            u_compact[i] = 2*np.exp(1)*np.exp(4/(x[i]**2-4))
        else:
            u_compact[i] = 0
    return u_compact


'''Build linear system of equations to solve the P-equation of the form AP = rhs'''

def P_matrix(dx, alpha, J):
    """
    Build system matrix to solve the elliptic equation for P
    :param dx: spatial discretization parameter
    :param alpha: filtering parameter
    :param J: amount of discrete spatial points
    :return: system matrix of the linear system (CSR format)
    """
    diag = (2 * alpha**2) + dx**2
    off_diag = -alpha**2

    # lil_matrix is more efficient for constructing sparse matrices
    A = lil_matrix((J, J))
    
    for i in range(J):
        A[i, i] = diag
        A[i, (i - 1) % J] = off_diag  # PBC
        A[i, (i + 1) % J] = off_diag  # PBC

    return A.tocsr()  # In CSR-Format

def get_rhs(u_list, dx, alpha):
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
    rhs[1:] = dx**2 * ((u[1:] + abs_u[1:])**2 / 4 + (u[:-1] - abs_u[:-1])**2 / 4) \
              + (alpha**2 / 2) * (u[1:] - u[:-1])**2
    rhs[0] = rhs[-1]
    return rhs

def linear_solver(A, b):
    """
    Solve the linear system Ax = b using conjugate gradient method
    :param A: system matrix
    :param b: right-hand side vector
    :return: solution vector x
    """
    x, info = scipy.sparse.linalg.cg(A, b)
    if info != 0:
        raise ValueError("Conjugate gradient solver did not converge")
    return x

def u_new(u_old, dx, dt, P_old, nu):
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
        (u_inner + np.abs(u_inner)) / 2 * (u_inner - u_minus) +
        (u_inner - np.abs(u_inner)) / 2 * (u_plus - u_inner)
    )
    pressure_term = P_plus - P_inner
    diffusion_term = nu * (u_plus - 2 * u_inner + u_minus) / dx

    u_next_inner = u_inner - dt / dx * (advective_flux + pressure_term - diffusion_term)

    # Add back the duplicated endpoint to preserve PBC
    u_next = np.append(u_next_inner, u_next_inner[0])

    return u_next



'''Compute the backward-difference q^n for given u'''
def backward_diff(u_list, dx):
    """
    Compute backward-difference q^n for discrete energy computation
    :param u_list: list containing approximation of u^n
    :param dx: spatial discretization parameter
    :return: list containing the backward-difference q^n
    """
    q = np.empty_like(u_list)
    q[1:] = (u_list[1:] - u_list[:-1]) / dx
    q[0] = (u_list[0] - u_list[-2]) / dx  # PBC
    return q

def compute_energy(u_list, q_list, alpha, dx):
    """
    Compute the discrete energy at time t^n
    :param u_list: list containing approximation of u^n
    :param q_list: list containing approximation of q^n
    :param dx: spatial discretization parameter
    :return: discrete energy at time t^n
    """
    return (0.5 * np.linalg.norm(u_list[:-1])**2 + 0.5 * alpha**2 * np.linalg.norm(q_list[:-1])**2) * dx

def compute_mass(u_list, dx):
    """
    Compute the discrete mass at time t^n
    :param u_list: list containing approximation of u^n
    :param dx: spatial discretization parameter
    :return: discrete mass at time t^n
    """
    return dx * (np.sum(u_list[:-1]))

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
    initial_energy = compute_energy(u_start, q_start, alpha, dx)
    initial_mass = compute_mass(u_start, dx)

    '''Solve P-equation for the initial time'''
    A_sparse = P_matrix(dx, alpha, J)
    P_0 = linear_solver(A_sparse, get_rhs(u_start, dx, alpha))


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
        u_list.append(u_new(u_list[-1], dx, dt, P_list[-1], nu))
        P_list.append(linear_solver(A_sparse, get_rhs(u_list[-1], dx, alpha)))
        energies.append(compute_energy(u_list[-1], q_list[-1], alpha, dx))
        mass.append(compute_mass(u_list[-1], dx))

    return q_list, u_list, P_list, energies, mass

def plot_discrete_approximation(plot_times, T, dt, u_list, x_list, a, b, alpha, nu, save=False):
    """
    Plot the discrete approximation at plot_times over space x
    :param plot_times: list of times to plot
    :param T: final time of the simulation
    :param alpha: filtering parameter
    :param nu: viscosity parameter
    :param save: boolean indicating whether to save the plot
    :param u_list: list containing the discrete approximation at each time step
    :param x_list: list containing the spatial points
    """
    ## Check for all plot times if they are less than T
    if not all(t <= T for t in plot_times):
        raise ValueError("Plot times cannot be greater than the final time T")
    # Convert in terms of time steps
    plot_times = [int(t/dt) for t in plot_times]
    linestyles = ['-', '--', '-.', ':'] 

    for i, t in enumerate(plot_times):
        linestyle = linestyles[i % len(linestyles)]  # Wiederholt Muster bei Bedarf
        plt.plot(x_list, u_list[t], label=fr'$t={t*dt:.2f}$', linestyle=linestyle)

    plt.xlabel(r'$x$')
    plt.ylabel(r'Discrete approximations to $u$')
    plt.xlim((a, b))
    plt.legend()
    plt.title(r'Discrete approximations for $\nu$' + f'={nu}' + r' and $\alpha$' + f'={alpha}')
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    if save:
        plt.savefig(file_path, format="pgf", bbox_inches="tight")
    plt.show()

def plot_energy(energies, t_list, T, alpha, nu, save=False):
    """
    Plot the discrete energy over time
    :param energies: list containing the discrete energy at each time step
    :param t_list: list containing the time steps
    :param alpha: filtering parameter
    :param nu: viscosity parameter
    :param save: boolean indicating whether to save the plot
    """
    plt.plot(t_list, energies)
    plt.xlabel(r'$t$')
    plt.ylabel(r'Discrete $\alpha$-energy')
    plt.title(r'Approximations of $E_\alpha[u(t)]$ over time for $\nu$' + f'={nu}' + r' and $\alpha$' + f'={alpha}')
    plt.xlim((0, T))
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    if save:
        plt.savefig(file_path, format="pgf", bbox_inches="tight")
    plt.show()

def plot_mass(mass, t_list, T, alpha, nu, save=False):
    """
    Plot the discrete mass over time
    :param mass: list containing the discrete mass at each time step
    :param t_list: list containing the time steps
    :param alpha: filtering parameter
    :param nu: viscosity parameter
    :param save: boolean indicating whether to save the plot
    """
    plt.plot(t_list, mass)
    plt.xlabel(r'$t$')
    plt.ylabel(r'Discrete integral')
    plt.title(r'Approximations of $U_\alpha[u(t)]$ over time for $\nu$' + f'={nu}' + r' and $\alpha$' + f'={alpha}')
    plt.xlim((0, T))
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    if save:
        plt.savefig(file_path, format="pgf", bbox_inches="tight")
    plt.show()

def compute_timestep(dx, alpha, nu):
    """
    Compute the time step based on the CFL condition
    :param dx: spatial discretization parameter
    :param alpha: filtering parameter
    :param nu: viscosity parameter
    :return: time step dt
    """
    return 1/2 * dx**2 / nu * alpha  # CFL condition for stability

# def compute_timestep(dx, u_list):
#     """
#     Compute the time step based on the CFL condition
#     :param dx: spatial discretization parameter
#     :param u_list: list containing the initial condition for u
#     :return: time step dt
#     """
#     max_speed = np.max(u_list)  # Maximum speed in the initial condition
#     return 0.5 * dx / max_speed  # CFL condition for stability, assuming nu=0


if __name__ == "__main__":

    """ENTER SIMULATION DATA HERE"""
    "----------------------------------------------------------------------------------------------------------------------"
    '''Define simulation paramaters'''
    # dx = 0.002
    # dt = 0.00001
    # alpha = 0.00001
    # nu = 0.01 # Make sure to set alpha = O(nu^2) to approximate Burgers equation
    dx = 0.02
    dt = 0.003
    alpha = 0.1
    nu = 0.01 # Make sure to set alpha = O(nu^2) to approximate Burgers equation
    a = -6
    b = 6
    T = 6
    J = int((b-a)/dx)
    N = int(T/dt)

    '''Initialize initial values for filtered velocity u'''
    x_list = np.linspace(a, b, J)
    t_list = np.linspace(0, T, N)
    u_start = u_compact(x_list) # compact support case
    # u_start = u_peakon(x_list) # peakon case
    # u_start = u_peakonantipeakon(x_list) # peakon-antipeakon example
    "----------------------------------------------------------------------------------------------------------------------"

    # measure time to solve
    start_time = time.time()
    q_list, u_list, P_list, energy_list, mass_list = solve_CamassaHolm(u_start, dx, dt, a, b, alpha, nu, T)
    stop_time = time.time()
    print(f'Time to solve: {stop_time - start_time} seconds. Simulation finished.')

    """Evaluate simulation results"""

    '''Plot the discrete approximation at different times'''
    # Make sure to choose times that are less than T
    plot_times = [0.1, 0.3, 0.4, 1.0, 2, 3, 4, 5, 5.5]
    plot_discrete_approximation(plot_times, T, dt, u_list, x_list, a, b, alpha, nu, False)


    '''Plot the discrete energy over time'''
    plot_energy(energy_list, t_list, T, alpha, nu, False)

    '''Plot the discrete mass over time'''
    plot_mass(mass_list, t_list, T, alpha, nu, False)


