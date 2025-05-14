import numpy as np
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Set style
plt.style.use("seaborn-whitegrid")  # Alternative: 'ggplot'
sns.set_palette("colorblind")       # Ensures good contrast

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",  # Use LaTeX for rendering text
    "font.family": "serif",        # Use serif fonts (matches LaTeX)
    "text.usetex": True,           # Use LaTeX for all text rendering
    "pgf.rcfonts": False,          # Use Matplotlib settings for font sizes
})

# Define the folder path
folder_path = r'C:\Users\dimit\Desktop\Abgabe BA'

# Ensure the folder exists
os.makedirs(folder_path, exist_ok=True)

# Define the full file path
file_path = os.path.join(folder_path, 'counterexample_energies.pgf')

'''
Implementation of a finite-difference scheme to solve the inviscid and viscous Camassa-Holm equation with PBC 
More precisely, the inviscid and viscous Scheme A
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
    :return: system matrix of the linear system
    """
    A = np.zeros((J, J))
    for i in range(J):
        A[i, i] = (2*alpha**2) + dx**2

    for j in range(J-1):
        A[j, j+1] = -alpha**2
        A[j+1, j] = -alpha**2

    A[0, -1] = -alpha**2
    A[-1, 0] = -alpha**2
    return A

def get_rhs(u_list, dx, alpha):
    """
    Construct the right-hand side of the linear system to solve elliptic P-equation AP = rhs
    :param u_list: u approximation of the current time step
    :param dx: spatial discretization parameter
    :param alpha: filtering parameter
    :param nu: viscosity parameter
    :return: right-hand side vector of the linear system
    """
    rhs = np.zeros_like(u_list)
    for j in range(1, len(u_list)):
        rhs[j] = dx**2 *(u_list[j] + np.abs(u_list[j]))**2/4 + dx**2 *(u_list[j-1] - np.abs(u_list[j-1]))**2/4 \
                 + alpha**2/(2) * (u_list[j] - u_list[j-1])**2

    rhs[0] = rhs[-1]
    return rhs


'''Compute the next discrete approximation to the filtered velocity u'''
def u_new(u_old, dx, dt, P_old, nu):
    """
    Perform one time step dt to obtain the next discrete approximation at t^(n+1)
    :param u_old: list of discrete u at the old time t^n
    :param dx: spatial discretization parameter
    :param dt: time step
    :param P_old: discrete solution of the elliptic problem at the old time t^n
    :return: list containing the discrete approximation at t^(n+1)
    """
    u_new = np.zeros_like(u_old)
    J = len(u_old)
    for j in range(J-1):
        if j == 0:
            u_new[j] = u_old[j] - dt/dx * ((u_old[j] + np.abs(u_old[j]))/2 * (u_old[j] - u_old[-2]) \
                                + (u_old[j] - np.abs(u_old[j]))/2 * (u_old[j+1] - u_old[j]) + (P_old[j+1] - P_old[j])- nu*(u_old[j+1] - 2*u_old[j] + u_old[-2])/dx)
        else:
            u_new[j] = u_old[j] - dt/dx * ((u_old[j] + np.abs(u_old[j]))/2 * (u_old[j] - u_old[j-1]) \
                                + (u_old[j] - np.abs(u_old[j]))/2 * (u_old[j+1] - u_old[j]) + (P_old[j+1] - P_old[j]) - nu*(u_old[j+1] - 2*u_old[j] + u_old[j-1])/dx)
    u_new[-1] = u_new[0] # PBC
    return u_new

'''Compute the backward-difference q^n for given u'''
def backward_diff(u_list, dx):
    """
    Compute backward-difference q^n for discrete energy computation
    :param u_list: list containing approximation of u^n
    :param dx: spatial discretization parameter
    :return: list containing the backward-difference q^n
    """
    q = np.zeros_like(u_list)
    for i in range(1, len(u_list)):
        q[i] = (u_list[i] - u_list[i - 1]) / dx
    q[0] = (u_list[0] - u_list[-2]) / dx # PBC
    return q
    
# Prepare figure
fig, ax = plt.subplots()

# Store all energy curves
energy_curves = []

alpha_list = [1, 0.1, 0.01, 0.001]
for alpha in alpha_list:
    """ENTER SIMULATION DATA HERE"""
    "----------------------------------------------------------------------------------------------------------------------"
    '''Define simulation paramaters'''
    dx = 0.02
    dt = 0.003
    # dx = 0.007  # seems good for alpha = 0.1 in the inviscid case
    # dt = 0.0001 # seems good for alpha = 0.1 in the inviscid case
    # alpha = 0.015
    nu = 0.01
    a = -6
    b = 6
    T = 0.3
    J = int((b-a)/dx)
    N = int(T/dt)
    x_list = np.linspace(a, b, J)
    t_list = np.linspace(0, T, N)

    '''Initialize initial values for filtered velocity u'''
    u_start = u_compact(x_list) # compact support case
    # u_start = u_peakon(x_list) # peakon case
    # u_start = u_peakonantipeakon(x_list) # peakon-antipeakon example
    "----------------------------------------------------------------------------------------------------------------------"

    '''Compute the backward-difference at each discrete point'''
    q_start = np.zeros_like(u_start)
    for i in range(1, len(u_start)):
        q_start[i] = (u_start[i] - u_start[i - 1]) / dx
    q_start[0] = - q_start[-1]

    '''Build initial discrete energy and discrete integral'''

    initial_energy = (0.5*np.linalg.norm(u_start[1:])**2 + 0.5*alpha**2*np.linalg.norm(q_start[1:])**2) * dx
    initial_mass = dx*(np.sum(u_start[1:]))

    '''Solve P-equation for the initial time'''
    A = P_matrix(dx, alpha, J)
    P_0 = np.linalg.solve(A, get_rhs(u_start, dx, alpha))



    '''Perform simulation'''
    u_list = [u_start]
    P_list = [P_0]
    energies = [initial_energy]
    mass = [initial_mass]
    q_list = [q_start]
    print(f'Solving Camassa-Holm equation with alpha={alpha} and nu={nu}')
    print('Starting simulation')
    for _ in tqdm.tqdm(range(len(t_list)-1)):
        q_list.append(backward_diff(u_list[-1], dx))
        u_list.append(u_new(u_list[-1], dx, dt, P_list[-1], nu))
        P_list.append(np.linalg.solve(A, get_rhs(u_list[-1], dx, alpha)))
        energies.append((0.5 * np.linalg.norm(u_list[-1][1:]) ** 2 \
                        + 0.5 * alpha ** 2 * np.linalg.norm(q_list[-1][1:]) ** 2) * dx)
        mass.append(dx * np.sum(u_list[-1][:-1]))
    print('Simulation finished')
     # Store energy curves
    energy_curves.append((alpha, list(energies)))
# Plot energy on main axis
    if alpha == 1:
        ax.plot(t_list, energies, label = r'$\alpha$' + f'={alpha}')
    elif alpha == 0.1:
        ax.plot(t_list, energies, '--' ,label = r'$\alpha$' + f'={alpha}')
    elif alpha == 0.01:
        ax.plot(t_list, energies, '-.',label = r'$\alpha$' + f'={alpha}')
    elif alpha == 0.001:
        ax.plot(t_list, energies, ':',label = r'$\alpha$' + f'={alpha}')

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
for _, energy in energy_curves:
    idx_zoom = np.where((t_list >= x1) & (t_list <= x2))
    all_zoom_ys.extend(np.array(energy)[idx_zoom])
y1, y2 = 3.9, 4
axins.set_ylim(y1, y2)

# Plot all curves in inset
for alpha, energy in energy_curves:
    if alpha == 1:
        axins.plot(t_list, energy)
    elif alpha == 0.1:
        axins.plot(t_list, energy, '--')
    elif alpha == 0.01:
        axins.plot(t_list, energy, '-.')
    elif alpha == 0.001:
        axins.plot(t_list, energy, ':')

# Format inset
axins.tick_params(left=False, bottom=False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
axins.set_xticklabels([])
axins.set_yticklabels([])

# Save and show
plt.savefig(file_path, format="pgf", bbox_inches="tight")
plt.show()