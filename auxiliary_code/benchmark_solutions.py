import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solver_core import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import pickle

# Define the folder path
folder_path = r'C:\Users\dimit\vscode\Camassa_Holm_Alpha_solver\auxiliary_code'

# Ensure the folder exists
os.makedirs(folder_path, exist_ok=True)

# Define the full file path
file_path = os.path.join(folder_path, 'counterexample_energies.pgf')

if __name__ == "__main__":

    # """ENTER SIMULATION DATA HERE"""
    # "----------------------------------------------------------------------------------------------------------------------"
    # # Fixed parameters
    # dx = 0.002
    # a = -6
    # b = 6
    # T = 1
    # J = int((b-a)/dx)


    # # Parameter grid for alpha and nu
    # nu_values = [0.01]

    # # Initialize initial values for filtered velocity u
    # x_list = np.linspace(a, b, J)
    # u_start = u_compact(x_list) # compact support case

    # # Prepare data storage
    # benchmark_data = {}

    # save_every = 100 # Save every n time steps

    # for nu in nu_values:
    #     alpha = 0.1

    #     print(f"Running simulation for alpha={alpha}, nu={nu}")
    #     # Compute time-step based on CFL condition
    #     dt = compute_timestep(dx, nu, alpha, u_start)
    #     N = int(T/dt)
    #     t_list = np.linspace(0, T, N)
    #     start_time = time.time()
    #     q_list, u_list, P_list, energy_list, mass_list = solve_CamassaHolm(
    #         u_start, dx, dt, a, b, alpha, nu, T, save_every=save_every
    #     )
    #     stop_time = time.time()
    #     print(f'Time to solve: {stop_time - start_time:.2f} seconds. Simulation finished.')

    #     # Store results in dictionary
    #     key = f"alpha_{alpha}_nu_{nu}"
    #     benchmark_data[key] = {
    #         "q_list": q_list,
    #         "u_list": u_list,
    #         "P_list": P_list,
    #         "energy_list": energy_list,
    #         "mass_list": mass_list,
    #         "params": {
    #         "alpha": alpha,
    #         "nu": nu,
    #         "dx": dx,
    #         "dt": dt,
    #         "a": a,
    #         "b": b,
    #         "T": T,
    #         "save_every": save_every
    #         }
    #     }

    # # Save benchmark data to file for later access, include dx in filename
    # data_save_path = os.path.join(folder_path, f"benchmark_solutions_nu={nu}_alpha={alpha}_dx={dx}.pkl")
    # with open(data_save_path, "wb") as f:
    #     pickle.dump(benchmark_data, f)
    # print(f"Benchmark data saved to {data_save_path}")

    # Load benchmark data from file
    data_save_path = os.path.join(folder_path, "benchmark_solutions_nu=0.01_alpha=0.0001_dx=0.0002.pkl")
    with open(data_save_path, "rb") as f:
        benchmark_data = pickle.load(f)

    import matplotlib.pyplot as plt

    # Plot solutions for each parameter set
    for key, data in benchmark_data.items():
        u_list = data["u_list"]
        q_list = data["q_list"]
        energy_list = data["energy_list"]
        mass_list = data["mass_list"]
        a = data["params"]["a"]
        b = data["params"]["b"]
        alpha = data["params"]["alpha"]
        nu = data["params"]["nu"]
        dt = data["params"]["dt"]
        T= data["params"]["T"]
        t_list = np.linspace(0, T, int(T/dt))
        x_list = np.linspace(data["params"]["a"], data["params"]["b"], len(u_list[0]))

        # Plot u(x, t) at several time slices
        plt.figure(figsize=(10, 6))
        for i, u in enumerate(u_list):
            if i % max(1, len(u_list)//5) == 0:  # Plot a few time slices
                plt.plot(x_list, u, label=f"t={t_list[i]:.2f}")
        plt.xlabel(r'$x$')
        plt.ylabel(r'Discrete approximations to $u$')
        plt.xlim((a, b))
        plt.legend()
        plt.title(r'Discrete approximations for $\nu$' + f'={nu}' + r' and $\alpha$' + f'={alpha}')
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.show()

        # Plot energy over time
        plt.figure(figsize=(8, 4))
        plt.plot(t_list, energy_list, label="Energy")
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.title(f"Energy evolution ($\\nu$={nu}, $\\alpha$={alpha})")
        plt.grid(True)
        plt.legend()
        plt.show()

        # # Plot mass over time
        # plt.figure(figsize=(8, 4))
        # plt.plot(t_list, mass_list, label="Mass", color="orange")
        # plt.xlabel("Time")
        # plt.ylabel("Mass")
        # plt.title(f"Mass evolution ($\\nu$={nu}, $\\alpha$={alpha})")
        # plt.grid(True)
        # plt.legend()
        # plt.show()

