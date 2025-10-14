import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from conservative_solver import *
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.interpolate import interp1d
import pickle

# Define the folder path
folder_path = r'C:\Users\dimit\vscode\Camassa_Holm_Alpha_solver\auxiliary_code'

# Ensure the folder exists
os.makedirs(folder_path, exist_ok=True)

# Define the full file path
file_path = os.path.join(folder_path, 'test.pgf')

import numpy as np
from scipy.interpolate import interp1d

def l2_error_over_time_interp(
    benchmark, test, x_bench, x_test, times_bench=None, times_test=None, times_eval=None
):
    benchmark = np.asarray(benchmark)
    test = np.asarray(test)
    x_bench = np.asarray(x_bench)
    x_test = np.asarray(x_test)

    nt_bench = benchmark.shape[0]
    nt_test = test.shape[0]

    if times_bench is None:
        times_bench = np.arange(nt_bench)
    if times_test is None:
        times_test = np.arange(nt_test)
    if times_eval is None:
        times_eval = times_bench

    # Precompute dx
    dx = np.mean(np.diff(x_bench))

    # Build time interpolators only once
    f_time_test = interp1d(times_test, test, axis=0, kind='linear', fill_value='extrapolate')
    f_time_bench = interp1d(times_bench, benchmark, axis=0, kind='linear', fill_value='extrapolate')

    errors = np.zeros(len(times_eval))

    for idx, t in enumerate(times_eval):
        # interpolate both in time
        test_interp_time = f_time_test(t)
        bench_at_t = f_time_bench(t)

        # interpolate test in space
        f_space = interp1d(x_test, test_interp_time, kind='linear', fill_value='extrapolate')
        test_on_bench = f_space(x_bench)

        diff = bench_at_t - test_on_bench
        errors[idx] = np.sqrt(np.sum(diff**2) * dx)

    return errors, np.array(times_eval)

def relative_l2_error_over_time_interp(
    benchmark, test, x_bench, x_test, times_bench=None, times_test=None, times_eval=None
):
    abs_errors, times_eval = l2_error_over_time_interp(
        benchmark, test, x_bench, x_test, times_bench, times_test, times_eval
    )
    # Compute L2 norm of benchmark at each eval time
    benchmark = np.asarray(benchmark)
    x_bench = np.asarray(x_bench)
    if times_bench is None:
        times_bench = np.arange(benchmark.shape[0])
    if times_eval is None:
        times_eval = times_bench
    dx = np.mean(np.diff(x_bench))
    f_time_bench = interp1d(times_bench, benchmark, axis=0, kind='linear', fill_value='extrapolate')
    norms = np.zeros(len(times_eval))
    for idx, t in enumerate(times_eval):
        bench_at_t = f_time_bench(t)
        norms[idx] = np.sqrt(np.sum(bench_at_t**2) * dx)
    # Avoid division by zero
    rel_errors = abs_errors / np.where(norms == 0, 1, norms)
    return rel_errors, np.array(times_eval)



if __name__ == "__main__":

    """ENTER SIMULATION DATA HERE"""
    "----------------------------------------------------------------------------------------------------------------------"
    '''Define simulation paramaters'''
    # dx = 0.002
    # dt = 0.00001
    # alpha = 0.00001
    # nu = 0.01 # Make sure to set alpha = O(nu^2) to approximate Burgers equation
    # dx = 0.002
    dx_list = [0.2, 0.02, 0.002, 0.001]
    alpha = .000001
    nu = 0.001 # Make sure to set alpha = O(nu^2) to approximate Burgers equation
    a = -6
    b = 6
    T = 1
    rel_l2_errors_dict = {}

    for dx in dx_list:
        J = int((b-a)/dx)

        '''Initialize initial values for filtered velocity u'''
        x_list = np.linspace(a, b, J)
        u_start = u_compact(x_list) # compact support case

        dt = compute_timestep(dx, nu, alpha, u_start)
        N = int(T/dt)
        t_list = np.linspace(0, T, N)
        save = False

        # measure time to solve
        start_time = time.time()
        sim_q_list, sim_u_list, sim_P_list, sim_energy_list, sim_mass_list = solve_CamassaHolm_conservative(u_start, dx, dt, a, b, alpha, nu, T)
        stop_time = time.time()
        print(f'Time to solve: {stop_time - start_time} seconds. Simulation finished.')

        # Load benchmark data from file
        data_save_path = os.path.join(folder_path, "benchmark_solutions_nu=0.001_alpha=1e-06_dx=0.0002.pkl")
        with open(data_save_path, "rb") as f:
            benchmark_data = pickle.load(f)

        # Compare simulation data with benchmark data
        for key, data in benchmark_data.items():
            bench_u_list = data["u_list"]
            bench_q_list = data["q_list"]
            bench_energy_list = data["energy_list"]
            bench_mass_list = data["mass_list"]
            bench_a = data["params"]["a"]
            bench_b = data["params"]["b"]
            bench_alpha = data["params"]["alpha"]
            bench_nu = data["params"]["nu"]
            bench_dt = data["params"]["dt"]
            bench_T = data["params"]["T"]
            bench_t_list = np.linspace(0, bench_T, len(bench_u_list))
            bench_x_list = np.linspace(bench_a, bench_b, len(bench_u_list[0]))
            dt_list = np.linspace(0, bench_T, int(bench_T/bench_dt))

            # Compute relative L2 error over time for u
            rel_l2_errors_u, _ = relative_l2_error_over_time_interp(
                benchmark=bench_u_list,
                test=sim_u_list,
                x_bench=bench_x_list,
                x_test=x_list,
                times_bench=bench_t_list,
                times_test=t_list,
                times_eval=t_list
            )
            rel_l2_errors_dict[dx] = (t_list, rel_l2_errors_u)

    # Plot all relative errors on the same plot
    plt.figure(figsize=(8, 4))
    for dx in dx_list:
        t_list, rel_l2_errors_u = rel_l2_errors_dict[dx]
        plt.plot(t_list, rel_l2_errors_u, label=f"$\\Delta x$={dx}")
    plt.yscale('log')
    plt.xlabel(r'$t$')
    plt.ylabel(r'Relative $\ell^2$-error')
    plt.title(r'Relative $\ell^2$-error over time for $\nu$' + f'={nu}' + r' and $\alpha$' + f'={alpha}')
    plt.xlim((0, T))
    plt.legend()
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    if save:
        plt.savefig(file_path, format="pgf", bbox_inches="tight")
    plt.show()

            # # Plot the solutions: benchmark vs simulation
            # plt.figure(figsize=(10, 5))
            # # Plot at a few selected times
            # selected_times = np.linspace(0, bench_T, 3)
            # for t in selected_times:
            #     # Interpolate simulation and benchmark at time t
            #     sim_interp = interp1d(t_list, sim_u_list, axis=0, kind='linear', fill_value='extrapolate')(t)
            #     bench_interp = interp1d(bench_t_list, bench_u_list, axis=0, kind='linear', fill_value='extrapolate')(t)
            #     plt.plot(x_list, sim_interp, '--', label=f'Sim t={t:.2f}')
            #     plt.plot(bench_x_list, bench_interp, '-', label=f'Bench t={t:.2f}')
            # plt.xlabel(r'$x$')
            # plt.ylabel(r'Discrete approximations to $u$')
            # plt.xlim((a, b))
            # plt.legend()
            # plt.title(r'Discrete approximations for $\nu$' + f'={nu}' + r' and $\alpha$' + f'={alpha}')
            # plt.gca().spines["top"].set_visible(False)
            # plt.gca().spines["right"].set_visible(False)
            # if save:
            #     plt.savefig(file_path, format="pgf", bbox_inches="tight")
            # plt.show()

            # plt.figure(figsize=(8, 4))
            # plt.plot(dt_list, bench_energy_list, label='Benchmark Energy')
            # plt.plot(t_list, sim_energy_list, label='Simulation Energy', linestyle='--')
            # plt.xlabel(r'$t$')
            # plt.ylabel('Energy')
            # plt.title(r'Benchmark Energy over time for $\nu$' + f'={bench_nu}' + r' and $\alpha$' + f'={bench_alpha}')
            # plt.legend()
            # plt.xlim((0, bench_T))
            # plt.gca().spines["top"].set_visible(False)
            # plt.gca().spines["right"].set_visible(False)
            # plt.show()

            # plt.figure(figsize=(8, 4))
            # plt.plot(dt_list, bench_mass_list, label='Benchmark Mass')
            # plt.plot(t_list, sim_mass_list, label='Simulation Mass', linestyle='--')
            # plt.xlabel(r'$t$')
            # plt.ylabel('Mass')
            # plt.title(r'Benchmark Mass over time for $\nu$' + f'={bench_nu}' + r' and $\alpha$' + f'={bench_alpha}')
            # plt.legend()
            # plt.xlim((0, bench_T))
            # plt.gca().spines["top"].set_visible(False)
            # plt.gca().spines["right"].set_visible(False)
            # plt.show()