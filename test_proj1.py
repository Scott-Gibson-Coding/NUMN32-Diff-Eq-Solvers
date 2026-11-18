### Project 1 - Test and plot scenarios for Runge-Kutta methods on the Linear Test Equation
#               and examine behavior of different solvers on sStiff vs. non-stiff problems
#               like nonlinear oscillatory systems, the non-stiff Lotka-Volterra population
#               dynamics problem, and the van der Pol equation.

### The Runge-Kutta RK4 method

import time as time
import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt

from ivp import *
from ivp_run import *
from plot import setup_plot


### Test RK4 explicit solver

def test_compare_RK4_method():
    """
    ## Scenario

    Plot a comparison of RK4 and Explicit Euler to the exact solution for a 2x2 LTE.
    """

    y0 = np.array([1, 2])
    A = np.array([[-1, 1], [1, -3]])
    ivp = LTE(0, 1, y0, A=A)

    Methods = [ExplicitEuler, RK4]

    fig, ax = setup_plot(
        title="LTE - 2d Phase Portrait",
    )

    run_ivp_methods(
        (fig, ax),
        Methods,
        ivp,
        N=40,
        include_exact=True,
    )

    ax.legend(loc="upper left")
    plt.show()

def test_verify_RK4_error_growth():
    """
    ## Scenario

    Compare the error as `h` decreases from a solver using the Trapezoidal rule, which should
    change relative to O(h^2), and RK4 which should be O(h^4).

    Scalar Case: A = -1
    """

    # Setup IVP
    y0 = np.array([2])
    A = np.array([[-1]])

    ivp = LTE(t0=0, T=1, y0=y0, A=A)
    Ns = 2**np.arange(1, 16)

    Methods = [TrapezoidalEuler, RK4]

    fig1, ax1 = setup_plot(
        title=f"Absolute Errors vs h - LTE Method Comparison",
    )
    fig2, ax2 = setup_plot(
        title=f"Relative Errors vs h - LTE Method Comparison",
    )

    kwargs = {
        "Methods": Methods,
        "ivp": ivp,
        "Ns": Ns,
    }

    # Absolute errors
    run_err_methods_vs_h(
        (fig1, ax1),
        err_type="absolute",
        **kwargs,
    )
    # Relative errors
    run_err_methods_vs_h(
        (fig2, ax2),
        err_type="relative",
        **kwargs,
    )

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")

    plt.show()


### Test RK34 adapative solver

def test_RK34_LTE_scalar():
    """
    ## Scenario

    Test the err increase is bounded by the tolerance
    """

    y0 = np.array([2])
    A = np.array([[-1]])
    ivp = LTE(t0=0, T=1, y0=y0, A=A)
    tols=[1e-2, 1e-4, 1e-6, 1e-8, 1e-10]

    fig, ax = setup_plot(
        title="RK34 on Scalar LTE",
        xlabel="t",
        ylabel="y"
    )

    for tol in tols:
        solver = RK34(ivp=ivp, tol=tol)

        us = solver.integrate()
        grid = solver.grid

        ax.plot(grid, us, linestyle="--", label=f"RK34 tol={tol:.3e} ({grid.shape[0]})")

    ts = np.linspace(ivp.t0, ivp.T, 100)
    ys = np.array([ivp.exact(t) for t in ts])
    ax.plot(ts, ys, color="black", alpha=0.5, linestyle="-", label=f"Exact Solution")
    
    plt.legend(loc="upper right")
    plt.show()

def test_RK34_LTE_2x2():
    """
    ## Scenario

    Test the err increase is bounded by the tolerance
    """

    y0 = np.array([1, 2])
    A = np.array([[-1, 1], [1, -3]])
    ivp = LTE(0, 1, y0, A=A)

    tols=[1e-2, 1e-4, 1e-6, 1e-8, 1e-10]

    fig, ax = setup_plot(
        title="RK34 on 2x2 LTE",
        xlabel="y[0]",
        ylabel="y[1]"
    )

    for tol in tols:
        solver = RK34(ivp=ivp, tol=tol)

        us = solver.integrate()
        grid = solver.grid

        ax.plot(us[:,0], us[:,1], linestyle="--", label=f"RK34 tol={tol:.3e} ({grid.shape[0]})")

    ts = np.linspace(ivp.t0, ivp.T, 400)
    ys = np.array([ivp.exact(t) for t in ts])
    ax.plot(ys[:,0], ys[:,1], color="black", alpha=0.5, linestyle="-", label=f"Exact Solution")
    
    plt.legend(loc="upper left")
    plt.show()


### The nonstiff Lotka-Volterra problem (population dynamics)

def test_lotka_is_periodic():
    a, b, c, d = 3, 9, 15, 15
    y0 = np.array([1, 1])
    t0, T = 0, 12
    ivp = LotkaVolterra(t0, T, y0, a, b, c, d)

    tol = 1e-8

    fig1, ax1 = setup_plot(
        title=f"Lotka Volterra - x/y vs time\nSolver: RK34, Tol={tol:.3e}",
        xlabel="t",
        ylabel="x/y",
    )
    fig2, ax2 = setup_plot(
        title=f"Lotka Volterra - Phase Portrait\nSolver: RK34, Tol={tol:.3e}",
        xlabel="x",
        ylabel="y"
    )

    solver = RK34(ivp=ivp, tol=tol)
    us = solver.integrate()
    grid = solver.grid

    # Draw x/y vs time
    plot_args = {
        "linestyle": "-",
        "alpha": 0.8,
    }
    ax1.plot(grid, us[:,0], label="x", **plot_args)
    ax1.plot(grid, us[:,1], label="y", **plot_args)

    # Draw x vs y
    ax2.scatter(y0[0], y0[1], label="Starting Point", color="blue", s=20, zorder=5)
    eq_x, eq_y = d/c, a/b
    ax2.scatter(eq_x, eq_y, label="Equilibrium Point", color="black", s=42, zorder=5)
    ax2.plot(us[:,0], us[:,1], label=f"RK34 - Tol={tol:.3e}")

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    plt.show()

def test_lotka_mult_starting_points():
    a, b, c, d = 3, 9, 15, 15
    y0s = [
        np.array([0.5, 0.6]),
        np.array([0.8, 0.9]),
        np.array([0.7, 0.2]),
        np.array([1, 0.2]),
        # np.array([1, 1]),
        # np.array([1.25, 0.8]),
        # np.array([1.3, 0.35]),
    ]
    tol = 1e-8

    fig1, ax1 = setup_plot(
        title=f"Lotka Volterra - Phase Portrait\nSolver: RK34, Tol={tol:.3e}",
        xlabel="x",
        ylabel="y"
    )
    fig2, ax2 = setup_plot(
        title=f"Lotka Volterra - x component vs time\nSolver: RK34, Tol={tol:.3e}",
        xlabel="t",
        ylabel="x"
    )

    t0, T = 0, 6

    for y0 in y0s:
        ivp = LotkaVolterra(t0, T, y0, a, b, c, d)
        solver = RK34(ivp=ivp, tol=tol)
        us = solver.integrate()
        grid = solver.grid

        # Draw x vs y
        line, = ax1.plot(us[:,0], us[:,1], label=f"Starting Point: {y0}", alpha=0.7)
        ax1.scatter(y0[0], y0[1], color=line.get_color(), alpha=1, s=20, zorder=5)

        # Draw x vs t
        ax2.plot(grid, us[:,0], label=f"Starting Point: {y0}", alpha=0.7)

    eq_x, eq_y = d/c, a/b
    ax1.scatter(eq_x, eq_y, label="Equilibrium Point", color="black", s=42, zorder=5)

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    plt.show()

def test_lotka_mult_ivp_values():
    # vals = np.random.random((6, 4))*15 + 3
    vals = np.array([
        [12, 3, 3, 3],
        [3, 12, 3, 3],
        [3, 3, 12, 3],
        [3, 3, 3, 12],
    ])
    y0s = np.array([
        [2, 4],
        [2, 0.25],
        [1, 1],
        [3, 3],
    ])
    tol = 1e-8

    fig1, ax1 = setup_plot(
        title=f"Lotka Volterra - Phase Portrait\nSolver: RK34, Tol={tol:.3e}",
        xlabel="x",
        ylabel="y"
    )
    fig2, ax2 = setup_plot(
        title=f"Lotka Volterra - x component vs time\nSolver: RK34, Tol={tol:.3e}",
        xlabel="t",
        ylabel="x"
    )

    t0, T = 0, 12

    for val, y0 in zip(vals, y0s):
        # Compute starting point close to eq point
        a, b, c, d = val
        eq_x, eq_y = d/c, a/b

        ivp = LotkaVolterra(t0, T, y0, a, b, c, d)
        solver = RK34(ivp=ivp, tol=tol)
        us = solver.integrate()
        grid = solver.grid
        
        # Draw x vs y, and equilibrium point
        line, = ax1.plot(us[:,0], us[:,1], label=f"a, b, c, d = {np.round(val, 2)}", alpha=0.7)
        ax1.scatter(y0[0], y0[1], color=line.get_color(), alpha=1, s=20, zorder=5)
        ax1.scatter(eq_x, eq_y, color=line.get_color(), s=24, zorder=5)

        # Draw x component vs time to show periodic behavior
        ax2.plot(grid, us[:,0], label=f"a, b, c, d = {np.round(val, 2)}", alpha=0.7)

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    plt.show()

def test_lotka_1000_cycles():
    a, b, c, d = 3, 9, 15, 15
    y0 = np.array([1.5, 0.4])
    t0, T = 0, 1200
    ivp = LotkaVolterra(t0, T, y0, a, b, c, d)

    tol = 1e-8

    fig1, ax1 = setup_plot(
        title=f"Lotka Volterra - x/y vs time\nSolver: RK34, Tol={tol:.3e}",
        xlabel="t",
        ylabel="x/y",
    )
    fig2, ax2 = setup_plot(
        title="Lotka Volterra - x vs y\nSolver: RK34, Tol={tol:.3e}",
        xlabel="x",
        ylabel="y"
    )

    solver = RK34(ivp=ivp, tol=tol)
    us = solver.integrate()
    grid = solver.grid

    # Draw x/y vs time
    plot_args = {
        "linestyle": "-",
        "alpha": 0.8,
    }
    ax1.semilogx(grid, us[:,0], label="x", **plot_args)
    ax1.semilogx(grid, us[:,1], label="y", **plot_args)

    # Draw x vs y
    ax2.plot(us[:,0], us[:,1], label=f"Solver: RK34 - Tol={tol:.3e}")

    fig1.legend(loc="upper right")
    fig2.legend(loc="upper right")
    plt.show()

def test_lotka_stays_near_initial_value():
    a, b, c, d = 3, 9, 15, 15
    y0 = np.array([1.5, 0.4])
    t0, T = 0, 100
    ivp = LotkaVolterra(t0, T, y0, a, b, c, d)

    tol = 1e-8

    fig1, ax1 = setup_plot(
        title=r"Lotka Volterra - |H(x, y)/H($x_0$, $y_0$) - 1| vs time",
        xlabel="t",
        ylabel=r"|H(x, y)/H($x_0$, $y_0$) - 1|",
    )
    fig2, ax2 = setup_plot(
        title=f"Lotka Volterra - H(x, y) vs time\nSolver: RK34, Tol={tol:.3e}",
        xlabel="t",
        ylabel="H(x, y)",
    )

    solver = RK34(ivp=ivp, tol=tol)
    us = solver.integrate()
    grid = solver.grid

    H0 = ivp.H(y0)
    Hs = [ivp.H(u) for u in us]
    H_deltas = [abs(ivp.H(u)/H0 - 1) for u in us]

    # Draw x/y vs time
    plot_args = {
        "linestyle": "-",
        "alpha": 0.8,
    }
    ax1.semilogy(grid[1:], H_deltas[1:], **plot_args)
    ax2.plot(grid, Hs, **plot_args)

    plt.show()


# The Van der Pol equation (elictric oscillator circuit)
def test_van_der_pol_mu_100():
    y0 = np.array([2, 0])
    mu = 100
    t0, T = 0, 2*mu
    ivp = VanderPol(t0=t0, T=T, y0=y0, mu=mu)

    tol = 1e-8

    fig1, ax1 = setup_plot(
        title=f"Van der Pol - Solution Component vs Time\n{mu=}",
        xlabel="t",
        ylabel="y[1]",
    )
    fig2, ax2 = setup_plot(
        title=f"Van der Pol - Phase Portrait\n{mu=}",
        xlabel="y[0]",
        ylabel="y[1]",
    )
    fig3, ax3 = setup_plot(
        title=f"Van der Pol - Step Sizes\n{mu=}",
        xlabel="t",
        ylabel="step (h)",
    )

    solver = RK34(ivp=ivp, tol=tol)
    us = solver.integrate()
    grid = solver.grid

    plot_args = {
        "linestyle": "-",
        "alpha": 0.8,
    }
    ax1.plot(grid, us[:,1], **plot_args)
    ax2.plot(us[:,0], us[:,1], **plot_args)

    step_sizes = np.diff(grid)
    ax3.semilogy(grid[:-1], step_sizes, **plot_args)

    plt.show()

def test_van_der_pol_stable_orbits():
    y0s = [
        np.array([2, 0]),
        np.array([1, 1]),
        np.array([-3, 4]),
        np.array([5, 7]),
        np.array([0, -2]),
    ]
    
    mu = 100
    t0, T = 0, 16*mu # Increased interval since some (e.g. [5, 7]) take longer to settle into a cycle

    tol = 1e-6

    plot_args = {
        "linestyle": "--",
        "alpha": 0.4,
    }

    for y0 in y0s:
        fig, ax = setup_plot(
            title=f"Van der Pol - Comparing initial conditions\n{y0=}",
            xlabel="y[0]",
            ylabel="y[1]",
        )
        
        ivp = VanderPol(t0=t0, T=T, y0=y0, mu=mu)
        solver = RK34(ivp=ivp, tol=tol)
        us = solver.integrate()

        ax.scatter(y0[0], y0[1], color="red", s=20, zorder=5)
        ax.plot(us[:,0], us[:,1], **plot_args)

    plt.show()

def test_van_der_pol_stiffness():
    y0 = np.array([2, 0])
    mus = [10, 15, 22, 33, 47, 68, 100, 150, 220, 330, 470, 680, 1000]
    
    tol = 1e-8

    fig, ax = setup_plot(
        title=f"Van der Pol - Comparing stiffness with μ",
        xlabel="μ",
        ylabel="Number of steps",
    )

    N_steps = []
    for mu in mus:
        t0, T = 0, 0.7*mu
        ivp = VanderPol(t0=t0, T=T, y0=y0, mu=mu)
        solver = RK34(ivp=ivp, tol=tol)
        us = solver.integrate()
        print(mu, us[-1])
        N_steps.append(us.shape[0])

    # Plot mu vs N_steps
    ax.loglog(mus, N_steps, label="RK34")
    ax.scatter(mus, N_steps)
    # Plot dashed O(mu**2)
    rat = N_steps[-1] / (mus[-1] ** 2)
    ax.loglog(mus, [mu**2*rat for mu in mus], label="$\mathcal{O}(\mu^2)$", linestyle="--", alpha=0.8)

    ax.legend(loc="upper left")
    plt.show()


# Test Scipy's solve_ivp compared to my RK34 implementation
def test_scipy_BDF_solver():
    y0 = np.array([2, 0])
    mus = [10, 15, 22, 33, 47, 68, 100, 150, 220, 330, 470, 680, 1000, 4000, 10_000, 25_000, 50_000]
    
    rtol = 1e-4
    atol = 1e-8

    fig, ax = setup_plot(
        title=f"Van der Pol - Using Scipy's BDF solver",
        xlabel="μ",
        ylabel="Number of steps",
    )

    N_steps = []
    for mu in mus:
        t0, T = 0, 0.7*mu
        ivp = VanderPol(t0=t0, T=T, y0=y0, mu=mu)

        result = si.solve_ivp(
            ivp.f, (t0, T), y0,
            method="BDF",
            rtol=rtol, atol=atol,
        )
        if not result.success:
            N_steps.append(0)
        else:
            N_steps.append(result.t.shape[0])

    # Plot mu vs N_steps
    ax.loglog(mus, N_steps, label="BDF")
    ax.scatter(mus, N_steps)
    # Plot dashed O(mu**2)
    rat = N_steps[-1] / (mus[-1] ** 2)
    ax.loglog(mus, [mu**2*rat for mu in mus], label="$\mathcal{O}(\mu^2)$", linestyle="--", alpha=0.8)

    ax.legend(loc="lower right")
    plt.show()

def test_scipy_RK45_solver():
    y0 = np.array([2, 0])
    mus = [10, 15, 22, 33, 47, 68, 100, 150, 220, 330, 470, 680]
    
    rtol = 1e-3
    atol = 1e-6

    fig, ax = setup_plot(
        title=f"Van der Pol - Scipy vs My Implementation",
        xlabel="μ",
        ylabel="Number of steps",
    )

    N_steps_mine = []
    N_steps_scipy = []
    for mu in mus:
        t0, T = 0, 0.7*mu
        ivp = VanderPol(t0=t0, T=T, y0=y0, mu=mu)
        solver = RK34(ivp=ivp, tol=atol)
        us = solver.integrate()
        N_steps_mine.append(us.shape[0])

        result = si.solve_ivp(
            ivp.f, (t0, T), y0,
            method="RK45",
            rtol=rtol, atol=atol,
        )
        if not result.success:
            N_steps_scipy.append(0)
        else:
            N_steps_scipy.append(result.t.shape[0])

    # Plot mu vs N_steps
    line, = ax.loglog(mus, N_steps_mine, label="My RK34")
    ax.scatter(mus, N_steps_mine, color=line.get_color())
    line, = ax.loglog(mus, N_steps_scipy, label="Scipy RK34")
    ax.scatter(mus, N_steps_scipy, color=line.get_color())
    # Plot dashed O(mu**2)
    rat = N_steps_scipy[-1] / (mus[-1] ** 2)
    ax.loglog(mus, [mu**2*rat for mu in mus], label="$\mathcal{O}(\mu^2)$", color="orange", linestyle="--", alpha=0.8)

    ax.legend(loc="upper left")
    plt.show()

def test_wall_clock_timing():
    # Run my RK34, Scipy RK45, Scipy BDF on the same problem with mu large, maybe 2500
    # Record the times of all three and 

    y0 = np.array([2, 0])
    mu = 750
    
    rtol = 1e-3
    atol = 1e-6

    fig, ax = setup_plot(
        title=r"Van der Pol - Method Timing $\mu = 750$",
        xlabel="Method",
        ylabel="Time (s)",
    )

    # Setup ivp
    t0, T = 0, 0.7*mu
    ivp = VanderPol(t0=t0, T=T, y0=y0, mu=mu)

    # My RK34 method
    start_time = time.time()
    solver = RK34(ivp=ivp, tol=atol)
    solver.integrate()
    my_rk34_time = time.time() - start_time

    # Scipy RK45 method
    start_time = time.time()
    result = si.solve_ivp(
        ivp.f, (t0, T), y0,
        method="RK45",
        rtol=rtol, atol=atol,
    )
    assert result.success
    scipy_rk45_time = time.time() - start_time

    # Scipy BDF method
    start_time = time.time()
    result = si.solve_ivp(
        ivp.f, (t0, T), y0,
        method="BDF",
        rtol=rtol, atol=atol,
    )
    assert result.success
    scipy_bdf_time = time.time() - start_time

    names = ["My RK34", "Scipy RK45", "Scipy BDF"]
    times = [my_rk34_time, scipy_rk45_time, scipy_bdf_time]
    ax.stem(names, times)
    ax.set_yscale("log")

    plt.show()
    