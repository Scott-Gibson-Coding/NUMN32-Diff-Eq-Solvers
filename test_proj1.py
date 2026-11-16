### Project 1 - Test and plot scenarios for Runge-Kutta methods on the Linear Test Equation
#               and examine behavior of different solvers on sStiff vs. non-stiff problems
#               like nonlinear oscillatory systems, the non-stiff Lotka-Volterra population
#               dynamics problem, and the van der Pol equation.

### The Runge-Kutta RK4 method

import numpy as np
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
        title="Method Comparison: 2x2 Case",
    )

    run_ivp_methods(
        (fig, ax),
        Methods,
        ivp,
        N=40,
        include_exact=True,
    )

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
        title=f"Absolute Errors vs h - Multiple Scalar Cases",
    )
    fig2, ax2 = setup_plot(
        title=f"Relative Errors vs h - Multiple Scalar Cases",
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
    
    plt.legend(loc="upper right")
    plt.show()


### The nonstiff Lotka-Volterra problem (population dynamics)

def test_lotka_is_periodic():
    a, b, c, d = 3, 9, 15, 15
    y0 = np.array([1.5, 0.2])
    t0, T = 0, 12
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
    ax1.plot(grid, us[:,0], label="x", **plot_args)
    ax1.plot(grid, us[:,1], label="y", **plot_args)

    # Draw x vs y
    ax2.plot(us[:,0], us[:,1], label=f"Solver: RK34 - Tol={tol:.3e}")

    fig1.legend(loc="upper right")
    fig2.legend(loc="upper right")
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
    t0, T = 0, 8000
    ivp = LotkaVolterra(t0, T, y0, a, b, c, d)

    tol = 1e-8

    fig1, ax1 = setup_plot(
        title=f"Lotka Volterra - |H(x, y)/H(x0, y0) - 1| vs time\nSolver: RK34, Tol={tol:.3e}",
        xlabel="t",
        ylabel="|H(x, y)/H(x0, y0) - 1|",
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
    ax1.loglog(grid[1:], H_deltas[1:], **plot_args)
    ax2.plot(grid[1:], Hs[1:], **plot_args)

    plt.show()


# The Van der Pol equation (elictric oscillator circuit)
def test_van_der_pol():
    pass
