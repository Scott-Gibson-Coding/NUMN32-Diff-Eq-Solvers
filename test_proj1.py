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
