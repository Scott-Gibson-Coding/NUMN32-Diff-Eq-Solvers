### Project 2 - Boundary Value Problems

### The Runge-Kutta RK4 method

import time as time
import numpy as np
import scipy.sparse.linalg as sparse_spla
import scipy.integrate as si
import matplotlib.pyplot as plt

from ivp import *
from plot import setup_plot
from utility import rms_norm

from bvp import *

def y_sin_cos(x):
    # Sample real solution of `y` using sin and cos
    return 4*np.sin(x/2) + 2*np.cos(x)
def f_sin_cos(x):
    # Sin/Cos second derivative
    return -1 * np.sin(x/2) - 2*np.cos(x)

def test_bvp_hom_dirichlet():
    """
    ## Scenario

    Test that homogenous dirichlet conditions can approximate a known function.
    """

    y, f = y_sin_cos, f_sin_cos

    L = 2*np.pi
    N = 50
    bc_left = Dirichlet(2)
    bc_right = Dirichlet(2)
    dx, grid, A, BCcorr = FDMd2dx2(L, bc_left, bc_right, N)
    RHS = dx**2 * f(grid) - BCcorr
    u = sparse_spla.spsolve(A, RHS)

    fig, ax = setup_plot(
        title="Sin/Cos - Homogenous Dirichlet",
        xlabel="x",
        ylabel="y",
    )

    # Draw real solution
    x = np.linspace(0, L, 100)
    y = np.array(y(x))

    ax.plot(x, y, color="green", label="Real solution", alpha=0.5)
    ax.plot(
        np.concat([np.array([0]), grid, np.array([L])]), 
        np.concat([np.array([bc_left.value]), u, np.array([bc_right.value])]), 
        color="orange", label=f"FDM | Dir-Dir | {N=}", linestyle="--", alpha=0.9,
    )

    ax.legend(loc="upper right")
    plt.show()

def test_bvp_hom_dirichlet_convergence():
    """
    ## Scenario

    Test that homogenous dirichlet conditions have error convergence of O(h^2)
    """

    y, f = y_sin_cos, f_sin_cos

    fig, ax = setup_plot(
        title="Sin/Cos - Homogenous Dirichlet - Global Error",
        xlabel="h",
        ylabel="error (RMS)",
    )

    L = 2*np.pi
    Ns = 2**np.arange(1, 16)
    bc_left = Dirichlet(2)
    bc_right = Dirichlet(2)

    hs = []
    errs = []
    for N in Ns:
        # Solve using FDM
        dx, grid, A, BCcorr = FDMd2dx2(L, bc_left, bc_right, N)
        RHS = dx**2 * f(grid) - BCcorr
        u = sparse_spla.spsolve(A, RHS)

        # Real solution
        y_sol = np.array(y(grid))

        # Global error (RMS)
        hs.append(dx)
        errs.append(rms_norm(u - y_sol, N))

    # Plot errors
    ax.loglog(hs, errs, "o-", label="Solution Errors")

    # Plot comparison line
    offset = 1.5
    ref_line = offset * errs[0] * (hs / hs[0]) ** (2)
    ax.loglog(hs, ref_line, "--", label="$\mathcal{O}(\mu^2)$", color="orange")

    ax.legend(loc="upper right")
    plt.show()
