### Project 0 - Test and plot scenarios for Explicit/Implicit Euler methods on
#               the Linear Test Equation: y' = Ay
#             
#               Additionally plot scenarios with the Trapezoidal Rule which should
#               demonstrate better error growth.

import numpy as np
import matplotlib.pyplot as plt

from ivp import *
from ivp_run import *
from plot import setup_plot

### Testing the LTE class is set up properly

def test_lte_exact(cycles: float = 2, damping: float = 0.):
    """
    ## Scenario
    
    Test that the LTE `exact()` function is working as expected for a known linear equation
    with given initial value.

    The function used in this test uses a specific rotation matrix with an optional damping factor
    and an initial value that simply draws a circle with radius 1 as time `t` increases. Reducing
    the damping factor below 0 should make the solution cycle inwards towards the origin as `t`
    increases.

    :param cycles: Number of full revolutions. Is multiplied by 2pi to simulate a full revolution
    around the origin.
    :param damping: Defaults to 0, reduce to a value between [-1, 0] to cause the solution to spiral
    towards the origin.
    """

    sigma = damping
    y0 = np.array([1.0, 0])
    A = np.array([[sigma, -1], [1, sigma]])
    ivp = LTE(0, 1, y0, A)

    ts = np.linspace(0, 2*cycles*np.pi, 400)
    ys = np.array([ivp.exact(t) for t in ts])
    # ys = get_exact_solution(ivp, ts)

    fig, ax = setup_plot(
        title="Test LTE Exact",
        xlabel="y[0]",
        ylabel="y[1]",
        draw_axes_lines=True,
    )

    # Plot data
    ax.plot(ys[:,0], ys[:,1])
    # Decorate plot
    ax.set_aspect("equal", adjustable="datalim")
    plt.show()

def test_lte_exact_with_damping():
    """
    ## Scenario
    
    Test the exact solution of the LTE spirals towards the origin as expected.
    """
    
    test_lte_exact(cycles=12, damping=-0.05)


### Comparing methods approximations of the Linear Test Equation

def test_run_ivp_methods():
    """
    ## Scenario

    Example usage for the `run_ivp_methods` function.
    """

    # Setup IVPs
    ivp_scalar = LTE(0, 1, np.array([2]), np.array([[-1]]))
    ivp_2x2 = LTE(0, 1, np.array([1, -1]), np.array([[-1, 3], [0, -1]]))
    ivp_large = LTE(0, 1, np.array([1, 1, 1]), np.array([
        [1, 0, -87],
        [0, 2, 4.5],
        [0, 0, 3],
    ]))

    # Setup plots
    fig1, ax1 = setup_plot(
        title="Method Comparison - Scalar Case",
    )
    fig2, ax2 = setup_plot(
        title="Method Comparison - 2x2 Case",
    )
    fig3, ax3 = setup_plot(
        title="Method Comparison - 3x3 Case",
    )

    # Shared args
    kwargs = {
        "Methods": [ExplicitEuler, ImplicitEuler, TrapezoidalEuler],
        "N": 24,
        "include_exact": True,
    }

    # Scalar case
    run_ivp_methods(
        (fig1, ax1),
        ivp=ivp_scalar,
        **kwargs,
    )
    # 2x2 case
    run_ivp_methods(
        (fig2, ax2),
        ivp=ivp_2x2,
        **kwargs,
    )
    # 3x3 case
    run_ivp_methods(
        (fig3, ax3),
        ivp=ivp_large,
        **kwargs,
    )

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    ax3.legend(loc="upper right")
    plt.show()

def test_approx_solutions_scalar():
    """
    ## Scenario

    Plot a comparison of the three methods from project 0 to the exact solution for
    a scalar LTE.
    """

    y0 = np.array([2])
    A = np.array([[-1]])
    ivp = LTE(0, 1, y0, A=A)

    Methods = [ExplicitEuler, ImplicitEuler, TrapezoidalEuler]

    fig, ax = setup_plot(
        title="Method Comparison: Scalar Case",
    )

    run_ivp_methods(
        (fig, ax),
        Methods,
        ivp,
        N=100,
        include_exact=True,
    )

    ax.legend(loc="upper right")
    plt.show()

def test_approx_solutions_2x2():
    """
    ## Scenario

    Plot a comparison of the three methods from project 0 to the exact solution for
    a 2x2 LTE.
    """

    y0 = np.array([1, 2])
    A = np.array([[-1, 1], [1, -3]])
    ivp = LTE(0, 1, y0, A=A)

    Methods = [ExplicitEuler, ImplicitEuler, TrapezoidalEuler]

    fig, ax = setup_plot(
        title="Method Comparison: 2x2 Case",
    )

    run_ivp_methods(
        (fig, ax),
        Methods,
        ivp,
        N=100,
        include_exact=True,
    )

    ax.legend(loc="upper right")
    plt.show()

def test_approx_solutions_2x2_2():
    """
    ## Scenario

    Setup initial conditions and discretization size such that the Implicit Euler solution 
    behaves considerably differently than Explicit Euler.
    """

    y0 = np.array([1, 1])
    A = np.array([[-1, 100], [0, -30]])
    ivp = LTE(0, 1, y0, A=A)

    Methods = [ExplicitEuler, ImplicitEuler, TrapezoidalEuler]

    fig, ax = setup_plot(
        title="Method Comparison: 2x2 Case",
    )

    run_ivp_methods(
        (fig, ax),
        Methods,
        ivp,
        N=20,
        include_exact=True,
    )

    ax.legend(loc="upper right")
    plt.show()

def test_approx_solutions_3x3():
    """
    ## Scenario

    Test that multiple methods can be evaluated and plotted side by side
    on a given 3x3 matrix.
    """

    y0 = np.random.randn(3)
    A = np.random.random((3, 3)) * 2
    ivp = LTE(0, 1, y0, A=A)

    Methods = [ExplicitEuler, ImplicitEuler, TrapezoidalEuler]

    fig, ax = setup_plot(
        title="Method Comparison: 3x3 Case",
    )

    run_ivp_methods(
        (fig, ax),
        Methods,
        ivp,
        N=2**12,
        include_exact=True,
    )

    ax.legend(loc="upper right")
    plt.show()

def test_run_mult_ivp_method():
    # Scalar cases
    y0s = np.linspace(-5, 5, 10).reshape(-1, 1)
    A = np.array([[-1]])

    # 2x2 cases
    # x, y = np.linspace(-5, 5, 8), np.linspace(-5, 5, 4)
    # X, Y = np.meshgrid(x, y, indexing="ij")
    
    # y0s = np.column_stack((X.ravel(), Y.ravel()))
    # A = np.array([[-1, 1], [1, -3]])

    ivps = [LTE(0, 4, y0, A) for y0 in y0s]
    Method = TrapezoidalEuler

    fig, ax = setup_plot(
        title="Single Method: Changing y0 for constant A",
    )

    run_mult_ivp_method(
        (fig, ax),
        Method=Method,
        ivps=ivps,
        N=100,
        include_exact=False,
    )

    plt.show()


### Visualizing Error Against Time

def test_err_vs_time_methods():
    """
    ## Scenario

    Plot to compare the absolute and relative error as time increases.

    Scalar Case: A = -1
    """
    y0 = np.array([2])
    A = np.array([[-1]])
    ivp = LTE(t0=0, T=1, y0=y0, A=A)
    N = 100

    Methods = [ExplicitEuler, ImplicitEuler, TrapezoidalEuler]

    fig1, ax1 = setup_plot(
        title="Absolute Errors vs Time - Scalar Case",
    )
    fig2, ax2 = setup_plot(
        title="Relative Errors vs Time - Scalar Case",
    )

    kwargs = {
        "Methods": Methods,
        "ivp": ivp,
        "N": N,
    }

    # Absolute errors
    run_err_methods_vs_time(
        (fig1, ax1),
        err_type="absolute",
        **kwargs,
    )
    # Relative errors
    run_err_methods_vs_time(
        (fig2, ax2),
        err_type="relative",
        **kwargs,
    )
    
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    plt.show()

def test_err_vs_time_ivps():
    """
    ## Scenario

    Plot to compare the absolute error as time increases when varying A.

    Scalar Cases: A = -10 / -5 / -1 / 1

    Larger positive values of A increase in error rapidly. So if we choose to add those we should switch to semilogy
    so that all curves are visible.
    """

    def label_gen(Method: type[Solver], ivp: IVP):
        return f"y0={ivp.y0[0]}, λ={ivp.A[0,0]}"

    ivps = [LTE(t0=0, T=4, y0=np.array([2]), A=np.array([[lamb]])) for lamb in [-10, -5, -1, 1, 5, 10]]
    N = 100

    Method = ExplicitEuler

    fig1, ax1 = setup_plot(
        title=f"{Method.__name__} - Absolute Errors vs Time\nMultiple Scalar Cases",
    )
    fig2, ax2 = setup_plot(
        title=f"{Method.__name__} - Relative Errors vs Time\nMultiple Scalar Cases",
    )

    kwargs = {
        "Method": Method,
        "ivps": ivps,
        "N": N,
        "label_gen": label_gen,
    }

    # Absolute errors
    run_err_ivps_vs_time(
        (fig1, ax1),
        err_type="absolute",
        **kwargs,
    )
    # Relative errors
    run_err_ivps_vs_time(
        (fig2, ax2),
        err_type="relative",
        **kwargs,
    )

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    
    plt.show()


### Visualizing Error Against [h = (t_{n+1} - t_{n})]

def test_err_vs_h_methods():
    """
    ## Scenario

    Plot the absolute error as `h` increases. `h` is the step size from the discretization 
    of the time interval, so it will reduce in size as N (number of steps) increases.

    Scalar Case: A = -1
    """

    # Setup IVP
    y0 = np.array([2])
    A = np.array([[-1]])

    ivp = LTE(t0=0, T=1, y0=y0, A=A)
    Ns = 2**np.arange(1, 16)

    Methods = [ExplicitEuler, ImplicitEuler, TrapezoidalEuler]

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

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    plt.show()

def test_err_vs_h_ivps():
    """
    ## Scenario

    Plot to compare the absolute error as versus h when varying A.

    Scalar Cases: A = -10 / -5 / -1 / 1 / 5 / 10
    """

    def label_gen(Method: type[Solver], ivp: IVP):
        return f"y0={ivp.y0[0]}, λ={ivp.A[0,0]}"

    ivps = [LTE(t0=0, T=4, y0=np.array([2]), A=np.array([[lamb]])) for lamb in [-10, -5, -1, 1, 5, 10]]
    Ns = 2**np.arange(2, 16)

    Method = TrapezoidalEuler

    fig1, ax1 = setup_plot(
        title=f"{Method.__name__} - Absolute Errors vs h\nMultiple Scalar Cases",
    )
    fig2, ax2 = setup_plot(
        title=f"{Method.__name__} - Relative Errors vs h\nMultiple Scalar Cases",
    )

    kwargs = {
        "Method": Method,
        "ivps": ivps,
        "Ns": Ns,
        "label_gen": label_gen,
    }

    # Absolute errors
    run_err_ivps_vs_h(
        (fig1, ax1),
        err_type="absolute",
        **kwargs,
    )
    # Relative errors
    run_err_ivps_vs_h(
        (fig2, ax2),
        err_type="relative",
        **kwargs,
    )
    
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    plt.show()
