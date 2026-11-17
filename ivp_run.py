###
# Note, for many functions in this file, you will only need to specify the title
# for the plot. The axis x and y labels will be determined by the function you call
# and occassionally also by the initial value problem used.

import numpy as np
import numpy.typing as t_np
import matplotlib.pyplot as plt

from ivp import *
from plot import decorate_axis
from utility import check_literal, ErrorTypes

def get_exact_solution(ivp: IVP, ts: t_np.NDArray) -> t_np.NDArray:
    """
    ## Description

    Takes in a grid of timesteps `ts`, and if the IVP implements an `exact()` method,
    runs that on each timestep and returns an nd array of solutions for each step.

    :param ivp: Initial Value Problem we expect to implement the `exact()` method. Will
    throw a NotImplementedError otherwise.
    :param ts: Grid of timesteps.
    """
    
    return np.array([ivp.exact(t) for t in ts])

def run_ivp_methods(
        fig_ax: tuple[plt.Figure, plt.Axes],
        Methods: list[type[Solver]] = [ExplicitEuler],
        ivp: IVP = LTE(0, 1, np.array([2]), np.array([[-1]])),
        N=100,
        include_exact=False,
        no_labels=False,
        label_rep="",
        color=None,
    ):
    """
    ## Description

    Accepts a plotting (fig, ax) tuple as returned by `plt.subplots()`.

    Pass a list of methods inheriting from `Solver` to integrate an approximate solution.
    Each one will be plot against the others as dashed lines. Only the exact solution
    will be solid gray line and higher opacity.

    For LTE's we have the following:
        Scalar Case: A = [[λ]]
                    y0 = [a] (scalar)
        Matrix Case: A = Square Matrix (n x n)
                    y0 = n-vector
    
    For plotting we divide the above into three cases. If the domain is in R1, we plot the
    value `y` by time. If the domain is in R2, then we plot the first component of `y` `y[0]`
    vs the second component `y[1]`. For all cases Rn, n > 2, we simply plot the first component
    `y[0]` vs time, similar to the first case.

    Note: Only set `include_exact` to True if the `IVP` implemented exposes an `exact()`
    function.

    :param fig_ax: (fig, ax) object from plt.subplots()
    :param Methods: Array of solver classes.
    :param ivp: Initial Value Problem.
    :param N: Discretization size.
    :param include_exact: Set to True to show the exact solution if possible. Defaults to False.
    :param no_labels: Set to True to avoid attaching a label to the plots. This is useful if you
    are plotting many different initial conditions on the same plot.
    :param label_rep: If labels are set to draw, you can set this value. It will replace the
    default label this function would ordinarily give to the plotted method.
    :param color: Set to draw the approximations using this one color string. Most useful if you
    are plotting many different initial conditions on the same plot, but only using one Method.
    """

    (fig, ax) = fig_ax
    y_dim = ivp.y0.shape[0] # Dim of y

    # Draw labels for x/y axes
    if y_dim == 1:
        decorate_axis(ax, xlabel="t", ylabel="y")
    elif y_dim == 2:
        decorate_axis(ax, xlabel="y[0]", ylabel="y[1]")
    else:
        decorate_axis(ax, xlabel="t", ylabel="y[0]")

    # Loop through methods
    for Method in Methods:
        solver = Method(ivp=ivp, N=N)
        solver.integrate()

        label = None
        if not no_labels:
            label = label_rep if label_rep else Method.__name__
        if y_dim == 1:
            # Draw time vs y0
            ax.plot(solver.grid, solver.us, color=color, linestyle="--", alpha=0.5, label=label)
        elif y_dim == 2:
            # Draw y0 vs y1
            ax.plot(solver.us[:,0], solver.us[:,1], color=color, linestyle="--", alpha=1, label=label)
        else:
            # Draw time vs y0
            ax.plot(solver.grid, solver.us[:,0], color=color, linestyle="--", alpha=0.5, label=label)
    if include_exact:
        # Rely on the grid from the last solver
        exact_sol = get_exact_solution(ivp=ivp, ts=solver.grid)
        label = None if no_labels else "Exact Solution"
        if y_dim == 1:
            # Draw time vs y0
            ax.plot(solver.grid, exact_sol, color="gray", alpha=0.7, label=label)
        elif y_dim == 2:
            # Draw y0 vs y1
            ax.plot(exact_sol[:,0], exact_sol[:,1], color="gray", alpha=0.7, label=label)
            # Draw a dot at the start point
            start_x, start_y = exact_sol[0,0], exact_sol[0,1]
            ax.scatter(start_x, start_y, color="gray", s=20, zorder=5)
            ax.text(start_x+0.01, start_y+0.01, "y(t0)")
        else:
            # Draw time vs y0
            ax.plot(solver.grid, exact_sol[:,0], color="gray", alpha=0.7, label=label)

def run_mult_ivp_method(
        fig_ax: tuple[plt.Figure, plt.Axes],
        Method: type[Solver] = ExplicitEuler,
        ivps: list[IVP] = [LTE(0, 1, np.array([2]), np.array([[-1]]))],
        N=100,
        include_exact=False,
        label_gen: function = lambda Method, ivp: None
    ):
    """
    ## Description

    Takes a fixed Method, and an array of ivps, and plots them all on the same graph. For instance
    you could fix the LTE `A` param, and plot a comparison of different starting points y0.

    To reduce clutter and noise, no labels are attached and the colors are all chosen to be a single
    one. This works well until the plots begin to intersect or cluster around each other.
    
    :param label_gen: Only new parameter compared to `run_ivp_methods()`. Defined by
    the caller, can create a label name from the method and ivp. Expect syntax of:

    `label_gen(Method: type[Solver], ivp: IVP) -> string`
    """

    # TODO label_gen has not yet been passed to `run_ivp_methods()`.

    for ivp in ivps:
        run_ivp_methods(
            fig_ax,
            Methods=[Method],
            ivp=ivp,
            N=N,
            include_exact=include_exact,
            no_labels=True,
            color="#2b7b9a8f",
        )

def run_err_methods_vs_time(
        fig_ax: tuple[plt.Figure, plt.Axes],
        Methods: list[type[Solver]] = [ExplicitEuler],
        ivp: IVP = LTE(0, 1, np.array([2]), np.array([[-1]])),
        N=100,
        err_type: ErrorTypes = "absolute",
        no_labels=False,
        label_rep="",
        color=None,
    ):
    """
    ## Description
    
    Takes a single ivp, and multiple methods, and plots a comparison of the error, either relative 
    or absolute, on a semilogy plot utilizing the `Evaluator` class.

    :param fig_ax: (fig, ax) object from plt.subplots()
    :param Methods: Array of solver classes.
    :param ivp: Initial Value Problem.
    :param N: Discretization size.
    :param err_type: "absolute" or "relative", defaults to "absolute".
    :param no_labels: Set to True to avoid attaching a label to the plots. This is useful if you
    are plotting many different initial conditions on the same plot.
    :param label_rep: If labels are set to draw, you can set this value. It will replace the
    default label this function would ordinarily give to the plotted method.
    :param color: Set to draw the approximations using this one color string. Most useful if you
    are plotting many different initial conditions on the same plot, but only using one Method.
    """

    check_literal(err_type, ErrorTypes, "Invalid error type passed.")

    (fig, ax) = fig_ax

    # Decorate axes
    match err_type:
        case "absolute":
            ylabel = "Absolute Error"
        case "relative":
            ylabel = "Relative Error"
        case _:
            ylabel = "Error"
    decorate_axis(ax, xlabel="t", ylabel=ylabel)

    # Loop through methods
    for Method in Methods:
        evaluator = Evaluator(ivp, Method)

        # Evaluate vs time
        errs, grid = evaluator.errvstime(N, err_type)
        
        label = None
        if not no_labels:
            label = label_rep if label_rep else Method.__name__

        # Draw errors, solid for absolute, dashed otherwise
        match err_type:
            case "absolute":
                linestyle = "-"
            case _:
                linestyle = "--"

        ax.semilogy(grid, errs, color=color, linestyle=linestyle, alpha=0.6, label=label)

def run_err_ivps_vs_time(
        fig_ax: tuple[plt.Figure, plt.Axes],
        Method: type[Solver] = ExplicitEuler,
        ivps: list[IVP] = [LTE(0, 1, np.array([2]), np.array([[-1]]))],
        N=100,
        err_type: ErrorTypes = "absolute",
        label_gen: function = lambda Method, ivp: None
    ):
    """
    ## Description

    Takes a fixed Method, and an array of ivps, and plots the errors all on the same graph. 
    For instance you could fix the LTE `y0` param, and plot a comparison of Linear Systems A.

    :param label_gen: Only new parameter compared to `run_err_methods_vs_time()`. Defined by
    the caller, can create a label name from the method and ivp. Expect syntax of:

    `label_gen(Method: type[Solver], ivp: IVP) -> string`
    """

    for ivp in ivps:
        run_err_methods_vs_time(
            fig_ax,
            Methods=[Method],
            ivp=ivp,
            N=N,
            err_type=err_type,
            label_rep=label_gen(Method, ivp)
        )

def run_err_methods_vs_h(
        fig_ax: tuple[plt.Figure, plt.Axes],
        Methods: list[type[Solver]] = [ExplicitEuler],
        ivp: IVP = LTE(0, 1, np.array([2]), np.array([[-1]])),
        Ns=2**np.arange(2, 10), # Discretization on increasing powers of 2
        err_type: ErrorTypes = "absolute",
        no_labels=False,
        label_rep="",
        color=None,
    ):
    """
    ## Description
    
    Takes a single ivp, and multiple methods, and plots a comparison of the error, either relative 
    or absolute, on a loglog plot utilizing the `Evaluator` class.

    :param fig_ax: (fig, ax) object from plt.subplots()
    :param Methods: Array of solver classes.
    :param ivp: Initial Value Problem.
    :param Ns: List of discretization sizes, for example increasing powers of 2.
    :param err_type: "absolute" or "relative", defaults to "absolute".
    :param no_labels: Set to True to avoid attaching a label to the plots. This is useful if you
    are plotting many different initial conditions on the same plot.
    :param label_rep: If labels are set to draw, you can set this value. It will replace the
    default label this function would ordinarily give to the plotted method.
    :param color: Set to draw the approximations using this one color string. Most useful if you
    are plotting many different initial conditions on the same plot, but only using one Method.
    """

    check_literal(err_type, ErrorTypes, "Invalid error type passed.")

    (fig, ax) = fig_ax

    # Decorate axes
    match err_type:
        case "absolute":
            ylabel = "Absolute Error"
        case "relative":
            ylabel = "Relative Error"
        case _:
            ylabel = "Error"
    decorate_axis(ax, xlabel="h", ylabel=ylabel)

    # Loop through methods
    for Method in Methods:
        evaluator = Evaluator(ivp, Method)

        # Evaluate vs h
        errs, grid = evaluator.errvsh(Ns, err_type)

        label = None
        if not no_labels:
            label = label_rep if label_rep else Method.__name__

        # Draw errors, solid for absolute, dashed otherwise
        match err_type:
            case "absolute":
                linestyle = "-"
            case _:
                linestyle = "--"
                
        ax.loglog(grid, errs, color=color, linestyle=linestyle, alpha=0.6, label=label)

def run_err_ivps_vs_h(
        fig_ax: tuple[plt.Figure, plt.Axes],
        Method: type[Solver] = ExplicitEuler,
        ivps: list[IVP] = [LTE(0, 1, np.array([2]), np.array([[-1]]))],
        Ns=2**np.arange(2, 10),
        err_type: ErrorTypes = "absolute",
        label_gen: function = lambda Method, ivp: None
    ):
    """
    ## Description

    Takes a fixed Method, and an array of ivps, and plots the errors all on the same graph. 
    For instance you could fix the LTE `y0` param, and plot a comparison of Linear Systems A.

    :param label_gen: Only new parameter compared to `run_err_methods_vs_time()`. Defined by
    the caller, can create a label name from the method and ivp. Expect syntax of:

    `label_gen(Method: type[Solver], ivp: IVP) -> string`
    """

    for ivp in ivps:
        run_err_methods_vs_h(
            fig_ax,
            Methods=[Method],
            ivp=ivp,
            Ns=Ns,
            err_type=err_type,
            label_rep=label_gen(Method, ivp)
        )
