# Numerical Methods for Differential Equations

## Project 0

The initial project introducing some of the simpler solvers for a specific set of Initial Value Problems (IVP). Namely, the Linear Test Equation $y' = Ay$, which has a known exact solution $y = y_0\cdot e^{At}$. The solution for the differential equation then depends on two terms $A$, and $y_0$.

### Notation

The differential equation is specified in terms of $y$ and $t$. $y(t)$ is the exact solution at time $t$, and the vector field $f = y'$ lets us use either $y'(y, t)$ or $f(y, t)$ to reference the value of $y'$ at a specific point and time. In my notation I tend to use both interchangeably.

The exact solution will use the $y$ variable, and an estimate achieved by some solver will use $u$. So if we are $n$ steps into the solver, we write $l_n = ||y_n - u_n||_2$ as the local error for the $n$-th step.

The discretization in code will be determined by a parameter $N$. This specifies the number of steps to take from initial time $t0$ to the final time $T$. So if all steps are stored, then the array of step estimates will be of length $N+1$ which includes the starting and ending points. $h$ refers to the length of the intervals, which are all the same, and is computed by $h = (T-t0) / N$.

### Solvers

In project 0 we developed three solvers:

- **Explicit Euler**: An explicit method which uses the value at $y'(u_{n},t_{n})$ to get the next step. Very efficient but can be shown to be unstable. Error reduces with $O(h)$.
- **Implicit Euler**: An implicit method which requires knowledge of $y'(u_{n+1},t_{n+1})$ to get the next step. This is in general a more expensive operation, but with the LTE just boils down to solving a linear system. Error reduces with $O(h)$.
    - Since the linear system is dependant only on $h$ and $A$, $h$ being the interval size determined by the discretization used, we save the linear system using the **LU** decomposition to acheive a speedup.
- **Trapezoidal Rule**: A combination of the values evaluated per step from both the Explicit and Implicit methods. This method, similar to the Implicit method, requires solving a linear system when applied to the LTE, so similar speedup techniques are implemented as described above. Error reduces with $O(h^2)$.

### Implementation Notes

There are three families of classes implemented.

The `IVP` class describing an initial value problem, and exposes methods to evaluate the problem at a certain point and time.

Then the `Solver` class which accepts an `IVP` and exposes an `integrate()` function which interacts with the `IVP` to estimate a solution from starting time to ending time. The discretization size $N$ determining the step size is passed to this solver.

And finally an `Evaluator` class then is used to evaluate four types of error, combinations of (err vs time/err vs h), against (absolute error/relative error).

---

Interfacing with the above classes included a lot of repeated code when writing individual scenarios in the `test_proj0` file. So a function to generate a single subplot is provided in the `plot.py` file, and several functions which execute a solver/evaluator on an ivp/method are exposed in the `ivp_run.py` file.

The `ivp_run.py` file has functions which allow the caller to choose either plotting multiple solver methods running on a single IVP, or passing multiple IVPs through a single method. Because of this, the functions in that file are split into pairs, with the bulk of the actual execuation and plotting done in the first function, and the second just modifying the parameters accepted before passing them to the first function. Reference the functions in `test_proj0` for a reference of how to use these.
