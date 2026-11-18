import math
import numpy as np
import scipy.linalg as sp_alg

from utility import get_error, check_literal, ErrorTypes

### IVP CLASSES ###

class IVP():
    # Stores initial conditions, and ODE.
    def __init__(self, t0, T, y0):
        self.t0 = t0    # Initial time
        self.T = T      # Total length
        self.y0 = y0    # Initial condition

    def f(self, t, y):
        # Evaluate the vector field
        raise NotImplementedError("Subclasses should implement this function!")

    def exact(self, t):
        # Return the exact solution at time t if it is known
        raise NotImplementedError("Subclasses should implement this function!")

class LTE(IVP):
    # Linear Test Equation
    # Inherits from IVP to support a linear operator A.
    def __init__(self, t0, T, y0, A):
        super().__init__(t0, T, y0)
        self.A = A

    def f(self, t, y):
        return self.A @ y

    def exact(self, t):
        # Compute e^(tA)*y0
        return sp_alg.expm(t*self.A) @ self.y0

class LotkaVolterra(IVP):
    def __init__(self, t0, T, y0, a, b, c, d):
        # Lotka-Volterra is R2 -> R2, so y0 should be a 2-vector
        super().__init__(t0, T, y0)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def f(self, t, y):
        a, b, c, d = self.a, self.b, self.c, self.d
        u0 = a*y[0] - b*y[0]*y[1]
        u1 = c*y[0]*y[1] - d*y[1]
        return np.array([u0, u1])

    def H(self, u):
        # Invariant function among solutions
        a, b, c, d = self.a, self.b, self.c, self.d
        x, y = u[0], u[1]
        return c*x + b*y - d*np.log(x) - a*np.log(y)

class VanderPol(IVP):
    def __init__(self, t0, T, y0, mu):
        # Vanderpol is also R2 -> R2, y0 should be a 2-vector
        super().__init__(t0, T, y0)
        self.mu = mu

    def f(self, t, y):
        mu = self.mu
        u0 = y[1]
        u1 = mu*(1 - y[0]**2)*y[1] - y[0]
        return np.array([u0, u1])

### SOLVER CLASSES ###

class Solver():
    # Handles the time-stepping process, given an IVP instance
    # TODO Default N to a value like 100, so that adaptive grid sizes do not need to specify one.
    def __init__(self, ivp: IVP, N: int, tol: float = 1e-4, adaptive: bool = False):
        self.ivp = ivp # Initial value problem
        self.u = ivp.y0 # Current step approximation
        self.N = N # Discretization intervals
        self.h = (ivp.T - ivp.t0) / N # Interval length for a fixed grid
        self.tol = tol # Goal error tolerance
        self.adaptive = adaptive # Set to False to ensure a fixed grid

        if not adaptive:
            # Fixed grid size
            self.grid = np.linspace(ivp.t0, ivp.T, N + 1)
            self.us = np.zeros((N+1, ivp.y0.shape[0]))
        else:
            # Adaptive grid size, populated during integration.
            # Note: At the end of integration, transform the grid and us into numpy arrays
            # matching how they'd behave in a non-adaptive solver.
            self.grid = []
            self.us = []


    def newstep(self, k, hn, errest, errestold):
        # Compute the next step size based on the current and previous local error.
        # Param `k`: Expected order of convergence: O(h^k)
        # Param `hn`: The current step size
        # Param `errest`: The current local error
        # Param `errestold`: The prior step local error
        raise NotImplementedError("Subclasses should implement this function!")

    def step(self, tn, un):
        # Compute the next estimate from the prior step
        raise NotImplementedError("Subclasses should implement this function!")

    def integrate(self):
        # Perform N steps using the solver step method.
        # Should return us, but the user can also access through class fields.
        raise NotImplementedError("Subclasses should implement this function!")

class ExplicitEuler(Solver):
    def __init__(self, ivp, N):
        super().__init__(ivp, N)

    def step(self, tn, un):
        return un + self.h * self.ivp.f(tn, un)

    def integrate(self):
        self.us[0,:] = self.u
        for i in range(self.N):
            self.u = self.step(self.grid[i], self.u)
            self.us[i+1,:] = self.u
        return self.us

class ImplicitEuler(Solver):
    """
    ## Description

    Only setup to work with the Linear Test Equation (LTE) class. Uses the linear system
    `A` directly to compute the next step.
    """

    def __init__(self, ivp, N):
        super().__init__(ivp, N)

        # Compute reusable LU decomp of LHS
        h = self.h
        A = self.ivp.A
        I = np.eye(A.shape[0])
        # Store (lu, piv) to pass to scipy lu_solve
        self.lu_piv = sp_alg.lu_factor(I - h*A)

    def step(self, tn, un):
        # Solve the linear system (I - A*h)u_{n+1} = u_{n}

        return sp_alg.lu_solve(self.lu_piv, un)
        # return sp_alg.solve(np.eye(A.shape[0]) - self.h * A, un)

    def integrate(self):
        self.us[0,:] = self.u
        for i in range(self.N):
            self.u = self.step(self.grid[i], self.u)
            self.us[i+1,:] = self.u
        return self.us

class TrapezoidalEuler(Solver):
    """
    ## Description

    Only setup to work with the Linear Test Equation (LTE) class. Uses the linear system
    `A` directly to compute the next step.
    
    Note: Using LU decomp provided a speedup of slightly over 2x.
    """

    def __init__(self, ivp, N):
        super().__init__(ivp, N)

        # Compute reusable LU decomp of LHS
        h = self.h
        A = self.ivp.A
        I = np.eye(A.shape[0])
        # Store (lu, piv) to pass to scipy lu_solve
        self.lu_piv = sp_alg.lu_factor(2*I - h*A)

    def step(self, tn, un):
        # Solve the linear system (2*I - h*A)u_{n+1} = (2*u_{n} + h*A*u_{n})

        h = self.h
        A = self.ivp.A
        return sp_alg.lu_solve(self.lu_piv, 2*un + h*A@un)

    def integrate(self):
        self.us[0,:] = self.u
        for i in range(self.N):
            self.u = self.step(self.grid[i], self.u)
            self.us[i+1,:] = self.u
        return self.us

class RK4(Solver):
    def __init__(self, ivp, N):
        super().__init__(ivp, N)

    def step(self, tn, un):
        h = self.h

        ivp = self.ivp
        Y1 = ivp.f(tn, un)
        Y2 = ivp.f(tn + h/2, un + h/2*Y1)
        Y3 = ivp.f(tn + h/2, un + h/2*Y2)
        Y4 = ivp.f(tn + h, un + h*Y3)
        
        return un + h/6 * (Y1 + 2*Y2 + 2*Y3 + Y4)

    def integrate(self):
        self.us[0,:] = self.u
        for i in range(self.N):
            self.u = self.step(self.grid[i], self.u)
            self.us[i+1,:] = self.u
        return self.us

class RK34(Solver):
    def __init__(self, ivp: IVP, tol: float = 1e-4):
        # TODO What is the purpose of `N` in an adaptive solver?
        super().__init__(ivp=ivp, N=100, tol=tol, adaptive=True)

        self.k = 4 # Order of convergence
        self.errest = None # Current local error
        self.errestold = self.tol # Last step local error

        # Compute starting step size
        self.h = np.abs(self.ivp.T - self.ivp.t0)*(self.tol**(1/self.k))/(100*(1+np.linalg.norm(self.ivp.f(self.ivp.t0,self.ivp.y0))))

    def newstep(self):
        tol = self.tol
        k = self.k
        hn = self.h
        e = self.errest
        e_prev = self.errestold

        fir = pow(tol/e, 2/(3*k))
        sec = pow(tol/e_prev, -1/(3*k))
        return fir * sec * hn

    def step(self, tn, un):
        h = self.h

        ivp = self.ivp
        Y1 = ivp.f(tn, un)
        Y2 = ivp.f(tn + h/2, un + h/2*Y1)
        Y3 = ivp.f(tn + h/2, un + h/2*Y2)
        Z3 = ivp.f(tn + h, un - h*Y1 + 2*h*Y2)
        Y4 = ivp.f(tn + h, un + h*Y3)

        # Store local error compared to RK4 using 2-norm: l_{n+1}
        self.errest = np.linalg.norm(h/6*(2*Y2 + Z3 - 2*Y3 - Y4))
        
        return un + h/6 * (Y1 + 4*Y2 + Z3)

    def integrate(self):
        tn = self.ivp.t0
        
        # Init grid and us with starting point
        self.grid.append(tn)
        self.us.append(self.u)

        # Loop until you hit T
        # TODO There is an issue with this implementation. We are incrementing tn first before
        # passing it to self.step(). This means we are effectively computing self.step(t_{n+1}, u_{n})
        # which is incorrect. We should do the following:
        #   - Figure out the step size, clamped by T
        #   - Perform the step, computing u_{n+1}
        #   - Compute t_{n+1} by adding the step size to tn clamped by T
        #   - Append both u_{n+1} and t_{n+1} to the respective arrays
        #   - Update h using self.newstep()
        while tn < self.ivp.T:
            # Increment tn by h, "clamp by T" and update h if needed
            tn = min(tn + self.h, self.ivp.T)
            if tn - self.grid[-1] < self.h:
                self.h = tn - self.grid[-1]

            # Perform step
            self.u = self.step(tn, self.u)
            self.grid.append(tn)
            self.us.append(self.u)

            # Update h
            self.h = self.newstep()

        # Convert the dynamic grid and us to static np arrays
        self.grid = np.array(self.grid)
        self.us = np.vstack(self.us)
        return self.us

### EVALUATOR CLASSES ###

class Evaluator():
    def __init__(self, ivp: IVP, Solver: Solver):
        self.ivp = ivp
        self.Solver = Solver

    def errvstime(self, N, err_type: ErrorTypes = "absolute"):
        check_literal(err_type, ErrorTypes)
        
        solver = self.Solver(self.ivp, N)
        grid = solver.grid
        us = solver.integrate()
        ys = np.zeros(us.shape)

        for i, tk in enumerate(grid):
            ys[i,:] = self.ivp.exact(tk)

        errs = np.array([get_error(us[i], ys[i], err_type) for i in range(N + 1)])
        return errs, grid

    def errvsh(self, Ns, err_type: ErrorTypes = "absolute"):
        check_literal(err_type, ErrorTypes)

        errs = []
        hs = []

        # Compute the exact solution at time T
        exact_sol = self.ivp.exact(self.ivp.T)

        for N in Ns:
            solver = self.Solver(self.ivp, N)
            us = solver.integrate()
            errs.append(get_error(us[-1], exact_sol, err_type))
            hs.append(solver.h)
        return np.array(errs), np.array(hs)
