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

### SOLVER CLASSES ###

class Solver():
    # Handles the time-stepping process, given an IVP instance
    def __init__(self, ivp: IVP, N: int):
        self.ivp = ivp
        self.u = ivp.y0
        self.N = N
        self.h = (ivp.T - ivp.t0) / N
        self.grid = np.linspace(ivp.t0, ivp.T, N + 1)
        self.us = np.zeros((N+1, ivp.y0.shape[0]))

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
        for i in range(1, self.N+1):
            self.u = self.step(self.grid[i], self.u)
            self.us[i,:] = self.u
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
        for i in range(1, self.N+1):
            self.u = self.step(self.grid[i], self.u)
            self.us[i,:] = self.u
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
        for i in range(1, self.N+1):
            self.u = self.step(self.grid[i], self.u)
            self.us[i,:] = self.u
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
        for i in range(1, self.N+1):
            self.u = self.step(self.grid[i], self.u)
            self.us[i,:] = self.u
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
