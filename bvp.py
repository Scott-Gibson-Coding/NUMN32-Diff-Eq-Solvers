import numpy as np
import scipy.sparse.linalg as sp_sparse_linalg

### BOUNDARY CONDITIONS

class BC():
    """
    Boundary condition
    """
    
    def __init__(self, value):
        self.value = value

    def _check_dir(self, dir):
        """Specify the allowed direction literals."""

        if dir not in ["left", "right"]:
            raise ValueError(f"Invalid boundary condition direction, {dir=}.")

    def update(self, dx, A, BCcorr, dir):
        """
        Pass dx, A and BCcorr with a direction. The values at the boundary of A and
        BCcorr will be updated by reference, then returned at the end of the function.

        BCcorr satisfies the following equation:

        `Au = dx^2 f - BCcorr`

        Inputs:
            dx: float interval size of points in the grid
            A: np.array (N x N) Coefficient matrix
            BCcorr: np.array (N,) Boundary value correction terms
            dir: "left" | "right" Side of the boundary to place the condition.

        Outputs:
            A: np.array (N x N)
            BCcorr: np.array (N,)
        """
        raise NotImplementedError("Expect boundary conditions to modify this")

class Dirichlet(BC):
    """
    Dirichlet boundary condition: `Dirichlet(3) --> y(x) = 3`
    """
    
    def __init__(self, value):
        super().__init__(value)

    def update(self, dx, A, BCcorr, dir):
        self._check_dir(dir)

        match dir:
            case "left":
                BCcorr[0] = self.value
            case "right":
                BCcorr[-1] = self.value

        return A, BCcorr

class Neumann(BC):
    """
    Neumann boundary condition: `Neumann(3) --> y'(x) = 3`
    """

    def __init__(self, value):
        super().__init__(value)

class Robin(BC):
    """
    Robin boundary condition: `Robin(a, b, 3) --> a*y(x) + b*y'(x) = 3`
    """

    def __init__(self, a, b, value):
        super().__init__(value)
        self.a = a
        self.b = b


### 

def FDMd2dx2 (L: float, BCleft: BC, BCright: BC, N: int = 100):
    """
    Compute the Finite Difference Method (FDM) discretization

    `u |--> Au + BCcorr`

    of the differential operator `y''` on `x ∈ [0 , L]`, given the boundary conditions 
    BCleft at x = 0 and BCright at x = L, using N computational points.

    Inputs:
        L: float > 0
        BCleft: BC object (e.g. Dirichlet, Neumann)
        BCright: BC object (e.g. Dirichlet, Neumann)
        N: integer > 0

    Outputs:
        dx: float, grid spacing
        grid: np.array (N,) the computational points
        A: np.array or scipy.sparse matrix (N, N)
        BCcorr: np.array (N, 1) or (N,) containing boundary condition correction terms.
    """

    if L <= 0 or N <= 0:
        raise ValueError("Invalid length or discretization size")

    # Compute grid spacing, and create grid
    dx = L / (N + 1)
    grid = np.linspace(dx, L-dx, N)

    # Create A
    sub_diag = np.ones(grid.shape[0] - 1)
    diag = np.ones(grid.shape[0]) * -2
    sup_diag = np.ones(grid.shape[0] - 1)
    A = np.diag(diag) + np.diag(sub_diag, -1) + np.diag(sup_diag, 1)

    # Create BCcorr
    BCcorr = np.zeros_like(grid)

    BCleft.update(dx, A, BCcorr, "left")
    BCright.update(dx, A, BCcorr, "right")

    return dx, grid, A, BCcorr
