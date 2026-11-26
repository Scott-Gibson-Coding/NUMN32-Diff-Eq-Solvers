import numpy as np
import scipy.sparse as sparse_sp
import scipy.sparse.linalg as sparse_spla

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

    def get_offset(self):
        """Return in units of `dx` the offset from the boundary node."""
        raise NotImplementedError("Expect boundary conditions to modify this")

    def update(self, dx, A, BCcorr, dir):
        """
        Pass dx, A and BCcorr with a direction. The values at the boundary of A and
        BCcorr will be updated by reference, then returned at the end of the function.

        BCcorr satisfies the following equation:

        `Au = dx^2 f - BCcorr`

        Inputs:
            dx: float interval size of points in the grid
            A: sparse array (N x N) Coefficient matrix
            BCcorr: np.array (N,) Boundary value correction terms
            dir: "left" | "right" Side of the boundary to place the condition.

        Outputs:
            A: sparse array (N x N)
            BCcorr: np.array (N,)
        """
        raise NotImplementedError("Expect boundary conditions to modify this")

class Dirichlet(BC):
    """
    Dirichlet boundary condition: `Dirichlet(3) --> y(x) = 3`
    """
    
    def __init__(self, value):
        super().__init__(value)

    def get_offset(self):
        return 1

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

    Using method 2 from the lecture notes, maintaining matrix A symmetry.
    """

    def __init__(self, value):
        super().__init__(value)

    def get_offset(self):
        return 1/2
    
    def update(self, dx, A, BCcorr, dir):
        self._check_dir(dir)

        match dir:
            case "left":
                BCcorr[0] = -dx*self.value
                A[0, 0] = A[0, 0] / 2
            case "right":
                BCcorr[-1] = dx*self.value
                A[-1, -1] = A[-1, -1] / 2

        return A, BCcorr

class Robin(BC):
    """
    Robin boundary condition: `Robin(a, b, 3) --> a*y(x) + b*y'(x) = 3`
    """

    def __init__(self, a, b, value):
        super().__init__(value)
        self.a = a
        self.b = b


### 

def FDMd2dx2(L: float, BCleft: BC, BCright: BC, N: int = 100):
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

    l_off = BCleft.get_offset()
    r_off = BCright.get_offset()

    # Compute grid spacing, and create grid
    dx = L / (N - 1 + l_off + r_off)
    grid = np.linspace(l_off*dx, L-r_off*dx, N)

    ### TODO How do we pass the above to the BCs to determine

    # Create A
    diag = np.ones(N) * -2
    off_diag = np.ones(N-1)
    A = sparse_sp.diags_array(
        [off_diag, diag, off_diag], offsets=[-1, 0, 1],
        shape=(N, N), format="csr",
    )

    # Create BCcorr
    BCcorr = np.zeros_like(grid)

    BCleft.update(dx, A, BCcorr, "left")
    BCright.update(dx, A, BCcorr, "right")

    return dx, grid, A, BCcorr
