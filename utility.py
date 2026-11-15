import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as t_np
from typing import get_args, Literal, TypeAlias

### TESTING UTILS

# Override plt.show to do nothing, useful for running coverage tests so that you don't
# have to manually close all plots.
# plt.show = lambda: None

### UTILITY FUNCTIONS

def get_error(y1: t_np.NDArray, y2: t_np.NDArray, err_type: ErrorTypes = "absolute") -> float:
    """
    ## Description

    Returns the error between y1 and y2 using the 2-norm, `err_type` controls how the error is computed.
    
    - "absolute" simply computes: ||y1 - y2||.
    - "relative" computes the error relative to `y2`, the "expected" value: (||y1 - y2||/||y2||).
    set use_absolute

    :param y1: Computed value.
    :param y2: Actual, expected value.
    :err_type: Specifies the type of error to compute.
    """

    # Utilize numpy norm() function, which defaults to p-2 norm on vectors.
    norm = np.linalg.norm

    match err_type:
        case "absolute":
            return norm(y1 - y2)
        case "relative":
            return norm(y1 - y2) / norm(y2)
        case _:
            # Default to "absolute"
            return norm(y1 - y2)

def check_literal(value: any, L, err_str=None):
    """
    ## Description

    Helper function working with type checking. Pass a value and a Literal type,
    and this function will raise a ValueError if the value is not in the Literal.

    :param value: Value to compare against Literal.
    :param L: Python Literal type.
    :param err_str: Optional error override.
    """

    if not err_str:
        err_str = f"Invalid value passed: {value}"

    if value not in get_args(L):
        raise ValueError(err_str)


### LITERAL TYPES

# Types of error calculations, usually some combination of absolute
# and relative errors.
ErrorTypes: TypeAlias = Literal["absolute", "relative"]