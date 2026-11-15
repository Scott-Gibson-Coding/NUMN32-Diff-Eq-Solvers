import pytest
import numpy as np

from utility import get_error, check_literal, ErrorTypes

### Tests for check_literal()

def test_check_literal():
    """
    ## Scenario

    Test that the check_literal raises the proper error on an invalid string literal.
    """

    value = "absolute"
    bug = "volt"

    # Expect this call to pass
    check_literal(value, ErrorTypes)

    with pytest.raises(ValueError):
        # Expect this call to raise a ValueError
        check_literal(bug, ErrorTypes)


### Tests for get_error

def test_get_error_2_norm():
    """
    ## Scenario

    1. Check that the `get_error()` function uses the 2-norm for absolute error.
    2. Check that the `get_error()` function when using relative error returns the same
       value if the norm of `y2` is 1.
    """

    y1 = np.array([0, 1])
    y2 = np.array([1, 0])
    assert get_error(y1, y2) == pytest.approx(np.sqrt(2))
    assert get_error(y1, y2) == pytest.approx(get_error(y1, y2, "relative"))

def test_get_error_relative():
    """
    ## Scenario

    Test that the relative error is less than the absolute error if `||y2|| > 1`, and larger
    if `||y2|| < 1`.
    """
    
    y1 = np.array([0.234, 0.57])
    y2 = np.array([3.4, -1.2])

    # expected value = y2, ||y2|| > 1
    assert get_error(y1, y2, "relative") < get_error(y1, y2, "absolute")

    # expected value = y1, ||y1|| < 1
    assert get_error(y2, y1, "relative") > get_error(y2, y1, "absolute")
