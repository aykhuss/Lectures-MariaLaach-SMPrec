#!/usr/bin/env python
import math
import vegas
import numpy as np
import sys

# the constants in the virtual and real corrections defined globally
a: float = 1.
b: float = 1.

# the measurement function (global flag as switch)
iJ: int = 1
def J(x: float) -> float:
    if iJ == 0:
        # total cross section (= a + b)
        return 1.
    elif iJ == 1:
        # linear dependence => linear error term
        return 1. - x
    elif iJ == 2:
        # quadratic dependence & error
        return 1. - x**2
    elif iJ == 3:
        # a more complex one (quadratic)
        return math.cosh(x)
    else:
        raise RuntimeError(
            "unknown measurement function switch: {}".format(iJ))

def subtr_dNLO(ncall: int) -> tuple[float, float]:

    def integrand(x: list[float]) -> float:
        return (1. + b * x[0]) / x[0] * (J(x[0]) - J(0))

    integ = vegas.Integrator([[0, 1]])
    integ(integrand, nitn=10, neval=(ncall / 10))  # 10% into adaption
    result = integ(integrand, nitn=10, neval=ncall)

    return ((a + b) * J(0) + result.mean, result.sdev)

def slice_dNLO(xi: float, ncall: int) -> tuple[float, float]:

    if (xi <= 0.) or (xi > 1.):
        raise ValueError("cutoff not in valid range [0,1]: {}".format(xi))

    def integrand(x: list[float]) -> float:
        return (1. + b * x[0]) / x[0] * J(x[0])

    integ = vegas.Integrator([[xi, 1]])
    integ(integrand, nitn=10, neval=(ncall / 10))  # 10% into adaption
    result = integ(integrand, nitn=10, neval=ncall)

    return ((a + math.log(xi)) * J(0) + result.mean, result.sdev)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise RuntimeError("I expect at least two arguments:  [subtr|slice]  ncall")
    ncall = int(sys.argv[2])
    if sys.argv[1].lower() == "subtr":
        res_subtr = subtr_dNLO(ncall=ncall)
        print("{:e} {:e}".format(*res_subtr))
    elif sys.argv[1].lower() == "slice":
        if len(sys.argv) == 6:
            ilow = int(sys.argv[3])
            iupp = int(sys.argv[4])
            nsteps = int(sys.argv[5])
        else:
            ilow = -3
            iupp = -3
            nsteps = 1
        res_subtr = subtr_dNLO(ncall=ncall)
        for xi in np.logspace(ilow, iupp, nsteps):
            res_slice = slice_dNLO(xi=xi, ncall=ncall)
            Del_val = res_slice[0] - res_subtr[0]
            Del_err = math.sqrt(res_slice[1]**2 + res_subtr[1]**2)
            print("{:e}  {:e} {:e}  {:+e} {:e}".format(
                xi, *res_slice, Del_val, Del_err))
    else:
        raise RuntimeError("unrecognised mode: {}".format(sys.argv[1]))
