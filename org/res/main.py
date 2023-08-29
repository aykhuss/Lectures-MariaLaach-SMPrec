#!/usr/bin/env python
import sys
from math import pi, exp, log, log10, ceil, floor
from scipy.special import jv  # Bessel function of the 1st kind
from scipy.integrate import quad
import numpy as np

alpha_s: float = 0.118


def res_integrand(b: float, QT: float, Q: float, CX: float) -> float:
    # b0: float = 2. * exp(-0.57721566490153286061)
    # blim: float = 5.  # should be > 1/Lambda_QCD ~ 5
    # bs2: float = b**2 * blim**2 / (b**2 + blim**2)
    # return (b / 2.) * jv(0, b * QT) * exp(
    #     -alpha_s / (2. * pi) * CX * log(Q**2 * bs2 / b0**2 + 1.)**2)
    return (b / 2.) * jv(0, b * QT) * exp(
        -alpha_s / (2. * pi) * CX * log(Q**2 * b**2)**2)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise RuntimeError("I expect at least two arguments:  Q [g|q]")
    Q = float(sys.argv[1])  # the hard scale
    pow_low = -4
    pow_upp = floor(log10(Q)) # ceil(log10(Q/2.))
    if sys.argv[2].lower() == "q":
        CX = 4. / 3.
    elif sys.argv[2].lower() == "g":
        CX = 3.
    else:
        raise RuntimeError("unrecognised parton: {}".format(sys.argv[2]))

    if len(sys.argv) >= 4:
        alpha_s = float(sys.argv[3])

    if len(sys.argv) >= 5:
        nsteps = int(sys.argv[4])
    else:
        nsteps = 51

    # print("# qt dSigQT2_val dSigQT2_err")
    for qt in np.logspace(pow_low, pow_upp, nsteps):
        val, err = quad(res_integrand,
                        0.,
                        np.inf,
                        args=(qt, Q, CX),
                        epsabs=0.,
                        epsrel=1e-3,
                        limit=50000)
        print("{}  {} {}".format(qt, val, err))
