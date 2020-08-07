import numpy as np
import pandas as pd


def BackgroundSNIPCalc(Power, nIterations=20, ApplyLLS=False, ProtectRange=100, ProtectIterations=5):
    # This function takes an input power spectrum and attempts to calculate the background
    # (i.e. everything that is 'smooth' and not a peak). This is done following the 1D
    # Sensitive Nonlinear Iterative Peak (SNIP) clipping algorithm. The inputs to this are
    # the number of iterations to do and the boolean ApplyLLS. If ApplyLLS: Apply the
    # Log-Log-Square root operator. This has the benefit of enhancing relatively small
    # peaks in the spectrum, though in some cases this may be undesirable (too sensitive).

    # First we create an empty numpy array that is meant to store each of the steps of the
    # iteration process. Afterwards we apply the LLS operator if desired.
    Iterations = np.empty((nIterations+1,len(Power)))
    if ApplyLLS: Iterations[0,:] = np.log( np.log( np.sqrt(Power + 1) + 1 ) + 1 )
    else: Iterations[0,:] = Power
    
    # Unfortunately I couldn't think of a pythonic way to do this elegantly. This is
    # immensely disatisfying, but it does appear to work okay. Fortunately this isn't being
    # used on any very large arrays where performance is important. Anyway, we begin the
    # iteration process. Essentially we are performing a 'smoothing' by looking at the
    # minimum of the current point and the average of its neighbors (not necessarily its
    # adjacent neighbors).
    for n in range(1,nIterations+1):
        for j in range(1, len(Power)):
            nProtect = n if n < 5 else ProtectRange
            if j >= nProtect and j+n < len(Power):
                Iterations[n,j] = min( (Iterations[n-1,j-n] + Iterations[n-1,j+n])/2.0,
                                       Iterations[n-1, j])
            else:
                Iterations[n,j] = Iterations[n-1,j]

    BG = Iterations[-1]
    if ApplyLLS: BG = np.square( np.exp( ( np.exp(BG) - 1 ) ) - 1 ) - 1

    # The last iteration should contain the best estimate for the background of the power
    # spectrum. Note that the more iterations are used, the 'smoother' the background
    # gets and more variation tends to get thrown out. In any case, this is our final
    # result
    return BG

