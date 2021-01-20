
"""A cubic spline function written in pure python to avoid dependencies on
non-standard libraries such as numpy or scipy.

Based on algorithms found in "Numerical Methods in Engineering with Python" by
Jaan Kiusalaas and in "Numerical Recipies".  Adapted by Seth M. Morton."""

def spline(x, y):
    """Splines the given data and returns the knots of the cubic spline.
    This spline is "natural", as in the bounary derivative is zero."""

    #####################################################
    # First set up the three bands of the matrix to solve
    #####################################################

    # Get the length of the data
    n = len(x) - 1

    # Set up the tridiagonal equations
    c = [x[i] - x[i-1] for i in range(1,n)] + [0.0]
    c[0] = 0.0
    d = [1.0] + [2.0 * ( x[i+1] - x[i-1] ) for i in range(1,n)] + [1.0]
    e = [0.0] + [x[i+1] - x[i] for i in range(1,n)]

    # Set up the knots to solve for
    k = [0.0] + [6.0 * ( ( y[i+1] - y[i] ) / ( x[i+1] - x[i] ) 
                       - ( y[i] - y[i-1] ) / ( x[i] - x[i-1] ) )
                                                 for i in range(1,n)] + [0.0]

    ##########################################################
    # Solve the tridiagonal system to return the knots
    # This is like a pure python drop-in for dgtsv from LAPACK
    ##########################################################

    # Get the diagonal length
    n += 1

    # Decompose the matrix
    for i in range(1, n):
        lam    = c[i-1] / d[i-1]
        d[i]  -= lam * e[i-1]
        c[i-1] = lam

    # Use the decomposed matrix to solve for the knots
    for i in range(1,n):
        k[i] = k[i] - c[i-1] * k[i-1]
    k[n-1] = k[n-1] / d[n-1]
    for i in range(n-2, -1, -1):
        k[i] = ( k[i] - e[i] * k[i+1] ) / d[i]

    return k

def interpolate(x, y, knots, xvalue):
    """Given the raw x and y data and the knots for that data, interpolate and
    return the y value associated with the requested x value"""

    ########################################################
    # Determine the index range between which our value lies
    ########################################################

    # The first two choices are for when the value matches our extremma
    if xvalue == x[0]:
        i = 0
    elif xvalue == x[-1]:
        i = len(x) - 2
    else:
        # Determine bounds
        ascending = x[-1] >= x[0]
        iLeft = -1
        iRight = len(x)

        # Find the index for the value immeadiately
        # below or equal to the requested value
        while iRight - iLeft > 1:
            # Compute a midpoint, and replace either the lower limit
            # or the upper limit, as appropriate.
            iMid = ( iLeft + iRight ) // 2
            if ascending == ( xvalue >= x[iMid] ):
                iLeft = iMid
            else:
                iRight = iMid

        # iLeft is the index
        i = iLeft

    # Make sure the index falls in the correct window
    i = max(min(i, len(x)-2), 0) 
    
    ###################################
    # Interpolate using the given knots
    ###################################

    h = x[i+1] - x[i]
    a = ( x[i+1] - xvalue ) / h
    b = ( xvalue - x[i] )   / h
    return ( a * y[i] + b * y[i+1] 
         + ( ( a**3 - a ) * knots[i] + ( b**3 - b ) * knots[i+1] )
         * ( h**2 ) / 6.0 )
