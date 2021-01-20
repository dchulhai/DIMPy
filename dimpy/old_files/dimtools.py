def calc_bonds(coords, radii, scalefac):
    '''
    Calculate all the bonds in the set of coordinates.
    '''

    from numpy import array, triu_indices, fill_diagonal, argwhere
    from scipy.spatial.distance import cdist

    natoms = coords.shape[0]
    maxbonds = natoms * (natoms - 1) / 2

    # to same some comparison, let's only look at a triangle matrix
    indx = triu_indices(natoms)

    # calculate the distance between all points
    dist = cdist(coords, coords)

    # calculate the bond length threshold
    r = array([radii]*natoms)
    bondthres = scalefac * (r + r.transpose())

    # find the bonds
    lbonds = dist < bondthres
    fill_diagonal(lbonds, False) # an atom does not form a bond with itself
    bonds_indx = argwhere(lbonds[indx])
    bonds = array([indx[0][bonds_indx], indx[1][bonds_indx]]).transpose()[0]

    # free up some memory
    # (probably don't need this)
    del(dist)
    del(bondthres)
    del(lbonds)
    del(indx)

    # return bond indices
    return bonds

def minmax_pdist(coords):
    '''
    Calculate the minimum and maximum distance between two sets of points.
    '''

    from numpy import array, triu_indices, nanmax, nanmin, fill_diagonal, nan
    from scipy.spatial.distance import cdist

    # find distance between all points
    dist = cdist(coords, coords)

    # remove diagonal
    fill_diagonal(dist, nan)
    
    # find min and max
    mindist = nanmin(dist)
    maxdist = nanmax(dist)

    return array([mindist, maxdist])
