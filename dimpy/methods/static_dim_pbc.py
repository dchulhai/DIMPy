import numpy as np
from numpy import linalg
import scipy as sp
from scipy import spatial

from .base import CalcMethodBase
from .static_dda_pbc import DDAsPBC
from ..tools.memory import check_memory
from ..tools.timer import check_time

class DIMsPBC(CalcMethodBase):
    """A static (without retardation effects) discrete dipole
    approximation (DDA) method for periodic nanoparticles.

    See :class:`dimpy.methods.base.CalcMethodBase` for full documentation.

    **Example:** (this is the same as running the example in
    ``DIMPy/examples/gold_chain_dim.dimpy``::

        >>> import dimpy
        >>> nano = dimpy.Nanoparticle('Au 0 0 0', pbc=[[0, 0, 3.32]],
        ...        atom_params={'Au': {'exp': 'Au_jc', 'rad': 1.66}})
        >>> nano.verbose = 0
        >>> nano.build()
        >>> calc = dimpy.DIMsPBC(nano, freqs=0.10125189)
        >>> calc.run()
        >>> calc.isotropic_polarizabilities[0]
        (24.582722+53.31772j)

    """

    interaction = 'DIM'
    """This is a discrete interaction model calculation."""

    @check_memory
    @check_time(log='debug')
    def t0_vec(self, vector):

        old_coordinates = self.nanoparticle.coordinates
        new_coordinates = old_coordinates + vector
        dists = sp.spatial.distance.cdist(old_coordinates, new_coordinates)

        # FIXME: 1.88973 is to convert Angstrom to Bohr
        R = (2/np.sqrt(np.pi)) * 1.88973 * self.nanoparticle.atomic_radii
        S = np.divide(dists, R, where=R!=0, dtype=np.float32)
        t = np.divide(1, dists, out=np.zeros_like(dists),
                      where=dists!=0, dtype=np.float32)
        return t * sp.special.erf(S)

    @check_memory
    @check_time(log='debug')
    def t2_vec(self, vector, r_inv=None):

        old_coordinates = self.nanoparticle.coordinates
        new_coordinates = old_coordinates + vector
        dists = sp.spatial.distance.cdist(old_coordinates, new_coordinates)

        # FIXME: 1.88973 is to convert Angstrom to Bohr
        R = (2/np.sqrt(np.pi)) * 1.88973 * self.nanoparticle.atomic_radii

        # do the off diagonal terms
        r_vec = self.nanoparticle.r_vec + vector
        if r_inv is None:
             r_inv = 1/dists
#            r_inv = self.t0_vec(vector)
        r3_inv = r_inv * r_inv * r_inv
        r5_inv = r3_inv * r_inv * r_inv

        S = np.divide(dists, R, where=R!=0, dtype=np.float32)
        S_cubed = S * S * S
        S_sq = S * S 

        einsum = np.einsum('ij,ijk,ijl->ikjl', r5_inv, r_vec,
                           r_vec, dtype=np.float32, casting='same_kind')
        t2 = 3 * einsum
        t2[:,0,:,0] -= r3_inv
        t2[:,1,:,1] -= r3_inv
        t2[:,2,:,2] -= r3_inv

        #t1 erf
        t1_erf = (sp.special.erf(S) - 
                  ((2/np.sqrt(np.pi)) * S * np.exp(-S_sq)))

        #multiplying t1 erf and t2
        p1 = np.einsum('ij,ikjl->ikjl', t1_erf, t2, 
                       dtype=np.float32, casting='same_kind')

        p2 = (4/np.sqrt(np.pi)) * np.einsum('ij,ij,ikjl->ikjl',
             S_cubed, np.exp(-S_sq), einsum, dtype=np.float32,
             casting='same_kind')

        return p1 - p2


    # use A-matrix from DDAsPBC
    A_matrix = DDAsPBC.A_matrix

