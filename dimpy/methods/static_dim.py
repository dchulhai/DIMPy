import numpy as np
import scipy as sp

from .base import CalcMethodBase
from ..tools.memory import check_memory
from ..tools.timer import check_time


class DIMs(CalcMethodBase):
    """A static (without retardation effects) discrete interaction
    model (DIM).

    The changed methods are: :meth:`t0`, :meth:`t1`, :meth:`t2`

    See :class:`dimpy.methods.base.CalcMethodBase` for full documentation.

    **Example:** (this is the same as running the example in 
    ``DIMPy/examples/minimal_input_dim.dimpy``::

        >>> import dimpy
        >>> nano = dimpy.Nanoparticle('Ag 0 0 0; Ag 0 0 1.89', verbose=0)
        >>> nano.build()
        >>> calc = dimpy.DIMs(nano)
        >>> calc.run()
        >>> calc.isotropic_polarizabilities[0]
        99.193596

    """


    interaction = 'DIM'
    """Interaction type for this class."""

    @property
    @check_memory
    @check_time(log='once')
    def t0(self):
        '''The screened zeroth-order interaction tensor.'''
        if self._t0 is None:

            # FIXME: 1.88973 is to convert Angstrom to Bohr
            R = (2/np.sqrt(np.pi)) * 1.88973 * self.nanoparticle.atomic_radii
            dists = self.nanoparticle.distances
    
            S = np.divide(dists, R, where=R!=0, dtype=np.float32)
            t = np.divide(1, dists, out=np.zeros_like(dists),
                          where=dists!=0, dtype=np.float32)
            self._t0 = t * sp.special.erf(S)

        return self._t0


    @property
    @check_memory
    @check_time(log='once')
    def t1(self):
        '''The screened first-order interaction tensor.'''
        if self._t1 is None:

            # FIXME: 1.88973 is to convert Angstrom to Bohr
            R = (2/np.sqrt(np.pi)) * 1.88973 * self.nanoparticle.atomic_radii
            r_vec = self.nanoparticle.r_vec
            r_inv = self.t0
            r3_inv = r_inv * r_inv * r_inv
    
            t = -1.0 * r3_inv[:,:,np.newaxis] * r_vec
    
            S = np.divide(dists, R, where=R!=0, dtype=np.float32)
            S_sq = S * S
            self._t1 = t * ( sp.special.erf(S) - ((2/np.sqrt(np.pi))
                           * S * np.exp(-S_sq)) )

        return self._t1


    @property
    @check_memory
    @check_time(log='once')
    def t2(self):
        '''The screened second-order interaction tensor.'''
        if self._t2 is None:

            # FIXME: 1.88973 is to convert Angstrom to Bohr
            R = (2/np.sqrt(np.pi)) * 1.88973 * self.nanoparticle.atomic_radii
            dists = self.nanoparticle.distances
            r_vec = self.nanoparticle.r_vec
            r_inv = np.divide(1, dists, out=np.zeros_like(dists),
                              where=dists!=0, dtype=np.float32)
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

            self._t2 = p1 - p2

        return self._t2

