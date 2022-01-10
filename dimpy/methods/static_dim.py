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
    """Discrete Interaction Model"""

    model = 'PIM'
    """Polarizability Interaction Model"""

    @check_memory
    @check_time(log='once')
    def t0(self, **kwargs):
        r"""The screened zeroth-order interaction tensor.

        .. math::

            T^{(0)} = \frac{\operatorname{erf}{\left(S \right)}}{r}

        """

        if self._t0 is None:

            # FIXME: 1.88973 is to convert Angstrom to Bohr
            R = (2/np.sqrt(np.pi)) * 1.88973 * self.nanoparticle.atomic_radii
            dists = self.nanoparticle.distances
    
            S = np.divide(dists, R, where=R!=0, dtype=np.float32)
            t = np.divide(1, dists, out=np.zeros_like(dists),
                          where=dists!=0, dtype=np.float32)
            self._t0 = t * sp.special.erf(S)

        return self._t0


    @check_memory
    @check_time(log='once')
    def t1(self, **kwargs):
        r"""The screened first-order interaction tensor.

        .. math::

            T^{(1)}_\alpha = \frac{2 S r_{\alpha} e^{- S^{2}}}{\sqrt{\pi} r^{3}}
            - \frac{r_{\alpha} \operatorname{erf}{\left(S \right)}}{r^{3}}

        """

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


    @check_memory
    @check_time(log='once')
    def t2(self, vector=None, **kwargs):
        r"""The screened second-order interaction tensor.

        .. math::

            T^{(2)}_{\alpha\beta}
            = - \frac{4 S^{3} r_{\alpha} r_{\beta} e^{- S^{2}}}{\sqrt{\pi}
            r^{5}} + \frac{2 S \delta_{\alpha\beta} e^{- S^{2}}}{\sqrt{\pi} r^{3}}
            - \frac{6 S r_{\alpha} r_{\beta} e^{- S^{2}}}{\sqrt{\pi} r^{5}}
            - \frac{\delta_{\alpha\beta} \operatorname{erf}{\left(S \right)}}
            {r^{3}} + \frac{3 r_{\alpha} r_{\beta} \operatorname{erf}
            {\left(S \right)}}{r^{5}}

        """

        # get dists, r_vec, and r_inv for any unit cell
        if vector is None:
            dists = self.nanoparticle.distances
            r_vec = self.nanoparticle.r_vec
        else:
            r_vec = self.nanoparticle.r_vec + vector
            c_old = self.nanoparticle.coordinates
            c_new = c_old + vector
            dists = sp.spatial.distance.cdist(c_old, c_new)
        r_inv = np.divide(1, dists, where=dists!=0, dtype=np.float32)

        # get the screening factor R
        # FIXME: 1.88973 is to convert Angstrom to Bohr
        R = (2/np.sqrt(np.pi)) * 1.88973 * self.nanoparticle.atomic_radii

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

