import numpy as np
import scipy as sp

from .base import CalcMethodBase
from ..tools.constants import HART2NM, NM2BOHR
from ..tools.memory import check_memory
from ..tools.timer import check_time

class DDAr(CalcMethodBase):
    """A discrete dipole approximation (DDA) method with
    field retardation effections.

    The changed methods are: :meth:`A_matrix` and :meth:`_get_Einc`

    See :class:`dimpy.methods.base.CalcMethodBase` for full documentation.

    **Example:** (same as ``DIMPy/examples/method_dda_retardation.dimpy``::

        >>> import dimpy
        >>> nano = dimpy.Nanoparticle('Au_147_ico.xyz', verbose=0,
        ...                           atom_params={'Au': {'exp': 'Au_jc'}})
        >>> # 'Au_147_ico.xyz' must be in the current directory
        >>> nano.build()
        >>> calc = dimpy.DDAr(nano, freqs=0.08360248, kdir=[1,0,0])
        >>> calc.run()
        >>> calc.isotropic_polarizabilities[0]
        (8905.193+6813j)

    """

    @check_memory
    @check_time(log='debug')
    def t2(self, omega=None, vector=None, **kwargs):

        natoms = self.nanoparticle.natoms
        k = 2 * np.pi / NM2BOHR(HART2NM(omega))

        # get r_vec and R for any unit cell
        if vector is None:
            R = self.nanoparticle.distances
            r_vec = self.nanoparticle.r_vec
        else:
            r_vec = self.nanoparticle.r_vec + vector
            c_old = self.nanoparticle.coordinates
            c_new = c_old + vector
            R = sp.spatial.distance.cdist(c_old, c_new)
        r_inv = np.divide(1, R, where=R!=0, dtype=np.float32)

        # get some common terms that are frequency used
        ikr = 1j * k * R
        eikr = np.exp(ikr)

        # initialize t2 matrix
        t2 = np.zeros((natoms, 3, natoms, 3), dtype=np.complex64)

        # generate t2 (diagonal parts)
        temp = eikr * r_inv * r_inv * r_inv
        t2[:,0,:,0] -= temp
        t2[:,1,:,1] -= temp
        t2[:,2,:,2] -= temp

        temp *= ikr
        t2[:,0,:,0] += temp
        t2[:,1,:,1] += temp
        t2[:,2,:,2] += temp

        # generate t2 (off-diagonal parts)
        temp = ikr * ikr * eikr * r_inv * r_inv * r_inv * r_inv * r_inv
        t2 += np.einsum('ij,ija,ijb->iajb', temp, r_vec, r_vec)

        temp = 3 * ikr * eikr * r_inv * r_inv * r_inv * r_inv * r_inv
        t2 -= np.einsum('ij,ija,ijb->iajb', temp, r_vec, r_vec)

        temp = 3 * eikr * r_inv * r_inv * r_inv * r_inv * r_inv
        t2 += np.einsum('ij,ija,ijb->iajb', temp, r_vec, r_vec)

        return t2

