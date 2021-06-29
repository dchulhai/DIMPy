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
        >>> calc = dimpy.DDAr(nano, freqs=0.08360248, kdir='x')
        >>> calc.run()
        >>> calc.isotropic_polarizabilities[0]
        (8906.452+6817.6953j)

    """

    @check_memory(log='debug')
    @check_time(log='debug')
    def _kR(self, omega):
        """Get k dot R."""
        return self.nanoparticle.distances * self._kmag(omega)

    @check_memory(log='debug')
    @check_time(log='debug')
    def _kmag(self, omega):
        """Get the magnitude of the k-vector."""
        return 2 * np.pi / NM2BOHR(HART2NM(omega))

    @check_memory
    @check_time
    def A_matrix(self, omega=None):
        r"""Returns the A-matrix accounting for
        a dynamic electric field.

        .. math::

            A_{ij,\alpha\beta} = \begin{cases}
                \alpha_{\alpha\beta}^{-1}, & \text{if $i=j$} \\
                -\text{exp}(ikr) \left[
                \delta_{\alpha\beta}k^2 T^{(0)}_{ij}
                + k^2 r_\beta T^{(1)}_{ij,\alpha}
                + (1 - ikr) T^{(2)}_{ij,\alpha\beta} \right]
                & \text{if $i\neq j$}
            \end{cases}

        :param omega: Incident frequency, default None
        :type omega: float, optional

        :returns: The A-matrix
        :rtype: numpy.ndarray (complex)

        """

        natoms = self.nanoparticle.natoms

        k = self._kmag(omega)
        kR = k * self.nanoparticle.distances
        eikr = np.exp(1j * kR)

        fac0 = eikr * k * k
        fac1 = fac0[:,:,np.newaxis] * self.nanoparticle.r_vec
        fac2 = eikr * ( 1 - 1j * kR )

        # initialize A-matrix
        A_matrix = np.zeros((natoms, 3, natoms, 3), dtype=self._dtype)

        # get off-diagonal terms
        for a in range(3):
            A_matrix[:,a,:,a] = A_matrix[:,a,:,a] - fac0 * self.t0
            for b in range(3):
                A_matrix[:,a,:,b] = ( A_matrix[:,a,:,b] - fac1[:,:,a]
                                    * self.t1[:,:,b] )
        A_matrix = A_matrix - fac2[:,np.newaxis,:,np.newaxis] * self.t2

        # diagonal terms is the inverse polarizability
        pol_inv = 1 / self.nanoparticle.atomic_polarizabilities(omega)
        for i in range(natoms):
            A_matrix[i,:,i,:] = 0 
            for a in range(3):
                A_matrix[i,a,i,a] = pol_inv[i]

        # reshape to a 2-dimensional matrix and return
        self._A_matrix = A_matrix.reshape(natoms * 3, natoms * 3)
        return self._A_matrix


    @check_memory(log='debug')
    @check_time(log='debug')
    def _get_Einc(self, dimension, natoms, omega):
        """Get incident field for this direction."""
        E = np.zeros((natoms, 3), dtype=np.complex64)

        # kdir is perpendicular to the field direction,
        # as it should be
        # if kdir and the field direction are the same, then
        # return no field
        if self.kdir != dimension:
            # get k dot r
            k = 2 * np.pi / NM2BOHR(HART2NM(omega))
            kvec = np.zeros((3), dtype=np.float32)
            kvec[self.kdir] = k
            print (kvec)

            # calculate k_dot_r
            k_dot_r = np.dot(self.nanoparticle.coordinates, kvec)

            # incorporate that into the field
            E[:,dimension] = np.exp(1j * k_dot_r)

        # reshape and return E
        E = E.reshape(natoms * 3)
        return E


