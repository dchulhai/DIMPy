import numpy as np
import scipy as sp

from .calc_method import CalcMethod
from .memory import check_memory
from .timer import check_time

class DDAr(CalcMethod):

    @check_memory(log='debug')
    @check_time(log='debug')
    def _kR(self, omega):
        k = 2 * np.pi * omega / 137.036 # 137.036 is the speed of light in a.u.
        kR = self.nanoparticle.distances * k
        return kR

    @check_memory
    @check_time
    def A_matrix(self, omega=None):
        '''Returns the A-matrix accounting for
        a dynamic electric field.'''

        natoms = self.nanoparticle.natoms
        pol_inv = 1 / self.nanoparticle.atomic_polarizabilities(omega)

        # calculate radiative components
        kR = self._kR(omega)
        temp = (1 - 1j * kR - kR * kR / 3)
        temp1 = np.exp(1j * kR) * temp
        del(temp)

        r_inv = self.t0
        temp = 2 * kR * kR * r_inv * r_inv * r_inv / 3
        temp2 = np.exp(1j * kR) * temp
        del(temp)

        A_matrix = np.zeros_like(self.t2, dtype=self._dtype)
        for a in range(3):
            A_matrix[:,a,:,a] -= temp2
            for b in range(3):
                A_matrix[:,a,:,b] -= temp1 * self.t2[:,a,:,b]

        for i in range(natoms):
            A_matrix[i,:,i,:] = 0 
            for a in range(3):
                A_matrix[i,a,i,a] = pol_inv[i]

        self._A_matrix = A_matrix.reshape(natoms * 3, natoms * 3)

        self.log('Amat Size (MB): {0:.1f}'.format(self._A_matrix.nbytes / (1024)**2))

        return self._A_matrix


    @check_memory(log='debug')
    @check_time(log='debug')
    def _get_Einc(self, dimension, natoms, omega):
        '''Get incident field for this direction.'''
        E = np.zeros((natoms, 3), dtype=np.complex64)

        # get k dot r
        k = 2 * np.pi * omega / 137.036 # 137 is speed of light
        kvec = np.zeros((3), dtype=np.float32)
        kvec[self.kdir] = k

        # calculate k_dot_r
        k_dot_r = np.dot(self.nanoparticle.coordinates, kvec)

        # incorporate that into the field
        E[:,dimension] = np.exp(1j * k_dot_r)

        # reshape and return E
        E = E.reshape(natoms * 3)
        return E


