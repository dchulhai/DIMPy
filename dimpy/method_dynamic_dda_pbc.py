import numpy as np
import scipy as sp

from .calc_method import CalcMethod
from .constants import HART2NM, NM2BOHR
from .memory import check_memory
from .method_dynamic_dda import DDAr
from .method_static_dda_pbc import DDAsPBC
from .timer import check_time

class DDArPBC(CalcMethod):
    """A discrete dipole approximation (DDA) method with
    field retardation effections and periodic boundary conditions.

    The changed methods are: :meth:`A_matrix` and :meth:`_get_Einc`

    See :class:`dimpy.calc_method.CalcMethod` for full documentation.

    """

    @check_memory(log='debug')
    @check_time(log='debug')
    def _kR(self, omega, vector=None, distances=None):
        k = 2 * np.pi / NM2BOHR(HART2NM(omega))
        if vector is None and distances is None:
            distances = self.nanoparticle.distances
        elif distances is None:
            distances = self._distances(vector)
        kR = k * distances
        return kR

    @check_memory
    @check_time(log='debug')
    def t0_vec(self, vector, distances=None):
        if distances is None:
            distances = self._distances(vector)
        return 1/distances

    @check_memory
    @check_time(log='debug')
    def t1_vec(self, vector, r_inv=None):
        if r_inv is None:
            r_inv = self.t0_vec(vector)
        r_vec = self.nanoparticle.r_vec + vector
        r3_inv = r_inv * r_inv * r_inv
        t1 = -1.0 * r3_inv[:,:,np.newaxis] * r_vec
        return t1

    t2_vec = DDAsPBC.t2_vec

    def _distances(self, vector):
        old_coordinates = self.nanoparticle.coordinates
        new_coordinates = old_coordinates + vector
        distances = sp.spatial.distance.cdist(old_coordinates, new_coordinates)
        return distances

    @check_memory(log='debug')
    @check_time(log='debug')
    def _kmag(self, omega):
        """Get the magnitude of the k-vector."""
        return 2 * np.pi / NM2BOHR(HART2NM(omega))

    @check_memory
    @check_time
    def A_matrix(self, omega=None):
        '''
        Returns the A-matrix accounting for a
        dynamic electric field and a periodically
        repeating unit cell in up to 2 dimensions.
        '''

        natoms = self.nanoparticle.natoms

        k = self._kmag(omega)
        kR = k * self.nanoparticle.distances
        eikr = np.exp(1j * kR)

        kvec = np.zeros((3), dtype=np.float32)
        kvec[self.kdir] = k

        fac0 = eikr * k * k 
        fac1 = fac0[:,:,np.newaxis] * self.nanoparticle.r_vec
        fac2 = eikr * ( 1 - 1j * kR )

        # initialize A-matrix
        A_matrix = np.zeros((natoms, 3, natoms, 3), dtype=self._dtype)

        # get off-diagonal terms
        for a in range(3):
            A_matrix[:,a,:,a] -= fac0 * self.t0
            for b in range(3):
                A_matrix[:,a,:,b] -= fac1[:,:,a] * self.t1[:,:,b]
        A_matrix -= fac2[:,np.newaxis,:,np.newaxis] * self.t2

        # diagonal terms is the inverse polarizability
        pol_inv = 1 / self.nanoparticle.atomic_polarizabilities(omega)
        for i in range(natoms):
            A_matrix[i,:,i,:] = 0 
            for a in range(3):
                A_matrix[i,a,i,a] = pol_inv[i]

        # get lattice vectors and max unit cells
        mvec = self.pbc[0]
        mmax = max(1, int(self.r_max_pbc / np.linalg.norm(self.pbc[0])))
        if len(self.pbc) > 1:
            nvec = self.pbc[1]
            nmax = max(1, int(self.r_max_pbc / np.linalg.norm(self.pbc[1])))
        else:
            nvec = np.zeros((3))
            nmax = 0

        # cycle over the first lattice vector
        for m in range(-mmax,mmax+1,1):

            for n in range(-nmax,nmax+1,1):

                # skip if this is the origin unit cell
                if m == 0 and n == 0:
                    continue

                # get new vector
                new_vec = m * mvec + n * nvec

                # skip if our distance is greater than needed
                dist = np.linalg.norm(new_vec)
                if dist > self.r_max_pbc:
                    continue

                distances = self._distances(new_vec)

                # calculate new interaction tensors
                t0 = self.t0_vec(new_vec, distances=distances)
                t1 = self.t1_vec(new_vec, r_inv=t0)
                t2 = self.t2_vec(new_vec, r_inv=t0)

                # get new radiative factors
                kR = k * distances
                eikr = np.exp(1j * kR) * np.exp(1j * np.dot(kvec, new_vec))

                fac0 = eikr * k * k 
                fac1 = ( fac0[:,:,np.newaxis]
                       * (self.nanoparticle.r_vec + new_vec) )
                fac2 = eikr * ( 1 - 1j * kR )

                # add these terms to the A-matrix
                for a in range(3):
                    A_matrix[:,a,:,a] -= fac0 * t0
                    for b in range(3):
                        A_matrix[:,a,:,b] -= fac1[:,:,a] * t1[:,:,b]
                A_matrix -= fac2[:,np.newaxis,:,np.newaxis] * t2

        # reshape to a 2-dimensional matrix and return
        self._A_matrix = A_matrix.reshape(natoms * 3, natoms * 3)
        return self._A_matrix

    _get_Einc = DDAr._get_Einc
