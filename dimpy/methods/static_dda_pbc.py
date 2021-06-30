import numpy as np
from numpy import linalg
import scipy as sp
from scipy import spatial

from .base import CalcMethodBase
from ..tools.memory import check_memory
from ..tools.timer import check_time

class DDAsPBC(CalcMethodBase):
    """A static (without retardation effects) discrete dipole
    approximation (DDA) method for periodic nanoparticles.

    See :class:`dimpy.methods.base.CalcMethodBase` for full documentation.

    **Example:** (this is the same as running the example in
    ``DIMPy/examples/gold_chain_dda.dimpy``::

        >>> import dimpy
        >>> nano = dimpy.Nanoparticle('Au 0 0 0', pbc=[[0, 0, 3.32]],
        ...        atom_params={'Au': {'exp': 'Au_jc', 'rad': 1.66}})
        >>> nano.verbose = 0
        >>> nano.build()
        >>> calc = dimpy.DDAsPBC(nano, freqs=0.10125189)
        >>> calc.run()
        >>> calc.isotropic_polarizabilities[0]
        (4.5884604+34.322826j)

    """


    @check_memory
    @check_time(log='debug')
    def t0_vec(self, vector):

        old_coordinates = self.nanoparticle.coordinates
        new_coordinates = old_coordinates + vector
        dist = sp.spatial.distance.cdist(old_coordinates, new_coordinates)
        return 1/dist

    @check_memory
    @check_time(log='debug')
    def t2_vec(self, vector, r_inv=None):

        # do the off diagonal terms
        r_vec = self.nanoparticle.r_vec + vector
        if r_inv is None:
            r_inv = self.t0_vec(vector)
        r3_inv = r_inv * r_inv * r_inv
        r5_inv = r3_inv * r_inv * r_inv

        t2 = 3 * np.einsum('ij,ija,ijb->iajb', r5_inv, r_vec,
                           r_vec, dtype=np.float32, casting='same_kind')
        t2[:,0,:,0] -= r3_inv
        t2[:,1,:,1] -= r3_inv
        t2[:,2,:,2] -= r3_inv

        return t2

    @check_memory
    @check_time
    def A_matrix(self, omega=None):
        '''
        Returns the A-matrix accounting for a
        dynamic electric field and a periodically
        repeating unit cell in up to 2 dimensions.
        '''

        natoms = self.nanoparticle.natoms
        pol_inv = 1 / self.nanoparticle.atomic_polarizabilities(omega)

        # create the A matrix for the origin unit cell
        A_matrix = np.zeros((natoms, 3, natoms, 3), dtype=self._dtype)
        A_matrix -= self.t2

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

                # calculate the interaction and add to the A-matrix
                t2 = self.t2_vec(new_vec)
                A_matrix -= t2

        self._A_matrix = A_matrix.reshape(natoms * 3, natoms * 3)

        return self._A_matrix
