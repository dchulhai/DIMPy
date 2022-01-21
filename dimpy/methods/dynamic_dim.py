import numpy as np
import scipy as sp

from .base import CalcMethodBase
from ..tools.memory import check_memory
from ..tools.timer import check_time
from ..tools.constants import HART2NM, NM2BOHR


class DIMr(CalcMethodBase):
    """A discrete interaction model (DIM) with
    field retardation effections.

    The changed methods are: :meth:`t2`

    See :class:`dimpy.methods.base.CalcMethodBase` for full documentation.

    **Example:** (this is the same as running the example in 
    ``DIMPy/examples/minimal_input_dim.dimpy``::

        >>> import dimpy
        >>> nano = dimpy.Nanoparticle('Ag 0 0 0; Ag 0 0 1.89', verbose=0, 
        >>>                           atom_params={'Ag': {'exp': 'Ag_jc'}})
        >>> nano.build()
        >>> calc = dimpy.DIMr(nano, freqs=0.0836, kdir=[1,0,0])
        >>> calc.run()
        >>> calc.isotropic_polarizabilities[0]
        (-32.781998+0.20258068j)

    """


    interaction = 'DIM'
    """Discrete Interaction Model"""

    model = 'PIM'
    """Polarizability Interaction Model"""

    @check_memory
    @check_time(log='once')
    def t2(self, omega=None, vector=None, **kwargs):
        '''The screened second-order interaction tensor.'''

        natoms = self.nanoparticle.natoms
        k = 2 * np.pi / NM2BOHR(HART2NM(omega))
        
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
        r3_inv = r_inv * r_inv * r_inv
        r_vec = r_vec.astype(np.float128)
        ikr = 1j * k * dists
        eikr = np.exp(ikr)
        
        # get the screening factor R
        # FIXME: 1.88973 is to convert Angstrom to Bohr
        R = (2/np.sqrt(np.pi)) * 1.88973 * self.nanoparticle.atomic_radii
        
        S = np.divide(dists, R, where=R!=0, dtype=np.float32)
        erf_S = sp.special.erf(S)
        S_cubed = S * S * S
        
        c=137.036 #speed of light
        c_sq = c*c 
        cSq_rSq = -c_sq*np.square(r_vec)
        
        t2 = np.zeros((natoms, 3, natoms, 3), dtype=np.complex64)
        sqrt_pi = np.sqrt(np.pi)
        
        #term2
        temp = (2 * S * eikr * r3_inv) / (sqrt_pi)
        t2[:,0,:,0] += ( temp * np.exp(-c_sq*r_vec[:,:,0]**2)
                       * np.exp(-c_sq*r_vec[:,:,0]**2) )
        t2[:,1,:,1] += ( temp * np.exp(-c_sq*r_vec[:,:,1]**2)
                       * np.exp(-c_sq*r_vec[:,:,1]**2) )
        t2[:,2,:,2] += ( temp * np.exp(-c_sq*r_vec[:,:,2]**2)
                       * np.exp(-c_sq*r_vec[:,:,2]**2) )
        
        #term6
        temp = (eikr*r3_inv*erf_S) 
        t2[:,0,:,0] += temp
        t2[:,1,:,1] += temp
        t2[:,2,:,2] += temp
        
        #term5
        temp *= ikr
        t2[:,0,:,0] += temp
        t2[:,1,:,1] += temp
        t2[:,2,:,2] += temp

        #term1
        temp = (4 * S_cubed * eikr * r3_inv * r_inv * r_inv) / sqrt_pi
        temp = np.einsum('ij,ija,ijb->iajb', temp, np.exp(cSq_rSq),np.exp(cSq_rSq))
        temp1 = np.einsum('iajb,ija,ijb->iajb', temp, r_vec, r_vec)
        t2 -= np.einsum('iajb,ija,ijb->iajb', temp, r_vec, r_vec)

        #term3
        temp = (4 * S * ikr * eikr * r3_inv * r_inv * r_inv) / sqrt_pi
        temp = np.einsum('ij,ija,ijb->iajb', temp, np.exp(cSq_rSq),np.exp(cSq_rSq))
        temp3 = np.einsum('iajb,ija,ijb->iajb', temp, r_vec, r_vec)
        t2 += np.einsum('iajb,ija,ijb->iajb', temp, r_vec, r_vec)

        #term4
        temp = (6 * S * eikr * r3_inv * r_inv * r_inv) / sqrt_pi
        temp = np.einsum('ij,ija,ijb->iajb', temp, np.exp(cSq_rSq),np.exp(cSq_rSq))
        temp4 = np.einsum('iajb,ija,ijb->iajb', temp, r_vec, r_vec)
        t2 += np.einsum('iajb,ija,ijb->iajb', temp, r_vec, r_vec)

        #term7
        temp = k * k * eikr * r3_inv * erf_S 
        temp7= np.einsum('ij,ija,ijb->iajb', temp, r_vec, r_vec)
        t2 -= np.einsum('ij,ija,ijb->iajb', temp, r_vec, r_vec)

        #term8
        temp = 3 * ikr * eikr * r3_inv * r_inv * r_inv * erf_S
        temp8 = np.einsum('ij,ija,ijb->iajb', temp, r_vec, r_vec)
        t2 -= np.einsum('ij,ija,ijb->iajb', temp, r_vec, r_vec)

        #term9
        temp = 3 * eikr * r3_inv * r_inv * r_inv * erf_S
        temp9 = np.einsum('ij,ija,ijb->iajb', temp, r_vec, r_vec)
        t2 += np.einsum('ij,ija,ijb->iajb', temp, r_vec, r_vec)

        return t2

