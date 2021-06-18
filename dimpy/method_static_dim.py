import numpy as np
import scipy as sp

from .calc_method import CalcMethod
from .memory import check_memory
from .timer import check_time


class DIMs(CalcMethod):


    @property
    @check_memory
    @check_time(log='once')
    def t0(self):
        '''The modified zeroth-order interaction tensor.'''
        if self._t0 is None:

            dists = self.nanoparticle.distances
            self._t0 = np.divide(1, dists, out=np.zeros_like(dists),
                                 where=dists!=0, dtype=np.float32)

        return self._t0


    @property
    @check_memory
    @check_time(log='once')
    def t1(self):
        '''The modified first-order interaction tensor.'''
        if self._t1 is None:

            r_vec = self.nanoparticle.r_vec
            r_inv = self.t0
            r3_inv = r_inv * r_inv * r_inv
            self._t1 = -1.0 * r3_inv[:,:,np.newaxis] * r_vec

        return self._t1


    @property
    @check_memory
    @check_time(log='once')
    def t2(self):
        '''The modified second-order interaction tensor.'''
        if self._t2 is None:

            # do the off diagonal terms
            r_vec = self.nanoparticle.r_vec
            r_inv = self.t0
            r3_inv = r_inv * r_inv * r_inv
            r5_inv = r3_inv * r_inv * r_inv

            self._t2 = 3 * np.einsum('ij,ija,ijb->iajb', r5_inv, r_vec,
                                     r_vec, dtype=np.float32, casting='same_kind')
            self._t2[:,0,:,0] -= r3_inv
            self._t2[:,1,:,1] -= r3_inv
            self._t2[:,2,:,2] -= r3_inv

        return self._t2
